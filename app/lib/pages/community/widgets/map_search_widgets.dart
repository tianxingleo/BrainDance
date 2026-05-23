import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_map/flutter_map.dart';

import '../amap_search.dart';

/// 顶部地图搜索输入栏。回车触发 `onSubmitted`，可清空。
class MapSearchBar extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool loading;
  final bool hasKeyword;
  final ValueChanged<String> onSubmitted;
  final VoidCallback onClear;
  final VoidCallback onFocusResults;
  final String hint;

  const MapSearchBar({
    super.key,
    required this.controller,
    required this.focusNode,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.loading,
    required this.hasKeyword,
    required this.onSubmitted,
    required this.onClear,
    required this.onFocusResults,
    this.hint = '搜索地点（回车确认）',
  });

  @override
  Widget build(BuildContext context) {
    return BDPanelCard(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Row(
        children: [
          Icon(Icons.search_rounded, color: hintColor, size: 22),
          const SizedBox(width: 10),
          Expanded(
            child: TextField(
              controller: controller,
              focusNode: focusNode,
              style: TextStyle(color: textColor, fontSize: 15),
              cursorColor: BDDesign.colorMutedBlue,
              textInputAction: TextInputAction.search,
              maxLength: 60,
              maxLengthEnforcement: MaxLengthEnforcement.enforced,
              onSubmitted: onSubmitted,
              onTap: onFocusResults,
              decoration: InputDecoration(
                isDense: true,
                counterText: '',
                border: InputBorder.none,
                hintText: hint,
                hintStyle: TextStyle(color: hintColor, fontSize: 14.5),
              ),
            ),
          ),
          if (loading)
            SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                valueColor: AlwaysStoppedAnimation(BDDesign.colorMutedBlue),
              ),
            )
          else if (hasKeyword)
            IconButton(
              tooltip: '清空搜索',
              splashRadius: 18,
              onPressed: onClear,
              icon: Icon(Icons.close_rounded, color: hintColor, size: 20),
            ),
        ],
      ),
    );
  }
}

/// 高亮的搜索结果定位 pin。叠加在 [FlutterMap] 上方，用区分色避免与社区标记混淆。
class SearchPinLayer extends StatelessWidget {
  final AmapPoi poi;
  const SearchPinLayer({super.key, required this.poi});  @override
  Widget build(BuildContext context) {
    return MarkerLayer(
      markers: [
        Marker(
          point: poi.location,
          width: 220,
          height: 78,
          alignment: Alignment.topCenter,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                constraints: const BoxConstraints(maxWidth: 220),
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: BDDesign.colorMutedBlue,
                  borderRadius: BorderRadius.circular(999),
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x33000000),
                      blurRadius: 6,
                      offset: Offset(0, 2),
                    ),
                  ],
                ),
                child: Text(
                  poi.name.isEmpty ? '搜索结果' : poi.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 2),
              Icon(
                Icons.location_on_rounded,
                color: BDDesign.colorMutedBlue,
                size: 38,
                shadows: const [
                  Shadow(
                    color: Color(0x55000000),
                    blurRadius: 6,
                    offset: Offset(0, 2),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// 搜索结果浮层。挂在地图上方，loading / error / empty / list 四种状态。
class SearchResultsOverlay extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool loading;
  final String? error;
  final List<AmapPoi> results;
  final ValueChanged<AmapPoi> onTap;
  final VoidCallback onClose;
  final double maxHeightFactor;

  const SearchResultsOverlay({
    super.key,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.loading,
    required this.error,
    required this.results,
    required this.onTap,
    required this.onClose,
    this.maxHeightFactor = 0.5,
  });

  @override
  Widget build(BuildContext context) {
    final maxH = MediaQuery.of(context).size.height * maxHeightFactor;
    return ConstrainedBox(
      constraints: BoxConstraints(maxHeight: maxH),
      child: BDPanelCard(
        padding: EdgeInsets.zero,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 6, 6),
              child: Row(
                children: [
                  Icon(Icons.place_rounded, color: hintColor, size: 18),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _headerText(),
                      style: TextStyle(
                        color: textColor,
                        fontSize: 13.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  IconButton(
                    tooltip: '关闭',
                    splashRadius: 18,
                    onPressed: onClose,
                    icon: Icon(Icons.close_rounded,
                        color: hintColor, size: 18),
                  ),
                ],
              ),
            ),
            Divider(
              height: 1,
              color: hintColor.withValues(alpha: 0.18),
            ),
            Flexible(child: _buildBody()),
          ],
        ),
      ),
    );
  }

  String _headerText() {
    if (loading) return '正在搜索…';
    if (error != null) return '搜索失败';
    if (results.isEmpty) return '没有匹配结果';
    return '共 ${results.length} 条结果';
  }

  Widget _buildBody() {
    if (loading) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 30),
        child: Center(
          child: SizedBox(
            width: 22,
            height: 22,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: AlwaysStoppedAnimation(BDDesign.colorMutedBlue),
            ),
          ),
        ),
      );
    }
    if (error != null) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 18, 16, 22),
        child: Text(
          error!,
          style: TextStyle(color: hintColor, fontSize: 13.5, height: 1.45),
        ),
      );
    }
    if (results.isEmpty) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 18, 16, 22),
        child: Text(
          '试试更具体的地名，或换一个关键词。',
          style: TextStyle(color: hintColor, fontSize: 13.5, height: 1.45),
        ),
      );
    }
    return ListView.separated(
      shrinkWrap: true,
      padding: EdgeInsets.zero,
      itemCount: results.length,
      separatorBuilder: (ctx, idx) => Divider(
        height: 1,
        indent: 16,
        endIndent: 16,
        color: hintColor.withValues(alpha: 0.12),
      ),
      itemBuilder: (ctx, i) {
        final poi = results[i];
        final region = poi.regionLabel;
        final subParts = <String>[];
        if (region.isNotEmpty) subParts.add(region);
        if (poi.address.isNotEmpty) subParts.add(poi.address);
        return InkWell(
          onTap: () => onTap(poi),
          child: Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.location_on_outlined,
                    color: BDDesign.colorMutedBlue, size: 20),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        poi.name.isEmpty ? '(未命名地点)' : poi.name,
                        style: TextStyle(
                          color: textColor,
                          fontSize: 14.5,
                          fontWeight: FontWeight.w600,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (subParts.isNotEmpty) ...[
                        const SizedBox(height: 3),
                        Text(
                          subParts.join(' · '),
                          style: TextStyle(
                            color: hintColor,
                            fontSize: 12.5,
                            height: 1.35,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ],
                  ),
                ),
                Icon(Icons.chevron_right_rounded,
                    color: hintColor, size: 20),
              ],
            ),
          ),
        );
      },
    );
  }
}
