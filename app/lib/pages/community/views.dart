import 'dart:math' as math;

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/community/map_marker.dart';
import 'package:braindance/pages/community/map_page.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:braindance/widgets/animated_network_image.dart';
import 'package:flutter/material.dart';

import 'models.dart';

// ============================================================
// Tab 1: 推荐 — 地图 + 模型列表
// ============================================================

class CommunityRecommendView extends StatelessWidget {
  final List<CommunityPost> posts;
  final int totalPosts;
  final int viewportPosts;
  final CommunityMapViewport mapViewport;
  final List<CommunityMapMarker> mapMarkers;
  final VoidCallback onOpenMap;
  final ValueChanged<CommunityPost> onTapPost;
  final List<String> availableTags;
  final String? selectedTag;
  final ValueChanged<String> onToggleTag;
  final VoidCallback onClearFilters;
  final double tagRadiusKm;

  const CommunityRecommendView({
    super.key,
    required this.posts,
    required this.mapViewport,
    required this.onOpenMap,
    required this.onTapPost,
    this.totalPosts = 0,
    this.viewportPosts = 0,
    this.mapMarkers = const [],
    this.availableTags = const [],
    this.selectedTag,
    required this.onToggleTag,
    required this.onClearFilters,
    this.tagRadiusKm = 0,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    if (totalPosts == 0) return const _CommunityEmptyState();

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
      child: Column(
        children: [
          // 地图面板
          BDPanelCard(
            padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'World Memory Map',
                            style: TextStyle(
                              color: textColor,
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            textLocalize('community_map_hint'),
                            style: TextStyle(color: hintColor, height: 1.35),
                          ),
                        ],
                      ),
                    ),
                    BDStatusPill(
                      label: 'ZOOM ${mapViewport.zoom}',
                      icon: Icons.public_rounded,
                      color: BDDesign.colorMutedBlue,
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                LayoutBuilder(
                  builder: (context, constraints) {
                    final mapWidth = constraints.maxWidth;
                    final mapHeight = math.max(200.0, mapWidth * 0.48);
                    return GestureDetector(
                      onTap: onOpenMap,
                      child: SizedBox(
                        height: mapHeight,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(26),
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              CommunityAmapPreview(
                                viewport: mapViewport,
                                width: mapWidth.round().clamp(320, 1024),
                                height: mapHeight.round().clamp(200, 1024),
                                markers: mapMarkers,
                              ),
                              Positioned(
                                right: 12,
                                top: 12,
                                child: DecoratedBox(
                                  decoration: BoxDecoration(
                                    color: (isDark
                                            ? Colors.black
                                            : Colors.white)
                                        .withValues(alpha: 0.82),
                                    borderRadius: BorderRadius.circular(999),
                                    border: Border.all(
                                      color: isDark
                                          ? Colors.white.withValues(alpha: 0.08)
                                          : BDDesign.colorMutedBlue
                                              .withValues(alpha: 0.10),
                                    ),
                                  ),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 12,
                                      vertical: 8,
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(
                                          Icons.open_in_full_rounded,
                                          color: hintColor,
                                          size: 16,
                                        ),
                                        const SizedBox(width: 6),
                                        Text(
                                          '调整地图',
                                          style: TextStyle(
                                            color: hintColor,
                                            fontWeight: FontWeight.w700,
                                            fontSize: 12.5,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          // 标签筛选栏
          _ExploreFilterBar(
            isDark: isDark,
            textColor: textColor,
            hintColor: hintColor,
            boundsReady: mapViewport.bounds != null,
            zoom: mapViewport.zoom,
            tagRadiusKm: tagRadiusKm,
            visibleCount: viewportPosts,
            totalCount: totalPosts,
            filteredCount: posts.length,
            availableTags: availableTags,
            selectedTag: selectedTag,
            onToggleTag: onToggleTag,
            onClearFilters: onClearFilters,
          ),
          const SizedBox(height: 12),
          // 模型列表
          if (posts.isEmpty)
            _ExploreEmptyHint(
              isDark: isDark,
              textColor: textColor,
              hintColor: hintColor,
              hasTag: selectedTag != null && selectedTag!.isNotEmpty,
              onClearFilters: onClearFilters,
            )
          else
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: posts.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, index) {
                final post = posts[index];
                return _RecommendPostCard(
                  post: post,
                  isDark: isDark,
                  textColor: textColor,
                  hintColor: hintColor,
                  onTap: () => onTapPost(post),
                );
              },
            ),
        ],
      ),
    );
  }
}

class _RecommendPostCard extends StatelessWidget {
  final CommunityPost post;
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final VoidCallback onTap;

  const _RecommendPostCard({
    required this.post,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BDDesign.radiusLarge,
      onTap: onTap,
      child: BDPanelCard(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            _CommunityThumbnail(
              imageUrl: post.coverUrl,
              height: 100,
              width: 88,
              icon: Icons.terrain_rounded,
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    post.placeName,
                    style: TextStyle(
                      color: textColor,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${post.latitude.toStringAsFixed(3)}, ${post.longitude.toStringAsFixed(3)}',
                    style: TextStyle(color: hintColor, fontSize: 12.5),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    post.title,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: textColor,
                      fontWeight: FontWeight.w700,
                      height: 1.2,
                    ),
                  ),
                  const SizedBox(height: 8),
                  BDStatusPill(
                    label: post.modelName,
                    icon: Icons.view_in_ar_rounded,
                    color: BDDesign.colorMutedBlue,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// Tab 2: 探索 — 搜索框 + 历史/推荐 + 搜索结果
// ============================================================

class CommunityExploreView extends StatefulWidget {
  final List<CommunityPost> posts;
  final List<String> searchHistory;
  final List<String> recommendedKeywords;
  final ValueChanged<String> onSearch;
  final VoidCallback onClearHistory;
  final ValueChanged<CommunityPost> onTapPost;

  const CommunityExploreView({
    super.key,
    required this.posts,
    required this.searchHistory,
    required this.recommendedKeywords,
    required this.onSearch,
    required this.onClearHistory,
    required this.onTapPost,
  });

  @override
  State<CommunityExploreView> createState() => _CommunityExploreViewState();
}

class _CommunityExploreViewState extends State<CommunityExploreView> {
  final _searchController = TextEditingController();
  final _focusNode = FocusNode();
  String _query = '';

  @override
  void dispose() {
    _searchController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _submit(String value) {
    final q = value.trim();
    if (q.isEmpty) return;
    setState(() => _query = q);
    widget.onSearch(q);
  }

  void _clear() {
    _searchController.clear();
    setState(() => _query = '');
    _focusNode.unfocus();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);
    final inputFill =
        isDark ? AppTheme.darkSurfaceElevated : const Color(0xFFF3F5F9);

    final hasQuery = _query.isNotEmpty;
    final posts = hasQuery
        ? widget.posts
            .where((p) =>
                p.title.contains(_query) ||
                p.modelName.contains(_query) ||
                p.caption.contains(_query) ||
                p.placeName.contains(_query) ||
                p.tags.any((t) => t.contains(_query)))
            .toList()
        : widget.posts;

    return Column(
      children: [
        // 搜索框
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 0),
          child: BDPanelCard(
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
            child: TextField(
              controller: _searchController,
              focusNode: _focusNode,
              onSubmitted: _submit,
              onChanged: (v) {
                if (v.trim().isEmpty && _query.isNotEmpty) {
                  setState(() => _query = '');
                }
              },
              decoration: InputDecoration(
                hintText: textLocalize('community_search_placeholder'),
                hintStyle: TextStyle(color: hintColor, fontSize: 14),
                prefixIcon:
                    Icon(Icons.search_rounded, color: hintColor, size: 20),
                suffixIcon: _query.isNotEmpty
                    ? IconButton(
                        icon: Icon(Icons.close_rounded, color: hintColor, size: 18),
                        onPressed: _clear,
                      )
                    : null,
                filled: true,
                fillColor: inputFill,
                border: OutlineInputBorder(
                  borderRadius: BDDesign.radiusLarge,
                  borderSide: BorderSide.none,
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 12,
                ),
              ),
            ),
          ),
        ),
        // 内容区
        Expanded(
          child: hasQuery
              ? _buildSearchResults(isDark, textColor, hintColor, posts)
              : _buildSuggestions(isDark, textColor, hintColor),
        ),
      ],
    );
  }

  Widget _buildSuggestions(bool isDark, Color textColor, Color hintColor) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 104),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 搜索历史
          if (widget.searchHistory.isNotEmpty) ...[
            Row(
              children: [
                Expanded(
                  child: Text(
                    textLocalize('community_search_history'),
                    style: TextStyle(
                      color: textColor,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                GestureDetector(
                  onTap: widget.onClearHistory,
                  child: Text(
                    textLocalize('community_search_clear_history'),
                    style: TextStyle(color: hintColor, fontSize: 12),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: widget.searchHistory.map((keyword) {
                return _SuggestionChip(
                  label: keyword,
                  icon: Icons.history_rounded,
                  isDark: isDark,
                  onTap: () {
                    _searchController.text = keyword;
                    _submit(keyword);
                  },
                );
              }).toList(),
            ),
            const SizedBox(height: 24),
          ],
          // 推荐关键词
          Text(
            textLocalize('community_search_recommended'),
            style: TextStyle(
              color: textColor,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: widget.recommendedKeywords.map((keyword) {
              return _SuggestionChip(
                label: keyword,
                icon: Icons.trending_up_rounded,
                isDark: isDark,
                onTap: () {
                  _searchController.text = keyword;
                  _submit(keyword);
                },
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchResults(
    bool isDark,
    Color textColor,
    Color hintColor,
    List<CommunityPost> results,
  ) {
    if (results.isEmpty) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 104),
        child: BDPanelCard(
          padding: const EdgeInsets.all(24),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.search_off_rounded, size: 48, color: hintColor),
                const SizedBox(height: 14),
                Text(
                  textLocalize('community_search_no_results'),
                  style: TextStyle(
                    color: textColor,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  textLocalize('community_search_try_other'),
                  style: TextStyle(color: hintColor, fontSize: 13),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 104),
      itemCount: results.length,
      separatorBuilder: (_, __) => const SizedBox(height: 10),
      itemBuilder: (context, index) {
        final post = results[index];
        return InkWell(
          borderRadius: BDDesign.radiusLarge,
          onTap: () => widget.onTapPost(post),
          child: BDPanelCard(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                _CommunityThumbnail(
                  imageUrl: post.coverUrl,
                  height: 80,
                  width: 72,
                  icon: Icons.terrain_rounded,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        post.title,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: textColor,
                          fontWeight: FontWeight.w700,
                          height: 1.25,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        '${post.placeName} · ${post.modelName}',
                        style: TextStyle(color: hintColor, fontSize: 12.5),
                      ),
                    ],
                  ),
                ),
                Icon(Icons.chevron_right_rounded, color: hintColor, size: 20),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _SuggestionChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool isDark;
  final VoidCallback onTap;

  const _SuggestionChip({
    required this.label,
    required this.icon,
    required this.isDark,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: BDMotion.durationFast,
        curve: BDMotion.curveFluid,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: isDark
              ? AppTheme.darkSurfaceElevated
              : const Color(0xFFF3F5F9),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isDark
                ? Colors.white.withValues(alpha: 0.08)
                : BDDesign.colorMutedBlue.withValues(alpha: 0.12),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: isDark ? Colors.white70 : BDDesign.colorMutedBlue),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// Tab 3: 投稿 — 搜索模型 + 紧凑选择 + 表单 + 发布
// ============================================================

class CommunitySubmitView extends StatefulWidget {
  final List<CommunityModelOption> shareableModels;
  final List<CommunityModelOption> selectedModels;
  final ValueChanged<CommunityModelOption> onToggleModel;
  final TextEditingController titleController;
  final TextEditingController captionController;
  final TextEditingController placeController;
  final TextEditingController latController;
  final TextEditingController lngController;
  final bool isSubmitting;
  final VoidCallback onSubmit;
  final VoidCallback onSaveDraft;

  static const _presets = <_LocationPreset>[
    _LocationPreset('西湖', 30.243, 120.150),
    _LocationPreset('外滩', 31.240, 121.490),
    _LocationPreset('东京塔', 35.659, 139.745),
    _LocationPreset('巴黎左岸', 48.853, 2.349),
    _LocationPreset('纽约中央公园', 40.782, -73.965),
  ];

  const CommunitySubmitView({
    super.key,
    required this.shareableModels,
    required this.selectedModels,
    required this.onToggleModel,
    required this.titleController,
    required this.captionController,
    required this.placeController,
    required this.latController,
    required this.lngController,
    required this.isSubmitting,
    required this.onSubmit,
    required this.onSaveDraft,
  });

  @override
  State<CommunitySubmitView> createState() => _CommunitySubmitViewState();
}

class _CommunitySubmitViewState extends State<CommunitySubmitView> {
  final _modelSearchController = TextEditingController();
  String _modelQuery = '';
  bool _hasAttemptedSubmit = false;

  @override
  void dispose() {
    _modelSearchController.dispose();
    super.dispose();
  }

  List<CommunityModelOption> get _filteredModels {
    if (_modelQuery.isEmpty) return widget.shareableModels;
    final q = _modelQuery.toLowerCase();
    return widget.shareableModels
        .where((m) =>
            m.sceneId.toLowerCase().contains(q) ||
            m.description.toLowerCase().contains(q))
        .toList();
  }

  bool get _isTitleValid => widget.titleController.text.trim().isNotEmpty;
  bool get _isCaptionValid => widget.captionController.text.trim().isNotEmpty;
  bool get _isPlaceValid => widget.placeController.text.trim().isNotEmpty;
  bool get _isLatValid =>
      double.tryParse(widget.latController.text.trim()) != null;
  bool get _isLngValid =>
      double.tryParse(widget.lngController.text.trim()) != null;
  bool get _isModelValid => widget.selectedModels.isNotEmpty;

  bool _validate() {
    setState(() => _hasAttemptedSubmit = true);
    return _isModelValid &&
        _isTitleValid &&
        _isCaptionValid &&
        _isPlaceValid &&
        _isLatValid &&
        _isLngValid;
  }

  void _handleSubmit() {
    if (_validate()) {
      widget.onSubmit();
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    final inputFill =
        isDark ? AppTheme.darkSurfaceElevated : const Color(0xFFF7FAFD);

    final inputBorder = OutlineInputBorder(
      borderRadius: BDDesign.radiusLarge,
      borderSide: BorderSide.none,
    );
    final errorBorder = OutlineInputBorder(
      borderRadius: BDDesign.radiusLarge,
      borderSide: const BorderSide(color: Color(0xFFD34C4C)),
    );

    InputBorder _fieldBorder(bool valid) {
      if (_hasAttemptedSubmit && !valid) return errorBorder;
      return inputBorder;
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 1. 模型搜索框
          BDPanelCard(
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
            child: TextField(
              controller: _modelSearchController,
              onChanged: (v) => setState(() => _modelQuery = v.trim()),
              decoration: InputDecoration(
                hintText: textLocalize('community_search_model'),
                hintStyle: TextStyle(color: hintColor, fontSize: 14),
                prefixIcon:
                    Icon(Icons.search_rounded, color: hintColor, size: 20),
                suffixIcon: _modelQuery.isNotEmpty
                    ? IconButton(
                        icon:
                            Icon(Icons.close_rounded, color: hintColor, size: 18),
                        onPressed: () {
                          _modelSearchController.clear();
                          setState(() => _modelQuery = '');
                        },
                      )
                    : null,
                filled: true,
                fillColor: inputFill,
                border: inputBorder,
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 10,
                ),
              ),
            ),
          ),
          if (_hasAttemptedSubmit && !_isModelValid)
            Padding(
              padding: const EdgeInsets.only(left: 4, top: 4),
              child: Text(
                textLocalize('community_fill_all'),
                style: const TextStyle(
                  color: Color(0xFFD34C4C),
                  fontSize: 12,
                ),
              ),
            ),
          const SizedBox(height: 12),

          // 2. 模型滑动栏（紧凑，带滚动条）
          if (widget.shareableModels.isEmpty)
            BDPanelCard(
              padding: const EdgeInsets.all(20),
              child: Text(
                textLocalize('community_submit_no_models'),
                style: TextStyle(color: hintColor, height: 1.4),
              ),
            )
          else ...[
            Text(
              textLocalize('community_select_model_label'),
              style: TextStyle(
                color: textColor,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            SizedBox(
              height: 120,
              child: BDPanelCard(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: RawScrollbar(
                  thumbVisibility: true,
                  trackVisibility: true,
                  thickness: 4,
                  radius: const Radius.circular(2),
                  thumbColor: isDark
                      ? Colors.white.withValues(alpha: 0.28)
                      : Colors.black.withValues(alpha: 0.18),
                  trackColor: isDark
                      ? Colors.white.withValues(alpha: 0.06)
                      : Colors.black.withValues(alpha: 0.04),
                  child: ListView.separated(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    itemCount: _filteredModels.length,
                    separatorBuilder: (_, __) =>
                        const Divider(height: 1, indent: 48),
                    itemBuilder: (context, index) {
                      final model = _filteredModels[index];
                      final isSelected =
                          widget.selectedModels.any((m) => m.id == model.id);
                      return GestureDetector(
                        onTap: () => widget.onToggleModel(model),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 8,
                          ),
                          decoration: BoxDecoration(
                            color: isSelected
                                ? const Color(0xFF2E7CF6).withValues(alpha: 0.08)
                                : Colors.transparent,
                            borderRadius: BDDesign.radiusLarge,
                          ),
                          child: Row(
                            children: [
                              _CommunityThumbnail(
                                imageUrl: model.coverUrl,
                                height: 32,
                                width: 32,
                                icon: Icons.view_in_ar_rounded,
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  model.sceneId,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color: textColor,
                                    fontSize: 13,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                              if (isSelected)
                                const Icon(
                                  Icons.check_circle_rounded,
                                  color: Color(0xFF2E7CF6),
                                  size: 20,
                                ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
            ),
          ],
          const SizedBox(height: 12),

          // 3. 已选模型小卡片（横向排列）
          if (widget.selectedModels.isNotEmpty) ...[
            Text(
              textLocalize('community_submit_selected_count').replaceFirst('%d', widget.selectedModels.length.toString()),
              style: TextStyle(
                color: hintColor,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 8),
            SizedBox(
              height: 64,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: widget.selectedModels.length,
                separatorBuilder: (_, __) => const SizedBox(width: 8),
                itemBuilder: (context, index) {
                  final model = widget.selectedModels[index];
                  return GestureDetector(
                    onTap: () => widget.onToggleModel(model),
                    child: Container(
                      width: 140,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFF2E7CF6).withValues(alpha: 0.10),
                        borderRadius: BDDesign.radiusLarge,
                        border: Border.all(
                          color:
                              const Color(0xFF2E7CF6).withValues(alpha: 0.28),
                        ),
                      ),
                      child: Row(
                        children: [
                          _CommunityThumbnail(
                            imageUrl: model.coverUrl,
                            height: 36,
                            width: 36,
                            icon: Icons.view_in_ar_rounded,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              model.sceneId,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                color: textColor,
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                height: 1.2,
                              ),
                            ),
                          ),
                          const SizedBox(width: 4),
                          Icon(
                            Icons.close_rounded,
                            size: 14,
                            color: hintColor,
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 16),
          ],

          // 4. 模型信息表（标题、简介、地点）
          Text(
            textLocalize('community_submit_post_info'),
            style: TextStyle(
              color: textColor,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: widget.titleController,
            onChanged: (_) {
              if (_hasAttemptedSubmit) setState(() {});
            },
            decoration: InputDecoration(
              labelText: textLocalize('community_input_title'),
              filled: true,
              fillColor: inputFill,
              border: _fieldBorder(_isTitleValid),
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 12,
              ),
            ),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: widget.captionController,
            minLines: 3,
            maxLines: 4,
            onChanged: (_) {
              if (_hasAttemptedSubmit) setState(() {});
            },
            decoration: InputDecoration(
              labelText: textLocalize('community_input_caption'),
              filled: true,
              fillColor: inputFill,
              border: _fieldBorder(_isCaptionValid),
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 12,
              ),
            ),
          ),
          const SizedBox(height: 14),
          // 地点预设
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: CommunitySubmitView._presets.map((preset) {
              return ActionChip(
                label: Text(preset.name, style: const TextStyle(fontSize: 12)),
                visualDensity: VisualDensity.compact,
                onPressed: () {
                  widget.placeController.text = preset.name;
                  widget.latController.text =
                      preset.latitude.toStringAsFixed(3);
                  widget.lngController.text =
                      preset.longitude.toStringAsFixed(3);
                },
              );
            }).toList(),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: widget.placeController,
            onChanged: (_) {
              if (_hasAttemptedSubmit) setState(() {});
            },
            decoration: InputDecoration(
              labelText: textLocalize('community_input_place'),
              filled: true,
              fillColor: inputFill,
              border: _fieldBorder(_isPlaceValid),
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 12,
              ),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: widget.latController,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true, signed: true),
                  onChanged: (_) {
                    if (_hasAttemptedSubmit) setState(() {});
                  },
                  decoration: InputDecoration(
                    labelText: textLocalize('community_input_lat'),
                    filled: true,
                    fillColor: inputFill,
                    border: _fieldBorder(_isLatValid),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 12,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextField(
                  controller: widget.lngController,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true, signed: true),
                  onChanged: (_) {
                    if (_hasAttemptedSubmit) setState(() {});
                  },
                  decoration: InputDecoration(
                    labelText: textLocalize('community_input_lng'),
                    filled: true,
                    fillColor: inputFill,
                    border: _fieldBorder(_isLngValid),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 12,
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // 5. 存草稿 + 发布按钮（同一行）
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: widget.onSaveDraft,
                  icon: const Icon(Icons.save_outlined, size: 18),
                  label: Text(textLocalize('community_save_draft')),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    shape: RoundedRectangleBorder(
                      borderRadius: BDDesign.radiusLarge,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 2,
                child: FilledButton.icon(
                  onPressed: widget.isSubmitting ? null : _handleSubmit,
                  icon: widget.isSubmitting
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.send_rounded, size: 18),
                  label: Text(
                    widget.isSubmitting
                        ? textLocalize('community_publishing')
                        : textLocalize('community_publish'),
                  ),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    shape: RoundedRectangleBorder(
                      borderRadius: BDDesign.radiusLarge,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ============================================================
// Shared Widgets
// ============================================================

class CommunityLocationHubRow extends StatelessWidget {
  final CommunityPost post;

  const CommunityLocationHubRow({super.key, required this.post});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isDark
            ? AppTheme.darkSurfaceElevated.withValues(alpha: 0.94)
            : const Color(0xFFF6F8FB),
        borderRadius: BDDesign.radiusLarge,
        border: Border.all(
          color: isDark
              ? Colors.white.withValues(alpha: 0.06)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
        ),
      ),
      child: Row(
        children: [
          _CommunityThumbnail(
            imageUrl: post.coverUrl,
            height: 72,
            width: 92,
            icon: Icons.landscape_rounded,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(post.title,
                    style: TextStyle(
                        color: textColor, fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text('${post.authorName} · ${post.modelName}',
                    style: TextStyle(color: hintColor, fontSize: 12.5)),
                const SizedBox(height: 8),
                Text(post.caption,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                        color: textColor.withValues(alpha: 0.82), height: 1.3)),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Icon(Icons.chevron_right_rounded, color: hintColor),
        ],
      ),
    );
  }
}

class CommunityMetricCard extends StatelessWidget {
  final String label;
  final String value;
  final String hint;

  const CommunityMetricCard({
    super.key,
    required this.label,
    required this.value,
    required this.hint,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.58)
                      : BDDesign.colorMutedBlue,
                  fontWeight: FontWeight.w700,
                  fontSize: 12.5)),
          const SizedBox(height: 8),
          Text(value,
              style: TextStyle(
                  color: isDark
                      ? BDDesign.colorPaperWhite
                      : BDDesign.colorInkBlack,
                  fontSize: 24,
                  fontWeight: FontWeight.w800)),
          const SizedBox(height: 6),
          Text(hint,
              style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.54)
                      : BDDesign.colorMutedBlue.withValues(alpha: 0.86),
                  height: 1.35,
                  fontSize: 12.5)),
        ],
      ),
    );
  }
}

// ------- Private Helpers -------

class _CommunityThumbnail extends StatelessWidget {
  final String? imageUrl;
  final double height;
  final double width;
  final IconData icon;

  const _CommunityThumbnail({
    required this.imageUrl,
    required this.height,
    required this.width,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final fallback = Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF8BA8C5),
            Color(0xFF536C8B),
            Color(0xFF38485F),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Center(
        child: Icon(
          icon,
          size: math.min(height, width) * 0.48,
          color: Colors.white.withValues(alpha: 0.88),
        ),
      ),
    );

    final url = imageUrl;
    return SizedBox(
      height: height,
      width: width,
      child: url == null || url.isEmpty
          ? ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: fallback,
            )
          : BDFadeInNetworkImage(
              imageUrl: url,
              placeholder: fallback,
              errorWidget: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: fallback,
              ),
              fit: BoxFit.cover,
              borderRadius: BorderRadius.circular(16),
              backgroundColor: Colors.transparent,
              duration: BDMotion.durationSlow,
              curve: BDMotion.curveEnter,
            ),
    );
  }
}

class _CommunityEmptyState extends StatelessWidget {
  const _CommunityEmptyState();

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 6, 20, 104),
      child: BDPanelCard(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.public_off_rounded,
                size: 48,
                color: isDark
                    ? Colors.white.withValues(alpha: 0.48)
                    : BDDesign.colorMutedBlue,
              ),
              const SizedBox(height: 14),
              Text(
                textLocalize('community_empty_feed'),
                style: TextStyle(
                  color: isDark
                      ? BDDesign.colorPaperWhite
                      : BDDesign.colorInkBlack,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                textLocalize('community_empty_hint'),
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.62)
                      : BDDesign.colorMutedBlue.withValues(alpha: 0.88),
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LocationPreset {
  final String name;
  final double latitude;
  final double longitude;

  const _LocationPreset(this.name, this.latitude, this.longitude);
}

// ============================================================
// Explore filter bar + empty hint (used by recommend tab)
// ============================================================

class _ExploreFilterBar extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool boundsReady;
  final int zoom;
  final double tagRadiusKm;
  final int visibleCount;
  final int totalCount;
  final int filteredCount;
  final List<String> availableTags;
  final String? selectedTag;
  final ValueChanged<String> onToggleTag;
  final VoidCallback onClearFilters;

  const _ExploreFilterBar({
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.boundsReady,
    required this.zoom,
    required this.tagRadiusKm,
    required this.visibleCount,
    required this.totalCount,
    required this.filteredCount,
    required this.availableTags,
    required this.selectedTag,
    required this.onToggleTag,
    required this.onClearFilters,
  });

  String _radiusLabel() {
    if (tagRadiusKm >= 1) {
      return '${tagRadiusKm.toStringAsFixed(tagRadiusKm >= 10 ? 0 : 1)} km';
    }
    return '${(tagRadiusKm * 1000).round()} m';
  }

  @override
  Widget build(BuildContext context) {
    final hasTag = selectedTag != null && selectedTag!.isNotEmpty;
    final summary = boundsReady
        ? (hasTag
            ? '当前区域 · 含 "$selectedTag" · ${_radiusLabel()} 内 $filteredCount 条'
            : '当前区域 $visibleCount/$totalCount 条 · ZOOM $zoom')
        : '调整地图后将按可视区域筛选 · 共 $totalCount 条';
    return BDPanelCard(
      padding: const EdgeInsets.fromLTRB(14, 12, 10, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.travel_explore_rounded, size: 18, color: hintColor),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  summary,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    height: 1.3,
                  ),
                ),
              ),
              if (hasTag)
                TextButton.icon(
                  onPressed: onClearFilters,
                  icon: const Icon(Icons.close_rounded, size: 16),
                  label: const Text('清除'),
                  style: TextButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    minimumSize: const Size(0, 32),
                    visualDensity: VisualDensity.compact,
                  ),
                ),
            ],
          ),
          if (availableTags.isNotEmpty) ...[
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: availableTags.map((tag) {
                final selected = tag == selectedTag;
                return _TagPill(
                  label: tag,
                  selected: selected,
                  isDark: isDark,
                  onTap: () => onToggleTag(tag),
                );
              }).toList(),
            ),
          ],
        ],
      ),
    );
  }
}

class _TagPill extends StatelessWidget {
  final String label;
  final bool selected;
  final bool isDark;
  final VoidCallback onTap;

  const _TagPill({
    required this.label,
    required this.selected,
    required this.isDark,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = selected
        ? Colors.white
        : (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack);
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: BDMotion.durationFast,
        curve: BDMotion.curveFluid,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: selected
              ? const Color(0xFF2E7CF6)
              : (isDark
                  ? AppTheme.darkSurfaceElevated
                  : const Color(0xFFF3F5F9)),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: selected
                ? const Color(0xFF2E7CF6)
                : (isDark
                    ? Colors.white.withValues(alpha: 0.08)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.12)),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: textColor,
            fontSize: 12.5,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _ExploreEmptyHint extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool hasTag;
  final VoidCallback onClearFilters;

  const _ExploreEmptyHint({
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.hasTag,
    required this.onClearFilters,
  });

  @override
  Widget build(BuildContext context) {
    return BDPanelCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                hasTag ? Icons.filter_alt_off_rounded : Icons.public_off_rounded,
                color: hintColor,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  hasTag ? '当前区域内没有匹配该标签的帖子' : '当前区域里还没有空间记忆',
                  style: TextStyle(
                    color: textColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            hasTag ? '试试切换其它标签，或清除筛选查看区域内全部帖子。' : '滑动或缩放地图，看看其它区域。',
            style: TextStyle(color: hintColor, height: 1.4, fontSize: 12.5),
          ),
          if (hasTag) ...[
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: onClearFilters,
                icon: const Icon(Icons.close_rounded, size: 16),
                label: const Text('清除筛选'),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
