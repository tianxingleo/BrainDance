import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/app_toast.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';

import '../../services/location_service.dart';
import 'amap_search.dart';
import 'location_picker.dart';
import 'models.dart';

Future<CommunityComposerResult?> showCommunityComposerSheet(
  BuildContext context, {
  required List<CommunityModelOption> models,
  String? initialModelId,
}) {
  return showModalBottomSheet<CommunityComposerResult>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) {
      return _CommunityComposerSheet(
        models: models,
        initialModelId: initialModelId,
      );
    },
  );
}

class _CommunityComposerSheet extends StatefulWidget {
  final List<CommunityModelOption> models;
  final String? initialModelId;

  const _CommunityComposerSheet({required this.models, this.initialModelId});

  @override
  State<_CommunityComposerSheet> createState() =>
      _CommunityComposerSheetState();
}

class _CommunityComposerSheetState extends State<_CommunityComposerSheet> {
  late final TextEditingController _titleController;
  late final TextEditingController _captionController;
  late final TextEditingController _placeController;
  CommunityModelOption? _selectedModel;
  bool _isSubmitting = false;

  // 选点状态（GCJ-02）
  double? _pickedLat;
  double? _pickedLng;
  bool _locating = false;

  @override
  void initState() {
    super.initState();
    _selectedModel = widget.models.isEmpty
        ? null
        : widget.models.firstWhere(
            (model) => model.id == widget.initialModelId,
            orElse: () => widget.models.first,
          );
    _titleController = TextEditingController(
      text: _selectedModel == null
          ? ''
          : '我在 ${_selectedModel!.sceneId} 留下的一段记忆',
    );
    _captionController = TextEditingController();
    _placeController = TextEditingController();
  }

  @override
  void dispose() {
    _titleController.dispose();
    _captionController.dispose();
    _placeController.dispose();
    super.dispose();
  }

  Future<void> _pickOnMap() async {
    final initial = (_pickedLat != null && _pickedLng != null)
        ? LatLng(_pickedLat!, _pickedLng!)
        : null;
    final result = await Navigator.of(context).push<LocationPickResult>(
      MaterialPageRoute(
        builder: (_) => LocationPickerPage(initialCenter: initial),
      ),
    );
    if (!mounted || result == null) return;
    setState(() {
      _pickedLat = result.latitude;
      _pickedLng = result.longitude;
      if (_placeController.text.trim().isEmpty &&
          result.placeName.isNotEmpty) {
        _placeController.text = result.placeName;
      } else if (result.placeName.isNotEmpty) {
        // 用户尚未自定义过 place 文案则覆盖；否则保留用户输入
        _placeController.text = result.placeName;
      }
    });
  }

  Future<void> _useCurrentLocation() async {
    if (_locating) return;
    setState(() => _locating = true);
    try {
      final p = await LocationService.instance.getCurrentGcj02();
      if (!mounted) return;
      setState(() {
        _pickedLat = p.latitude;
        _pickedLng = p.longitude;
      });
      // 调高德逆地理编码把坐标转成可读地点名；失败则回退到坐标占位。
      String placeName = '';
      try {
        final regeo = await AmapSearchService.instance.regeoSearch(p);
        placeName = regeo.placeName.isNotEmpty
            ? regeo.placeName
            : regeo.formattedAddress;
      } on AmapSearchException catch (_) {
        // 静默降级，不打断"已获取当前位置"的主流程
      } catch (_) {
        // 同上
      }
      if (!mounted) return;
      if (placeName.isNotEmpty) {
        _placeController.text = placeName;
      } else if (_placeController.text.trim().isEmpty) {
        _placeController.text =
            '${p.latitude.toStringAsFixed(4)}, ${p.longitude.toStringAsFixed(4)}';
      }
      showAppToast(context, '已获取当前位置');
    } on LocationException catch (e) {
      if (!mounted) return;
      showAppToast(context, e.message);
    } catch (e) {
      if (!mounted) return;
      showAppToast(context, '定位失败：$e');
    } finally {
      if (mounted) setState(() => _locating = false);
    }
  }

  void _resetPicked() {
    setState(() {
      _pickedLat = null;
      _pickedLng = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final bottomInset = MediaQuery.viewInsetsOf(context).bottom;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Padding(
      padding: EdgeInsets.only(bottom: bottomInset),
      child: DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.82,
        maxChildSize: 0.92,
        minChildSize: 0.62,
        builder: (context, scrollController) {
          return Padding(
            padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
            child: BDPanelCard(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
              child: SafeArea(
                top: false,
                child: SingleChildScrollView(
                  controller: scrollController,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildHeader(textColor, hintColor),
                      const SizedBox(height: 18),
                      if (widget.models.isEmpty)
                        _buildEmptyModelsHint(isDark, hintColor)
                      else
                        DropdownButtonFormField<CommunityModelOption>(
                          initialValue: _selectedModel,
                          decoration: InputDecoration(
                            labelText:
                                textLocalize('community_select_model'),
                          ),
                          items: widget.models
                              .map((model) => DropdownMenuItem(
                                    value: model,
                                    child: Text(model.sceneId),
                                  ))
                              .toList(),
                          onChanged: (value) {
                            setState(() {
                              _selectedModel = value;
                            });
                          },
                        ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _titleController,
                        decoration: InputDecoration(
                          labelText:
                              textLocalize('community_input_title'),
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _captionController,
                        minLines: 3,
                        maxLines: 5,
                        decoration: InputDecoration(
                          labelText:
                              textLocalize('community_input_caption'),
                        ),
                      ),
                      const SizedBox(height: 14),
                      _buildLocationSection(textColor, hintColor),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _placeController,
                        decoration: InputDecoration(
                          labelText:
                              textLocalize('community_input_place'),
                        ),
                      ),
                      const SizedBox(height: 18),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed:
                              widget.models.isEmpty || _isSubmitting
                                  ? null
                                  : _submit,
                          icon: _isSubmitting
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Icon(Icons.send_rounded),
                          label: Text(
                            _isSubmitting
                                ? textLocalize('community_publishing')
                                : textLocalize('community_publish'),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildHeader(Color textColor, Color hintColor) {
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                textLocalize('community_share_title'),
                style: TextStyle(
                  color: textColor,
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                textLocalize('community_share_hint'),
                style: TextStyle(color: hintColor, height: 1.4),
              ),
            ],
          ),
        ),
        IconButton(
          onPressed: () => Navigator.pop(context),
          icon: Icon(Icons.close_rounded, color: textColor),
        ),
      ],
    );
  }

  Widget _buildEmptyModelsHint(bool isDark, Color hintColor) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isDark
            ? AppTheme.darkSurfaceElevated
            : const Color(0xFFF7FAFD),
        borderRadius: BDDesign.radiusLarge,
      ),
      child: Text(
        '还没有可分享的模型。先在“过往回忆”里生成一个 3D 模型，再回来发布社区贴文。',
        style: TextStyle(color: hintColor, height: 1.4),
      ),
    );
  }

  Widget _buildLocationSection(Color textColor, Color hintColor) {
    final hasPicked = _pickedLat != null && _pickedLng != null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          textLocalize('community_location_preset'),
          style: TextStyle(
            color: textColor,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _pickOnMap,
                icon: const Icon(Icons.map_outlined, size: 18),
                label: Text(textLocalize('community_pick_on_map')),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _locating ? null : _useCurrentLocation,
                icon: _locating
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child:
                            CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.my_location_rounded, size: 18),
                label: Text(textLocalize('community_use_current_location')),
              ),
            ),
          ],
        ),
        if (hasPicked) ...[
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(12, 10, 6, 10),
            decoration: BoxDecoration(
              color: BDDesign.colorMutedBlue.withValues(alpha: 0.08),
              borderRadius: BDDesign.radiusNormal,
              border: Border.all(
                color: BDDesign.colorMutedBlue.withValues(alpha: 0.25),
                width: 1,
              ),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Icon(Icons.place_rounded,
                    size: 18, color: BDDesign.colorMutedBlue),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    '${_pickedLat!.toStringAsFixed(6)}, ${_pickedLng!.toStringAsFixed(6)}',
                    style: TextStyle(
                      color: textColor,
                      fontSize: 12.5,
                      fontFeatures: const [
                        FontFeature.tabularFigures(),
                      ],
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                IconButton(
                  tooltip: '清除',
                  splashRadius: 18,
                  onPressed: _resetPicked,
                  icon: Icon(Icons.close_rounded,
                      size: 18, color: hintColor),
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }

  Future<void> _submit() async {
    final model = _selectedModel;
    final title = _titleController.text.trim();
    final caption = _captionController.text.trim();
    final place = _placeController.text.trim();
    final lat = _pickedLat;
    final lng = _pickedLng;

    if (model == null ||
        lat == null ||
        lng == null ||
        title.isEmpty ||
        caption.isEmpty ||
        place.isEmpty) {
      showAppToast(context, textLocalize('community_fill_all'));
      return;
    }

    setState(() {
      _isSubmitting = true;
    });

    Navigator.pop(
      context,
      CommunityComposerResult(
        title: title,
        caption: caption,
        placeName: place,
        latitude: lat,
        longitude: lng,
        models: [model],
        tags: [],
      ),
    );
  }
}
