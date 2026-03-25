import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

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
  late final TextEditingController _latitudeController;
  late final TextEditingController _longitudeController;
  CommunityModelOption? _selectedModel;
  bool _isSubmitting = false;

  static const _presets = <_LocationPreset>[
    _LocationPreset('西湖', 30.243, 120.150),
    _LocationPreset('外滩', 31.240, 121.490),
    _LocationPreset('东京塔', 35.659, 139.745),
    _LocationPreset('巴黎左岸', 48.853, 2.349),
    _LocationPreset('纽约中央公园', 40.782, -73.965),
  ];

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
    _placeController = TextEditingController(text: '西湖');
    _latitudeController = TextEditingController(text: '30.243');
    _longitudeController = TextEditingController(text: '120.150');
  }

  @override
  void dispose() {
    _titleController.dispose();
    _captionController.dispose();
    _placeController.dispose();
    _latitudeController.dispose();
    _longitudeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final mediaQuery = MediaQuery.of(context);
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Padding(
      padding: EdgeInsets.only(bottom: mediaQuery.viewInsets.bottom),
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
                      Row(
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
                                  style: TextStyle(
                                    color: hintColor,
                                    height: 1.4,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          IconButton(
                            onPressed: () => Navigator.pop(context),
                            icon: Icon(Icons.close_rounded, color: textColor),
                          ),
                        ],
                      ),
                      const SizedBox(height: 18),
                      if (widget.models.isEmpty)
                        Container(
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
                        )
                      else
                        DropdownButtonFormField<CommunityModelOption>(
                          initialValue: _selectedModel,
                          decoration: InputDecoration(
                            labelText: textLocalize('community_select_model'),
                          ),
                          items: widget.models
                              .map(
                                (model) => DropdownMenuItem(
                                  value: model,
                                  child: Text(model.sceneId),
                                ),
                              )
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
                          labelText: textLocalize('community_input_title'),
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _captionController,
                        minLines: 3,
                        maxLines: 5,
                        decoration: InputDecoration(
                          labelText: textLocalize('community_input_caption'),
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        textLocalize('community_location_preset'),
                        style: TextStyle(
                          color: textColor,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: _presets.map((preset) {
                          return ActionChip(
                            label: Text(preset.name),
                            onPressed: () {
                              setState(() {
                                _placeController.text = preset.name;
                                _latitudeController.text = preset.latitude
                                    .toStringAsFixed(3);
                                _longitudeController.text = preset.longitude
                                    .toStringAsFixed(3);
                              });
                            },
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _placeController,
                        decoration: InputDecoration(
                          labelText: textLocalize('community_input_place'),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _latitudeController,
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                    decimal: true,
                                    signed: true,
                                  ),
                              decoration: InputDecoration(
                                labelText: textLocalize('community_input_lat'),
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: TextField(
                              controller: _longitudeController,
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                    decimal: true,
                                    signed: true,
                                  ),
                              decoration: InputDecoration(
                                labelText: textLocalize('community_input_lng'),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 18),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: widget.models.isEmpty || _isSubmitting
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

  Future<void> _submit() async {
    final model = _selectedModel;
    final latitude = double.tryParse(_latitudeController.text.trim());
    final longitude = double.tryParse(_longitudeController.text.trim());
    final title = _titleController.text.trim();
    final caption = _captionController.text.trim();
    final place = _placeController.text.trim();

    if (model == null ||
        latitude == null ||
        longitude == null ||
        title.isEmpty ||
        caption.isEmpty ||
        place.isEmpty) {
      TDToast.showText(context: context, textLocalize('community_fill_all'));
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
        latitude: latitude,
        longitude: longitude,
        model: model,
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
