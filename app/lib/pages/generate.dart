import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/gen_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
import 'package:braindance/main.dart' show pageIndexProvider;
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../configs/supabase_config.dart';

part 'generate/generate_media.dart';
part 'generate/generate_submission.dart';
part 'generate/generate_widgets.dart';

class GeneratePage extends ConsumerStatefulWidget {
  const GeneratePage({super.key});

  @override
  ConsumerState<GeneratePage> createState() => _GeneratePageState();
}

class _GeneratePageState extends ConsumerState<GeneratePage>
    with TickerProviderStateMixin {
  late final TabController _tabController;
  late final ScrollController _scrollController;
  late final TextEditingController _textEditingController;
  final ImagePicker _picker = ImagePicker();

  static Key _uploadKey = UniqueKey();
  static Key _uploadKey2 = UniqueKey();
  static const TextStyle tabTextStyle = TextStyle(
    fontSize: 16,
    fontFamily: AppConfig.fontFamily,
  );
  static const int maxImageCount = 1;
  static const int sizeLimit = 40960;
  static bool firstCheck = false;
  static final Random _rdg = Random();

  bool _isUploading = false;
  double _uploadProgress = 0.0;
  int _uploadedBytes = 0;
  int _totalFileSize = 0;
  String? _generatedImageUrl;
  bool _isGenerating = false;

  static String _generateSceneId() {
    final time = DateTime.now();
    return 'scene_'
        '${time.year.toString().padLeft(4, '0')}'
        '${time.month.toString().padLeft(2, '0')}'
        '${time.day.toString().padLeft(2, '0')}'
        '_'
        '${_rdg.nextInt(1000000).toString().padLeft(6, '0')}';
  }

  void _refresh(VoidCallback fn) {
    setState(fn);
  }

  static String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    if (bytes < 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
  }

  @override
  void initState() {
    super.initState();
    _tabController = TabController(
      length: 3,
      vsync: this,
      animationDuration: const Duration(milliseconds: 200),
    );
    _scrollController = ScrollController();
    _textEditingController = TextEditingController();

    if (firstCheck) {
      return;
    }
    firstCheck = true;
    loadCache();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _scrollController.dispose();
    _textEditingController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    const double floatingNavHeight = 68;
    const double floatingNavBottomMargin = 20;
    const double submitBottomGap = 10;
    const double submitAreaBottomPadding =
        floatingNavHeight + floatingNavBottomMargin + submitBottomGap;
    const double contentBottomPadding = 36;
    const double keyboardSubmitBottomPadding = 8;
    final textColor = isDark ? Colors.white : BDDesign.colorInkBlack;
    final bgCardColor = isDark
        ? const Color(0xFF1C1C1E)
        : BDDesign.colorPaperWhite;
    final keyboardInset = MediaQuery.of(context).viewInsets.bottom;
    final submitBottomPadding = keyboardInset > 0
        ? keyboardSubmitBottomPadding
        : submitAreaBottomPadding;

    final currentSelectionCount = switch (_tabController.index) {
      0 => GenConfig.uploadedImages.length,
      1 => _textEditingController.text.trim().isEmpty ? 0 : 1,
      _ => GenConfig.uploadedVideos.length,
    };
    final modeLabel = switch (_tabController.index) {
      0 => textLocalize('gen_pic'),
      1 => textLocalize('gen_text'),
      _ => textLocalize('gen_video'),
    };
    final uploadLabel = _isGenerating
        ? textLocalize('gen_text_generating')
        : _isUploading
        ? textLocalize(
            'gen_upload_progress',
          ).replaceAll('%s', (_uploadProgress * 100).toStringAsFixed(0))
        : currentSelectionCount == 0
        ? textLocalize('gen_waiting_material')
        : textLocalize('gen_ready_submit');

    final List<Widget> tabContents = [
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: BDPanelCard(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          child: ClipRRect(
            borderRadius: BDDesign.radiusNormal,
            child: TDTabBar(
              tabs: [
                TDTab(text: textLocalize('gen_pic')),
                TDTab(text: textLocalize('gen_text')),
                TDTab(text: textLocalize('gen_video')),
              ],
              controller: _tabController,
              outlineType: TDTabBarOutlineType.capsule,
              showIndicator: false,
              backgroundColor: Colors.transparent,
              selectedBgColor: isDark
                  ? Colors.white.withValues(alpha: 0.10)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.14),
              unSelectedBgColor: Colors.transparent,
              labelPadding: const EdgeInsets.all(4),
              onTap: (index) {
                setState(() {});
              },
              labelStyle: tabTextStyle.copyWith(
                fontWeight: FontWeight.w600,
                color: isDark
                    ? BDDesign.colorPaperWhite
                    : BDDesign.colorMutedBlue,
              ),
              unselectedLabelStyle: tabTextStyle.copyWith(
                fontWeight: FontWeight.w400,
                color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
              ),
            ),
          ),
        ),
      ),
    ];

    Widget? currentTabContent;
    switch (_tabController.index) {
      case 0:
        final upload = TDUpload(
          key: _uploadKey,
          files: GenConfig.uploadedImages,
          multiple: true,
          max: maxImageCount,
          onUploadTap: () => _showActionSheet(context, true),
          onChange: (files, type) {
            switch (type) {
              case TDUploadType.add:
                GenConfig.uploadedImages = [
                  ...GenConfig.uploadedImages,
                  ...files,
                ];
                break;
              case TDUploadType.remove:
                for (final f in files) {
                  GenConfig.uploadedImages.remove(f);
                }
                break;
              case TDUploadType.replace:
                break;
            }
            setState(() {
              _uploadKey = UniqueKey();
            });
          },
          mediaType: const [TDUploadMediaType.image],
          width: 150,
          height: 150,
        );
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(0),
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.fromLTRB(
              24,
              24,
              24,
              contentBottomPadding,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _GenerateSectionHeading(
                  title: textLocalize('gen_section_image'),
                  description: textLocalize(
                    'gen_tip_pic',
                  ).replaceAll('[FILE_SIZE]', _formatBytes(sizeLimit * 1024)),
                ),
                const SizedBox(height: 18),
                BDPanelCard(
                  padding: const EdgeInsets.all(20),
                  child: Container(
                    decoration: BoxDecoration(
                      color: bgCardColor,
                      borderRadius: BDDesign.radiusNormal,
                    ),
                    child: upload,
                  ),
                ),
              ],
            ),
          ),
        );
        break;
      case 1:
        _textEditingController.text = GenConfig.uploadedText;
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(1),
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.fromLTRB(
              24,
              24,
              24,
              contentBottomPadding,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _GenerateSectionHeading(
                  title: textLocalize('gen_section_text'),
                  description: textLocalize('gen_tip_text'),
                ),
                const SizedBox(height: 18),
                BDPanelCard(
                  child: Container(
                    decoration: BoxDecoration(
                      color: bgCardColor,
                      borderRadius: BDDesign.radiusNormal,
                    ),
                    child: TDTextarea(
                      controller: _textEditingController,
                      hintText: textLocalize('gen_tip_textbox'),
                      minLines: 8,
                      maxLines: 20,
                      onChanged: (value) {
                        GenConfig.uploadedText = value;
                        setState(() {});
                      },
                      decoration: BoxDecoration(
                        color: Colors.transparent,
                        borderRadius: BDDesign.radiusNormal,
                        border: Border.all(color: Colors.transparent),
                      ),
                      textStyle: TextStyle(color: textColor, fontSize: 16),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
        break;
      case 2:
        final upload = TDUpload(
          type: TDUploadBoxType.circle,
          key: _uploadKey2,
          files: GenConfig.uploadedVideos,
          multiple: false,
          onUploadTap: () => _showActionSheet(context, false),
          onChange: (files, type) {
            switch (type) {
              case TDUploadType.add:
                GenConfig.uploadedVideos = [
                  ...GenConfig.uploadedVideos,
                  ...files,
                ];
                break;
              case TDUploadType.remove:
                for (final f in files) {
                  GenConfig.uploadedVideos.remove(f);
                }
                break;
              case TDUploadType.replace:
                break;
            }
            setState(() {
              _uploadKey2 = UniqueKey();
            });
          },
          mediaType: const [TDUploadMediaType.video],
          width: 150,
          height: 150,
        );
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(2),
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.fromLTRB(
              24,
              24,
              24,
              contentBottomPadding,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _GenerateSectionHeading(
                  title: textLocalize('gen_section_video'),
                  description: textLocalize('gen_tip_video'),
                ),
                const SizedBox(height: 18),
                BDPanelCard(
                  padding: const EdgeInsets.all(20),
                  child: Container(
                    decoration: BoxDecoration(
                      color: bgCardColor,
                      borderRadius: BDDesign.radiusNormal,
                    ),
                    child: upload,
                  ),
                ),
              ],
            ),
          ),
        );
        break;
    }

    if (currentTabContent != null) {
      tabContents.add(
        Expanded(
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            switchInCurve: Curves.easeOutCubic,
            switchOutCurve: Curves.easeInCubic,
            transitionBuilder: (child, animation) {
              return FadeTransition(
                opacity: animation,
                child: SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0.05, 0.0),
                    end: Offset.zero,
                  ).animate(animation),
                  child: child,
                ),
              );
            },
            child: currentTabContent,
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: Stack(
            children: [
              Column(
                children: [
                  BDPageHeader(
                    title: textLocalize('gen_top'),
                    //subtitle: textLocalize('gen_subtitle'),
                    trailing: BDStatusPill(
                      label: uploadLabel,
                      icon: _isUploading || _isGenerating
                          ? Icons.sync_rounded
                          : Icons.layers_outlined,
                      color: _isUploading || _isGenerating
                          ? BDDesign.colorFadedOlive
                          : BDDesign.colorMutedBlue,
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: BDPanelCard(
                      padding: const EdgeInsets.all(18),
                      child: Row(
                        children: [
                          Expanded(
                            child: _GenerateMetric(
                              label: textLocalize('gen_label_mode'),
                              value: modeLabel,
                            ),
                          ),
                          Expanded(
                            child: _GenerateMetric(
                              label: textLocalize('gen_label_material'),
                              value: currentSelectionCount.toString(),
                            ),
                          ),
                          Expanded(
                            child: _GenerateMetric(
                              label: textLocalize('gen_label_status'),
                              value: uploadLabel,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  if (_isUploading)
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20),
                      child: BDPanelCard(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 18,
                          vertical: 14,
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Icon(
                                  Icons.cloud_upload_outlined,
                                  size: 14,
                                  color: isDark
                                      ? const Color(0xFFFFB74D)
                                      : const Color(0xFFF57C00),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  '${textLocalize('gen_uploading')} ${(_uploadProgress * 100).toStringAsFixed(1)}%',
                                  style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                    color: isDark
                                        ? const Color(0xFFFFB74D)
                                        : const Color(0xFFF57C00),
                                  ),
                                ),
                                const Spacer(),
                                Text(
                                  '${_formatBytes(_uploadedBytes)} / ${_formatBytes(_totalFileSize)}',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: isDark
                                        ? Colors.white.withValues(alpha: 0.58)
                                        : BDDesign.colorMutedBlue,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(3),
                              child: LinearProgressIndicator(
                                value: _uploadProgress,
                                minHeight: 5,
                                backgroundColor: isDark
                                    ? Colors.white.withAlpha(20)
                                    : Colors.black.withAlpha(15),
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  isDark
                                      ? const Color(0xFFFFB74D)
                                      : const Color(0xFFF57C00),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  Expanded(child: Column(children: tabContents)),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(
                      20,
                      8,
                      20,
                      0,
                    ),
                    child: TDButton(
                      onTap: (_isUploading || _isGenerating)
                          ? () {}
                          : () => _submit(),
                      style: TDButtonStyle(
                        backgroundColor: isDark
                            ? const Color(0xFF2A2A2E)
                            : BDDesign.colorMutedBlue,
                        textColor: Colors.white,
                        radius: BorderRadius.circular(22),
                      ),
                      type: TDButtonType.fill,
                      shape: TDButtonShape.rectangle,
                      theme: TDButtonTheme.primary,
                      size: TDButtonSize.large,
                      width: double.infinity,
                      text: (_isUploading || _isGenerating)
                          ? (_isGenerating
                                ? textLocalize('gen_text_generating')
                                : '${textLocalize('gen_uploading')} ${(_uploadProgress * 100).toStringAsFixed(1)}%')
                          : textLocalize('gen_button'),
                    ),
                  ),
                  SizedBox(height: submitBottomPadding),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
