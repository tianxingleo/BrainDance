import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';
import 'model_card.dart';

String _modelDisplayName(
  Map<String, dynamic> model, {
  String fallback = 'Unknown Scene',
}) {
  final displayName = model['display_name']?.toString().trim() ?? '';
  if (displayName.isNotEmpty) {
    return displayName;
  }

  final tags = model['tags'];
  if (tags is List) {
    for (final tag in tags) {
      final value = tag?.toString().trim() ?? '';
      if (value.isNotEmpty) {
        return value;
      }
    }
  }

  final sceneId = model['scene_id']?.toString().trim() ?? '';
  if (sceneId.isNotEmpty) {
    return sceneId;
  }

  return fallback;
}

Color _resolveTextColor(bool isDark) =>
    isDark ? const Color(0xFFFFFFFF) : const Color(0xFF333333);

Color _resolveHintTextColor(bool isDark, TDThemeData theme) =>
    isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;

String _formatSimilarity(double value) =>
    '${(value * 100).toStringAsFixed(1)}%';

Widget _buildSimilarityBadge({required Widget child}) {
  return Tooltip(
    message: '\u5339\u914D\u5EA6',
    preferBelow: false,
    verticalOffset: 12,
    child: child,
  );
}

class RecallModelGrid extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final List<Map<String, dynamic>> models;
  final Map<String, dynamic>? activeModelAction;
  final GlobalKey Function(Map<String, dynamic>) modelCardKeyFor;
  final bool Function(Map<String, dynamic>?, Map<String, dynamic>?) isSameModel;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final void Function(Map<String, dynamic> model, {bool imageOnly})
  onShowModelActions;
  final String Function(String) toPublicUrl;

  const RecallModelGrid({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.models,
    required this.activeModelAction,
    required this.modelCardKeyFor,
    required this.isSameModel,
    required this.onNavigateToViewer,
    required this.onShowModelActions,
    required this.toPublicUrl,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = _resolveTextColor(isDark);
    final hintTextColor = _resolveHintTextColor(isDark, theme);

    final isSearchWithFrames =
        models.isNotEmpty && models.first.containsKey('matched_frames');

    if (isSearchWithFrames) {
      return SliverPadding(
        padding: const EdgeInsets.fromLTRB(16.0, 6.0, 16.0, 16.0),
        sliver: SliverList(
          delegate: SliverChildBuilderDelegate(
            (context, index) {
              final model = models[index];
              final cardKey = modelCardKeyFor(model);
              final isActionTarget = isSameModel(activeModelAction, model);
              final displayName = _modelDisplayName(model);
              final sceneStorageId = model['scene_id']?.toString() ?? '';
              final desc =
                  model['description'] ?? textLocalize("recall_no_desc");
              final similarity = model['similarity'] as double?;
              final userId = model['user_id'] ?? '';
              final matchedFrames =
                  model['matched_frames'] as List<dynamic>? ?? [];

              return RepaintBoundary(
                child: IgnorePointer(
                  ignoring: isActionTarget,
                  child: Opacity(
                    opacity: isActionTarget ? 0.0 : 1.0,
                    child: Container(
                      key: cardKey,
                      margin: const EdgeInsets.only(bottom: 16.0),
                      decoration: BoxDecoration(
                        color: isDark ? darkCard : BDDesign.colorPaperWhite,
                        borderRadius: BDDesign.radiusLarge,
                        boxShadow: isDark ? [] : [BDDesign.shadowLight],
                        border: Border.all(
                          color: isDark
                              ? const Color(0xFF2A2A30)
                              : Colors.transparent,
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          GestureDetector(
                            onTap: () => onNavigateToViewer(model, null),
                            onLongPressStart: (_) => onShowModelActions(
                              model,
                              imageOnly: false,
                            ), // Here we are sending false for standard view cards
                            child: Padding(
                              padding: const EdgeInsets.all(16.0),
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        TDText(
                                          displayName,
                                          font: theme.fontTitleMedium,
                                          fontWeight: FontWeight.w600,
                                          maxLines: 1,
                                          textColor: textColor,
                                        ),
                                        const SizedBox(height: 4),
                                        TDText(
                                          desc,
                                          font: theme.fontBodySmall,
                                          textColor: hintTextColor,
                                          maxLines: 2,
                                        ),
                                      ],
                                    ),
                                  ),
                                  if (similarity != null)
                                    _buildSimilarityBadge(
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 8,
                                          vertical: 4,
                                        ),
                                        decoration: BoxDecoration(
                                          color: theme.brandColor4.withAlpha(
                                            220,
                                          ),
                                          borderRadius: BorderRadius.circular(
                                            6,
                                          ),
                                        ),
                                        child: TDText(
                                          _formatSimilarity(similarity),
                                          font: theme.fontBodySmall,
                                          textColor: isDark
                                              ? const Color(0xFFFFFFFF)
                                              : Colors.white,
                                        ),
                                      ),
                                    ),
                                ],
                              ),
                            ),
                          ),
                          if (matchedFrames.isNotEmpty)
                            SizedBox(
                              height: 120,
                              child: ListView.builder(
                                scrollDirection: Axis.horizontal,
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16.0,
                                ).copyWith(bottom: 16.0),
                                itemCount: matchedFrames.length,
                                itemBuilder: (context, frameIndex) {
                                  final frame = matchedFrames[frameIndex];
                                  final imageName = frame['image_name'];
                                  final frameSim =
                                      frame['similarity'] as double?;

                                  final imageUrl = Supabase
                                      .instance
                                      .client
                                      .storage
                                      .from('braindance-assets')
                                      .getPublicUrl(
                                        '$userId/$sceneStorageId/output/images/$imageName',
                                      );

                                  return GestureDetector(
                                    onTap: () =>
                                        onNavigateToViewer(model, frame),
                                    child: _AdaptiveFrameThumbnail(
                                      imageUrl: imageUrl,
                                      frameSim: frameSim,
                                      height: 104,
                                      backgroundColor: isDark
                                          ? darkInput
                                          : theme.grayColor3,
                                    ),
                                  );
                                },
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              );
            },
            childCount: models.length,
            addAutomaticKeepAlives: false,
          ),
        ),
      );
    }

    return SliverPadding(
      padding: const EdgeInsets.only(
        left: 16.0,
        right: 16.0,
        top: 14.0,
        bottom: 16.0,
      ),
      sliver: SliverGrid(
        delegate: SliverChildBuilderDelegate(
          (context, index) {
            final model = models[index];
            final cardKey = modelCardKeyFor(model);
            final isActionTarget = isSameModel(activeModelAction, model);

            return RepaintBoundary(
              child: IgnorePointer(
                ignoring: isActionTarget,
                child: Opacity(
                  opacity: isActionTarget ? 0.0 : 1.0,
                  child: GestureDetector(
                    onTap: () => onNavigateToViewer(model, null),
                    onLongPressStart: (_) => onShowModelActions(
                      model,
                      imageOnly: false,
                    ), // Here we are sending false for standard view cards
                    child: Container(
                      key: cardKey,
                      child: RecallModelTile(
                        model: model,
                        theme: theme,
                        isDark: isDark,
                        darkCard: darkCard,
                        darkInput: darkInput,
                        textColor: textColor,
                        hintTextColor: hintTextColor,
                        toPublicUrl: toPublicUrl,
                      ),
                    ),
                  ),
                ),
              ),
            );
          },
          childCount: models.length,
          addAutomaticKeepAlives: false,
        ),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          crossAxisSpacing: 16.0,
          mainAxisSpacing: 16.0,
          childAspectRatio: 0.85,
        ),
      ),
    );
  }
}

class _AdaptiveFrameThumbnail extends StatefulWidget {
  final String imageUrl;
  final double? frameSim;
  final double height;
  final Color backgroundColor;

  const _AdaptiveFrameThumbnail({
    required this.imageUrl,
    required this.frameSim,
    required this.height,
    required this.backgroundColor,
  });

  @override
  State<_AdaptiveFrameThumbnail> createState() =>
      _AdaptiveFrameThumbnailState();
}

/// Shared image resolution logic for widgets that load a network image by URL.
mixin _NetworkImageResolverMixin<T extends StatefulWidget> on State<T> {
  static final Map<String, ui.Image> _imageCache = <String, ui.Image>{};

  ui.Image? resolvedImage;
  Object? lastError;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  String get imageUrl;

  @override
  void initState() {
    super.initState();
    _resolveImage();
  }

  @override
  void dispose() {
    _stopListening();
    super.dispose();
  }

  void onImageUrlChanged(String oldUrl) {
    if (oldUrl != imageUrl) {
      _stopListening();
      resolvedImage = null;
      lastError = null;
      _resolveImage();
    }
  }

  void _resolveImage() {
    final cached = _imageCache[imageUrl];
    if (cached != null) {
      resolvedImage = cached;
      return;
    }

    final provider = NetworkImage(imageUrl);
    final stream = provider.resolve(const ImageConfiguration());
    _imageStream = stream;
    _imageStreamListener = ImageStreamListener(
      (ImageInfo info, bool synchronousCall) {
        _imageCache[imageUrl] = info.image;
        if (!mounted) {
          return;
        }
        setState(() {
          resolvedImage = info.image;
          lastError = null;
        });
      },
      onError: (Object error, StackTrace? stackTrace) {
        if (!mounted) {
          return;
        }
        setState(() {
          lastError = error;
        });
      },
    );
    stream.addListener(_imageStreamListener!);
  }

  void _stopListening() {
    final stream = _imageStream;
    final listener = _imageStreamListener;
    if (stream != null && listener != null) {
      stream.removeListener(listener);
    }
    _imageStream = null;
    _imageStreamListener = null;
  }
}

class _AdaptiveFrameThumbnailState extends State<_AdaptiveFrameThumbnail>
    with _NetworkImageResolverMixin {
  @override
  String get imageUrl => widget.imageUrl;

  @override
  void didUpdateWidget(covariant _AdaptiveFrameThumbnail oldWidget) {
    super.didUpdateWidget(oldWidget);
    onImageUrlChanged(oldWidget.imageUrl);
  }

  @override
  Widget build(BuildContext context) {
    final img = resolvedImage;
    final aspectRatio = img == null ? 4 / 3 : img.width / img.height;
    final width = (widget.height * aspectRatio).clamp(76.0, 220.0);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      curve: Curves.easeOutCubic,
      width: width,
      margin: const EdgeInsets.only(right: 12.0),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8.0),
        color: widget.backgroundColor,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8.0),
        child: Stack(
          fit: StackFit.expand,
          children: [
            ColoredBox(
              color: widget.backgroundColor,
              child: img != null
                  ? TweenAnimationBuilder<double>(
                      tween: Tween<double>(begin: 0.0, end: 1.0),
                      duration: BDMotion.durationSlow,
                      curve: BDMotion.curveEnter,
                      builder: (context, value, child) {
                        return Opacity(
                          opacity: value,
                          child: Transform.translate(
                            offset: Offset(0, 8 * (1 - value)),
                            child: Transform.scale(
                              scale: 0.985 + 0.015 * value,
                              child: child,
                            ),
                          ),
                        );
                      },
                      child: FittedBox(
                        fit: BoxFit.cover,
                        clipBehavior: Clip.hardEdge,
                        child: SizedBox(
                          width: img.width.toDouble(),
                          height: img.height.toDouble(),
                          child: RawImage(
                            image: img,
                            filterQuality: FilterQuality.low,
                          ),
                        ),
                      ),
                    )
                  : lastError != null
                  ? const Center(
                      child: Icon(Icons.broken_image, color: Colors.grey),
                    )
                  : const Center(
                      child: SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
            ),
            if (widget.frameSim != null)
              Positioned(
                bottom: 4,
                left: 4,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 4,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withAlpha(100),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    _formatSimilarity(widget.frameSim!),
                    style: const TextStyle(color: Colors.white, fontSize: 10),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _CoverNetworkImage extends StatefulWidget {
  final String imageUrl;
  final Color backgroundColor;
  final Widget errorWidget;

  const _CoverNetworkImage({
    required this.imageUrl,
    required this.backgroundColor,
    required this.errorWidget,
  });

  @override
  State<_CoverNetworkImage> createState() => _CoverNetworkImageState();
}

class _CoverNetworkImageState extends State<_CoverNetworkImage>
    with _NetworkImageResolverMixin {
  @override
  String get imageUrl => widget.imageUrl;

  @override
  void didUpdateWidget(covariant _CoverNetworkImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    onImageUrlChanged(oldWidget.imageUrl);
  }

  @override
  Widget build(BuildContext context) {
    final img = resolvedImage;
    if (lastError != null) {
      return widget.errorWidget;
    }

    return ColoredBox(
      color: widget.backgroundColor,
      child: img != null
          ? TweenAnimationBuilder<double>(
              tween: Tween<double>(begin: 0.0, end: 1.0),
              duration: BDMotion.durationSlow,
              curve: BDMotion.curveEnter,
              builder: (context, value, child) {
                return Opacity(
                  opacity: value,
                  child: Transform.translate(
                    offset: Offset(0, 6 * (1 - value)),
                    child: Transform.scale(
                      scale: 0.985 + 0.015 * value,
                      child: child,
                    ),
                  ),
                );
              },
              child: ClipRect(
                child: FittedBox(
                  fit: BoxFit.cover,
                  clipBehavior: Clip.hardEdge,
                  child: SizedBox(
                    width: img.width.toDouble(),
                    height: img.height.toDouble(),
                    child: RawImage(image: img, filterQuality: FilterQuality.low),
                  ),
                ),
              ),
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }
}

class RecallModelTile extends StatelessWidget {
  final Map<String, dynamic> model;
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Color textColor;
  final Color hintTextColor;
  final bool elevated;
  final double? elevationProgress;
  final String Function(String)? toPublicUrl;
  final bool imageOnly;

  const RecallModelTile({
    super.key,
    required this.model,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.textColor,
    required this.hintTextColor,
    this.elevated = false,
    this.elevationProgress,
    this.toPublicUrl,
    this.imageOnly = false,
  });

  @override
  Widget build(BuildContext context) {
    final sceneId = _modelDisplayName(model);
    final desc = model['description'] ?? textLocalize("recall_no_desc");
    final similarity = model['similarity'] as double?;
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty && toPublicUrl != null
        ? toPublicUrl!(plyPath)
        : '';
    final radius = BorderRadius.circular(28.0);

    return Container(
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: radius,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(
              elevationProgress != null
                  ? (elevationProgress! > 0
                        ? (20 + (46 - 20) * elevationProgress!).round()
                        : 20)
                  : (elevated ? 46 : 20),
            ),
            blurRadius: elevationProgress != null
                ? (10 + (26 - 10) * elevationProgress!)
                : (elevated ? 26 : 10),
            spreadRadius: elevationProgress != null
                ? (0 + (2 - 0) * elevationProgress!)
                : (elevated ? 2 : 0),
            offset: Offset(
              0,
              elevationProgress != null
                  ? (4 + (16 - 4) * elevationProgress!)
                  : (elevated ? 16 : 4),
            ),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                Container(
                  decoration: BoxDecoration(
                    color: isDark ? darkInput : theme.grayColor3,
                    borderRadius: imageOnly
                        ? radius
                        : const BorderRadius.vertical(
                            top: Radius.circular(28.0),
                          ),
                  ),
                  clipBehavior: Clip.hardEdge,
                  child:
                      model['preview_img_path'] != null &&
                          model['preview_img_path'].toString().isNotEmpty
                      ? _CoverNetworkImage(
                          imageUrl: model['preview_img_path'].toString(),
                          backgroundColor: isDark
                              ? darkInput
                              : theme.grayColor3,
                          errorWidget: RecallModelMockCover(
                            isDark: isDark,
                            theme: theme,
                          ),
                        )
                      : RecallModelMockCover(isDark: isDark, theme: theme),
                ),
                if (similarity != null)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: _buildSimilarityBadge(
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: theme.brandColor4.withAlpha(220),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: TDText(
                          _formatSimilarity(similarity),
                          font: theme.fontBodyExtraSmall,
                          textColor: isDark
                              ? const Color(0xFFFFFFFF)
                              : Colors.white,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          if (!imageOnly)
            Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TDText(
                    sceneId,
                    font: theme.fontTitleMedium,
                    fontWeight: FontWeight.w600,
                    maxLines: 1,
                    textColor: textColor,
                  ),
                  const SizedBox(height: 4),
                  TDText(
                    desc,
                    font: theme.fontBodySmall,
                    textColor: hintTextColor,
                    maxLines: 2,
                  ),
                  if (toPublicUrl != null) ...[
                    const SizedBox(height: 6),
                    ModelDownloadBadge(
                      modelUrl: modelUrl,
                      isDark: isDark,
                      theme: theme,
                    ),
                  ],
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class RecallModelMockCover extends StatelessWidget {
  final bool isDark;
  final TDThemeData theme;

  const RecallModelMockCover({
    super.key,
    required this.isDark,
    required this.theme,
  });

  @override
  Widget build(BuildContext context) {
    final accent = isDark ? const Color(0xFF7AA2FF) : BDDesign.colorMutedBlue;

    return Container(
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1A1E27) : const Color(0xFFF6F8FC),
        border: Border.all(
          color: isDark ? Colors.white.withAlpha(18) : accent.withAlpha(35),
        ),
      ),
      child: Center(
        child: Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: isDark
                ? Colors.white.withAlpha(6)
                : Colors.white.withAlpha(190),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: isDark ? Colors.white.withAlpha(18) : accent.withAlpha(28),
            ),
          ),
          child: Icon(
            Icons.auto_awesome_mosaic_rounded,
            size: 28,
            color: accent.withAlpha(210),
          ),
        ),
      ),
    );
  }
}

