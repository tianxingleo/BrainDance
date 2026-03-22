import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';

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
  final void Function(Map<String, dynamic>) onShowModelActions;

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
  });

  @override
  Widget build(BuildContext context) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;

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
              final sceneId = model['display_name'] ?? model['scene_id'] ?? 'Unknown Scene';
              final desc = model['description'] ?? '没有描述信息';
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
                            onLongPressStart: (_) => onShowModelActions(model),
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
                                      ],
                                    ),
                                  ),
                                  if (similarity != null)
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8,
                                        vertical: 4,
                                      ),
                                      decoration: BoxDecoration(
                                        color: theme.brandColor4.withAlpha(220),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: TDText(
                                        '${(similarity * 100).toStringAsFixed(1)}%',
                                        font: theme.fontBodySmall,
                                        textColor: isDark
                                            ? const Color(0xFFFFFFFF)
                                            : Colors.white,
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
                                        '$userId/$sceneId/output/images/$imageName',
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
                    onLongPressStart: (_) => onShowModelActions(model),
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

class RecallModelActionOverlay extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Map<String, dynamic> model;
  final Rect rect;
  final VoidCallback onDismiss;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final Future<void> Function(Map<String, dynamic>) onShowModelDetails;
  final Future<void> Function(Map<String, dynamic>) onShareModelToCommunity;
  final Future<void> Function(Map<String, dynamic>) onRenameModel;
  final Future<void> Function(Map<String, dynamic>) onDeleteLocalModel;

  const RecallModelActionOverlay({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.model,
    required this.rect,
    required this.onDismiss,
    required this.onNavigateToViewer,
    required this.onShowModelDetails,
    required this.onShareModelToCommunity,
    required this.onRenameModel,
    required this.onDeleteLocalModel,
  });

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    const horizontalGap = 12.0;
    const actionWidth = 128.0;
    final actionLeft =
        (rect.right + horizontalGap + actionWidth <= screenWidth - 16)
        ? rect.right + horizontalGap
        : rect.left + rect.width - actionWidth;

    return Positioned.fill(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onDismiss,
        child: Stack(
          children: [
            TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: 1),
              duration: const Duration(milliseconds: 220),
              curve: Curves.easeOutCubic,
              builder: (context, value, child) {
                return Opacity(
                  opacity: value,
                  child: ClipRect(
                    child: BackdropFilter(
                      filter: ui.ImageFilter.blur(
                        sigmaX: 10 * value,
                        sigmaY: 10 * value,
                      ),
                      child: Container(
                        color: Colors.black.withValues(alpha: 0.12 * value),
                      ),
                    ),
                  ),
                );
              },
            ),
            Positioned(
              left: rect.left,
              top: rect.top,
              width: rect.width,
              height: rect.height,
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: 1),
                duration: const Duration(milliseconds: 240),
                curve: Curves.easeOutCubic,
                builder: (context, value, child) {
                  return Transform.translate(
                    offset: Offset(0, -10 * value),
                    child: Transform.scale(
                      scale: 1 + (0.045 * value),
                      alignment: Alignment.center,
                      child: child,
                    ),
                  );
                },
                child: GestureDetector(
                  onTap: () => onNavigateToViewer(model, null),
                  onLongPress: onDismiss,
                  child: RecallModelTile(
                    model: model,
                    theme: theme,
                    isDark: isDark,
                    darkCard: darkCard,
                    darkInput: darkInput,
                    textColor: isDark
                        ? const Color(0xFFFFFFFF)
                        : BDDesign.colorInkBlack,
                    hintTextColor: isDark
                        ? const Color(0xFF888888)
                        : theme.fontGyColor3,
                    elevated: true,
                  ),
                ),
              ),
            ),
            Positioned(
              left: actionLeft,
              top: rect.top + 24,
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: 1),
                duration: const Duration(milliseconds: 260),
                curve: Curves.easeOutBack,
                builder: (context, value, child) {
                  final safeOpacity = value.clamp(0.0, 1.0);
                  return Opacity(
                    opacity: safeOpacity,
                    child: Transform.translate(
                      offset: Offset(18 * (1 - value), 0),
                      child: child,
                    ),
                  );
                },
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(18),
                    onTap: () async {
                      onDismiss();
                      await onShareModelToCommunity(model);
                    },
                    child: Ink(
                      width: actionWidth,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        color: isDark
                            ? const Color(0xEE1F2430)
                            : Colors.white.withAlpha(236),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(
                          color: isDark
                              ? Colors.white.withValues(alpha: 0.08)
                              : BDDesign.colorMutedBlue.withValues(alpha: 0.14),
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withAlpha(20),
                            blurRadius: 18,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          _ActionMenuItem(
                            icon: Icons.info_outline_rounded,
                            label: '查看详情',
                            isDark: isDark,
                            onTap: () async {
                              onDismiss();
                              await onShowModelDetails(model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.edit_rounded,
                            label: '重命名',
                            isDark: isDark,
                            onTap: () async {
                              onDismiss();
                              await onRenameModel(model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.delete_outline_rounded,
                            label: '删除本地模型',
                            isDark: isDark,
                            destructive: true,
                            onTap: () async {
                              onDismiss();
                              await onDeleteLocalModel(model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.public_rounded,
                            label: '分享到社区',
                            isDark: isDark,
                            onTap: () async {
                              onDismiss();
                              await onShareModelToCommunity(model);
                            },
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ActionMenuItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isDark;
  final bool destructive;
  final Future<void> Function() onTap;

  const _ActionMenuItem({
    required this.icon,
    required this.label,
    required this.isDark,
    required this.onTap,
    this.destructive = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = destructive
        ? const Color(0xFFD34C4C)
        : (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack);

    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 8),
        child: Row(
          children: [
            Icon(icon, size: 18, color: color),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: color,
                ),
              ),
            ),
          ],
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

class _AdaptiveFrameThumbnailState extends State<_AdaptiveFrameThumbnail> {
  static final Map<String, ui.Image> _imageCache = <String, ui.Image>{};

  ui.Image? _resolvedImage;
  Object? _lastError;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  @override
  void initState() {
    super.initState();
    _resolveImage();
  }

  @override
  void didUpdateWidget(covariant _AdaptiveFrameThumbnail oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.imageUrl != widget.imageUrl) {
      _stopListening();
      _resolvedImage = null;
      _lastError = null;
      _resolveImage();
    }
  }

  @override
  void dispose() {
    _stopListening();
    super.dispose();
  }

  void _resolveImage() {
    final cached = _imageCache[widget.imageUrl];
    if (cached != null) {
      _resolvedImage = cached;
      return;
    }

    final provider = NetworkImage(widget.imageUrl);
    final stream = provider.resolve(const ImageConfiguration());
    _imageStream = stream;
    _imageStreamListener = ImageStreamListener(
      (ImageInfo info, bool synchronousCall) {
        _imageCache[widget.imageUrl] = info.image;
        if (!mounted) {
          return;
        }
        setState(() {
          _resolvedImage = info.image;
          _lastError = null;
        });
      },
      onError: (Object error, StackTrace? stackTrace) {
        if (!mounted) {
          return;
        }
        setState(() {
          _lastError = error;
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

  @override
  Widget build(BuildContext context) {
    final resolvedImage = _resolvedImage;
    final aspectRatio = resolvedImage == null
        ? 4 / 3
        : resolvedImage.width / resolvedImage.height;
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
              child: resolvedImage != null
                  ? FittedBox(
                      fit: BoxFit.cover,
                      clipBehavior: Clip.hardEdge,
                      child: SizedBox(
                        width: resolvedImage.width.toDouble(),
                        height: resolvedImage.height.toDouble(),
                        child: RawImage(
                          image: resolvedImage,
                          filterQuality: FilterQuality.low,
                        ),
                      ),
                    )
                  : _lastError != null
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
                    '${(widget.frameSim! * 100).toStringAsFixed(1)}%',
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

class _CoverNetworkImageState extends State<_CoverNetworkImage> {
  static final Map<String, ui.Image> _imageCache = <String, ui.Image>{};

  ui.Image? _resolvedImage;
  Object? _lastError;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  @override
  void initState() {
    super.initState();
    _resolveImage();
  }

  @override
  void didUpdateWidget(covariant _CoverNetworkImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.imageUrl != widget.imageUrl) {
      _stopListening();
      _resolvedImage = null;
      _lastError = null;
      _resolveImage();
    }
  }

  @override
  void dispose() {
    _stopListening();
    super.dispose();
  }

  void _resolveImage() {
    final cached = _imageCache[widget.imageUrl];
    if (cached != null) {
      _resolvedImage = cached;
      return;
    }

    final provider = NetworkImage(widget.imageUrl);
    final stream = provider.resolve(const ImageConfiguration());
    _imageStream = stream;
    _imageStreamListener = ImageStreamListener(
      (ImageInfo info, bool synchronousCall) {
        _imageCache[widget.imageUrl] = info.image;
        if (!mounted) {
          return;
        }
        setState(() {
          _resolvedImage = info.image;
          _lastError = null;
        });
      },
      onError: (Object error, StackTrace? stackTrace) {
        if (!mounted) {
          return;
        }
        setState(() {
          _lastError = error;
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

  @override
  Widget build(BuildContext context) {
    final resolvedImage = _resolvedImage;
    if (_lastError != null) {
      return widget.errorWidget;
    }

    return ColoredBox(
      color: widget.backgroundColor,
      child: resolvedImage != null
          ? ClipRect(
              child: FittedBox(
                fit: BoxFit.cover,
                clipBehavior: Clip.hardEdge,
                child: SizedBox(
                  width: resolvedImage.width.toDouble(),
                  height: resolvedImage.height.toDouble(),
                  child: RawImage(
                    image: resolvedImage,
                    filterQuality: FilterQuality.low,
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
    this.imageOnly = false,
  });

  @override
  Widget build(BuildContext context) {
    final sceneId = model['display_name'] ?? model['scene_id'] ?? 'Unknown Scene';
    final desc = model['description'] ?? textLocalize("recall_no_desc");
    final similarity = model['similarity'] as double?;
    final radius = BorderRadius.circular(28.0);

    return Container(
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: radius,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(elevated ? 46 : 20),
            blurRadius: elevated ? 26 : 10,
            spreadRadius: elevated ? 2 : 0,
            offset: Offset(0, elevated ? 16 : 4),
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
                        '${(similarity * 100).toStringAsFixed(1)}%',
                        font: theme.fontBodyExtraSmall,
                        textColor: isDark
                            ? const Color(0xFFFFFFFF)
                            : Colors.white,
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

/// Time Peeling: 按模型名称分组，每组一个水平滑动时间槽
class TimePeelingList extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Map<String, List<Map<String, dynamic>>> groupedModels;
  final Map<String, dynamic>? activeModelAction;
  final GlobalKey Function(Map<String, dynamic>) modelCardKeyFor;
  final bool Function(Map<String, dynamic>?, Map<String, dynamic>?) isSameModel;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final void Function(Map<String, dynamic>) onShowModelActions;
  final void Function(String name) onAddNewTask;

  const TimePeelingList({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.groupedModels,
    required this.activeModelAction,
    required this.modelCardKeyFor,
    required this.isSameModel,
    required this.onNavigateToViewer,
    required this.onShowModelActions,
    required this.onAddNewTask,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark
        ? const Color(0xFFCCCCCC)
        : theme.fontGyColor3;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue;

    final sortedKeys = groupedModels.keys.toList()..sort((a, b) {
      final ta = _newestTime(groupedModels[a]!);
      final tb = _newestTime(groupedModels[b]!);
      return tb.compareTo(ta);
    });

    return SliverPadding(
      padding: const EdgeInsets.fromLTRB(0, 6, 0, 16),
      sliver: SliverList(
        delegate: SliverChildBuilderDelegate(
          (context, index) {
            final name = sortedKeys[index];
            final models = groupedModels[name]!;
            return _TimePeelingSlot(
              name: name,
              models: models,
              theme: theme,
              isDark: isDark,
              darkCard: darkCard,
              darkInput: darkInput,
              textColor: textColor,
              hintTextColor: hintTextColor,
              hintColor: hintColor,
              activeModelAction: activeModelAction,
              modelCardKeyFor: modelCardKeyFor,
              isSameModel: isSameModel,
              onNavigateToViewer: onNavigateToViewer,
              onShowModelActions: onShowModelActions,
              onAddNewTask: onAddNewTask,
            );
          },
          childCount: sortedKeys.length,
        ),
      ),
    );
  }

  DateTime _newestTime(List<Map<String, dynamic>> models) {
    return models
        .map((m) => DateTime.tryParse(m['created_at']?.toString() ?? '') ?? DateTime(0))
        .reduce((a, b) => a.isAfter(b) ? a : b);
  }
}

const double _kCardWidth = 170.0;
const double _kCardGap = 12.0;
const double _kSlotWidth = _kCardWidth + _kCardGap;
const double _kTimelineHeight = 58.0;
const double _kNodeRadius = 7.0;
const double _kLineHeight = 3.5;
const Color _kTimelineColor = Color(0xFFCC9A5C); // 橙色微灰

class _TimePeelingSlot extends StatelessWidget {
  final String name;
  final List<Map<String, dynamic>> models;
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Color textColor;
  final Color hintTextColor;
  final Color hintColor;
  final Map<String, dynamic>? activeModelAction;
  final GlobalKey Function(Map<String, dynamic>) modelCardKeyFor;
  final bool Function(Map<String, dynamic>?, Map<String, dynamic>?) isSameModel;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final void Function(Map<String, dynamic>) onShowModelActions;
  final void Function(String name) onAddNewTask;

  const _TimePeelingSlot({
    required this.name,
    required this.models,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.textColor,
    required this.hintTextColor,
    required this.hintColor,
    required this.activeModelAction,
    required this.modelCardKeyFor,
    required this.isSameModel,
    required this.onNavigateToViewer,
    required this.onShowModelActions,
    required this.onAddNewTask,
  });

  @override
  Widget build(BuildContext context) {
    final lineColor = _kTimelineColor.withValues(alpha: 0.6);
    final nodeColor = _kTimelineColor.withValues(alpha: 0.95);
    final timeStyle = TextStyle(
      fontSize: 11,
      fontWeight: FontWeight.w500,
      color: _kTimelineColor.withValues(alpha: 0.8),
    );

    final slotBg = isDark
        ? Colors.white.withValues(alpha: 0.04)
        : Colors.white.withValues(alpha: 0.55);
    final slotBorder = isDark
        ? Colors.white.withValues(alpha: 0.07)
        : Colors.black.withValues(alpha: 0.06);

    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 16),
      child: Container(
        decoration: BoxDecoration(
          color: slotBg,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: slotBorder, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: isDark ? 0.12 : 0.04),
              blurRadius: 12,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 标题行
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: textColor,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: hintColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${models.length}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: hintColor,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            // 卡片 + 时间轴，统一水平滚动
            SizedBox(
              height: 220 + 10 + _kTimelineHeight,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 14),
                itemCount: models.length + 1, // +1 for add card
              itemBuilder: (context, i) {
                // 第一个位置：添加卡片
                if (i == 0) {
                  return SizedBox(
                    width: _kSlotWidth,
                    child: Column(
                      children: [
                        Expanded(
                          child: Padding(
                            padding: const EdgeInsets.only(right: _kCardGap),
                            child: GestureDetector(
                              onTap: () => onAddNewTask(name),
                              child: Container(
                                decoration: BoxDecoration(
                                  color: isDark
                                      ? Colors.white.withValues(alpha: 0.06)
                                      : Colors.white.withValues(alpha: 0.7),
                                  borderRadius: BorderRadius.circular(28),
                                  border: Border.all(
                                    color: _kTimelineColor.withValues(alpha: 0.35),
                                    width: 1.5,
                                  ),
                                ),
                                child: Center(
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.add_rounded,
                                        size: 36,
                                        color: _kTimelineColor.withValues(alpha: 0.7),
                                      ),
                                      const SizedBox(height: 6),
                                      Text(
                                        textLocalize("create"),
                                        style: TextStyle(
                                          fontSize: 12,
                                          fontWeight: FontWeight.w600,
                                          color: _kTimelineColor.withValues(alpha: 0.7),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                        // 空白占位，与时间轴对齐
                        const SizedBox(height: 10),
                        SizedBox(height: _kTimelineHeight),
                      ],
                    ),
                  );
                }
                final modelIndex = i - 1;
                final model = models[modelIndex];
                final cardKey = modelCardKeyFor(model);
                final isActionTarget = isSameModel(activeModelAction, model);
                final dt = DateTime.tryParse(
                  model['created_at']?.toString() ?? '',
                );
                final timeLabel = dt != null
                    ? '${dt.toLocal().month.toString().padLeft(2, '0')}/${dt.toLocal().day.toString().padLeft(2, '0')} ${dt.toLocal().hour.toString().padLeft(2, '0')}:${dt.toLocal().minute.toString().padLeft(2, '0')}'
                    : '--';

                return SizedBox(
                  width: _kSlotWidth,
                  child: Column(
                    children: [
                      // 卡片
                      Expanded(
                        child: Padding(
                          padding: const EdgeInsets.only(right: _kCardGap),
                          child: RepaintBoundary(
                            child: IgnorePointer(
                              ignoring: isActionTarget,
                              child: Opacity(
                                opacity: isActionTarget ? 0.0 : 1.0,
                                child: GestureDetector(
                                  onTap: () => onNavigateToViewer(model, null),
                                  onLongPressStart: (_) =>
                                      onShowModelActions(model),
                                  child: Container(
                                    key: cardKey,
                                    decoration: BoxDecoration(
                                      borderRadius: BorderRadius.circular(28),
                                      border: Border.all(
                                        color: isDark
                                            ? Colors.white.withValues(alpha: 0.08)
                                            : Colors.black.withValues(alpha: 0.06),
                                        width: 1,
                                      ),
                                    ),
                                    child: RecallModelTile(
                                      model: model,
                                      theme: theme,
                                      isDark: isDark,
                                      darkCard: darkCard,
                                      darkInput: darkInput,
                                      textColor: textColor,
                                      hintTextColor: hintTextColor,
                                      imageOnly: true,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      // 卡片与时间轴间距
                      const SizedBox(height: 10),
                      // 时间轴
                      SizedBox(
                        height: _kTimelineHeight,
                        child: _TimelineNode(
                          isFirst: modelIndex == 0,
                          isLast: modelIndex == models.length - 1,
                          lineColor: lineColor,
                          nodeColor: nodeColor,
                          timeLabel: timeLabel,
                          timeStyle: timeStyle,
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    ));
  }
}

/// 单个时间轴节点：横线 + 圆点 + 时间标签
class _TimelineNode extends StatelessWidget {
  final bool isFirst;
  final bool isLast;
  final Color lineColor;
  final Color nodeColor;
  final String timeLabel;
  final TextStyle timeStyle;

  const _TimelineNode({
    required this.isFirst,
    required this.isLast,
    required this.lineColor,
    required this.nodeColor,
    required this.timeLabel,
    required this.timeStyle,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // 横线 + 节点
        SizedBox(
          height: _kNodeRadius * 2 + 4,
          child: Stack(
            alignment: Alignment.center,
            children: [
              // 横线
              Positioned(
                left: isFirst ? _kSlotWidth / 2 : 0,
                right: isLast ? _kSlotWidth / 2 : 0,
                child: Container(
                  height: _kLineHeight,
                  color: lineColor,
                ),
              ),
              // 圆点，居中于卡片（含 gap 的一半偏移）
              Positioned(
                left: (_kCardWidth - _kNodeRadius * 2) / 2,
                child: Container(
                  width: _kNodeRadius * 2,
                  height: _kNodeRadius * 2,
                  decoration: BoxDecoration(
                    color: nodeColor,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        // 时间标签，居中于卡片
        SizedBox(
          width: _kCardWidth,
          child: Text(
            timeLabel,
            textAlign: TextAlign.center,
            style: timeStyle,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}
