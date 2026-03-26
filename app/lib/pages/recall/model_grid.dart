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
  final void Function(Map<String, dynamic> model, {bool imageOnly}) onShowModelActions;
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
                            onLongPressStart: (_) => onShowModelActions(model, imageOnly: false), // Here we are sending false for standard view cards
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
                    onLongPressStart: (_) => onShowModelActions(model, imageOnly: false), // Here we are sending false for standard view cards
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

class RecallModelActionOverlay extends StatefulWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Map<String, dynamic> model;
  final Rect rect;
  final VoidCallback onDismiss;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final Future<void> Function(Map<String, dynamic>) onShowModelDetails;
  final Future<void> Function(Map<String, dynamic>) onDownloadModel;
  final Future<void> Function(Map<String, dynamic>) onShareModelToCommunity;
  final Future<void> Function(Map<String, dynamic>) onRenameModel;
  final Future<void> Function(Map<String, dynamic>) onDeleteCloudModel;
  final String Function(String) toPublicUrl;

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
    required this.onDownloadModel,
    required this.onShareModelToCommunity,
    required this.onRenameModel,
    required this.onDeleteCloudModel,
    required this.toPublicUrl,
  });

  @override
  State<RecallModelActionOverlay> createState() =>
      RecallModelActionOverlayState();
}

class RecallModelActionOverlayState extends State<RecallModelActionOverlay>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _blurOpacityAnimation;
  late final Animation<double> _cardScaleAnimation;
  late final Animation<double> _cardTranslateAnimation;
  late final Animation<double> _menuOpacityAnimation;
  late final Animation<double> _menuTranslateAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 320),
      reverseDuration: const Duration(milliseconds: 240),
    );

    _blurOpacityAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    );

    _cardScaleAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
      reverseCurve: Curves.easeInCubic,
    );

    _cardTranslateAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    );

    _menuOpacityAnimation = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.1, 1.0, curve: Curves.easeOutCubic),
      reverseCurve: Curves.easeInCubic,
    );

    _menuTranslateAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
      reverseCurve: Curves.easeInCubic,
    );

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> hide() async {
    if (mounted) {
      await _controller.reverse();
    }
  }

  @override
  Widget build(BuildContext context) {
    const deleteLabel = '删除云端模型';
    final screenWidth = MediaQuery.of(context).size.width;
    const screenPadding = 16.0;
    const horizontalGap = 12.0;
    const actionWidth = 128.0;
    final maxLeft = screenWidth - screenPadding - actionWidth;
    final actionLeft = (widget.rect.right + horizontalGap)
        .clamp(screenPadding, maxLeft)
        .toDouble();

    return Positioned.fill(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: widget.onDismiss,
        child: Stack(
          children: [
            AnimatedBuilder(
              animation: _blurOpacityAnimation,
              builder: (context, child) {
                final value = _blurOpacityAnimation.value;
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
              left: widget.rect.left,
              top: widget.rect.top,
              width: widget.rect.width,
              height: widget.rect.height,
              child: AnimatedBuilder(
                animation: _controller,
                builder: (context, child) {
                  final tValue = _cardTranslateAnimation.value;
                  final sValue = _cardScaleAnimation.value;
                  return Transform.translate(
                    offset: Offset(0, -10 * tValue),
                    child: Transform.scale(
                      scale: 1 + (0.045 * sValue),
                      alignment: Alignment.center,
                      child: GestureDetector(
                  onTap: () => widget.onNavigateToViewer(widget.model, null),
                  onLongPress: widget.onDismiss,
                  child: RecallModelTile(
                    model: widget.model,
                    theme: widget.theme,
                    isDark: widget.isDark,
                    darkCard: widget.darkCard,
                    darkInput: widget.darkInput,
                    textColor: widget.isDark
                        ? const Color(0xFFFFFFFF)
                        : BDDesign.colorInkBlack,
                    hintTextColor: widget.isDark
                        ? const Color(0xFF888888)
                        : widget.theme.fontGyColor3,
                    elevated: true,
                    elevationProgress: sValue,
                    toPublicUrl: widget.toPublicUrl,
                    imageOnly: widget.model['_imageOnly'] == true, // Correctly read boolean
                  ),
                ),
                    ),
                  );
                },
              ),
            ),
            Positioned.fill(
              child: AnimatedBuilder(
                animation: _blurOpacityAnimation,
                builder: (context, child) {
                  final bValue = _blurOpacityAnimation.value;
                  return IgnorePointer(
                    child: Opacity(
                      opacity: bValue,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              widget.theme.brandColor4.withValues(alpha: widget.isDark ? 0.25 : 0.15),
                              Colors.transparent,
                            ],
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
            Positioned(
              left: actionLeft,
              top: widget.rect.top + 24,
              child: AnimatedBuilder(
                animation: _controller,
                builder: (context, child) {
                  final mValue = _menuTranslateAnimation.value;
                  final oValue = _menuOpacityAnimation.value;
                  return Opacity(
                    opacity: oValue.clamp(0.0, 1.0),
                    child: Transform.translate(
                      offset: Offset(18 * (1 - mValue), 0),
                      child: child,
                    ),
                  );
                },
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(18),
                    onTap: () async {
                      widget.onDismiss();
                      await widget.onShareModelToCommunity(widget.model);
                    },
                    child: Ink(
                      width: actionWidth,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        color: widget.isDark
                            ? const Color(0xEE1F2430)
                            : Colors.white.withAlpha(236),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(
                          color: widget.isDark
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
                            isDark: widget.isDark,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onShowModelDetails(widget.model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.edit_rounded,
                            label: '重命名',
                            isDark: widget.isDark,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onRenameModel(widget.model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.download_rounded,
                            label: textLocalize('recall_download_model'),
                            isDark: widget.isDark,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onDownloadModel(widget.model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.delete_outline_rounded,
                            label: deleteLabel,
                            isDark: widget.isDark,
                            destructive: true,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onDeleteCloudModel(widget.model);
                            },
                          ),
                          const SizedBox(height: 6),
                          _ActionMenuItem(
                            icon: Icons.public_rounded,
                            label: '分享到社区',
                            isDark: widget.isDark,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onShareModelToCommunity(
                                widget.model,
                              );
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
                  ? FittedBox(
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
          ? ClipRect(
              child: FittedBox(
                fit: BoxFit.cover,
                clipBehavior: Clip.hardEdge,
                child: SizedBox(
                  width: img.width.toDouble(),
                  height: img.height.toDouble(),
                  child: RawImage(image: img, filterQuality: FilterQuality.low),
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
        : './models/scene_auto_sync_raw.ply';
    final radius = BorderRadius.circular(28.0);

    return Container(
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: radius,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(
              elevationProgress != null
                  ? (elevationProgress! > 0 ? (20 + (46 - 20) * elevationProgress!).round() : 20)
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

/// Time Peeling: group models by name, with a horizontal time strip for each group
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
  final void Function(Map<String, dynamic> model, {bool imageOnly}) onShowModelActions;
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
    final textColor = _resolveTextColor(isDark);
    final hintTextColor = _resolveHintTextColor(isDark, theme);
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue;

    final sortedKeys = groupedModels.keys.toList()
      ..sort((a, b) {
        final ta = _newestTime(groupedModels[a]!);
        final tb = _newestTime(groupedModels[b]!);
        return tb.compareTo(ta);
      });

    return SliverPadding(
      padding: const EdgeInsets.fromLTRB(0, 6, 0, 16),
      sliver: SliverList(
        delegate: SliverChildBuilderDelegate((context, index) {
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
        }, childCount: sortedKeys.length),
      ),
    );
  }

  DateTime _newestTime(List<Map<String, dynamic>> models) {
    return models
        .map(
          (m) =>
              DateTime.tryParse(m['created_at']?.toString() ?? '') ??
              DateTime(0),
        )
        .reduce((a, b) => a.isAfter(b) ? a : b);
  }
}

const Color _kTimelineColor = Color(0xFFCC9A5C); // muted orange
const double _kCarouselHeight = 240.0;
const double _kViewportFraction = 0.52;
const double _kEdgeFadeWidth = 36.0;

class _TimePeelingSlot extends StatefulWidget {
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
  final void Function(Map<String, dynamic> model, {bool imageOnly}) onShowModelActions;
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
  State<_TimePeelingSlot> createState() => _TimePeelingSlotState();
}

class _TimePeelingSlotState extends State<_TimePeelingSlot> {
  late PageController _pageController;
  double _currentPage = 1.0; // matches initialPage: 1 (first model card)

  /// Total items: create button first, then models (newest→oldest).
  int get _totalCount => widget.models.length + 1;

  @override
  void initState() {
    super.initState();
    _pageController = PageController(
      viewportFraction: _kViewportFraction,
      initialPage: 1, // index 0 is create card, 1 is newest model
    );
    _pageController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _pageController.removeListener(_onScroll);
    _pageController.dispose();
    super.dispose();
  }

  void _onScroll() {
    if (_pageController.page != null) {
      setState(() => _currentPage = _pageController.page!);
    }
  }

  String _timeLabelFor(int modelIndex) {
    if (modelIndex < 0 || modelIndex >= widget.models.length) return '';
    final dt = DateTime.tryParse(
      widget.models[modelIndex]['created_at']?.toString() ?? '',
    );
    if (dt == null) return '--';
    final local = dt.toLocal();
    return '${local.month.toString().padLeft(2, '0')}/${local.day.toString().padLeft(2, '0')} '
        '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final slotBg = widget.isDark
        ? Colors.white.withValues(alpha: 0.04)
        : Colors.white.withValues(alpha: 0.55);
    final slotBorder = widget.isDark
        ? Colors.white.withValues(alpha: 0.07)
        : Colors.black.withValues(alpha: 0.06);

    // Current selected page index (rounded page)
    final selectedPage = _currentPage.round().clamp(0, _totalCount - 1);
    // index 0 = create card, index 1+ = model cards
    final isModelSelected = selectedPage >= 1;
    final timeLabel = isModelSelected ? _timeLabelFor(selectedPage - 1) : '';

    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 16),
      child: Container(
        decoration: BoxDecoration(
          color: slotBg,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: slotBorder, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(
                alpha: widget.isDark ? 0.12 : 0.04,
              ),
              blurRadius: 12,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title row
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      widget.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: widget.textColor,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: widget.hintColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${widget.models.length}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: widget.hintColor,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            // Carousel with ShaderMask edge fade
            SizedBox(
              height: _kCarouselHeight,
              child: ShaderMask(
                shaderCallback: (Rect bounds) {
                  return LinearGradient(
                    begin: Alignment.centerLeft,
                    end: Alignment.centerRight,
                    colors: const [
                      Colors.transparent,
                      Colors.white,
                      Colors.white,
                      Colors.transparent,
                    ],
                    stops: [
                      0.0,
                      _kEdgeFadeWidth / bounds.width,
                      1.0 - _kEdgeFadeWidth / bounds.width,
                      1.0,
                    ],
                  ).createShader(bounds);
                },
                blendMode: BlendMode.dstIn,
                child: PageView.builder(
                  controller: _pageController,
                  itemCount: _totalCount,
                  clipBehavior: Clip.hardEdge,
                  itemBuilder: (context, index) {
                    final distance = (index - _currentPage).abs().clamp(
                      0.0,
                      1.0,
                    );
                    final scale = ui.lerpDouble(1.0, 0.82, distance)!;
                    final opacity = ui.lerpDouble(1.0, 0.5, distance)!;

                    // First item is the create button
                    if (index == 0) {
                      return _buildCarouselItem(
                        scale: scale,
                        opacity: opacity,
                        isSelected: selectedPage == index,
                        child: _buildCreateCard(),
                      );
                    }

                    final model = widget.models[index - 1];
                    final cardKey = widget.modelCardKeyFor(model);
                    final isActionTarget = widget.isSameModel(
                      widget.activeModelAction,
                      model,
                    );

                    return _buildCarouselItem(
                      scale: scale,
                      opacity: isActionTarget ? 0.0 : opacity,
                      isSelected: selectedPage == index,
                      child: IgnorePointer(
                        ignoring: isActionTarget,
                        child: GestureDetector(
                          onTap: () => widget.onNavigateToViewer(model, null),
                          onLongPressStart: (_) =>
                              widget.onShowModelActions(model, imageOnly: true),
                          child: Container(
                            key: cardKey,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(28),
                              border: Border.all(
                                color: widget.isDark
                                    ? Colors.white.withValues(alpha: 0.08)
                                    : Colors.black.withValues(alpha: 0.06),
                                width: 1,
                              ),
                            ),
                            child: RecallModelTile(
                              model: model,
                              theme: widget.theme,
                              isDark: widget.isDark,
                              darkCard: widget.darkCard,
                              darkInput: widget.darkInput,
                              textColor: widget.textColor,
                              hintTextColor: widget.hintTextColor,
                              imageOnly: true,
                            ),
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
            // Timeline with connected nodes
            SizedBox(
              height: 48,
              child: ClipRect(
                child: CustomPaint(
                  painter: _TimelinePainter(
                    modelCount: widget.models.length,
                    currentPage: _currentPage,
                    timeLabel: timeLabel,
                    color: _kTimelineColor,
                    viewportFraction: _kViewportFraction,
                  ),
                  size: Size.infinite,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCarouselItem({
    required double scale,
    required double opacity,
    required bool isSelected,
    required Widget child,
  }) {
    return Center(
      child: Transform.scale(
        scale: scale,
        child: Opacity(
          opacity: opacity.clamp(0.0, 1.0),
          child: Container(
            decoration: isSelected
                ? BoxDecoration(
                    borderRadius: BorderRadius.circular(30),
                    boxShadow: [
                      BoxShadow(
                        color: _kTimelineColor.withValues(alpha: 0.18),
                        blurRadius: 18,
                        offset: const Offset(0, 6),
                      ),
                    ],
                  )
                : null,
            child: child,
          ),
        ),
      ),
    );
  }

  Widget _buildCreateCard() {
    return GestureDetector(
      onTap: () => widget.onAddNewTask(widget.name),
      child: Container(
        decoration: BoxDecoration(
          color: widget.isDark
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
    );
  }
}

class _TimelinePainter extends CustomPainter {
  /// Number of model cards (excludes the create card at index 0).
  final int modelCount;

  /// Current page position from PageController (index 0 = create card).
  final double currentPage;
  final String timeLabel;
  final Color color;
  final double viewportFraction;

  _TimelinePainter({
    required this.modelCount,
    required this.currentPage,
    required this.color,
    required this.timeLabel,
    required this.viewportFraction,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (modelCount == 0) return;

    final lineY = 10.0;
    final slotWidth = size.width * viewportFraction;
    final centerX = size.width / 2;

    // Model cards occupy PageView indices 1..modelCount.
    // Node i (0-based model index) corresponds to PageView index (i + 1).
    // Its center X = centerX + (pageIndex - currentPage) * slotWidth.
    List<double> nodeXs = [];
    for (int i = 0; i < modelCount; i++) {
      final pageIndex = i + 1; // offset by create card
      final dx = centerX + (pageIndex - currentPage) * slotWidth;
      nodeXs.add(dx);
    }

    // Draw the connecting line between first and last node, clamped to bounds
    final linePaint = Paint()
      ..color = color.withValues(alpha: 0.25)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final lineLeft = nodeXs.first.clamp(0.0, size.width);
    final lineRight = nodeXs.last.clamp(0.0, size.width);
    if (lineRight > lineLeft) {
      canvas.drawLine(
        Offset(lineLeft, lineY),
        Offset(lineRight, lineY),
        linePaint,
      );
    }

    // Selected model index (0-based): PageView selectedPage - 1
    final selectedPageIndex = currentPage.round().clamp(0, modelCount);
    // Convert to model index; -1 means create card is selected (no highlight)
    final selectedModelIndex = selectedPageIndex - 1;

    final normalRadius = 3.5;
    final selectedRadius = 5.5;

    for (int i = 0; i < modelCount; i++) {
      final x = nodeXs[i];
      if (x < -20 || x > size.width + 20) continue;

      // Distance from the selected model node (continuous for smooth animation)
      final distFromSelected = ((i + 1) - currentPage).abs().clamp(0.0, 1.0);
      final radius = ui.lerpDouble(
        selectedRadius,
        normalRadius,
        distFromSelected,
      )!;
      final alpha = ui.lerpDouble(0.95, 0.4, distFromSelected)!;

      final dotPaint = Paint()
        ..color = color.withValues(alpha: alpha)
        ..style = PaintingStyle.fill;

      canvas.drawCircle(Offset(x, lineY), radius, dotPaint);
    }

    // Draw time label below the selected model node
    if (timeLabel.isNotEmpty &&
        selectedModelIndex >= 0 &&
        selectedModelIndex < modelCount) {
      final labelX = nodeXs[selectedModelIndex];
      final textPainter = TextPainter(
        text: TextSpan(
          text: timeLabel,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: color.withValues(alpha: 0.85),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final textX = (labelX - textPainter.width / 2).clamp(
        4.0,
        size.width - textPainter.width - 4,
      );
      textPainter.paint(canvas, Offset(textX, lineY + selectedRadius + 6));
    }
  }

  @override
  bool shouldRepaint(covariant _TimelinePainter old) =>
      old.currentPage != currentPage ||
      old.modelCount != modelCount ||
      old.timeLabel != timeLabel;
}
