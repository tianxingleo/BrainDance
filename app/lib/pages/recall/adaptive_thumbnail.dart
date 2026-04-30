/// 自适应缩略图组件
///
/// 包含自适应宽高比的帧缩略图、网络图片解析混入和封面网络图片组件。
library;

import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'model_grid_helpers.dart';

/// 自适应帧缩略图
///
/// 根据图片的宽高比自适应宽度，用于搜索结果中的匹配帧水平列表。
class AdaptiveFrameThumbnail extends StatefulWidget {
  final String imageUrl;
  final double? frameSim;
  final double height;
  final Color backgroundColor;

  const AdaptiveFrameThumbnail({
    super.key,
    required this.imageUrl,
    required this.frameSim,
    required this.height,
    required this.backgroundColor,
  });

  @override
  State<AdaptiveFrameThumbnail> createState() =>
      AdaptiveFrameThumbnailState();
}

/// 网络图片解析混入
///
/// 为需要按 URL 加载网络图片的 StatefulWidget 提供图片加载、缓存和错误处理能力。
mixin NetworkImageResolverMixin<T extends StatefulWidget> on State<T> {
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

class AdaptiveFrameThumbnailState extends State<AdaptiveFrameThumbnail>
    with NetworkImageResolverMixin {
  @override
  String get imageUrl => widget.imageUrl;

  @override
  void didUpdateWidget(covariant AdaptiveFrameThumbnail oldWidget) {
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
                    formatSimilarity(widget.frameSim!),
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

/// 封面网络图片组件
///
/// 使用 [NetworkImageResolverMixin] 加载并展示封面图片，支持错误回退。
class CoverNetworkImage extends StatefulWidget {
  final String imageUrl;
  final Color backgroundColor;
  final Widget errorWidget;

  const CoverNetworkImage({
    super.key,
    required this.imageUrl,
    required this.backgroundColor,
    required this.errorWidget,
  });

  @override
  State<CoverNetworkImage> createState() => CoverNetworkImageState();
}

class CoverNetworkImageState extends State<CoverNetworkImage>
    with NetworkImageResolverMixin {
  @override
  String get imageUrl => widget.imageUrl;

  @override
  void didUpdateWidget(covariant CoverNetworkImage oldWidget) {
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
