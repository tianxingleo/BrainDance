import 'package:braindance/configs/motion_tokens.dart';
import 'package:flutter/material.dart';

class BDFadeInNetworkImage extends StatefulWidget {
  final String imageUrl;
  final Widget placeholder;
  final Widget errorWidget;
  final BoxFit fit;
  final Alignment alignment;
  final BorderRadius? borderRadius;
  final EdgeInsetsGeometry padding;
  final Color? backgroundColor;
  final Duration duration;
  final Curve curve;
  final ValueChanged<Size>? onImageLoaded;

  const BDFadeInNetworkImage({
    super.key,
    required this.imageUrl,
    required this.placeholder,
    required this.errorWidget,
    this.fit = BoxFit.cover,
    this.alignment = Alignment.center,
    this.borderRadius,
    this.padding = EdgeInsets.zero,
    this.backgroundColor,
    this.duration = BDMotion.durationNormal,
    this.curve = BDMotion.curveEnter,
    this.onImageLoaded,
  });

  @override
  State<BDFadeInNetworkImage> createState() => _BDFadeInNetworkImageState();
}

class _BDFadeInNetworkImageState extends State<BDFadeInNetworkImage>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _opacity;
  late final Animation<double> _scale;
  late ImageProvider _provider;
  ImageStream? _imageStream;
  ImageStreamListener? _listener;
  bool _hasImage = false;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: widget.duration);
    _opacity = CurvedAnimation(parent: _controller, curve: widget.curve);
    _scale = Tween<double>(begin: 0.985, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: widget.curve),
    );
    _provider = NetworkImage(widget.imageUrl);
    _resolveImage();
  }

  @override
  void didUpdateWidget(covariant BDFadeInNetworkImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.imageUrl != widget.imageUrl) {
      _detachStream();
      _provider = NetworkImage(widget.imageUrl);
      _controller.reset();
      _hasImage = false;
      _hasError = false;
      _resolveImage();
    }
  }

  @override
  void dispose() {
    _detachStream();
    _controller.dispose();
    super.dispose();
  }

  void _resolveImage() {
    final stream = _provider.resolve(const ImageConfiguration());
    _imageStream = stream;
    _listener = ImageStreamListener(
      (ImageInfo info, bool synchronousCall) {
        widget.onImageLoaded?.call(
          Size(info.image.width.toDouble(), info.image.height.toDouble()),
        );
        if (!mounted) return;
        setState(() {
          _hasImage = true;
          _hasError = false;
        });
        if (synchronousCall) {
          _controller.value = 1.0;
        } else {
          _controller.forward(from: 0);
        }
      },
      onError: (Object error, StackTrace? stackTrace) {
        if (!mounted) return;
        setState(() {
          _hasError = true;
          _hasImage = false;
        });
      },
    );
    stream.addListener(_listener!);
  }

  void _detachStream() {
    final stream = _imageStream;
    final listener = _listener;
    if (stream != null && listener != null) {
      stream.removeListener(listener);
    }
    _imageStream = null;
    _listener = null;
  }

  @override
  Widget build(BuildContext context) {
    final content = _hasError
        ? widget.errorWidget
        : Stack(
            fit: StackFit.expand,
            children: [
              widget.placeholder,
              if (_hasImage)
                AnimatedBuilder(
                  animation: _controller,
                  builder: (context, child) {
                    return Opacity(
                      opacity: _opacity.value,
                      child: Transform.translate(
                        offset: Offset(0, 6 * (1 - _opacity.value)),
                        child: Transform.scale(
                          scale: _scale.value,
                          child: child,
                        ),
                      ),
                    );
                  },
                  child: Image(
                    image: _provider,
                    fit: widget.fit,
                    alignment: widget.alignment,
                    filterQuality: FilterQuality.low,
                    gaplessPlayback: true,
                  ),
                ),
            ],
          );

    Widget result = Padding(padding: widget.padding, child: content);
    if (widget.borderRadius != null) {
      result = ClipRRect(borderRadius: widget.borderRadius!, child: result);
    }
    if (widget.backgroundColor != null) {
      result = ColoredBox(color: widget.backgroundColor!, child: result);
    }
    return result;
  }
}
