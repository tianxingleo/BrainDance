import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../extra_func/theme_animation_notifier.dart';
import 'dart:ui' as ui;

class ThemeAnimationOverlay extends ConsumerStatefulWidget {
  final Widget child;

  const ThemeAnimationOverlay({super.key, required this.child});

  @override
  ConsumerState<ThemeAnimationOverlay> createState() =>
      _ThemeAnimationOverlayState();
}

class _ThemeAnimationOverlayState extends ConsumerState<ThemeAnimationOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        duration: const Duration(milliseconds: 800), vsync: this);
    
    _animation = CurvedAnimation(
        parent: _controller, 
        curve: Curves.easeOutQuint,
    );
    
    // Listen to provider changes to trigger animation
    // But ref.listen in build() is safer? No, usually in build or initState/dispose
    // But since it's a provider change triggering animation, using ref.listen in build is better practice.
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Listen to trigger animation
    ref.listen<ThemeAnimationState>(themeAnimationProvider, (previous, next) {
      if (next.isAnimating && !(previous?.isAnimating ?? false)) {
        _controller.reset();
        _controller.forward().then((_) {
          ref.read(themeAnimationProvider.notifier).end();
        });
      }
    });

    final animationState = ref.watch(themeAnimationProvider);
    final isAnimating = animationState.isAnimating;
    final screenshot = animationState.screenshot;

    return Stack(
      children: [
        // Layer 0: The real app (which will update to the NEW theme deeply inside)
        // We wrap child with RepaintBoundary here to make sure we can capture it
        RepaintBoundary(
            key: themeAnimationKey,
            child: widget.child,
        ),

        // Layer 1: The OLD theme screenshot with a HOLE being punched out
        if (isAnimating && screenshot != null)
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(
                painter: _ScreenshotPainter(
                  image: screenshot,
                  center: animationState.center,
                  animation: _animation,
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class _ScreenshotPainter extends CustomPainter {
  final ui.Image image;
  final Offset center;
  final Animation<double> animation;

  _ScreenshotPainter({
    required this.image,
    required this.center,
    required this.animation,
  }) : super(repaint: animation);

  @override
  void paint(Canvas canvas, Size size) {
    final radiusValid = animation.value;
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    final double specificRadius = _maxDistance(center, size) * radiusValid;
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());

    canvas.saveLayer(dst, Paint());
    canvas.drawImageRect(image, src, dst, Paint()..filterQuality = FilterQuality.low);
    canvas.drawCircle(center, specificRadius, Paint()..blendMode = BlendMode.clear);
    canvas.restore();
  }

  double _maxDistance(Offset p, Size size) {
    double dist(Offset a, Offset b) => sqrt(pow(a.dx - b.dx, 2) + pow(a.dy - b.dy, 2));
    final tl = Offset(0, 0);
    final tr = Offset(size.width, 0);
    final bl = Offset(0, size.height);
    final br = Offset(size.width, size.height);
    return [dist(p, tl), dist(p, tr), dist(p, bl), dist(p, br)].reduce(max);
  }

  @override
  bool shouldRepaint(covariant _ScreenshotPainter oldDelegate) {
    return oldDelegate.image != image || oldDelegate.center != center;
  }
}
