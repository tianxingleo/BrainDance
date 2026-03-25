import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../extra_func/theme_animation_notifier.dart';
import 'dart:ui' as ui;

class ThemeAnimationOverlay extends ConsumerStatefulWidget {
  final Widget child;

  const ThemeAnimationOverlay({
    Key? key,
    required this.child,
  }) : super(key: key);

  @override
  ConsumerState<ThemeAnimationOverlay> createState() =>
      _ThemeAnimationOverlayState();
}

class _ThemeAnimationOverlayState extends ConsumerState<ThemeAnimationOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  
  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        duration: const Duration(milliseconds: 500), vsync: this);
    
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
              // Allow touches to pass through during animation? 
              // Usually safer to block touches or ignore. Ignoring is better if the animation is purely visual overlay.
              child: AnimatedBuilder(
                animation: _controller,
                builder: (context, child) {
                  return CustomPaint(
                    painter: _ScreenshotPainter(
                      image: screenshot,
                      center: animationState.center,
                      radiusValid: _controller.value,
                    ),
                  );
                },
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
  final double radiusValid; // 0.0 to 1.0

  _ScreenshotPainter({
    required this.image,
    required this.center,
    required this.radiusValid,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // We want to draw the image BUT with a hole in it.
    // The hole allows the content below (new theme) to show through.
    
    final paint = Paint();
    
    // Determine max radius needed to cover the screen from center
    // Distance to farthest corner
    final double maxRadius = _maxDistance(center, size);
    final double specificRadius = maxRadius * radiusValid;

    // We can use saveLayer with a blend mode or path operation.
    // Simplest approach: Use a Path that covers the whole screen minus the circle.
    // Then clip to that path and draw the image.

    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addOval(Rect.fromCircle(center: center, radius: specificRadius))
      ..fillType = PathFillType.evenOdd; // This subtracts the circle from the rect

    canvas.save();
    canvas.clipPath(path);
    
    // Draw the image to fit the screen size.
    // Assuming the screenshot matches the screen size exactly.
    // If pixel ratio makes image larger, we need to scale.
    // ui.Image dimensions are in pixels, canvas is in logical pixels.
    final double scaleX = size.width / image.width.toDouble();
    final double scaleY = size.height / image.height.toDouble();
    
    // Typically screenshots from toImage(pixelRatio: view.devicePixelRatio) have higher res.
    // We should draw it scaled down into the canvas rect.
    
    // Better: use drawImageRect
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    
    canvas.drawImageRect(image, src, dst, paint);
    
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
    return oldDelegate.image != image ||
           oldDelegate.center != center ||
           oldDelegate.radiusValid != radiusValid;
  }
}
