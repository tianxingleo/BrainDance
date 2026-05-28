import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../extra_func/theme_animation_notifier.dart';
import '../configs/set_config.dart';
import '../configs/app_config.dart';
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

  static const int _baseDurationMs = 1200;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        duration: const Duration(milliseconds: _baseDurationMs), vsync: this);

    _animation = CurvedAnimation(
      parent: _controller,
      curve: const Cubic(0.33, 0.0, 0.67, 1.0),
    );

    _animation.addListener(_updateFraction);
  }

  void _updateFraction() {
    final state = ref.read(themeAnimationProvider);
    if (!state.isAnimating) return;
    final curveValue = _animation.value;
    if (state.isReversing) {
      themeAnimationFraction.value = state.startFraction * (1.0 - curveValue);
    } else {
      themeAnimationFraction.value = state.startFraction + (1.0 - state.startFraction) * curveValue;
    }
  }

  @override
  void dispose() {
    _animation.removeListener(_updateFraction);
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    ref.listen<ThemeAnimationState>(themeAnimationProvider, (previous, next) {
      if (next.isAnimating && !(previous?.isAnimating ?? false)) {
        _controller.duration = const Duration(milliseconds: _baseDurationMs);
        _controller.reset();
        _controller.forward().orCancel.then((_) {
          ref.read(themeAnimationProvider.notifier).end();
        }, onError: (_) {});
      } else if (next.isAnimating && (previous?.isReversing != next.isReversing)) {
        final distance = next.isReversing
            ? next.startFraction
            : (1.0 - next.startFraction);
        final ms = (distance * _baseDurationMs).round().clamp(100, _baseDurationMs);
        _controller.duration = Duration(milliseconds: ms);
        _controller.reset();
        _controller.forward().orCancel.then((_) {
          final endState = ref.read(themeAnimationProvider);
          if (endState.isReversing) {
            SetConfig.setNightMode(!AppConfig.isNightMode, ref);
            SetConfig.saveMsgToFile();
            WidgetsBinding.instance.addPostFrameCallback((_) {
              ref.read(themeAnimationProvider.notifier).end();
            });
          } else {
            ref.read(themeAnimationProvider.notifier).end();
          }
        }, onError: (_) {});
      }
    });

    final animationState = ref.watch(themeAnimationProvider);
    final isAnimating = animationState.isAnimating;
    final screenshot = animationState.screenshot;

    return Stack(
      children: [
        RepaintBoundary(
          key: themeAnimationKey,
          child: widget.child,
        ),
        if (isAnimating && screenshot != null)
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(
                painter: _ScreenshotPainter(
                  image: screenshot,
                  center: animationState.center,
                  animation: _animation,
                  mode: animationState.mode,
                  isReversing: animationState.isReversing,
                  startFraction: animationState.startFraction,
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
  final ThemeTransitionMode mode;
  final bool isReversing;
  final double startFraction;

  _ScreenshotPainter({
    required this.image,
    required this.center,
    required this.animation,
    required this.mode,
    required this.isReversing,
    required this.startFraction,
  }) : super(repaint: animation);

  @override
  void paint(Canvas canvas, Size size) {
    final curveValue = animation.value;
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final maxDist = _maxDistance(center, size);

    double fraction;
    if (isReversing) {
      fraction = startFraction * (1.0 - curveValue);
    } else {
      fraction = startFraction + (1.0 - startFraction) * curveValue;
    }

    if (mode == ThemeTransitionMode.expandHole) {
      final radius = maxDist * fraction;
      final holePath = Path()
        ..addRect(dst)
        ..addOval(Rect.fromCircle(center: center, radius: radius));
      holePath.fillType = PathFillType.evenOdd;
      canvas.save();
      canvas.clipPath(holePath);
      canvas.drawImageRect(image, src, dst, Paint()..filterQuality = FilterQuality.none);
      canvas.restore();
    } else {
      final radius = maxDist * (1.0 - fraction);
      canvas.save();
      canvas.clipPath(Path()..addOval(Rect.fromCircle(center: center, radius: radius)));
      canvas.drawImageRect(image, src, dst, Paint()..filterQuality = FilterQuality.none);
      canvas.restore();
    }
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
    return oldDelegate.image != image || oldDelegate.center != center || oldDelegate.mode != mode;
  }
}
