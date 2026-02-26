import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../configs/app_config.dart';

class DynamicGradientBackground extends StatefulWidget {
  final Widget child;
  const DynamicGradientBackground({super.key, required this.child});

  @override
  State<DynamicGradientBackground> createState() => _DynamicGradientBackgroundState();
}

class _DynamicGradientBackgroundState extends State<DynamicGradientBackground> with SingleTickerProviderStateMixin {
  late final AnimationController _bgAnimController;
  late final Animation<Alignment> _topAlignment;
  late final Animation<Alignment> _bottomAlignment;

  @override
  void initState() {
    super.initState();
    _bgAnimController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 40),
    )..repeat(reverse: true);

    _topAlignment = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topLeft, end: Alignment.topRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topRight, end: Alignment.bottomRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomRight, end: Alignment.bottomLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomLeft, end: Alignment.topLeft),
        weight: 1,
      ),
    ]).animate(CurvedAnimation(parent: _bgAnimController, curve: Curves.easeInOut));

    _bottomAlignment = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomRight, end: Alignment.bottomLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomLeft, end: Alignment.topLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topLeft, end: Alignment.topRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topRight, end: Alignment.bottomRight),
        weight: 1,
      ),
    ]).animate(CurvedAnimation(parent: _bgAnimController, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _bgAnimController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: AnimatedBuilder(
            animation: _bgAnimController,
            builder: (context, child) {
              final isDark = Theme.of(context).brightness == Brightness.dark;
              return Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: _topAlignment.value,
                    end: _bottomAlignment.value,
                    colors: isDark
                        ? [
                            AppConfig.primaryColor.withValues(alpha: 0.25),
                            TDTheme.of(context).brandColor8.withValues(alpha: 0.15),
                            TDTheme.of(context).grayColor14,
                            AppConfig.primaryColor.withValues(alpha: 0.08),
                          ]
                        : [
                            TDTheme.of(context).brandColor4.withValues(alpha: 0.2),
                            AppConfig.primaryColor.withValues(alpha: 0.1),
                            TDTheme.of(context).grayColor1,
                            AppConfig.primaryColor.withValues(alpha: 0.05),
                          ],
                    stops: const [0.0, 0.4, 0.8, 1.0],
                  ),
                ),
              );
            },
          ),
        ),
        widget.child,
      ],
    );
  }
}
