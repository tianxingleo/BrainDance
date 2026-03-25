import 'package:flutter/material.dart';
import '../configs/app_config.dart';
import '../configs/app_theme.dart';
import '../configs/motion_tokens.dart';
import '../widgets/bd_surfaces.dart';

/// TimePeeling 页面 — 占位页面，后续开发
class TimePeelingPage extends StatelessWidget {
  const TimePeelingPage({super.key});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.38)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.48);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.layers_rounded,
                  size: 48,
                  color: hintColor,
                ),
                const SizedBox(height: 16),
                Text(
                  'Time Peeling',
                  style: TextStyle(
                    color: hintColor,
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  textLocalize('coming_soon'),
                  style: TextStyle(
                    color: hintColor.withValues(alpha: 0.6),
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
