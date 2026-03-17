import 'package:flutter/material.dart';
import 'dart:ui';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// BrainDance 的悬浮式“导航岛”组件
/// 风格特点：半透明玻璃质感、重心偏移切换、统一动效曲线
class FloatingNavBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;
  final List<NavIslandItem> items;

  const FloatingNavBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
    required this.items,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final navBackground = isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.82)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.82);
    final navBorder = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
    final navShadow = Colors.black.withValues(alpha: isDark ? 0.22 : 0.05);
    final selectedBackground = isDark
        ? const Color(0xFFAEBAC7).withValues(alpha: 0.16)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.12);
    final selectedColor = isDark
        ? const Color(0xFFF4F7FA)
        : BDDesign.colorInkBlack;
    final unselectedColor = isDark
        ? const Color(0xFF98A3AF)
        : const Color(0xFF7F878D);

    return Positioned(
      bottom: 20.0,
      left: 20.0,
      right: 20.0,
      child: ClipRRect(
        borderRadius: BDDesign.radiusLarge,
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 18.0, sigmaY: 18.0),
          child: Container(
            height: 68.0,
            padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 8.0),
            decoration: BoxDecoration(
              color: navBackground,
              borderRadius: BDDesign.radiusLarge,
              border: Border.all(color: navBorder, width: 1.0),
              boxShadow: [
                BoxShadow(
                  color: navShadow,
                  blurRadius: 28,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final compact = constraints.maxWidth < 420;

                return Row(
                  children: List.generate(items.length, (index) {
                    final isSelected = currentIndex == index;
                    final item = items[index];

                    return Expanded(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 2.0),
                        child: GestureDetector(
                          onTap: () => onTap(index),
                          behavior: HitTestBehavior.opaque,
                          child: AnimatedContainer(
                            duration: BDMotion.durationNormal,
                            curve: BDMotion.curveFluid,
                            padding: EdgeInsets.symmetric(
                              horizontal: compact
                                  ? (isSelected ? 8.0 : 6.0)
                                  : (isSelected ? 12.0 : 8.0),
                              vertical: 8.0,
                            ),
                            decoration: BoxDecoration(
                              color: isSelected
                                  ? selectedBackground
                                  : Colors.transparent,
                              borderRadius: BorderRadius.circular(22.0),
                            ),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                AnimatedScale(
                                  scale: isSelected ? 1.1 : 1.0,
                                  duration: BDMotion.durationNormal,
                                  curve: BDMotion.curveFluid,
                                  child: Icon(
                                    item.icon,
                                    color: isSelected
                                        ? selectedColor
                                        : unselectedColor,
                                    size: 22,
                                  ),
                                ),
                                if (isSelected) const SizedBox(width: 6.0),
                                if (isSelected)
                                  Expanded(
                                    child: AnimatedSwitcher(
                                      duration: BDMotion.durationNormal,
                                      switchInCurve: BDMotion.curveFluid,
                                      switchOutCurve: BDMotion.curveFluid,
                                      transitionBuilder: (child, animation) {
                                        return FadeTransition(
                                          opacity: animation,
                                          child: SizeTransition(
                                            sizeFactor: animation,
                                            axis: Axis.horizontal,
                                            child: child,
                                          ),
                                        );
                                      },
                                      child: Text(
                                        item.label,
                                        key: ValueKey<String>(item.label),
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        softWrap: false,
                                        style: TextStyle(
                                          color: selectedColor,
                                          fontWeight: FontWeight.w600,
                                          fontSize: compact ? 11.0 : 12.0,
                                        ),
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    );
                  }),
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}

class NavIslandItem {
  final IconData icon;
  final String label;

  NavIslandItem({required this.icon, required this.label});
}
