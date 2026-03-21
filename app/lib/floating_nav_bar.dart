import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';

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
                final itemCount = items.length;
                final slotWidth = constraints.maxWidth / itemCount;
                final pillWidth = math.min(
                  constraints.maxWidth - 4.0,
                  slotWidth + (compact ? 28.0 : 42.0),
                );
                final pillCenter = slotWidth * (currentIndex + 0.5);
                final pillLeft = (pillCenter - pillWidth / 2).clamp(
                  2.0,
                  constraints.maxWidth - pillWidth - 2.0,
                );

                return Stack(
                  children: [
                    Positioned.fill(
                      child: IgnorePointer(
                        child: TweenAnimationBuilder<double>(
                          tween: Tween<double>(begin: pillLeft, end: pillLeft),
                          duration: BDMotion.durationSlow,
                          curve: Curves.easeOutCubic,
                          builder: (context, animatedLeft, child) {
                            return Stack(
                              children: [
                                Transform.translate(
                                  offset: Offset(animatedLeft, 0.0),
                                  child: SizedBox(
                                    width: pillWidth,
                                    height: double.infinity,
                                    child: DecoratedBox(
                                      decoration: BoxDecoration(
                                        color: selectedBackground,
                                        borderRadius: BorderRadius.circular(
                                          22.0,
                                        ),
                                        boxShadow: [
                                          BoxShadow(
                                            color: selectedColor.withValues(
                                              alpha: 0.05,
                                            ),
                                            blurRadius: 16,
                                            offset: const Offset(0, 6),
                                          ),
                                        ],
                                      ),
                                      child: _SelectedPillContent(
                                        item: items[currentIndex],
                                        compact: compact,
                                        color: selectedColor,
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            );
                          },
                        ),
                      ),
                    ),
                    Row(
                      children: List.generate(items.length, (index) {
                        final item = items[index];
                        final distance = (index - currentIndex).abs();
                        final direction = index < currentIndex ? -1.0 : 1.0;
                        final shiftBase = compact ? 8.0 : 10.0;
                        final visualShift = index == currentIndex
                            ? 0.0
                            : direction *
                                  (distance == 1
                                      ? shiftBase
                                      : distance == 2
                                      ? shiftBase * 0.42
                                      : 0.0);

                        return Expanded(
                          child: Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 2.0,
                            ),
                            child: _FloatingNavBarItem(
                              item: item,
                              compact: compact,
                              isSelected: currentIndex == index,
                              visualShift: visualShift,
                              onTap: () => onTap(index),
                              selectedColor: selectedColor,
                              unselectedColor: unselectedColor,
                            ),
                          ),
                        );
                      }),
                    ),
                  ],
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

class _SelectedPillContent extends StatelessWidget {
  const _SelectedPillContent({
    required this.item,
    required this.compact,
    required this.color,
  });

  final NavIslandItem item;
  final bool compact;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final iconSize = compact ? 21.0 : 22.0;
    final horizontalPadding = compact ? 10.0 : 12.0;
    final labelGap = compact ? 3.0 : 4.0;

    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0.0, end: 1.0),
      duration: BDMotion.durationSlow,
      curve: Curves.easeOutCubic,
      builder: (context, progress, child) {
        final revealProgress = Curves.easeOutCubic.transform(progress);
        final bounceWave = math.sin(revealProgress * math.pi);

        return Padding(
          padding: EdgeInsets.symmetric(
            horizontal: horizontalPadding,
            vertical: compact ? 7.0 : 8.0,
          ),
          child: Row(
            children: [
              Transform.translate(
                offset: Offset(-1.2 * revealProgress, -1.4 * bounceWave),
                child: Transform.scale(
                  scale: 1.0 + 0.08 * bounceWave,
                  child: _AnimatedFillIcon(
                    icon: item.icon,
                    progress: revealProgress,
                    selectedColor: color,
                    unselectedColor: color.withValues(alpha: 0.68),
                    size: iconSize,
                  ),
                ),
              ),
              SizedBox(width: labelGap),
              Expanded(
                child: ClipRect(
                  child: Transform.translate(
                    offset: Offset((1 - revealProgress) * -8.0, 0.0),
                    child: Opacity(
                      opacity: revealProgress,
                      child: Text(
                        item.label,
                        maxLines: 1,
                        overflow: TextOverflow.fade,
                        softWrap: false,
                        style: TextStyle(
                          color: color,
                          fontWeight: FontWeight.w600,
                          fontSize: compact ? 13.0 : 14.0,
                          height: 1.0,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _FloatingNavBarItem extends StatelessWidget {
  const _FloatingNavBarItem({
    required this.item,
    required this.compact,
    required this.isSelected,
    required this.visualShift,
    required this.onTap,
    required this.selectedColor,
    required this.unselectedColor,
  });

  final NavIslandItem item;
  final bool compact;
  final bool isSelected;
  final double visualShift;
  final VoidCallback onTap;
  final Color selectedColor;
  final Color unselectedColor;

  @override
  Widget build(BuildContext context) {
    final iconSize = compact ? 21.0 : 22.0;

    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0.0, end: isSelected ? 1.0 : 0.0),
      duration: BDMotion.durationSlow,
      curve: Curves.easeOutCubic,
      builder: (context, progress, child) {
        final revealProgress = Curves.easeOutCubic.transform(
          progress.clamp(0.0, 1.0),
        );
        final bounceWave = math.sin(revealProgress * math.pi);

        return GestureDetector(
          onTap: onTap,
          behavior: HitTestBehavior.opaque,
          child: Center(
            child: Transform.translate(
              offset: Offset(
                lerpDouble(visualShift, 0.0, revealProgress)!,
                -1.1 * bounceWave,
              ),
              child: Opacity(
                opacity: isSelected ? 0.0 : 1.0,
                child: Icon(item.icon, size: iconSize, color: unselectedColor),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _AnimatedFillIcon extends StatelessWidget {
  const _AnimatedFillIcon({
    required this.icon,
    required this.progress,
    required this.selectedColor,
    required this.unselectedColor,
    required this.size,
  });

  final IconData icon;
  final double progress;
  final Color selectedColor;
  final Color unselectedColor;
  final double size;

  @override
  Widget build(BuildContext context) {
    final easedProgress = Curves.easeOutCubic.transform(
      progress.clamp(0.0, 1.0),
    );
    final fillProgress = Curves.easeInOutCubic.transform(
      progress.clamp(0.0, 1.0),
    );

    return SizedBox(
      width: size + 8.0,
      height: size + 8.0,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Icon(
            icon,
            size: size,
            color: Color.lerp(unselectedColor, selectedColor, easedProgress),
            fill: 0.0,
            weight: 500,
          ),
          if (fillProgress > 0.0)
            ClipRect(
              child: Align(
                alignment: Alignment.bottomCenter,
                heightFactor: easedProgress,
                child: Opacity(
                  opacity: fillProgress,
                  child: Icon(
                    icon,
                    size: size,
                    color: selectedColor,
                    fill: 1.0,
                    weight: lerpDouble(560, 700, easedProgress),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
