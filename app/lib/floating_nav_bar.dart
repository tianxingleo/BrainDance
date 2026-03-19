import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// BrainDance 的悬浮式底部导航栏组件
/// 保留 dev 分支的大按钮布局，并融合更细腻的选中 pill 与图标动效。

const double _kCreateSize = 66.0;
const double _kNavBarInnerHeight = 60.0;
const double _kCreateDockWidth = 88.0;
const double _kCreateDockRightInset =
    (_kCreateDockWidth - _kCreateSize) / 2;
const double _kCreateDockBottomOffset =
    (_kNavBarInnerHeight - _kCreateSize) / 2;

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
        ? AppTheme.darkSurface.withValues(alpha: 0.55)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.52);
    final navBorder = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
    final navShadow = Colors.black.withValues(alpha: isDark ? 0.22 : 0.05);
    final selectedBackground = isDark
        ? const Color(0xFFAEBAC7).withValues(alpha: 0.14)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.09);
    final selectedColor = isDark
        ? const Color(0xFFF4F7FA)
        : BDDesign.colorInkBlack;
    final unselectedColor = isDark
        ? const Color(0xFFB4BEC9)
        : const Color(0xFF9AA3AD);

    return Positioned(
      bottom: 20.0,
      left: 20.0,
      right: 20.0,
      child: SizedBox(
        height: _kCreateSize / 2 + _kNavBarInnerHeight / 2 + 12.0,
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.bottomCenter,
          children: [
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: ClipRRect(
                borderRadius: BDDesign.radiusLarge,
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 24.0, sigmaY: 24.0),
                  child: Container(
                    height: _kNavBarInnerHeight,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8.0,
                      vertical: 8.0,
                    ),
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
                        final contentWidth =
                            constraints.maxWidth - _kCreateDockWidth;
                        final slotWidths = _buildSlotWidths(
                          contentWidth,
                          items,
                        );
                        final selectedItem = items[currentIndex];
                        final selectedLeft = _slotLeft(
                          slotWidths,
                          currentIndex,
                        );
                        final selectedWidth = slotWidths[currentIndex];
                        final showSelectedPill = !selectedItem.isLarge;
                        final labelStyle = TextStyle(
                          color: selectedColor,
                          fontWeight: FontWeight.w600,
                          fontSize: compact ? 13.0 : 14.0,
                          height: 1.0,
                        );
                        final measuredLabelWidth = _measureLabelWidth(
                          text: selectedItem.label,
                          style: labelStyle,
                        );
                        final minPillWidth =
                            (compact ? 76.0 : 88.0) + measuredLabelWidth;
                        final pillWidth = math.min(
                          contentWidth - 8.0,
                          math.max(
                            minPillWidth,
                            selectedWidth - (compact ? 8.0 : 10.0),
                          ),
                        );
                        final pillLeft =
                            (selectedLeft + (selectedWidth - pillWidth) / 2)
                                .clamp(2.0, contentWidth - pillWidth - 2.0);

                        return Stack(
                          children: [
                            if (showSelectedPill)
                              Positioned.fill(
                                child: IgnorePointer(
                                  child: TweenAnimationBuilder<double>(
                                    tween: Tween<double>(
                                      begin: pillLeft,
                                      end: pillLeft,
                                    ),
                                    duration: BDMotion.durationSlow,
                                    curve: Curves.easeOutCubic,
                                    builder: (context, animatedLeft, child) {
                                      return Stack(
                                        children: [
                                          Positioned(
                                            left: animatedLeft,
                                            top: 0,
                                            bottom: 0,
                                            child: SizedBox(
                                              width: pillWidth,
                                              child: DecoratedBox(
                                                decoration: BoxDecoration(
                                                  color: selectedBackground,
                                                  borderRadius:
                                                      BorderRadius.circular(
                                                        22.0,
                                                      ),
                                                  boxShadow: [
                                                    BoxShadow(
                                                      color: selectedColor
                                                          .withValues(
                                                            alpha: 0.05,
                                                          ),
                                                      blurRadius: 16,
                                                      offset: const Offset(
                                                        0,
                                                        6,
                                                      ),
                                                    ),
                                                  ],
                                                ),
                                                child: _SelectedPillContent(
                                                  item: selectedItem,
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
                              children: [
                                ...List.generate(items.length, (index) {
                                  final item = items[index];
                                  final slotWidth = slotWidths[index];
                                  if (item.isLarge) {
                                    return const SizedBox.shrink();
                                  }

                                  final distance = (index - currentIndex).abs();
                                  final direction = index < currentIndex
                                      ? -1.0
                                      : 1.0;
                                  final shiftBase = compact ? 5.0 : 6.0;
                                  final visualShift = index == currentIndex
                                      ? 0.0
                                      : direction *
                                            (distance == 1
                                                ? shiftBase
                                                : distance == 2
                                                ? shiftBase * 0.28
                                                : 0.0);

                                  return SizedBox(
                                    width: slotWidth,
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
                                const SizedBox(width: _kCreateDockWidth),
                              ],
                            ),
                          ],
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),
            ...List.generate(items.length, (index) {
              final item = items[index];
              if (!item.isLarge) return const SizedBox.shrink();
              return Positioned(
                bottom: _kCreateDockBottomOffset,
                right: _kCreateDockRightInset,
                child: _CreateButton(
                  isSelected: currentIndex == index,
                  isDark: isDark,
                  onTap: () => onTap(index),
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  List<double> _buildSlotWidths(double totalWidth, List<NavIslandItem> items) {
    final normalCount = items.where((item) => !item.isLarge).length;
    final normalWidth = normalCount <= 0
        ? 0.0
        : math.max(0.0, totalWidth / normalCount);

    return items.map((item) => item.isLarge ? 0.0 : normalWidth).toList();
  }

  double _slotLeft(List<double> slotWidths, int index) {
    var left = 0.0;
    for (var i = 0; i < index; i++) {
      left += slotWidths[i];
    }
    return left;
  }

  double _measureLabelWidth({required String text, required TextStyle style}) {
    final painter = TextPainter(
      text: TextSpan(text: text, style: style),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    )..layout();
    return painter.width;
  }
}

class NavIslandItem {
  final IconData icon;
  final String label;
  final bool isLarge;

  NavIslandItem({
    required this.icon,
    required this.label,
    this.isLarge = false,
  });
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

class _CreateButton extends StatelessWidget {
  final bool isSelected;
  final bool isDark;
  final VoidCallback onTap;

  const _CreateButton({
    required this.isSelected,
    required this.isDark,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final inverseColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final themeColor = isDark
        ? BDDesign.colorInkBlack
        : BDDesign.colorPaperWhite;
    final fillColor = isSelected ? themeColor : inverseColor;
    final iconColor = isSelected ? inverseColor : themeColor;

    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: TweenAnimationBuilder<double>(
        tween: Tween<double>(begin: 0.0, end: isSelected ? 1.0 : 0.0),
        duration: BDMotion.durationSlow,
        curve: Curves.easeOutCubic,
        builder: (context, progress, child) {
          final lift = math.sin(progress * math.pi) * 3.0;
          return Transform.translate(
            offset: Offset(0.0, -lift),
            child: SizedBox(
              width: _kCreateSize,
              height: _kCreateSize,
              child: AnimatedContainer(
                duration: BDMotion.durationNormal,
                curve: BDMotion.curveFluid,
                decoration: BoxDecoration(
                  color: fillColor,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(
                        alpha: isDark ? 0.32 : 0.12,
                      ),
                      blurRadius: 16,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Center(
                  child: Transform.translate(
                    offset: const Offset(0, 1),
                    child: AnimatedRotation(
                      turns: isSelected ? 0.125 : 0.0,
                      duration: BDMotion.durationNormal,
                      curve: BDMotion.curveFluid,
                      child: Icon(
                        Icons.add_rounded,
                        color: iconColor,
                        size: 30,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
