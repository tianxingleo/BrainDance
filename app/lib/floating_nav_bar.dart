import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// BrainDance 的悬浮式底部导航栏组件
/// create 按钮嵌入导航栏中间，recall 和 manage 在两侧。

const double _kCreateSize = 68.0;
const double _kNavBarInnerHeight = 60.0;

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
    final createOverflow = (_kCreateSize - _kNavBarInnerHeight) / 2;

    return Positioned(
      bottom: 20.0,
      left: 20.0,
      right: 20.0,
      child: SizedBox(
        height: _kNavBarInnerHeight + math.max(0, createOverflow),
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.bottomCenter,
          children: [
            // 导航栏背景层
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
                        final normalItems = items.where((i) => !i.isLarge).toList();
                        final normalCount = normalItems.length;
                        final createDockWidth = _kCreateSize + 16.0;
                        final contentWidth = constraints.maxWidth - createDockWidth;
                        final normalSlotWidth = normalCount > 0
                            ? math.max(0.0, contentWidth / normalCount)
                            : 0.0;
                // Build pill position for normal (non-create) items
                final selectedItem = items[currentIndex];
                final showSelectedPill = !selectedItem.isLarge;

                // Calculate slot positions accounting for create button in middle
                double selectedLeft = 0.0;
                double selectedWidth = 0.0;
                if (showSelectedPill) {
                  // Find position among normal items only
                  int normalIdx = 0;
                  for (int i = 0; i < currentIndex; i++) {
                    if (!items[i].isLarge) normalIdx++;
                  }
                  // Items before create button are on the left
                  final createIdx = items.indexWhere((i) => i.isLarge);
                  if (currentIndex < createIdx) {
                    selectedLeft = normalIdx * normalSlotWidth;
                  } else {
                    selectedLeft = normalIdx * normalSlotWidth + createDockWidth;
                  }
                  selectedWidth = normalSlotWidth;
                }

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
                final pillWidth = showSelectedPill
                    ? math.min(
                        contentWidth - 8.0,
                        math.max(minPillWidth, selectedWidth - (compact ? 8.0 : 10.0)),
                      )
                    : 0.0;
                final pillLeft = showSelectedPill
                    ? (selectedLeft + (selectedWidth - pillWidth) / 2)
                        .clamp(2.0, constraints.maxWidth - pillWidth - 2.0)
                    : 0.0;

                return Stack(
                  children: [
                    if (showSelectedPill)
                      Positioned.fill(
                        child: IgnorePointer(
                          child: TweenAnimationBuilder<double>(
                            tween: Tween<double>(begin: pillLeft, end: pillLeft),
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
                                          borderRadius: BorderRadius.circular(22.0),
                                          boxShadow: [
                                            BoxShadow(
                                              color: selectedColor.withValues(alpha: 0.05),
                                              blurRadius: 16,
                                              offset: const Offset(0, 6),
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
                      children: List.generate(items.length, (index) {
                        final item = items[index];
                        if (item.isLarge) {
                          // 占位，真正的按钮浮在 Stack 上层
                          return SizedBox(width: createDockWidth);
                        }
                        final distance = (index - currentIndex).abs();
                        final direction = index < currentIndex ? -1.0 : 1.0;
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
                          width: normalSlotWidth,
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 2.0),
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
    ),
            // Create 按钮浮在上层，可溢出导航栏
            Positioned(
              bottom: (_kNavBarInnerHeight - _kCreateSize) / 2 + 4,
              left: 0,
              right: 0,
              child: Center(
                child: _CreateButton(
                  isSelected: currentIndex == items.indexWhere((i) => i.isLarge),
                  isDark: isDark,
                  onTap: () => onTap(items.indexWhere((i) => i.isLarge)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
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

/// Create 按钮：圆形边框，底色与系统颜色对比，图标为加号且颜色与底色对比。
/// 选中时图标旋转45度，同时图标颜色与按钮底色渐变反转。
class _CreateButton extends StatefulWidget {
  final bool isSelected;
  final bool isDark;
  final VoidCallback onTap;

  const _CreateButton({
    required this.isSelected,
    required this.isDark,
    required this.onTap,
  });

  @override
  State<_CreateButton> createState() => _CreateButtonState();
}

class _CreateButtonState extends State<_CreateButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _rotation;
  late final Animation<double> _colorProgress;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 520),
    );
    _rotation = Tween<double>(begin: 0.0, end: 0.125).animate(
      CurvedAnimation(parent: _controller, curve: BDMotion.curveFluid),
    );
    _colorProgress = CurvedAnimation(
      parent: _controller,
      curve: BDMotion.curveFluid,
    );

    if (widget.isSelected) {
      _controller.value = 1.0;
    }
  }

  @override
  void didUpdateWidget(covariant _CreateButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isSelected != oldWidget.isSelected) {
      if (widget.isSelected) {
        _controller.forward(from: 0.0);
      } else {
        _controller.reverse();
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 未选中：底色与系统对比，图标与底色对比
    final baseFill = widget.isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final baseIcon = widget.isDark
        ? BDDesign.colorInkBlack
        : BDDesign.colorPaperWhite;

    return GestureDetector(
      onTap: widget.onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          final t = _colorProgress.value;
          // 选中时颜色反转
          final fillColor = Color.lerp(baseFill, baseIcon, t)!;
          final iconColor = Color.lerp(baseIcon, baseFill, t)!;

          return SizedBox(
            width: _kCreateSize,
            height: _kCreateSize,
            child: Container(
              decoration: BoxDecoration(
                color: fillColor,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(
                      alpha: widget.isDark ? 0.32 : 0.12,
                    ),
                    blurRadius: 12,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Center(
                child: Transform.rotate(
                  angle: _rotation.value * 2 * math.pi,
                  child: Icon(
                    Icons.add_rounded,
                    color: iconColor,
                    size: 28,
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


