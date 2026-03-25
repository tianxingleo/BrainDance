import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// BrainDance 的悬浮式底部导航栏组件
/// create 按钮嵌入导航栏中间，recall 和 manage 在两侧。

const double _kCreateSize = 68.0;
const double _kNavBarInnerHeight = 60.0;
const double _kSelectedContentOpticalOffset = -1.5;

class FloatingNavBar extends StatefulWidget {
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
  State<FloatingNavBar> createState() => _FloatingNavBarState();
}

class _FloatingNavBarState extends State<FloatingNavBar>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pillController;
  late Animation<double> _pillLeftAnim;
  late Animation<double> _pillWidthAnim;
  double _prevPillLeft = 0.0;
  double _prevPillWidth = 0.0;
  bool _initialized = false;
  int _prevIndex = 0;

  @override
  void initState() {
    super.initState();
    _prevIndex = widget.currentIndex;
    _pillController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 420),
    );
    _pillLeftAnim = AlwaysStoppedAnimation(0.0);
    _pillWidthAnim = AlwaysStoppedAnimation(0.0);
  }

  @override
  void didUpdateWidget(covariant FloatingNavBar oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.currentIndex != oldWidget.currentIndex) {
      _prevIndex = oldWidget.currentIndex;
    }
  }

  @override
  void dispose() {
    _pillController.dispose();
    super.dispose();
  }

  void _animatePill(double targetLeft, double targetWidth) {
    _pillLeftAnim = Tween<double>(begin: _prevPillLeft, end: targetLeft)
        .animate(
          CurvedAnimation(parent: _pillController, curve: BDMotion.curveFluid),
        );
    _pillWidthAnim = Tween<double>(begin: _prevPillWidth, end: targetWidth)
        .animate(
          CurvedAnimation(parent: _pillController, curve: BDMotion.curveFluid),
        );
    _pillController.forward(from: 0.0);
    _prevPillLeft = targetLeft;
    _prevPillWidth = targetWidth;
  }

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
                        final normalItems = widget.items
                            .where((i) => !i.isLarge)
                            .toList();
                        final normalCount = normalItems.length;
                        final createDockWidth = _kCreateSize + 16.0;
                        final contentWidth =
                            constraints.maxWidth - createDockWidth;
                        final normalSlotWidth = normalCount > 0
                            ? math.max(0.0, contentWidth / normalCount)
                            : 0.0;

                        final selectedItem = widget.items[widget.currentIndex];
                        final showSelectedPill = !selectedItem.isLarge;

                        // Calculate pill target position
                        double targetLeft = 0.0;
                        double targetWidth = 0.0;
                        if (showSelectedPill) {
                          int normalIdx = 0;
                          for (int i = 0; i < widget.currentIndex; i++) {
                            if (!widget.items[i].isLarge) normalIdx++;
                          }
                          final createIdx = widget.items.indexWhere(
                            (i) => i.isLarge,
                          );
                          if (widget.currentIndex < createIdx) {
                            targetLeft = normalIdx * normalSlotWidth;
                          } else {
                            targetLeft =
                                normalIdx * normalSlotWidth + createDockWidth;
                          }
                          targetWidth = normalSlotWidth;
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
                        // pill 宽度需要和选中态真实内容宽度保持一致，
                        // 否则右侧补偿过大时会造成内容中心偏离 pill 中心。
                        final minPillWidth = _estimateSelectedContentWidth(
                          compact: compact,
                          labelWidth: measuredLabelWidth,
                        );
                        final pillWidth = showSelectedPill
                            ? math.min(
                                contentWidth - 8.0,
                                math.max(
                                  minPillWidth,
                                  targetWidth - (compact ? 8.0 : 10.0),
                                ),
                              )
                            : 0.0;
                        final pillLeft = showSelectedPill
                            ? (targetLeft + (targetWidth - pillWidth) / 2)
                                  .clamp(
                                    2.0,
                                    constraints.maxWidth - pillWidth - 2.0,
                                  )
                            : 0.0;

                        // Trigger animation when pill position changes
                        if (!_initialized) {
                          _prevPillLeft = pillLeft;
                          _prevPillWidth = pillWidth;
                          _pillLeftAnim = AlwaysStoppedAnimation(pillLeft);
                          _pillWidthAnim = AlwaysStoppedAnimation(pillWidth);
                          _initialized = true;
                        } else if (widget.currentIndex != _prevIndex ||
                            (_prevPillLeft - pillLeft).abs() > 1.0) {
                          _animatePill(pillLeft, pillWidth);
                          _prevIndex = widget.currentIndex;
                        }

                        return AnimatedBuilder(
                          animation: _pillController,
                          builder: (context, _) {
                            final animLeft = _pillLeftAnim.value;
                            final animWidth = _pillWidthAnim.value;

                            return Stack(
                              children: [
                                if (showSelectedPill && animWidth > 0)
                                  Positioned(
                                    left: animLeft,
                                    top: 0,
                                    bottom: 0,
                                    child: SizedBox(
                                      key: const ValueKey('floating-nav-pill'),
                                      width: animWidth,
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
                                      ),
                                    ),
                                  ),
                                Row(
                                  children: List.generate(widget.items.length, (
                                    index,
                                  ) {
                                    final item = widget.items[index];
                                    if (item.isLarge) {
                                      return SizedBox(width: createDockWidth);
                                    }
                                    final isSelected =
                                        widget.currentIndex == index;
                                    return SizedBox(
                                      width: normalSlotWidth,
                                      child: Padding(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 2.0,
                                        ),
                                        child: _NavBarItem(
                                          item: item,
                                          compact: compact,
                                          isSelected: isSelected,
                                          onTap: () => widget.onTap(index),
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
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),
            // Create 按钮浮在上层
            Positioned(
              bottom: (_kNavBarInnerHeight - _kCreateSize) / 2 + 4,
              left: 0,
              right: 0,
              child: Center(
                child: _CreateButton(
                  isSelected:
                      widget.currentIndex ==
                      widget.items.indexWhere((i) => i.isLarge),
                  isDark: isDark,
                  onTap: () =>
                      widget.onTap(widget.items.indexWhere((i) => i.isLarge)),
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

  double _estimateSelectedContentWidth({
    required bool compact,
    required double labelWidth,
  }) {
    final iconSize = compact ? 21.0 : 22.0;
    final labelGap = compact ? 3.0 : 4.0;
    final horizontalPadding = compact ? 10.0 : 12.0;
    final itemOuterPadding = 4.0;
    final iconBoxWidth = iconSize + 8.0;
    return horizontalPadding * 2 +
        iconBoxWidth +
        labelGap +
        labelWidth +
        itemOuterPadding;
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

/// 普通导航项：选中时显示图标+标签，未选中时仅图标
class _NavBarItem extends StatelessWidget {
  const _NavBarItem({
    required this.item,
    required this.compact,
    required this.isSelected,
    required this.onTap,
    required this.selectedColor,
    required this.unselectedColor,
  });

  final NavIslandItem item;
  final bool compact;
  final bool isSelected;
  final VoidCallback onTap;
  final Color selectedColor;
  final Color unselectedColor;

  @override
  Widget build(BuildContext context) {
    final iconSize = compact ? 21.0 : 22.0;
    final labelGap = compact ? 3.0 : 4.0;
    final horizontalPadding = compact ? 10.0 : 12.0;

    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Stack(
        alignment: Alignment.center,
        children: [
          IgnorePointer(
            ignoring: isSelected,
            child: AnimatedOpacity(
              duration: BDMotion.durationNormal,
              curve: BDMotion.curveExit,
              opacity: isSelected ? 0.0 : 1.0,
              child: Center(
                child: Icon(item.icon, size: iconSize, color: unselectedColor),
              ),
            ),
          ),
          IgnorePointer(
            ignoring: !isSelected,
            child: AnimatedOpacity(
              duration: BDMotion.durationNormal,
              curve: BDMotion.curveFluid,
              opacity: isSelected ? 1.0 : 0.0,
              child: Center(
                child: Transform.translate(
                  offset: const Offset(_kSelectedContentOpticalOffset, 0),
                  child: Padding(
                    padding: EdgeInsets.symmetric(
                      horizontal: horizontalPadding,
                      vertical: compact ? 7.0 : 8.0,
                    ),
                    child: Row(
                      key: isSelected
                          ? ValueKey(
                              'floating-nav-selected-content-${item.label}',
                            )
                          : null,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _AnimatedFillIcon(
                          icon: item.icon,
                          progress: 1.0,
                          selectedColor: selectedColor,
                          unselectedColor: selectedColor.withValues(
                            alpha: 0.68,
                          ),
                          size: iconSize,
                        ),
                        SizedBox(width: labelGap),
                        Text(
                          item.label,
                          maxLines: 1,
                          overflow: TextOverflow.fade,
                          softWrap: false,
                          style: TextStyle(
                            color: selectedColor,
                            fontWeight: FontWeight.w600,
                            fontSize: compact ? 13.0 : 14.0,
                            height: 1.0,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
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
    return SizedBox(
      width: size + 8.0,
      height: size + 8.0,
      child: Center(
        child: Icon(
          icon,
          size: size,
          color: selectedColor,
          fill: 1.0,
          weight: 700,
        ),
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
    _rotation = Tween<double>(
      begin: 0.0,
      end: 0.125,
    ).animate(CurvedAnimation(parent: _controller, curve: BDMotion.curveFluid));
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
                  child: Icon(Icons.add_rounded, color: iconColor, size: 28),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
