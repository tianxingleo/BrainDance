import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// BrainDance 的悬浮式底部导航栏组件
/// create 按钮嵌入导航栏中间，recall 和 manage 在两侧。

const double _kCreateSize = 60.0;
const double _kNavBarInnerHeight = 48.0;

class FloatingNavBar extends StatefulWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;
  final List<NavIslandItem> items;
  final bool skipBlur;

  const FloatingNavBar({
    super.key,
    required this.currentIndex,
    required this.onTap,
    required this.items,
    this.skipBlur = false,
  });

  @override
  State<FloatingNavBar> createState() => _FloatingNavBarState();
}

class _FloatingNavBarState extends State<FloatingNavBar> {
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
                child: Builder(
                  builder: (ctx) {
                    final bg = Container(
                      height: _kNavBarInnerHeight,
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
                    );
                    return widget.skipBlur
                        ? bg
                        : BackdropFilter(
                            filter: ImageFilter.blur(
                              sigmaX: 24.0,
                              sigmaY: 24.0,
                            ),
                            child: bg,
                          );
                  },
                ),
              ),
            ),
            // 导航项层（允许溢出）
            Positioned(
              bottom: 10,
              left: 0,
              right: 0,
              child: SizedBox(
                height: _kNavBarInnerHeight,
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final compact = constraints.maxWidth < 500;
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

                    return Row(
                      children: List.generate(widget.items.length, (index) {
                        final item = widget.items[index];
                        if (item.isLarge) {
                          return SizedBox(width: createDockWidth);
                        }
                        final isSelected = widget.currentIndex == index;
                        return SizedBox(
                          width: normalSlotWidth,
                          child: _NavBarItem(
                            item: item,
                            compact: compact,
                            isSelected: isSelected,
                            onTap: () => widget.onTap(index),
                            selectedColor: selectedColor,
                            unselectedColor: unselectedColor,
                          ),
                        );
                      }),
                    );
                  },
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

/// 普通导航项：垂直布局，图标在上、文字在下，图标可超出底栏上边界
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
    final iconSize = compact ? 30.0 : 32.0;
    final fontSize = compact ? 11.0 : 12.0;
    final color = isSelected ? selectedColor : unselectedColor;

    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          AnimatedContainer(
            duration: BDMotion.durationNormal,
            curve: BDMotion.curveFluid,
            transform: Matrix4.translationValues(
              0,
              isSelected ? -4.0 : 0.0,
              0,
            ),
            child: Icon(
              item.icon,
              size: iconSize,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            item.label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: color,
              fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
              fontSize: fontSize,
              height: 1.0,
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
    ).animate(_controller);

    if (widget.isSelected) {
      _controller.value = 1.0;
    }
  }

  @override
  void didUpdateWidget(covariant _CreateButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isSelected != oldWidget.isSelected) {
      if (widget.isSelected) {
        _controller.animateTo(1.0, curve: BDMotion.curveFluid);
      } else {
        _controller.animateTo(0.0, curve: BDMotion.curveFluid);
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
          final t = _controller.value;
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
