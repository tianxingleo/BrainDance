import 'package:flutter/material.dart';
import 'dart:ui';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// BrainDance 的悬浮式底部导航栏组件
/// 毛玻璃质感、四周间距悬浮、不对称布局（create 按钮更大）

const double _kCreateSize = 66.0;
const double _kNavBarInnerHeight = 54.0;

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
      child: SizedBox(
        // 给 create 按钮的溢出留空间
        height: _kCreateSize / 2 + _kNavBarInnerHeight / 2 + 8.0,
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.bottomCenter,
          children: [
            // ── 毛玻璃导航栏主体 ──
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: ClipRRect(
                borderRadius: BDDesign.radiusLarge,
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 24.0, sigmaY: 24.0),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 6.0, vertical: 8.0),
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
                            final item = items[index];
                            if (item.isLarge) {
                              // 占位，实际按钮在 Stack 上层
                              return const SizedBox(width: 72);
                            }
                            final isSelected = currentIndex == index;
                            return Expanded(
                              child: _buildNormalItem(
                                item: item,
                                isSelected: isSelected,
                                onTap: () => onTap(index),
                                selectedBackground: selectedBackground,
                                selectedColor: selectedColor,
                                unselectedColor: unselectedColor,
                                compact: compact,
                              ),
                            );
                          }),
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),
            // ── Create 圆形按钮（溢出导航栏） ──
            ...List.generate(items.length, (index) {
              final item = items[index];
              if (!item.isLarge) return const SizedBox.shrink();
              return Positioned(
                bottom: 0,
                right: 8.0,
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

  Widget _buildNormalItem({
    required NavIslandItem item,
    required bool isSelected,
    required VoidCallback onTap,
    required Color selectedBackground,
    required Color selectedColor,
    required Color unselectedColor,
    required bool compact,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2.0),
      child: GestureDetector(
        onTap: onTap,
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
            color: isSelected ? selectedBackground : Colors.transparent,
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
                  color: isSelected ? selectedColor : unselectedColor,
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
    );
  }

}

/// Create 按钮 — 圆形、溢出导航栏
/// 未选中：反色填充 + 主题色加号
/// 选中：主题色填充 + 反色加号旋转 45° 变 ×
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
    // 未选中：外圆 = 反色，图标 = 主题色
    // 选中：外圆 = 主题色，图标 = 反色，旋转 45°
    final inverseColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final themeColor =
        isDark ? BDDesign.colorInkBlack : BDDesign.colorPaperWhite;

    final fillColor = isSelected ? themeColor : inverseColor;
    final iconColor = isSelected ? inverseColor : themeColor;

    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
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
                color: Colors.black.withValues(alpha: isDark ? 0.32 : 0.12),
                blurRadius: 16,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Center(
            child: AnimatedRotation(
              turns: isSelected ? 0.125 : 0, // 45°
              duration: BDMotion.durationNormal,
              curve: BDMotion.curveFluid,
              child: AnimatedSwitcher(
                duration: BDMotion.durationFast,
                child: Icon(
                  Icons.add_rounded,
                  key: ValueKey(isSelected),
                  color: iconColor,
                  size: 30,
                ),
              ),
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
  final bool isLarge;
  NavIslandItem({required this.icon, required this.label, this.isLarge = false});
}