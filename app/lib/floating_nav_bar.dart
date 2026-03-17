import 'package:flutter/material.dart';
import 'dart:ui';

/// BrainDance 的悬浮式“导航岛”组件
/// 风格特点：半透明玻璃质感、重心偏移切换、统一动效曲线
class FloatingNavBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onTap;
  final List<NavIslandItem> items;

  const FloatingNavBar({
    Key? key,
    required this.currentIndex,
    required this.onTap,
    required this.items,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 24.0, // 距离底部浮动
      left: 32.0,
      right: 32.0,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(32.0),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 16.0, sigmaY: 16.0),
          child: Container(
            height: 64.0,
            padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 8.0),
            decoration: BoxDecoration(
              // 石灰/纸白系的半透明背景，拒绝廉价纯白或大面积高亮玻璃
              color: const Color(0xFFF0F2F5).withOpacity(0.75), 
              borderRadius: BorderRadius.circular(32.0),
              border: Border.all(
                color: Colors.white.withOpacity(0.4),
                width: 1.0,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 24,
                  offset: const Offset(0, 8),
                )
              ],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: List.generate(items.length, (index) {
                final isSelected = currentIndex == index;
                final item = items[index];

                return GestureDetector(
                  onTap: () => onTap(index),
                  behavior: HitTestBehavior.opaque,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 280), // 卡片浮现速率 Token
                    curve: Curves.easeOutQuart, // 漂浮跟随曲线
                    padding: EdgeInsets.symmetric(
                      horizontal: isSelected ? 20.0 : 12.0,
                      vertical: 8.0,
                    ),
                    decoration: BoxDecoration(
                      // 高亮选用“钝蓝灰” (Muted Blue-Gray)
                      color: isSelected
                          ? const Color(0xFF6B7A8F).withOpacity(0.15)
                          : Colors.transparent,
                      borderRadius: BorderRadius.circular(24.0),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        AnimatedScale(
                          scale: isSelected ? 1.1 : 1.0,
                          duration: const Duration(milliseconds: 280),
                          curve: Curves.easeOutQuart,
                          child: Icon(
                            item.icon,
                            color: isSelected
                                ? const Color(0xFF4A5C70) // 选中的深灰蓝
                                : const Color(0xFF909A9E), // 未选中的石灰
                            size: 24,
                          ),
                        ),
                        AnimatedSize(
                          duration: const Duration(milliseconds: 280),
                          curve: Curves.easeOutQuart,
                          child: isSelected
                              ? Padding(
                                  padding: const EdgeInsets.only(left: 8.0),
                                  child: Text(
                                    item.label,
                                    style: const TextStyle(
                                      color: Color(0xFF4A5C70),
                                      fontWeight: FontWeight.w600,
                                      fontSize: 14,
                                    ),
                                  ),
                                )
                              : const SizedBox.shrink(),
                        )
                      ],
                    ),
                  ),
                );
              }),
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
