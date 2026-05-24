import 'package:flutter/material.dart';

import '../../configs/motion_tokens.dart';

class RecallModelSectionHeader extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String title;
  final int count;
  final bool isExpanded;
  final bool isDark;
  final Color textColor;
  final VoidCallback onToggle;

  const RecallModelSectionHeader({
    super.key,
    required this.icon,
    required this.color,
    required this.title,
    required this.count,
    required this.isExpanded,
    required this.isDark,
    required this.textColor,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    final arrowColor = isDark
        ? Colors.white.withValues(alpha: 0.45)
        : const Color(0xFF8899BB);

    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 16, 24, 6),
      child: InkWell(
        onTap: onToggle,
        borderRadius: BorderRadius.circular(12),
        child: Row(
          children: [
            Icon(icon, color: color, size: 18),
            const SizedBox(width: 8),
            Text(
              title,
              style: TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: textColor,
              ),
            ),
            const SizedBox(width: 10),
            _CountBadge(count: count, color: color, isDark: isDark),
            const Spacer(),
            AnimatedRotation(
              turns: isExpanded ? 0.5 : 0,
              duration: BDMotion.durationFast,
              child: Icon(Icons.keyboard_arrow_down, color: arrowColor, size: 20),
            ),
          ],
        ),
      ),
    );
  }
}

class _CountBadge extends StatelessWidget {
  final int count;
  final Color color;
  final bool isDark;

  const _CountBadge({
    required this.count,
    required this.color,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: isDark ? 0.16 : 0.10),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        '$count',
        style: TextStyle(
          fontSize: 11.5,
          fontWeight: FontWeight.w600,
          color: color,
        ),
      ),
    );
  }
}
