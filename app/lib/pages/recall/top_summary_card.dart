import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import '../../configs/motion_tokens.dart';

class TopSummaryCard extends StatelessWidget {
  final int recordCount;
  final int completedCount;
  final VoidCallback onTaskTap;
  final bool isDark;

  const TopSummaryCard({
    Key? key,
    required this.recordCount,
    required this.completedCount,
    required this.onTaskTap,
    required this.isDark,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final bgColor = isDark ? const Color(0xFF18181C) : BDDesign.colorPaperWhite;
    final textColor = isDark ? Colors.white : BDDesign.colorInkBlack;
    final subTextColor = isDark ? Colors.white70 : BDDesign.colorInkBlack.withOpacity(0.6);

    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0.0, end: 1.0),
      duration: BDMotion.durationNormal,
      curve: BDMotion.curveEnter,
      builder: (context, value, child) {
        return Transform.translate(
          offset: Offset(0, 20 * (1 - value)),
          child: Opacity(
            opacity: value,
            child: child,
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 8.0),
        padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 16.0),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: BDDesign.radiusNormal,
          boxShadow: [if (!isDark) BDDesign.shadowLight],
          border: Border.all(
            color: isDark ? const Color(0xFF2A2A30) : Colors.transparent,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  textLocalize("recall_summary_title"),
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: isDark ? const Color(0xFFA0AAB5) : BDDesign.colorMutedBlue,
                  ),
                ),
                IconButton(
                  onPressed: onTaskTap,
                  icon: Icon(
                    Icons.task_alt_rounded,
                    color: isDark ? const Color(0xFFA0AAB5) : BDDesign.colorMutedBlue,
                    size: 20,
                  ),
                  tooltip: textLocalize("recall_task_list"),
                  splashRadius: 20,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                _buildStatItem(textLocalize("recall_recent_added"), recordCount.toString(), textColor, subTextColor),
                const SizedBox(width: 48),
                _buildStatItem(textLocalize("recall_ready_spaces"), completedCount.toString(), textColor, subTextColor),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String label, String value, Color textColor, Color subTextColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: textColor,
            height: 1.2,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: subTextColor,
          ),
        ),
      ],
    );
  }
}
