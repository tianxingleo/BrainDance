import 'package:braindance/configs/app_config.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/motion_tokens.dart';
import '../../widgets/bd_surfaces.dart';

class RecallOverviewCard extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final int recentCount;
  final int allModelCount;
  final int processingTaskCount;
  final VoidCallback onOpenTasks;

  const RecallOverviewCard({
    super.key,
    required this.isDark,
    required this.textColor,
    required this.recentCount,
    required this.allModelCount,
    required this.processingTaskCount,
    required this.onOpenTasks,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: BDPanelCard(
        glass: true,
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              textLocalize('recall_summary_title'),
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                color: textColor,
              ),
            ),
            const SizedBox(height: 18),
            Row(
              children: [
                Expanded(
                  child: _RecallMetric(
                    label: textLocalize('recall_label_space'),
                    value: allModelCount.toString(),
                  ),
                ),
                Expanded(
                  child: _RecallMetric(
                    label: textLocalize('recall_label_processing'),
                    value: processingTaskCount.toString(),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _OverviewChip(
              label: textLocalize('recall_recent_added'),
              value: recentCount.toString(),
              isDark: isDark,
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: _TaskButton(isDark: isDark, onTap: onOpenTasks),
            ),
          ],
        ),
      ),
    );
  }
}

class _TaskButton extends StatelessWidget {
  final bool isDark;
  final VoidCallback onTap;

  const _TaskButton({required this.isDark, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final accent = isDark
        ? BDDesign.colorMutedBlueLight.withValues(alpha: 0.18)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.11);

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Ink(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          decoration: BoxDecoration(
            color: accent,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: accent),
          ),
          child: Row(
            children: [
              Icon(Icons.task_alt_rounded, color: textColor, size: 20),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  textLocalize('recall_task_list'),
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: textColor,
                  ),
                ),
              ),
              Icon(Icons.chevron_right_rounded, color: textColor, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}

class _OverviewChip extends StatelessWidget {
  final String label;
  final String value;
  final bool isDark;

  const _OverviewChip({
    required this.label,
    required this.value,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    final bgColor = isDark
        ? Colors.white.withValues(alpha: 0.06)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.08);
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: textColor,
            ),
          ),
          const SizedBox(height: 2),
          Text(label, style: TextStyle(fontSize: 12, color: textColor)),
        ],
      ),
    );
  }
}

class _RecallMetric extends StatelessWidget {
  final String label;
  final String value;

  const _RecallMetric({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: textColor,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: textColor,
          ),
        ),
      ],
    );
  }
}
