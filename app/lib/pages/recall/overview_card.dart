import 'package:flutter/material.dart';

import '../../configs/motion_tokens.dart';
import '../../widgets/bd_surfaces.dart';

class RecallOverviewCard extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final int recentCount;
  final int allModelCount;
  final int processingTaskCount;
  final int ragCount;
  final bool isLocalIndexing;
  final VoidCallback onOpenTasks;

  const RecallOverviewCard({
    super.key,
    required this.isDark,
    required this.textColor,
    required this.recentCount,
    required this.allModelCount,
    required this.processingTaskCount,
    required this.ragCount,
    required this.isLocalIndexing,
    required this.onOpenTasks,
  });

  @override
  Widget build(BuildContext context) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: BDPanelCard(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '空间档案概览',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: hintColor,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        '把空间、任务和检索线索压进同一条记忆流里。',
                        style: TextStyle(
                          fontSize: 12.5,
                          height: 1.4,
                          color: hintColor.withValues(alpha: 0.88),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                IconButton(
                  onPressed: onOpenTasks,
                  icon: Icon(
                    Icons.task_alt_rounded,
                    color: hintColor,
                    size: 20,
                  ),
                  tooltip: '任务列表',
                  splashRadius: 20,
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                ),
              ],
            ),
            const SizedBox(height: 18),
            Row(
              children: [
                Expanded(
                  child: _RecallMetric(
                    label: '空间',
                    value: allModelCount.toString(),
                  ),
                ),
                Expanded(
                  child: _RecallMetric(
                    label: '处理中',
                    value: processingTaskCount.toString(),
                  ),
                ),
                Expanded(
                  child: _RecallMetric(
                    label: 'RAG',
                    value: isLocalIndexing ? '...' : ragCount.toString(),
                    accent: textColor,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                _OverviewChip(
                  label: '近日新增',
                  value: recentCount.toString(),
                  isDark: isDark,
                ),
                _OverviewChip(
                  label: '已就绪空间',
                  value: allModelCount.toString(),
                  isDark: isDark,
                ),
              ],
            ),
          ],
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
    final labelColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;
    final valueColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: valueColor,
            ),
          ),
          const SizedBox(height: 2),
          Text(label, style: TextStyle(fontSize: 12, color: labelColor)),
        ],
      ),
    );
  }
}

class _RecallMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color? accent;

  const _RecallMetric({required this.label, required this.value, this.accent});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isDark
                ? Colors.white.withValues(alpha: 0.58)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color:
                accent ??
                (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack),
          ),
        ),
      ],
    );
  }
}
