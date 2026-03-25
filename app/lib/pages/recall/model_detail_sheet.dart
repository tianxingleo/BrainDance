import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';

Future<void> showRecallModelDetailSheet(
  BuildContext context, {
  required String displayName,
  required String createdAt,
  required String updatedAt,
  required String taskType,
  required Object? qualityScore,
  required String sizeLabel,
  required Widget? qualityScoreTrailing,
}) {
  final isDark = AppConfig.isNightMode;
  final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
  final hintColor = isDark
      ? Colors.white.withValues(alpha: 0.62)
      : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

  return showModalBottomSheet<void>(
    context: context,
    backgroundColor: Colors.transparent,
    builder: (context) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
        child: BDPanelCard(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 16),
          child: SafeArea(
            top: false,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.info_outline_rounded,
                      size: 22,
                      color: textColor,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        displayName,
                        style: TextStyle(
                          color: textColor,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 18),
                RecallDetailRow(
                  icon: Icons.calendar_today_rounded,
                  label: textLocalize('recall_detail_created_at'),
                  value: createdAt,
                  textColor: textColor,
                  hintColor: hintColor,
                ),
                const SizedBox(height: 12),
                RecallDetailRow(
                  icon: Icons.update_rounded,
                  label: textLocalize('recall_detail_updated_at'),
                  value: updatedAt,
                  textColor: textColor,
                  hintColor: hintColor,
                ),
                const SizedBox(height: 12),
                RecallDetailRow(
                  icon: Icons.category_rounded,
                  label: textLocalize('recall_detail_task_type'),
                  value: taskType,
                  textColor: textColor,
                  hintColor: hintColor,
                ),
                const SizedBox(height: 12),
                RecallDetailRow(
                  icon: Icons.star_rounded,
                  label: textLocalize('recall_detail_quality_score'),
                  value: qualityScore != null
                      ? '$qualityScore / 100'
                      : textLocalize('recall_detail_unknown'),
                  textColor: textColor,
                  hintColor: hintColor,
                  valueTrailing: qualityScoreTrailing,
                ),
                if (sizeLabel.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  RecallDetailRow(
                    icon: Icons.storage_rounded,
                    label: textLocalize('recall_detail_local_size'),
                    value: sizeLabel.replaceAll(RegExp(r'[()]'), ''),
                    textColor: textColor,
                    hintColor: hintColor,
                  ),
                ],
              ],
            ),
          ),
        ),
      );
    },
  );
}

class RecallDetailRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color textColor;
  final Color hintColor;
  final Widget? trailing;
  final Widget? valueTrailing;

  const RecallDetailRow({
    super.key,
    required this.icon,
    required this.label,
    required this.value,
    required this.textColor,
    required this.hintColor,
    this.trailing,
    this.valueTrailing,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: hintColor),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  color: hintColor,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Flexible(
                    child: Text(
                      value,
                      style: TextStyle(
                        color: textColor,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  if (valueTrailing != null) ...[
                    const SizedBox(width: 10),
                    valueTrailing!,
                  ],
                ],
              ),
            ],
          ),
        ),
        if (trailing != null) ...[const SizedBox(width: 12), trailing!],
      ],
    );
  }
}
