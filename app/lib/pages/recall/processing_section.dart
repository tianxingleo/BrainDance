import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';
import '../../widgets/bd_surfaces.dart';
import 'model_grid_helpers.dart';

class RecallProcessingSection extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color textColor;
  final Color darkInput;
  final bool isExpanded;
  final List<Map<String, dynamic>> processingTasks;
  final Map<String, List<String>> taskAllLogs;
  final Set<String> expandedTaskLogs;
  final VoidCallback onToggleExpanded;
  final ValueChanged<String> onToggleTaskLogs;

  const RecallProcessingSection({
    super.key,
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.darkInput,
    required this.isExpanded,
    required this.processingTasks,
    required this.taskAllLogs,
    required this.expandedTaskLogs,
    required this.onToggleExpanded,
    required this.onToggleTaskLogs,
  });

  @override
  Widget build(BuildContext context) {
    final hintTextColor = isDark ? const Color(0xFF888888) : theme.fontGyColor3;

    return BDGlassSurface(
      noBlur: true,
      variant: BDGlassVariant.panel,
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          InkWell(
            onTap: onToggleExpanded,
            borderRadius: BDDesign.radiusLarge,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
              child: Row(
                children: [
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: BDDesign.colorMutedBlue.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Center(
                      child: SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            BDDesign.colorMutedBlue,
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          textLocalize('status_processing'),
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: textColor,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '这个场景还在重建，共 ${processingTasks.length} 项任务正在推进。',
                          style: TextStyle(
                            fontSize: 12.5,
                            color: hintTextColor,
                            height: 1.35,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  BDStatusPill(
                    label: '${processingTasks.length}',
                    icon: Icons.motion_photos_on_rounded,
                    color: BDDesign.colorMutedBlue,
                  ),
                  const SizedBox(width: 8),
                  AnimatedRotation(
                    turns: isExpanded ? 0.5 : 0,
                    duration: BDMotion.durationFast,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: isDark
                          ? Colors.white.withValues(alpha: 0.56)
                          : BDDesign.colorMutedBlue,
                    ),
                  ),
                ],
              ),
            ),
          ),
          AnimatedCrossFade(
            firstChild: const SizedBox.shrink(),
            secondChild: Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Column(
                children: processingTasks.asMap().entries.map((entry) {
                  final index = entry.key;
                  final task = entry.value;
                  return _RecallProcessingTaskItem(
                    task: task,
                    theme: theme,
                    isDark: isDark,
                    textColor: textColor,
                    hintTextColor: hintTextColor,
                    darkInput: darkInput,
                    allLogs: taskAllLogs[task['id'].toString()] ?? const [],
                    isExpanded: expandedTaskLogs.contains(
                      task['id'].toString(),
                    ),
                    isFirst: index == 0,
                    isLast: index == processingTasks.length - 1,
                    onToggleLogs: () => onToggleTaskLogs(task['id'].toString()),
                  );
                }).toList(),
              ),
            ),
            crossFadeState: isExpanded
                ? CrossFadeState.showSecond
                : CrossFadeState.showFirst,
            duration: BDMotion.durationNormal,
          ),
        ],
      ),
    );
  }
}

class _RecallProcessingTaskItem extends StatelessWidget {
  final Map<String, dynamic> task;
  final TDThemeData theme;
  final bool isDark;
  final Color textColor;
  final Color hintTextColor;
  final Color darkInput;
  final List<String> allLogs;
  final bool isExpanded;
  final bool isFirst;
  final bool isLast;
  final VoidCallback onToggleLogs;

  const _RecallProcessingTaskItem({
    required this.task,
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.hintTextColor,
    required this.darkInput,
    required this.allLogs,
    required this.isExpanded,
    required this.isFirst,
    required this.isLast,
    required this.onToggleLogs,
  });

  @override
  Widget build(BuildContext context) {
    final displayName = modelDisplayName(task, fallback: 'Unknown');
    final latestLog = allLogs.isNotEmpty ? allLogs.last : null;

    return Container(
      margin: EdgeInsets.fromLTRB(12, isFirst ? 1 : 4, 12, isLast ? 12 : 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark
            ? darkInput.withValues(alpha: 0.86)
            : BDDesign.colorMutedBlueLight.withValues(alpha: 0.38),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: isDark
              ? Colors.white.withValues(alpha: 0.05)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: BDDesign.colorMutedBlue.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Center(
                  child: SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        BDDesign.colorMutedBlue,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      displayName,
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: textColor,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      latestLog ?? textLocalize('status_processing'),
                      style: TextStyle(
                        fontSize: 12.5,
                        color: hintTextColor,
                        height: 1.35,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              if (allLogs.length > 1)
                IconButton(
                  icon: AnimatedRotation(
                    turns: isExpanded ? 0.5 : 0,
                    duration: BDMotion.durationFast,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: isDark
                          ? Colors.white.withValues(alpha: 0.56)
                          : BDDesign.colorMutedBlue,
                      size: 20,
                    ),
                  ),
                  onPressed: onToggleLogs,
                ),
            ],
          ),
          if (allLogs.length > 1)
            AnimatedCrossFade(
              firstChild: const SizedBox.shrink(),
              secondChild: Container(
                margin: const EdgeInsets.only(top: 10),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: isDark
                      ? const Color(0xFF1A1A20).withValues(alpha: 0.94)
                      : Colors.white.withValues(alpha: 0.7),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: allLogs.reversed
                      .map(
                        (log) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 2),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                margin: const EdgeInsets.only(top: 6, right: 8),
                                width: 5,
                                height: 5,
                                decoration: BoxDecoration(
                                  color: BDDesign.colorMutedBlue.withValues(
                                    alpha: isDark ? 0.72 : 0.55,
                                  ),
                                  shape: BoxShape.circle,
                                ),
                              ),
                              Expanded(
                                child: Text(
                                  log,
                                  style: TextStyle(
                                    fontSize: 11.5,
                                    color: isDark
                                        ? Colors.white.withValues(alpha: 0.7)
                                        : theme.fontGyColor2,
                                    height: 1.35,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      )
                      .toList(),
                ),
              ),
              crossFadeState: isExpanded
                  ? CrossFadeState.showSecond
                  : CrossFadeState.showFirst,
              duration: BDMotion.durationFast,
            ),
        ],
      ),
    );
  }
}
