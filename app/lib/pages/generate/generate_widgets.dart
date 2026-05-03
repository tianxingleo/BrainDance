part of '../generate.dart';

class _VideoTaskTypeSheet extends StatelessWidget {
  final String selectedTaskType;
  final String Function(String taskType) labelBuilder;
  final String Function(String taskType) hintBuilder;
  final List<String> options;
  final ValueChanged<String> onSelect;

  const _VideoTaskTypeSheet({
    required this.selectedTaskType,
    required this.labelBuilder,
    required this.hintBuilder,
    required this.options,
    required this.onSelect,
  });

  static IconData _iconForTaskType(String taskType) {
    switch (taskType) {
      case 'video_dual_chain':
        return Icons.account_tree_rounded;
      case 'video_3dgs':
        return Icons.view_in_ar_rounded;
      case 'da3_feed_forward_3dgs':
        return Icons.rocket_launch_rounded;
      case 'da3_sugar':
        return Icons.star_rounded;
      case 'da3_2dgs':
        return Icons.grid_view_rounded;
      case 'sparse2dgs':
        return Icons.filter_frames_rounded;
      default:
        return Icons.video_library_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AppConfig.isNightMode;
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;
    final darkInput = const Color(0xFF23232A);

    Widget modeTile(String taskType) {
      final selected = selectedTaskType == taskType;
      final icon = _iconForTaskType(taskType);
      return InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: () => onSelect(taskType),
        child: AnimatedContainer(
          duration: BDMotion.durationFast,
          curve: Curves.easeOutCubic,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: selected
                ? BDDesign.colorMutedBlue.withValues(
                    alpha: isDark ? 0.22 : 0.10,
                  )
                : (isDark ? darkInput : const Color(0xFFF6F8FC)),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: selected
                  ? BDDesign.colorMutedBlue
                  : (isDark
                      ? Colors.white.withValues(alpha: 0.08)
                      : BDDesign.colorMutedBlue.withValues(alpha: 0.14)),
            ),
          ),
          child: Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: selected
                      ? BDDesign.colorMutedBlue.withValues(alpha: 0.18)
                      : (isDark
                          ? Colors.white.withValues(alpha: 0.05)
                          : Colors.white),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: textColor, size: 20),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      labelBuilder(taskType),
                      style: TextStyle(
                        color: textColor,
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      hintBuilder(taskType),
                      style: TextStyle(
                        color: hintColor,
                        fontSize: 12.5,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
              if (selected)
                const Icon(
                  Icons.check_circle_rounded,
                  color: BDDesign.colorMutedBlue,
                ),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
      child: BDPanelCard(
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
        child: SafeArea(
          top: false,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  textLocalize('gen_video_task_title'),
                  style: TextStyle(
                    color: textColor,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  textLocalize('gen_video_task_sheet_desc'),
                  style: TextStyle(
                    color: hintColor,
                    fontSize: 12.5,
                    height: 1.35,
                  ),
                ),
                const SizedBox(height: 16),
                ...options.map((taskType) => Padding(
                      padding: EdgeInsets.only(
                        bottom: taskType == options.last ? 0 : 10,
                      ),
                      child: modeTile(taskType),
                    )),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _GenerateSectionHeading extends StatelessWidget {
  final String title;
  final String description;

  const _GenerateSectionHeading({
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          description,
          style: TextStyle(
            fontSize: 13,
            height: 1.45,
            color: isDark
                ? Colors.white.withValues(alpha: 0.62)
                : BDDesign.colorMutedBlue,
          ),
        ),
      ],
    );
  }
}
