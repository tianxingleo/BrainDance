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
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
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
                ...options.map(
                  (taskType) => Padding(
                    padding: EdgeInsets.only(
                      bottom: taskType == options.last ? 0 : 10,
                    ),
                    child: modeTile(taskType),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _GenerateTabBar extends StatefulWidget {
  final TabController controller;
  final List<String> labels;
  final VoidCallback onChanged;

  const _GenerateTabBar({
    required this.controller,
    required this.labels,
    required this.onChanged,
  });

  @override
  State<_GenerateTabBar> createState() => _GenerateTabBarState();
}

class _GenerateTabBarState extends State<_GenerateTabBar> {
  @override
  Widget build(BuildContext context) {
    final isDark = AppConfig.isNightMode;

    final navBackground = isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.55)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.52);
    final navBorder = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
    final navShadow = Colors.black.withValues(alpha: isDark ? 0.22 : 0.05);

    final selectedBackground = isDark
        ? const Color(0xFFAEBAC7).withValues(alpha: 0.14)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.09);
    final selectedColor = isDark
        ? const Color(0xFFF4F7FA)
        : BDDesign.colorInkBlack;
    final unselectedColor = isDark
        ? const Color(0xFFB4BEC9)
        : const Color(0xFF9AA3AD);

    return ClipRRect(
      borderRadius: BDDesign.radiusLarge,
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 24.0, sigmaY: 24.0),
        child: Container(
          height: 56,
          padding: const EdgeInsets.all(4.0),
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
              final tabWidth = constraints.maxWidth / widget.labels.length;
              return Stack(
                children: [
                  AnimatedBuilder(
                    animation: widget.controller.animation!,
                    builder: (context, child) {
                      final double offset =
                          widget.controller.animation!.value * tabWidth;
                      return Positioned(
                        left: offset,
                        width: tabWidth,
                        top: 0,
                        bottom: 0,
                        child: Container(
                          decoration: BoxDecoration(
                            color: selectedBackground,
                            borderRadius: BorderRadius.circular(22),
                          ),
                        ),
                      );
                    },
                  ),
                  Row(
                    children: List.generate(widget.labels.length, (index) {
                      return Expanded(
                        child: GestureDetector(
                          onTap: () {
                            if (widget.controller.index != index) {
                              widget.controller.animateTo(index);
                              widget.onChanged();
                            }
                          },
                          behavior: HitTestBehavior.opaque,
                          child: Center(
                            child: AnimatedBuilder(
                              animation: widget.controller.animation!,
                              builder: (ctx, child) {
                                final selected =
                                    index == widget.controller.index;
                                return Text(
                                  widget.labels[index],
                                  style: TextStyle(
                                    color: selected
                                        ? selectedColor
                                        : unselectedColor,
                                    fontWeight: FontWeight.w600,
                                    fontSize: 14,
                                  ),
                                );
                              },
                            ),
                          ),
                        ),
                      );
                    }),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    );
  }
}

class _GenerateSectionHeading extends StatelessWidget {
  final String title;
  final String? description;

  const _GenerateSectionHeading({
    required this.title,
    this.description,
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
        if (description != null) ...[
          const SizedBox(height: 6),
          Text(
            description!,
            style: TextStyle(
              fontSize: 13,
              height: 1.45,
              color: isDark
                  ? Colors.white.withValues(alpha: 0.62)
                  : BDDesign.colorMutedBlue,
            ),
          ),
        ],
      ],
    );
  }
}
