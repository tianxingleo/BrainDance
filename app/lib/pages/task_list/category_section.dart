import 'package:braindance/configs/motion_tokens.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

class ExpandableCategorySection extends StatefulWidget {
  final String title;
  final IconData icon;
  final Color color;
  final List<Map<String, dynamic>> tasks;
  final Map<String, List<String>> taskLogs;
  final bool initiallyExpanded;
  final Function(Map<String, dynamic>)? onTaskTap;
  final Color? textColor;
  final bool isDark;

  const ExpandableCategorySection({
    super.key,
    required this.title,
    required this.icon,
    required this.color,
    required this.tasks,
    this.taskLogs = const {},
    this.initiallyExpanded = true,
    this.onTaskTap,
    this.textColor,
    required this.isDark,
  });

  @override
  State<ExpandableCategorySection> createState() =>
      _ExpandableCategorySectionState();
}

class _ExpandableCategorySectionState extends State<ExpandableCategorySection>
    with SingleTickerProviderStateMixin {
  late bool _isExpanded;
  late AnimationController _controller;
  late Animation<double> _iconTurns;
  late Animation<double> _heightFactor;

  @override
  void initState() {
    super.initState();
    _isExpanded = widget.initiallyExpanded;
    _controller = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );
    _iconTurns = Tween<double>(
      begin: 0.0,
      end: 0.5,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
    _heightFactor = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
    if (_isExpanded) {
      _controller.value = 1.0;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _toggleExpand() {
    setState(() {
      _isExpanded = !_isExpanded;
      if (_isExpanded) {
        _controller.forward();
      } else {
        _controller.reverse();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final darkCard = BDDesign.colorInkBlack;
    final darkInput = BDDesign.colorInkBlack.withAlpha(200);
    final bgColor = widget.isDark ? darkCard : theme.whiteColor1.withAlpha(220);
    final borderColor = widget.color.withAlpha(80);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: borderColor, width: 1.5),
        boxShadow: widget.isDark
            ? []
            : [
                BoxShadow(
                  color: widget.color.withAlpha(15),
                  blurRadius: 16,
                  offset: const Offset(0, 4),
                ),
              ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          InkWell(
            onTap: _toggleExpand,
            borderRadius: BorderRadius.circular(24),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(4),
                    color: Colors.transparent,
                    child: Icon(widget.icon, size: 20, color: widget.color),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.title,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: widget.textColor ?? (widget.isDark ? Colors.white : Colors.black87),
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: widget.color.withAlpha(20),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      '${widget.tasks.length}',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                        color: widget.color,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  RotationTransition(
                    turns: _iconTurns,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: widget.isDark
                          ? const Color(0xFF888888)
                          : theme.fontGyColor3,
                    ),
                  ),
                ],
              ),
            ),
          ),
          AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              return ClipRect(
                child: Align(heightFactor: _heightFactor.value, child: child),
              );
            },
            child: Column(
              children: widget.tasks
                  .map(
                    (task) =>
                        _buildTaskItem(task, theme, widget.isDark, darkInput),
                  )
                  .toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTaskItem(
    Map<String, dynamic> task,
    TDThemeData theme,
    bool isDark,
    Color darkInput,
  ) {
    final taskId = task['id'].toString();
    final sceneId = task['scene_id']?.toString() ?? 'Unknown';
    final description = task['description']?.toString() ?? '';
    final displayName = task['display_name']?.toString();
    final createdAt = task['created_at'] != null
        ? DateTime.tryParse(task['created_at'].toString())
        : null;
    final taskType = task['task_type']?.toString() ?? 'video_3dgs';

    final allLogs = widget.taskLogs[taskId] ?? [];
    final latestLog = allLogs.isNotEmpty ? allLogs.last : null;
    final taskTypeIcon = _getTaskTypeIcon(taskType);
    final isProcessing = widget.color == BDDesign.colorMutedBlue;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: isDark ? BDDesign.colorInkBlack : BDDesign.colorPaperWhite,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: widget.color.withAlpha(isDark ? 80 : 40),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: widget.color.withAlpha(isDark ? 10 : 20),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: InkWell(
        onTap: widget.onTaskTap != null ? () => widget.onTaskTap!(task) : null,
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                margin: const EdgeInsets.only(top: 2),
                child: isProcessing
                    ? SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            widget.color,
                          ),
                        ),
                      )
                    : Icon(taskTypeIcon, color: widget.color, size: 18),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Text(
                            displayName ?? sceneId,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: isDark
                                  ? BDDesign.colorPaperWhite
                                  : BDDesign.colorInkBlack,
                              letterSpacing: 0.5,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        if (createdAt != null)
                          Text(
                            _formatDate(createdAt),
                            style: TextStyle(
                              fontSize: 12,
                              color: isDark ? BDDesign.colorAshGray : theme.fontGyColor3,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'ID: ${taskId.length > 8 ? taskId.substring(0, 8) : taskId} • ${_getTaskTypeLabel(taskType)}',
                      style: TextStyle(
                        fontSize: 12,
                        color: isDark
                            ? BDDesign.colorAshGray
                            : BDDesign.colorInkBlack.withAlpha(150),
                      ),
                    ),
                    if (latestLog != null || description.isNotEmpty) ...[
                      const SizedBox(height: 6),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: isDark
                              ? darkInput
                              : BDDesign.colorAshGray.withAlpha(28),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          latestLog ?? description,
                          style: TextStyle(
                            fontSize: 12,
                            color: isDark ? BDDesign.colorAshGray : theme.fontGyColor2,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  IconData _getTaskTypeIcon(String taskType) {
    switch (taskType) {
      case 'video_3dgs':
        return Icons.videocam;
      case 'video_dual_chain':
        return Icons.hub;
      case 'single_image_sam3d':
        return Icons.image;
      case 'single_image_sharp':
        return Icons.auto_fix_high;
      case 'sparse2dgs':
        return Icons.photo_library;
      default:
        return Icons.view_in_ar;
    }
  }

  String _getTaskTypeLabel(String taskType) {
    switch (taskType) {
      case 'video_3dgs':
        return 'Video 3DGS';
      case 'video_dual_chain':
        return 'Dual Chain';
      case 'single_image_sam3d':
        return 'SAM3D';
      case 'single_image_sharp':
        return 'Sharp 3D';
      case 'sparse2dgs':
        return 'Sparse2DGS';
      default:
        return taskType;
    }
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final diff = now.difference(date);
    if (diff.inDays == 0) {
      if (diff.inHours == 0) {
        return '${diff.inMinutes}m ago';
      }
      return '${diff.inHours}h ago';
    } else if (diff.inDays < 7) {
      return '${diff.inDays}d ago';
    } else {
      return '${date.month}/${date.day}';
    }
  }
}
