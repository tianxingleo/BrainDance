import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../../configs/app_config.dart';

/// 可展开收起的分类组件
class ExpandableCategorySection extends StatefulWidget {
  final String title;
  final IconData icon;
  final Color color;
  final List<Map<String, dynamic>> tasks;
  final Map<String, List<String>> taskLogs; // taskId -> logs
  final bool initiallyExpanded;
  final String status; // 状态标识
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
    required this.status,
    this.onTaskTap,
    this.textColor,
    required this.isDark,
  });

  @override
  State<ExpandableCategorySection> createState() => _ExpandableCategorySectionState();
}

class _ExpandableCategorySectionState extends State<ExpandableCategorySection>
    with SingleTickerProviderStateMixin {
  late bool _isExpanded;
  late AnimationController _controller;
  late Animation<double> _iconTurns;
  late Animation<double> _heightFactor;
  
  // 跟踪每个任务的logs展开状态
  final Map<String, bool> _logsExpanded = {};

  @override
  void initState() {
    super.initState();
    _isExpanded = widget.initiallyExpanded;
    _controller = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );
    _iconTurns = Tween<double>(begin: 0.0, end: 0.5).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
    _heightFactor = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
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
    final darkCard = const Color(0xFF18181C);
    final bgColor = widget.isDark ? darkCard : theme.whiteColor1.withAlpha(220);
    final borderColor = widget.isDark ? const Color(0xFF333333) : theme.grayColor3;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: borderColor, width: 1),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(15),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 标题栏（可点击展开/收起）
          InkWell(
            onTap: _toggleExpand,
            borderRadius: BorderRadius.circular(16),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: widget.color.withAlpha(30),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Icon(
                      widget.icon,
                      size: 20,
                      color: widget.color,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      widget.title,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: widget.textColor,
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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
                      color: widget.isDark ? const Color(0xFF888888) : theme.fontGyColor3,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // 任务列表（可展开/收起）
          AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              return ClipRect(
                child: Align(
                  heightFactor: _heightFactor.value,
                  child: child,
                ),
              );
            },
            child: Column(
              children: widget.tasks.map((task) => _buildTaskItem(task, theme, widget.isDark)).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTaskItem(Map<String, dynamic> task, TDThemeData theme, bool isDark) {
    final darkInput = const Color(0xFF23232A);
    final taskId = task['id'].toString();
    final sceneId = task['scene_id']?.toString() ?? 'Unknown';
    final description = task['description']?.toString() ?? '';
    final displayName = task['display_name']?.toString();
    final createdAt = task['created_at'] != null
        ? DateTime.tryParse(task['created_at'].toString())
        : null;
    final taskType = task['task_type']?.toString() ?? 'video_3dgs';
    
    // 获取该任务的 logs
    final allLogs = widget.taskLogs[taskId] ?? [];
    final latestLog = allLogs.isNotEmpty ? allLogs.last : null;
    final hasLogs = allLogs.isNotEmpty;

    // 任务类型图标映射
    final taskTypeIcon = _getTaskTypeIcon(taskType);

    // 判断是否为 processing 状态（显示加载动画）
    final isProcessing = widget.status == 'processing';
    
    // 获取当前任务的logs展开状态
    final isLogsExpanded = _logsExpanded[taskId] ?? false;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark ? darkInput : theme.grayColor1,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 主行：任务信息
          InkWell(
            onTap: widget.onTaskTap != null ? () => widget.onTaskTap!(task) : null,
            borderRadius: BorderRadius.circular(12),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: widget.color.withAlpha(20),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: isProcessing
                      ? const Center(
                          child: SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
                            ),
                          ),
                        )
                      : Icon(
                          taskTypeIcon,
                          color: widget.color,
                          size: 24,
                        ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        displayName ?? sceneId,
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          color: widget.textColor,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      // 显示最新日志或描述
                      Text(
                        latestLog ?? (description.isNotEmpty ? description : ''),
                        style: TextStyle(
                          fontSize: 12,
                          color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                if (createdAt != null)
                  Text(
                    _formatDate(createdAt),
                    style: TextStyle(
                      fontSize: 11,
                      color: isDark ? const Color(0xFF666666) : theme.fontGyColor4,
                    ),
                  ),
              ],
            ),
          ),
          // Logs展开按钮和内容
          if (hasLogs) ...[
            const SizedBox(height: 8),
            // 展开按钮
            InkWell(
              onTap: () {
                setState(() {
                  _logsExpanded[taskId] = !isLogsExpanded;
                });
              },
              borderRadius: BorderRadius.circular(8),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: isDark 
                      ? const Color(0xFF2A2A30) 
                      : theme.grayColor2.withAlpha(150),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isLogsExpanded 
                          ? Icons.keyboard_arrow_up 
                          : Icons.keyboard_arrow_down,
                      size: 16,
                      color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      isLogsExpanded 
                          ? '${textLocalize('logs_collapse')} (${allLogs.length})'
                          : '${textLocalize('logs_expand')} (${allLogs.length})',
                      style: TextStyle(
                        fontSize: 12,
                        color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            // 展开的日志列表
            if (isLogsExpanded) ...[
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: isDark 
                      ? const Color(0xFF1A1A1E) 
                      : theme.grayColor1,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: isDark 
                        ? const Color(0xFF333333) 
                        : theme.grayColor3,
                    width: 0.5,
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: allLogs.reversed.map((log) => 
                    Padding(
                      padding: const EdgeInsets.only(bottom: 6),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            margin: const EdgeInsets.only(top: 6),
                            width: 6,
                            height: 6,
                            decoration: BoxDecoration(
                              color: widget.color.withAlpha(150),
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: SelectableText(
                              log,
                              style: TextStyle(
                                fontSize: 12,
                                height: 1.4,
                                color: isDark 
                                    ? const Color(0xFFBBBBBB) 
                                    : theme.fontGyColor2,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ).toList(),
                ),
              ),
            ],
          ],
        ],
      ),
    );
  }

  IconData _getTaskTypeIcon(String taskType) {
    switch (taskType) {
      case 'video_3dgs':
        return Icons.videocam;
      case 'single_image_sam3d':
        return Icons.image;
      case 'single_image_sharp':
        return Icons.auto_fix_high;
      default:
        return Icons.view_in_ar;
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