import 'dart:async';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/app_config.dart';
import '../services/task_notification_service.dart';
import 'webgl_viewer.dart';

/// 状态分类配置
class StatusCategory {
  final String status;
  final String labelKey;
  final IconData icon;
  final Color color;
  final int priority;

  const StatusCategory({
    required this.status,
    required this.labelKey,
    required this.icon,
    required this.color,
    required this.priority,
  });
}

/// 预定义的状态分类
const List<StatusCategory> statusCategories = [
  StatusCategory(
    status: 'pending',
    labelKey: 'status_pending',
    icon: Icons.schedule,
    color: Colors.orange,
    priority: 1,
  ),
  StatusCategory(
    status: 'processing',
    labelKey: 'status_processing',
    icon: Icons.sync,
    color: Colors.blue,
    priority: 2,
  ),
  StatusCategory(
    status: 'completed',
    labelKey: 'status_completed',
    icon: Icons.check_circle,
    color: Colors.green,
    priority: 3,
  ),
  StatusCategory(
    status: 'failed',
    labelKey: 'status_failed',
    icon: Icons.error,
    color: Colors.red,
    priority: 4,
  ),
];

/// 可展开收起的分类组件
class ExpandableCategorySection extends StatefulWidget {
  final String title;
  final IconData icon;
  final Color color;
  final List<Map<String, dynamic>> tasks;
  final Map<String, List<String>> taskLogs; // taskId -> logs
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
  State<ExpandableCategorySection> createState() => _ExpandableCategorySectionState();
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
    final darkInput = const Color(0xFF23232A);
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
              children: widget.tasks.map((task) => _buildTaskItem(task, theme, widget.isDark, darkInput)).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTaskItem(Map<String, dynamic> task, TDThemeData theme, bool isDark, Color darkInput) {
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

    // 任务类型图标映射
    final taskTypeIcon = _getTaskTypeIcon(taskType);

    // 判断是否为 processing 状态（显示加载动画）
    final isProcessing = widget.color == Colors.blue;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark ? darkInput : theme.grayColor1,
        borderRadius: BorderRadius.circular(12),
      ),
      child: InkWell(
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

/// 任务列表页面 - 使用Supabase Realtime轮询
class TaskListPage extends StatefulWidget {
  const TaskListPage({super.key});

  @override
  State<TaskListPage> createState() => _TaskListPageState();
}

class _TaskListPageState extends State<TaskListPage> {
  Map<String, List<Map<String, dynamic>>> _tasksByStatus = {};
  Map<String, List<String>> _taskLogs = {}; // taskId -> logs
  Map<String, bool> _expandedStatus = {};
  bool _isLoading = true;
  String? _error;
  Timer? _refreshTimer;
  StreamSubscription<AuthState>? _authSubscription;

  // 颜色配置
  final darkBg = const Color(0xFF101014);
  final darkCard = const Color(0xFF18181C);

  /// 解析 logs JSON，返回 msg 列表
  List<String> _parseLogMsgs(dynamic logs) {
    if (logs == null) return [];
    if (logs is! List) return [];
    
    try {
      final List<String> result = [];
      for (final log in logs) {
        if (log is Map) {
          final msg = log['msg']?.toString() ?? '';
          if (msg.isNotEmpty) {
            result.add(msg);
          }
        }
      }
      return result;
    } catch (e) {
      return [];
    }
  }

  @override
  void initState() {
    super.initState();
    _initializeData();
  }

  Future<void> _initializeData() async {
    await _fetchTasks();
    _setupAutoRefresh();
    _listenAuthChanges();
  }

  void _listenAuthChanges() {
    _authSubscription = Supabase.instance.client.auth.onAuthStateChange.listen((event) {
      if (event.event == AuthChangeEvent.signedOut) {
        // 用户登出时清空数据
        setState(() {
          _tasksByStatus = {};
          _error = textLocalize('error_not_logged_in');
        });
      } else if (event.event == AuthChangeEvent.signedIn) {
        // 用户登录时刷新数据
        _fetchTasks();
      }
    });
  }

  void _setupAutoRefresh() {
    // 每5秒自动刷新一次，实现类似realtime的效果
    _refreshTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      // 调试阶段：不检查登录状态
      if (mounted) {
        _fetchTasksSilent();
      }
      // if (mounted && Supabase.instance.client.auth.currentSession != null) {
      //   _fetchTasksSilent();
      // }
    });
  }

  Future<void> _fetchTasksSilent() async {
    try {
      final response = await Supabase.instance.client
          .from('processing_tasks')
          .select('*')
          .order('created_at', ascending: false);

      if (mounted) {
        final Map<String, List<Map<String, dynamic>>> grouped = {};
        final Map<String, List<String>> logMap = {};
        
        for (final task in response) {
          final status = task['status']?.toString() ?? 'pending';
          final taskId = task['id'].toString();
          
          grouped.putIfAbsent(status, () => []);
          grouped[status]!.add(Map<String, dynamic>.from(task));
          
          // 解析 logs
          final logs = task['logs'];
          if (logs is List) {
            final parsedLogs = _parseLogMsgs(logs);
            if (parsedLogs.isNotEmpty) {
              logMap[taskId] = parsedLogs;
            }
          }
        }

        for (final status in grouped.keys) {
          _expandedStatus.putIfAbsent(status, () => true);
        }

        setState(() {
          _tasksByStatus = grouped;
          _taskLogs = logMap;
        });

        // 标记所有任务为已通知（更新缓存）
        final allTasks = grouped.values.expand((list) => list).toList();
        taskNotificationService.markAllTasksAsNotified(allTasks);
      }
    } catch (e) {
      // 静默刷新失败不显示错误
    }
  }

  Future<void> _fetchTasks() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final response = await Supabase.instance.client
          .from('processing_tasks')
          .select('*')
          .order('created_at', ascending: false);

      if (mounted) {
        final Map<String, List<Map<String, dynamic>>> grouped = {};
        final Map<String, List<String>> logMap = {};
        
        for (final task in response) {
          final status = task['status']?.toString() ?? 'pending';
          final taskId = task['id'].toString();
          
          grouped.putIfAbsent(status, () => []);
          grouped[status]!.add(Map<String, dynamic>.from(task));
          
          // 解析 logs
          final logs = task['logs'];
          if (logs is List) {
            final parsedLogs = _parseLogMsgs(logs);
            if (parsedLogs.isNotEmpty) {
              logMap[taskId] = parsedLogs;
            }
          }
        }

        // 初始化展开状态
        for (final status in grouped.keys) {
          _expandedStatus.putIfAbsent(status, () => true);
        }

        setState(() {
          _tasksByStatus = grouped;
          _taskLogs = logMap;
          _isLoading = false;
        });

        // 标记所有任务为已通知（更新缓存）
        final allTasks = grouped.values.expand((list) => list).toList();
        taskNotificationService.markAllTasksAsNotified(allTasks);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = e.toString();
        });
        TDToast.showText(
          '${textLocalize('error_fetch_tasks')}: $e',
          context: context,
        );
      }
    }
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _authSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final textColor = isDark ? const Color(0xFFFFFFFF) : const Color(0xFF333333);

    return Scaffold(
      backgroundColor: isDark ? darkBg : theme.grayColor1,
      appBar: AppBar(
        backgroundColor: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        elevation: 0,
        scrolledUnderElevation: 0,
        surfaceTintColor: Colors.transparent,
        centerTitle: true,
        title: Text(
          textLocalize('task_list_title'),
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: textColor,
          ),
        ),
        actions: [
          IconButton(
            icon: AnimatedRotation(
              turns: _isLoading ? 1 : 0,
              duration: const Duration(milliseconds: 600),
              child: Icon(
                Icons.refresh,
                color: isDark ? const Color(0xFFEEEEEE) : const Color(0xFF333333),
              ),
            ),
            tooltip: textLocalize('refresh'),
            onPressed: _isLoading ? null : _fetchTasks,
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: isDark
                ? [darkBg, darkCard]
                : [theme.grayColor1, theme.whiteColor1],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: _buildBody(theme, isDark, textColor),
      ),
    );
  }

  Widget _buildBody(TDThemeData theme, bool isDark, Color textColor) {
    if (_isLoading) {
      return const Center(
        child: TDLoading(
          size: TDLoadingSize.large,
          icon: TDLoadingIcon.circle,
        ),
      );
    }

    if (_error != null) {
      return _buildErrorState(theme, isDark, textColor);
    }

    if (_tasksByStatus.isEmpty) {
      return _buildEmptyState(theme, isDark, textColor);
    }

    return RefreshIndicator(
      onRefresh: _fetchTasks,
      child: ListView(
        padding: const EdgeInsets.only(top: 8, bottom: 100),
        children: _buildCategorySections(theme, isDark, textColor),
      ),
    );
  }

  List<Widget> _buildCategorySections(TDThemeData theme, bool isDark, Color textColor) {
    // 按优先级排序状态
    final sortedStatuses = _tasksByStatus.keys.toList()..sort((a, b) {
      final priorityA = statusCategories.firstWhere(
        (c) => c.status == a,
        orElse: () => StatusCategory(
          status: a,
          labelKey: a,
          icon: Icons.folder,
          color: Colors.grey,
          priority: 99,
        ),
      ).priority;
      final priorityB = statusCategories.firstWhere(
        (c) => c.status == b,
        orElse: () => StatusCategory(
          status: b,
          labelKey: b,
          icon: Icons.folder,
          color: Colors.grey,
          priority: 99,
        ),
      ).priority;
      return priorityA.compareTo(priorityB);
    });

    return sortedStatuses.map((status) {
      final category = statusCategories.firstWhere(
        (c) => c.status == status,
        orElse: () => StatusCategory(
          status: status,
          labelKey: status,
          icon: Icons.folder,
          color: Colors.grey,
          priority: 99,
        ),
      );

      return ExpandableCategorySection(
        key: ValueKey(status),
        title: textLocalize(category.labelKey),
        icon: category.icon,
        color: category.color,
        tasks: _tasksByStatus[status]!,
        taskLogs: _taskLogs,
        initiallyExpanded: _expandedStatus[status] ?? true,
        isDark: isDark,
        textColor: textColor,
        onTaskTap: (task) => _onTaskTap(task),
      );
    }).toList();
  }

  void _onTaskTap(Map<String, dynamic> task) {
    final status = task['status']?.toString();
    final sceneId = task['scene_id']?.toString();
    
    // 只有completed状态的任务才能查看
    if (status == 'completed' && sceneId != null) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => WebGLViewerPage(sceneId: sceneId),
        ),
      );
    } else {
      // 显示任务详情或状态提示
      TDToast.showText(
        textLocalize('task_status_${status ?? 'unknown'}'),
        context: context,
      );
    }
  }

  Widget _buildEmptyState(TDThemeData theme, bool isDark, Color textColor) {
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32.0),
          border: Border.all(
            color: isDark ? const Color(0xFF23232A) : theme.whiteColor1,
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withAlpha(20),
              blurRadius: 20,
              spreadRadius: 5,
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.inbox_outlined,
              size: 80,
              color: isDark ? const Color(0xFF666666) : theme.fontGyColor4,
            ),
            const SizedBox(height: 24),
            Text(
              textLocalize('task_list_empty_title'),
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: textColor,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              textLocalize('task_list_empty_desc'),
              style: TextStyle(
                fontSize: 14,
                color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorState(TDThemeData theme, bool isDark, Color textColor) {
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32.0),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.error_outline,
              size: 64,
              color: Colors.red.withAlpha(180),
            ),
            const SizedBox(height: 16),
            Text(
              textLocalize('error_title'),
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: textColor,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              _error ?? textLocalize('error_unknown'),
              style: TextStyle(
                fontSize: 14,
                color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            TDButton(
              text: textLocalize('retry'),
              icon: Icons.refresh,
              type: TDButtonType.fill,
              theme: TDButtonTheme.primary,
              shape: TDButtonShape.round,
              size: TDButtonSize.medium,
              onTap: _fetchTasks,
            ),
          ],
        ),
      ),
    );
  }
}
