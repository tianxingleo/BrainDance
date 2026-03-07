import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../configs/app_config.dart';
import '../extra_func/language.dart';
import '../pages/task_list.dart';

/// 全局任务通知服务
/// 单例模式，全局管理任务状态监听
class TaskNotificationService {
  static final TaskNotificationService _instance =
      TaskNotificationService._internal();
  factory TaskNotificationService() => _instance;
  TaskNotificationService._internal();

  Timer? _timer;
  Set<String> _notifiedCompletedTasks = {};
  Set<String> _notifiedFailedTasks = {};
  OverlayEntry? _currentOverlay;
  GlobalKey<NavigatorState>? _navigatorKey;

  // 本地缓存 key
  static const String _kNotifiedCompletedTasks = 'notified_completed_tasks';
  static const String _kNotifiedFailedTasks = 'notified_failed_tasks';

  /// 初始化服务
  Future<void> init() async {
    await _loadNotifiedTasksFromCache();
    startMonitoring();
  }

  /// 设置 NavigatorKey（用于导航）
  void setNavigatorKey(GlobalKey<NavigatorState> key) {
    _navigatorKey = key;
  }

  /// 从本地缓存加载已通知过的任务ID
  Future<void> _loadNotifiedTasksFromCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final completedJson = prefs.getString(_kNotifiedCompletedTasks);
      final failedJson = prefs.getString(_kNotifiedFailedTasks);

      if (completedJson != null) {
        final List<dynamic> completedList = jsonDecode(completedJson);
        _notifiedCompletedTasks = Set<String>.from(completedList);
      }
      if (failedJson != null) {
        final List<dynamic> failedList = jsonDecode(failedJson);
        _notifiedFailedTasks = Set<String>.from(failedList);
      }
    } catch (e) {
      // 静默失败，使用空集合
    }
  }

  /// 保存已通知过的任务ID到本地缓存
  Future<void> _saveNotifiedTasksToCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(
        _kNotifiedCompletedTasks,
        jsonEncode(_notifiedCompletedTasks.toList()),
      );
      await prefs.setString(
        _kNotifiedFailedTasks,
        jsonEncode(_notifiedFailedTasks.toList()),
      );
    } catch (e) {
      // 静默失败
    }
  }

  /// 启动任务状态监听（每5秒检查一次）
  void startMonitoring() {
    if (_timer != null) return; // 已经在运行

    _fetchTaskStatuses(); // 立即获取一次
    _timer = Timer.periodic(const Duration(seconds: 5), (timer) {
      _fetchTaskStatuses();
    });
  }

  /// 停止任务状态监听
  void stopMonitoring() {
    _timer?.cancel();
    _timer = null;
  }

  /// 获取任务状态并检测变化
  Future<void> _fetchTaskStatuses() async {
    try {
      final response = await Supabase.instance.client
          .from('processing_tasks')
          .select('id, status, scene_id, display_name')
          .order('created_at', ascending: false);

      final List<String> newlyCompletedIds = [];
      final List<String> newlyFailedIds = [];

      for (final task in response) {
        final id = task['id'].toString();
        final status = task['status']?.toString() ?? 'pending';

        // 检测 completed 状态变化，且未被通知过
        if (status == 'completed' && !_notifiedCompletedTasks.contains(id)) {
          newlyCompletedIds.add(id);
        }
        // 检测 failed 状态变化，且未被通知过
        if (status == 'failed' && !_notifiedFailedTasks.contains(id)) {
          newlyFailedIds.add(id);
        }
      }

      // 如果有新完成或失败的任务，显示通知
      if (newlyCompletedIds.isNotEmpty || newlyFailedIds.isNotEmpty) {
        // 将新任务ID加入已通知集合
        _notifiedCompletedTasks.addAll(newlyCompletedIds);
        _notifiedFailedTasks.addAll(newlyFailedIds);

        // 保存到本地缓存
        _saveNotifiedTasksToCache();

        // 显示通知
        _showTaskNotification(
          completedCount: newlyCompletedIds.length,
          failedCount: newlyFailedIds.length,
        );
      }
    } catch (e) {
      // 静默失败
    }
  }

  /// 显示任务状态变化通知
  void _showTaskNotification({
    required int completedCount,
    required int failedCount,
  }) {
    final context = _navigatorKey?.currentContext;
    if (context == null) return;

    _hideNotification(); // 先隐藏之前的

    _currentOverlay = OverlayEntry(
      builder: (context) => _TaskNotificationWidget(
        completedCount: completedCount,
        failedCount: failedCount,
        onTap: () {
          _hideNotification();
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => const TaskListPage()),
          );
        },
        onDismiss: () {
          _hideNotification();
        },
      ),
    );

    Overlay.of(context).insert(_currentOverlay!);

    // 5秒后自动隐藏
    Future.delayed(const Duration(seconds: 5), () {
      _hideNotification();
    });
  }

  /// 隐藏通知
  void _hideNotification() {
    _currentOverlay?.remove();
    _currentOverlay = null;
  }

  /// 销毁服务
  void dispose() {
    stopMonitoring();
    _hideNotification();
  }
}

/// 全局实例
final taskNotificationService = TaskNotificationService();

/// 任务状态变化通知组件（类似 Edge 浏览器下载提示）
class _TaskNotificationWidget extends StatefulWidget {
  final int completedCount;
  final int failedCount;
  final VoidCallback? onTap;
  final VoidCallback? onDismiss;

  const _TaskNotificationWidget({
    required this.completedCount,
    required this.failedCount,
    this.onTap,
    this.onDismiss,
  });

  @override
  State<_TaskNotificationWidget> createState() => _TaskNotificationWidgetState();
}

class _TaskNotificationWidgetState extends State<_TaskNotificationWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, -1),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(_controller);
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AppConfig.isNightMode;
    final hasCompleted = widget.completedCount > 0;
    final hasFailed = widget.failedCount > 0;

    // 构建通知内容
    String message = '';
    IconData icon = Icons.check_circle;
    Color iconColor = Colors.green;

    if (hasCompleted && hasFailed) {
      message =
          '${widget.completedCount} ${textLocalize('task_completed')}，${widget.failedCount} ${textLocalize('task_failed')}';
      icon = Icons.info;
      iconColor = Colors.orange;
    } else if (hasCompleted) {
      message =
          '${widget.completedCount} ${textLocalize('task_notification_completed')}';
      icon = Icons.check_circle;
      iconColor = Colors.green;
    } else if (hasFailed) {
      message =
          '${widget.failedCount} ${textLocalize('task_notification_failed')}';
      icon = Icons.error;
      iconColor = Colors.red;
    }

    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: SafeArea(
        child: SlideTransition(
          position: _slideAnimation,
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: Material(
              color: Colors.transparent,
              child: Center(
                child: Container(
                  margin:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: isDark ? const Color(0xFF2A2A30) : Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withAlpha(30),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                    border: Border.all(
                      color: isDark
                          ? const Color(0xFF3A3A40)
                          : const Color(0xFFE0E0E0),
                      width: 1,
                    ),
                  ),
                  child: InkWell(
                    onTap: widget.onTap,
                    borderRadius: BorderRadius.circular(12),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: iconColor.withAlpha(20),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(icon, color: iconColor, size: 20),
                        ),
                        const SizedBox(width: 12),
                        Flexible(
                          child: Text(
                            message,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: isDark
                                  ? Colors.white
                                  : const Color(0xFF333333),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Icon(
                          Icons.keyboard_arrow_right,
                          color: isDark
                              ? const Color(0xFF888888)
                              : const Color(0xFF999999),
                          size: 20,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}