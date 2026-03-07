import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// 任务状态类型
enum TaskStatusType { completed, failed }

/// 通知数据类
class TaskNotificationData {
  final int completedCount;
  final int failedCount;

  TaskNotificationData({
    required this.completedCount,
    required this.failedCount,
  });
}

/// 全局任务通知服务
/// 单例模式，使用 Supabase Realtime 监听任务状态变化
class TaskNotificationService extends ChangeNotifier {
  static final TaskNotificationService _instance =
      TaskNotificationService._internal();
  factory TaskNotificationService() => _instance;
  TaskNotificationService._internal();

  RealtimeChannel? _channel;
  
  // 已通知的任务ID缓存（按状态分组）
  final Map<TaskStatusType, Set<String>> _notifiedTasks = {
    TaskStatusType.completed: {},
    TaskStatusType.failed: {},
  };
  
  // 待通知的计数器
  final Map<TaskStatusType, int> _pendingCount = {
    TaskStatusType.completed: 0,
    TaskStatusType.failed: 0,
  };
  
  GlobalKey<NavigatorState>? _navigatorKey;

  // 当前显示的通知数据
  TaskNotificationData? _currentNotification;
  TaskNotificationData? get currentNotification => _currentNotification;

  // 本地缓存 key
  static const Map<TaskStatusType, String> _cacheKeys = {
    TaskStatusType.completed: 'notified_completed_tasks',
    TaskStatusType.failed: 'notified_failed_tasks',
  };

  /// 初始化服务
  Future<void> init() async {
    await _loadNotifiedTasksFromCache();
    startMonitoring();
  }

  /// 设置 NavigatorKey（用于导航）
  void setNavigatorKey(GlobalKey<NavigatorState> key) {
    _navigatorKey = key;
  }

  /// 隐藏当前通知
  void hideNotification() {
    _currentNotification = null;
    notifyListeners();
  }

  /// 导航到任务列表
  void navigateToTaskList() {
    hideNotification();
    final context = _navigatorKey?.currentContext;
    if (context != null) {
      Navigator.pushNamed(context, '/tasks');
    }
  }

  /// 获取当前路由
  String? get currentRoute {
    final context = _navigatorKey?.currentContext;
    if (context == null) return null;
    return ModalRoute.of(context)?.settings.name;
  }

  /// 检查当前路由是否允许显示通知
  bool isNotificationEnabledForRoute(String? route) {
    if (route == null) return true;
    // 在任务列表页面不显示通知
    return route != '/tasks';
  }

  /// 从本地缓存加载已通知过的任务ID
  Future<void> _loadNotifiedTasksFromCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      for (final entry in _cacheKeys.entries) {
        final json = prefs.getString(entry.value);
        if (json != null) {
          _notifiedTasks[entry.key] = Set<String>.from(jsonDecode(json) as List<dynamic>);
        }
      }
    } catch (e) {
      // 静默失败，使用空集合
    }
  }

  /// 保存已通知过的任务ID到本地缓存
  Future<void> _saveNotifiedTasksToCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await Future.wait(
        _cacheKeys.entries.map((entry) => prefs.setString(
          entry.value,
          jsonEncode(_notifiedTasks[entry.key]!.toList()),
        )),
      );
    } catch (e) {
      // 静默失败
    }
  }

  /// 启动 Realtime 监听
  void startMonitoring() {
    if (_channel != null) return;

    _channel = Supabase.instance.client.channel('public:processing_tasks');
    _channel!.onPostgresChanges(
      event: PostgresChangeEvent.update,
      schema: 'public',
      table: 'processing_tasks',
      callback: _handleTaskChange,
    );
    _channel!.subscribe();
  }

  /// 停止 Realtime 监听
  void stopMonitoring() {
    if (_channel != null) {
      Supabase.instance.client.removeChannel(_channel!);
      _channel = null;
    }
  }

  /// 处理任务状态变化
  void _handleTaskChange(PostgresChangePayload payload) {
    try {
      final newData = payload.newRecord;
      final oldData = payload.oldRecord;
      final String id = newData['id'].toString();
      final String newStatus = newData['status']?.toString() ?? 'pending';
      final String? oldStatus = oldData['status']?.toString();

      // 检测状态变化
      _checkStatusChange(id, newStatus, oldStatus, 'completed', TaskStatusType.completed);
      _checkStatusChange(id, newStatus, oldStatus, 'failed', TaskStatusType.failed);
    } catch (e) {
      // 静默失败
    }
  }

  /// 检查状态变化并更新计数器
  void _checkStatusChange(
    String id,
    String newStatus,
    String? oldStatus,
    String targetStatus,
    TaskStatusType type,
  ) {
    if (newStatus == targetStatus && oldStatus != targetStatus) {
      if (!_notifiedTasks[type]!.contains(id)) {
        _pendingCount[type] = _pendingCount[type]! + 1;
        _updateNotification();
      }
    }
  }

  /// 更新通知显示
  void _updateNotification() {
    final completedCount = _pendingCount[TaskStatusType.completed]!;
    final failedCount = _pendingCount[TaskStatusType.failed]!;
    
    if (completedCount > 0 || failedCount > 0) {
      _currentNotification = TaskNotificationData(
        completedCount: completedCount,
        failedCount: failedCount,
      );
      notifyListeners();
    }
  }

  /// 标记所有任务为已通知（在打开任务页面时调用）
  Future<void> markAllTasksAsNotified(List<Map<String, dynamic>> tasks) async {
    bool hasChanges = false;

    for (final task in tasks) {
      final String id = task['id'].toString();
      final String status = task['status']?.toString() ?? 'pending';
      
      final type = status == 'completed' 
          ? TaskStatusType.completed 
          : status == 'failed' 
              ? TaskStatusType.failed 
              : null;
      
      if (type != null && !_notifiedTasks[type]!.contains(id)) {
        _notifiedTasks[type]!.add(id);
        hasChanges = true;
      }
    }

    if (hasChanges) {
      await _saveNotifiedTasksToCache();
    }

    // 重置待通知计数器
    _pendingCount[TaskStatusType.completed] = 0;
    _pendingCount[TaskStatusType.failed] = 0;
    hideNotification();
  }

  @override
  void dispose() {
    stopMonitoring();
    super.dispose();
  }
}

/// 全局实例
final taskNotificationService = TaskNotificationService();