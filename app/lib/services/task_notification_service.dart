import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

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
  Set<String> _notifiedCompletedTasks = {};
  Set<String> _notifiedFailedTasks = {};
  
  // 待通知的计数器（尚未被缓存记录的新任务）
  int _pendingCompletedCount = 0;
  int _pendingFailedCount = 0;
  
  GlobalKey<NavigatorState>? _navigatorKey;

  // 禁用通知的路由集合
  final Set<String> _disabledRoutes = {};

  // 当前显示的通知数据
  TaskNotificationData? _currentNotification;
  TaskNotificationData? get currentNotification => _currentNotification;

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

  /// 禁用指定路由的通知
  void disableNotificationForRoutes(List<String> routes) {
    _disabledRoutes.addAll(routes);
  }

  /// 启用指定路由的通知
  void enableNotificationForRoutes(List<String> routes) {
    _disabledRoutes.removeAll(routes);
  }

  /// 检查当前路由是否允许显示通知
  bool isNotificationEnabledForRoute(String? route) {
    if (route == null) return true;
    return !_disabledRoutes.contains(route);
  }

  /// 获取当前路由
  String? get currentRoute {
    final context = _navigatorKey?.currentContext;
    if (context == null) return null;
    return ModalRoute.of(context)?.settings.name;
  }

  /// 从本地缓存加载已通知过的任务ID
  Future<void> _loadNotifiedTasksFromCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _notifiedCompletedTasks =
          _loadSetFromPrefs(prefs, _kNotifiedCompletedTasks);
      _notifiedFailedTasks = _loadSetFromPrefs(prefs, _kNotifiedFailedTasks);
    } catch (e) {
      // 静默失败，使用空集合
    }
  }

  /// 从 SharedPreferences 加载 Set
  Set<String> _loadSetFromPrefs(SharedPreferences prefs, String key) {
    final json = prefs.getString(key);
    if (json == null) return {};
    return Set<String>.from(jsonDecode(json) as List<dynamic>);
  }

  /// 保存已通知过的任务ID到本地缓存
  Future<void> _saveNotifiedTasksToCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await Future.wait([
        prefs.setString(
            _kNotifiedCompletedTasks, jsonEncode(_notifiedCompletedTasks.toList())),
        prefs.setString(
            _kNotifiedFailedTasks, jsonEncode(_notifiedFailedTasks.toList())),
      ]);
    } catch (e) {
      // 静默失败
    }
  }

  /// 启动 Realtime 监听
  void startMonitoring() {
    if (_channel != null) return; // 已经在运行

    _channel = Supabase.instance.client.channel('public:processing_tasks');

    _channel!.onPostgresChanges(
      event: PostgresChangeEvent.update,
      schema: 'public',
      table: 'processing_tasks',
      callback: (payload) => _handleTaskChange(payload),
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

      // 检测状态变化为 completed 或 failed
      // 增加待通知计数器，显示累积数量
      if (newStatus == 'completed' && oldStatus != 'completed') {
        if (!_notifiedCompletedTasks.contains(id)) {
          _pendingCompletedCount++;
          _showPendingNotification();
        }
      } else if (newStatus == 'failed' && oldStatus != 'failed') {
        if (!_notifiedFailedTasks.contains(id)) {
          _pendingFailedCount++;
          _showPendingNotification();
        }
      }
    } catch (e) {
      // 静默失败
    }
  }

  /// 显示待通知任务的累积数量
  void _showPendingNotification() {
    if (_pendingCompletedCount > 0 || _pendingFailedCount > 0) {
      _currentNotification = TaskNotificationData(
        completedCount: _pendingCompletedCount,
        failedCount: _pendingFailedCount,
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

      if (status == 'completed' && !_notifiedCompletedTasks.contains(id)) {
        _notifiedCompletedTasks.add(id);
        hasChanges = true;
      } else if (status == 'failed' && !_notifiedFailedTasks.contains(id)) {
        _notifiedFailedTasks.add(id);
        hasChanges = true;
      }
    }

    if (hasChanges) {
      await _saveNotifiedTasksToCache();
    }

    // 重置待通知计数器
    _pendingCompletedCount = 0;
    _pendingFailedCount = 0;
    hideNotification();
  }

  /// 销毁服务
  void dispose() {
    stopMonitoring();
    super.dispose();
  }
}

/// 全局实例
final taskNotificationService = TaskNotificationService();