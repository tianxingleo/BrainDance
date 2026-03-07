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
      if (newStatus == 'completed' && oldStatus != 'completed') {
        if (!_notifiedCompletedTasks.contains(id)) {
          _notifiedCompletedTasks.add(id);
          _saveNotifiedTasksToCache();
          _showTaskNotification(completedCount: 1, failedCount: 0);
        }
      } else if (newStatus == 'failed' && oldStatus != 'failed') {
        if (!_notifiedFailedTasks.contains(id)) {
          _notifiedFailedTasks.add(id);
          _saveNotifiedTasksToCache();
          _showTaskNotification(completedCount: 0, failedCount: 1);
        }
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
    _currentNotification = TaskNotificationData(
      completedCount: completedCount,
      failedCount: failedCount,
    );
    notifyListeners();

    // 5秒后自动隐藏
    Future.delayed(const Duration(seconds: 5), () {
      hideNotification();
    });
  }

  /// 销毁服务
  void dispose() {
    stopMonitoring();
    super.dispose();
  }
}

/// 全局实例
final taskNotificationService = TaskNotificationService();