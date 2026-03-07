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
/// 单例模式，全局管理任务状态监听
class TaskNotificationService extends ChangeNotifier {
  static final TaskNotificationService _instance =
      TaskNotificationService._internal();
  factory TaskNotificationService() => _instance;
  TaskNotificationService._internal();

  Timer? _timer;
  Set<String> _notifiedCompletedTasks = {};
  Set<String> _notifiedFailedTasks = {};
  GlobalKey<NavigatorState>? _navigatorKey;

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