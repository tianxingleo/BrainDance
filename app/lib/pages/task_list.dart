import 'dart:async';

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../configs/app_config.dart';
import '../configs/motion_tokens.dart';
import '../configs/supabase_config.dart';
import '../services/task_notification_service.dart';
import '../services/viewer_navigation.dart';
import '../widgets/bd_surfaces.dart';
import 'task_list/category_section.dart';
import 'task_list/status_category.dart';

/// 任务列表页面 - 使用Supabase Realtime轮询
class TaskListPage extends StatefulWidget {
  const TaskListPage({super.key});

  @override
  State<TaskListPage> createState() => _TaskListPageState();
}

class _TaskListPageState extends State<TaskListPage> {
  Map<String, List<Map<String, dynamic>>> _tasksByStatus = {};
  Map<String, List<String>> _taskLogs = {}; // taskId -> logs
  final Map<String, bool> _expandedStatus = {};
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
    if (SupabaseConfig.isAdminMode) {
      return;
    }

    _authSubscription = Supabase.instance.client.auth.onAuthStateChange.listen((
      event,
    ) {
      if (event.event == AuthChangeEvent.signedOut) {
        _refreshTimer?.cancel();
        _refreshTimer = null;
        setState(() {
          _tasksByStatus = {};
          _error = textLocalize('error_not_logged_in');
        });
      } else if (event.event == AuthChangeEvent.signedIn) {
        _fetchTasks();
        _setupAutoRefresh();
      }
    });
  }

  void _setupAutoRefresh() {
    _refreshTimer?.cancel();
    _refreshTimer = Timer.periodic(const Duration(seconds: 15), (timer) {
      if (mounted) {
        _fetchTasksSilent();
      }
      // if (mounted && Supabase.instance.client.auth.currentSession != null) {
      //   _fetchTasksSilent();
      // }
    });
  }

  /// 查询 processing_tasks 并分组，返回分组结果和日志映射
  Future<
    ({
      Map<String, List<Map<String, dynamic>>> grouped,
      Map<String, List<String>> logMap,
    })
  >
  _queryAndGroupTasks() async {
    final query = Supabase.instance.client.from('processing_tasks').select('*');

    final response = SupabaseConfig.isAdminMode
        ? await query.order('created_at', ascending: false)
        : await query
              .eq(
                'user_id',
                Supabase.instance.client.auth.currentUser?.id ??
                    (throw Exception(textLocalize('error_not_logged_in'))),
              )
              .order('created_at', ascending: false);

    final Map<String, List<Map<String, dynamic>>> grouped = {};
    final Map<String, List<String>> logMap = {};

    for (final task in response) {
      final status = task['status']?.toString() ?? 'pending';
      final taskId = task['id'].toString();

      grouped.putIfAbsent(status, () => []);
      grouped[status]!.add(Map<String, dynamic>.from(task));

      final logs = task['logs'];
      if (logs is List) {
        final parsedLogs = _parseLogMsgs(logs);
        if (parsedLogs.isNotEmpty) {
          logMap[taskId] = parsedLogs;
        }
      }
    }

    return (grouped: grouped, logMap: logMap);
  }

  Future<void> _fetchTasksSilent() async {
    try {
      final result = await _queryAndGroupTasks();

      if (mounted) {
        for (final status in result.grouped.keys) {
          _expandedStatus.putIfAbsent(status, () => true);
        }

        setState(() {
          _tasksByStatus = result.grouped;
          _taskLogs = result.logMap;
        });

        final allTasks = result.grouped.values.expand((list) => list).toList();
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
      final result = await _queryAndGroupTasks();

      if (mounted) {
        for (final status in result.grouped.keys) {
          _expandedStatus.putIfAbsent(status, () => true);
        }

        setState(() {
          _tasksByStatus = result.grouped;
          _taskLogs = result.logMap;
          _isLoading = false;
        });

        final allTasks = result.grouped.values.expand((list) => list).toList();
        taskNotificationService.markAllTasksAsNotified(allTasks);
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[TaskList] fetch error: $e');
        setState(() {
          _isLoading = false;
          _error = textLocalize('error_unknown');
        });
        TDToast.showText(
          textLocalize('error_fetch_tasks'),
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
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          bottom: false,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              BDPageHeader(
                title: textLocalize('task_list_title'),
                trailing: IconButton(
                  icon: AnimatedRotation(
                    turns: _isLoading ? 1 : 0,
                    duration: const Duration(milliseconds: 600),
                    child: Icon(
                      Icons.refresh,
                      color: isDark
                          ? BDDesign.colorPaperWhite
                          : BDDesign.colorInkBlack,
                      size: 20,
                    ),
                  ),
                  tooltip: textLocalize('refresh'),
                  onPressed: _isLoading ? null : _fetchTasks,
                ),
              ),
              Expanded(child: _buildBody(theme, isDark, textColor)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBody(TDThemeData theme, bool isDark, Color textColor) {
    if (_isLoading) {
      return const Center(
        child: TDLoading(size: TDLoadingSize.large, icon: TDLoadingIcon.circle),
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

  List<Widget> _buildCategorySections(
    TDThemeData theme,
    bool isDark,
    Color textColor,
  ) {
    // 按优先级排序状态
    final sortedStatuses = _tasksByStatus.keys.toList()
      ..sort((a, b) {
        final priorityA = statusCategories
            .firstWhere(
              (c) => c.status == a,
              orElse: () => StatusCategory(
                status: a,
                labelKey: a,
                icon: Icons.folder,
                color: BDDesign.colorAshGray,
                priority: 99,
              ),
            )
            .priority;
        final priorityB = statusCategories
            .firstWhere(
              (c) => c.status == b,
              orElse: () => StatusCategory(
                status: b,
                labelKey: b,
                icon: Icons.folder,
                color: BDDesign.colorAshGray,
                priority: 99,
              ),
            )
            .priority;
        return priorityA.compareTo(priorityB);
      });

    return sortedStatuses.map<Widget>((status) {
      final category = statusCategories.firstWhere(
        (c) => c.status == status,
        orElse: () => StatusCategory(
          status: status,
          labelKey: status,
          icon: Icons.folder,
          color: BDDesign.colorAshGray,
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

  String _resolveDisplayName(Map<String, dynamic> task, String fallback) {
    final displayName = task['display_name']?.toString().trim() ?? '';
    if (displayName.isNotEmpty) return displayName;

    final tags = task['tags'];
    if (tags is List) {
      for (final tag in tags) {
        final value = tag?.toString().trim() ?? '';
        if (value.isNotEmpty) return value;
      }
    }

    return fallback;
  }

  Future<void> _onTaskTap(Map<String, dynamic> task) async {
    final status = task['status']?.toString();
    final sceneId = task['scene_id']?.toString();

    // 只有completed状态的任务才能查看
    if (status != 'completed' || sceneId == null) {
      TDToast.showText(
        textLocalize('task_status_${status ?? 'unknown'}'),
        context: context,
      );
      return;
    }

    // 查询 model_assets 获取 ply_path，确保 Viewer 能触发下载
    String modelUrl = '';
    String? posesUrl;
    try {
      final asset = await Supabase.instance.client
          .from('model_assets')
          .select('ply_path')
          .eq('scene_id', sceneId)
          .order('created_at', ascending: false)
          .limit(1)
          .maybeSingle();
      final plyPath = asset?['ply_path']?.toString() ?? '';
      if (plyPath.isNotEmpty) {
        modelUrl = toPublicUrl(plyPath);
        posesUrl = toPosesUrl(plyPath);
      }
    } catch (_) {
      // 查询失败时使用默认值，Viewer 仍可打开
    }

    if (!mounted) return;
    final displayName = _resolveDisplayName(task, sceneId);
    await openViewer(
      context,
      initialModelUrl: modelUrl,
      posesUrl: posesUrl,
      sceneId: displayName,
    );
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
              color: BDDesign.colorDarkRed.withAlpha(180),
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
