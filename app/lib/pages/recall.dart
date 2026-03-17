import 'dart:async';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/app_config.dart';
import '../configs/supabase_config.dart';
import '../configs/motion_tokens.dart';
import '../services/local_rag_index.dart';
import '../widgets/bd_surfaces.dart';
import 'community.dart';
import 'webgl_viewer.dart';
import 'task_list.dart';
import 'recall/top_summary_card.dart';

enum _RecallSearchMode { local, cloud }

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
  List<Map<String, dynamic>> _models = [];
  List<Map<String, dynamic>> _allModels = [];
  List<Map<String, dynamic>> _processingTasks = [];
  Map<String, List<String>> _taskAllLogs = {}; // taskId -> all log msgs
  final Set<String> _expandedTaskLogs = {}; // 展开的任务ID集合
  bool _isLoading = true;
  bool _isLocalIndexing = false;
  bool _isProcessingExpanded = true;
  final TextEditingController _searchController = TextEditingController();
  final LocalRagIndexService _localRagIndex = LocalRagIndexService();
  RealtimeChannel? _realtimeChannel;
  Timer? _searchDebounce;
  LocalRagIndexStats? _indexStats;
  _RecallSearchMode _searchMode = _RecallSearchMode.local;

  @override
  void initState() {
    super.initState();
    _fetchModels();
    _fetchProcessingTasks();
    _setupRealtimeListener();
  }

  @override
  void dispose() {
    _searchDebounce?.cancel();
    _searchController.dispose();
    _realtimeChannel?.unsubscribe();
    super.dispose();
  }

  /// 设置 Realtime 监听 processing_tasks 表的变化
  void _setupRealtimeListener() {
    _realtimeChannel = Supabase.instance.client.channel(
      'public:processing_tasks:recall',
    );

    _realtimeChannel!.onPostgresChanges(
      event: PostgresChangeEvent.all,
      schema: 'public',
      table: 'processing_tasks',
      callback: (payload) => _handleRealtimeChange(payload),
    );

    _realtimeChannel!.subscribe();
  }

  /// 处理 Realtime 变化
  void _handleRealtimeChange(PostgresChangePayload payload) {
    final newData = payload.newRecord;
    final oldData = payload.oldRecord;
    final taskId = (newData['id'] ?? oldData['id'])?.toString();
    final String? status =
        newData['status']?.toString() ?? oldData['status']?.toString();

    if (taskId == null) return;

    if (status == 'processing') {
      // 更新或添加 processing 任务
      final logsJson = newData['logs'] as List<dynamic>?;
      final allLogs = _parseAllLogMsgs(logsJson);

      setState(() {
        // 移除旧版本（如果存在）
        _processingTasks.removeWhere((t) => t['id'].toString() == taskId);
        // 添加更新后的任务
        _processingTasks.add(Map<String, dynamic>.from(newData));
        if (allLogs.isNotEmpty) {
          _taskAllLogs[taskId] = allLogs;
        }
      });
    } else if (status != 'processing' && oldData['status'] == 'processing') {
      // 任务从 processing 变为其他状态，移除
      setState(() {
        _processingTasks.removeWhere((t) => t['id'].toString() == taskId);
        _taskAllLogs.remove(taskId);
        _expandedTaskLogs.remove(taskId);
      });
    }
  }

  /// 解析所有 logs，返回 msg 列表
  List<String> _parseAllLogMsgs(List<dynamic>? logs) {
    if (logs == null || logs.isEmpty) return [];

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
  }

  /// 获取 processing 状态的任务
  Future<void> _fetchProcessingTasks() async {
    try {
      final response = await Supabase.instance.client
          .from('processing_tasks')
          .select('*')
          .eq('status', 'processing')
          .order('created_at', ascending: false);

      if (mounted) {
        final Map<String, List<String>> logMap = {};
        for (final task in response) {
          final taskId = task['id'].toString();
          final logs = task['logs'];

          if (logs is List) {
            final allLogs = _parseAllLogMsgs(List<dynamic>.from(logs));
            if (allLogs.isNotEmpty) {
              logMap[taskId] = allLogs;
            }
          }
        }

        setState(() {
          _processingTasks = List<Map<String, dynamic>>.from(response);
          _taskAllLogs = logMap;
        });
      }
    } catch (e) {
      // 静默失败
    }
  }

  /// 将 Storage 内的相对路径转为可访问的公开 URL。
  /// ply_path 示例: "my_scene/point_cloud.splat"
  String _toPublicUrl(String storagePath) {
    try {
      return Supabase.instance.client.storage
          .from('braindance-assets')
          .getPublicUrl(storagePath);
    } catch (_) {
      return storagePath; // 兜底：原样返回，让 viewer 显示错误提示
    }
  }

  /// 根据模型路径推导同场景的 webgl_poses.json 公开 URL。
  /// ply_path 格式：{user_id}/{scene_id}/output/point_cloud.(ply|splat|ksplat)
  /// poses 路径：{user_id}/{scene_id}/output/webgl_poses.json
  String? _toPosesUrl(String? plyPath) {
    if (plyPath == null || plyPath.isEmpty) return null;
    try {
      // 将 point_cloud.xxx 替换为 webgl_poses.json
      final posesPath = plyPath.replaceAll(
        RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
        'webgl_poses.json',
      );
      if (posesPath == plyPath) return null; // 替换失败，路径格式不符
      return Supabase.instance.client.storage
          .from('braindance-assets')
          .getPublicUrl(posesPath);
    } catch (_) {
      return null;
    }
  }

  Future<void> _fetchModels() async {
    try {
      final response = await Supabase.instance.client
          .from('model_assets')
          .select(
            'id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info, created_at',
          )
          .order('created_at', ascending: false);

      final models = List<Map<String, dynamic>>.from(response);
      if (models.isEmpty) {
        models.add(_buildDemoModel());
      }

      if (mounted) {
        setState(() {
          _allModels = models;
          _models = models;
          _isLoading = false;
        });
      }
      await _syncLocalIndex(models);
    } catch (e) {
      final demoModels = [_buildDemoModel()];
      if (mounted) {
        setState(() {
          _allModels = demoModels;
          _models = demoModels;
          _isLoading = false;
        });
        TDToast.showText(
          '${textLocalize('recall_error_offline')} [${SupabaseConfig.modeLabel}] $e',
          context: context,
        );
      }
      await _syncLocalIndex(demoModels);
    }
  }

  Map<String, dynamic> _buildDemoModel() {
    return {
      'id': 'local_demo',
      'scene_id': textLocalize('recall_demo_title'),
      'description': textLocalize('recall_demo_desc'),
      'tags': const ['offline', 'demo'],
      'objects': const ['3dgs', 'memory'],
      'ply_path': '',
      'meta_info': {'search_summary': textLocalize('recall_demo_desc')},
    };
  }

  Future<void> _syncLocalIndex(List<Map<String, dynamic>> models) async {
    if (mounted) {
      setState(() {
        _isLocalIndexing = true;
      });
    }

    try {
      final stats = await _localRagIndex.syncModels(models);
      if (!mounted) return;
      setState(() {
        _indexStats = stats;
        _isLocalIndexing = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _isLocalIndexing = false;
      });
    }
  }

  // 更黑的夜间色值
  final darkBg = const Color(0xFF101014);
  final darkCard = const Color(0xFF18181C);
  final darkInput = const Color(0xFF23232A);
  final darkBorder = const Color(0xFF23232A);

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final textColor = isDark ? const Color(0xFFFFFFFF) : BDDesign.colorInkBlack;
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.only(bottom: 96.0),
            child: Column(
              children: [
                BDPageHeader(
                  title: textLocalize("home_page"),
                  subtitle: textLocalize('recall_subtitle'),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      BDStatusPill(
                        label: SupabaseConfig.isAdminMode ? 'ADMIN' : 'RLS',
                        icon: SupabaseConfig.isAdminMode
                            ? Icons.admin_panel_settings_rounded
                            : Icons.verified_user_rounded,
                        color: SupabaseConfig.isAdminMode
                            ? BDDesign.colorDarkRed
                            : BDDesign.colorMutedBlue,
                      ),
                      const SizedBox(width: 8),
                      IconButton(
                        icon: AnimatedRotation(
                          turns: _isLoading ? 1 : 0,
                          duration: const Duration(milliseconds: 600),
                          child: Icon(
                            Icons.sync_rounded,
                            color: isDark
                                ? BDDesign.colorPaperWhite
                                : BDDesign.colorInkBlack,
                          ),
                        ),
                        tooltip: textLocalize("recall_refresh"),
                        onPressed: () {
                          setState(() {
                            _isLoading = true;
                          });
                          _fetchModels();
                        },
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: BDPanelCard(
                    padding: const EdgeInsets.all(18),
                    child: Row(
                      children: [
                        Expanded(
                          child: _RecallMetric(
                            label: textLocalize('recall_label_space'),
                            value: _allModels.length.toString(),
                          ),
                        ),
                        Expanded(
                          child: _RecallMetric(
                            label: textLocalize('recall_label_processing'),
                            value: _processingTasks.length.toString(),
                          ),
                        ),
                        Expanded(
                          child: _RecallMetric(
                            label: 'RAG',
                            value: _isLocalIndexing
                                ? '...'
                                : (_indexStats?.totalItems ?? _allModels.length)
                                      .toString(),
                            accent: textColor,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                TopSummaryCard(
                  recordCount: _allModels.isNotEmpty ? 1 : 0,
                  completedCount: _allModels.length,
                  isDark: isDark,
                  onTaskTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const TaskListPage(),
                      ),
                    );
                  },
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 6, 20, 8),
                  child: Column(
                    children: [
                      BDPanelCard(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        child: TextField(
                          controller: _searchController,
                          style: TextStyle(color: textColor, fontSize: 15),
                          decoration: InputDecoration(
                            hintText: textLocalize("recall_search_hint"),
                            hintStyle: TextStyle(
                              color: isDark
                                  ? Colors.white.withValues(alpha: 0.45)
                                  : BDDesign.colorMutedBlue.withValues(
                                      alpha: 0.78,
                                    ),
                              fontSize: 15,
                            ),
                            prefixIcon: Icon(
                              Icons.search_rounded,
                              color: isDark
                                  ? Colors.white.withValues(alpha: 0.5)
                                  : BDDesign.colorMutedBlue,
                            ),
                            suffixIcon: _searchController.text.trim().isEmpty
                                ? null
                                : IconButton(
                                    onPressed: () {
                                      _searchController.clear();
                                      _searchDebounce?.cancel();
                                      _searchModels('');
                                      setState(() {});
                                    },
                                    icon: Icon(
                                      Icons.close_rounded,
                                      color: isDark
                                          ? Colors.white.withValues(alpha: 0.5)
                                          : BDDesign.colorMutedBlue,
                                    ),
                                  ),
                            filled: true,
                            fillColor: Colors.transparent,
                            contentPadding: const EdgeInsets.symmetric(
                              vertical: 14,
                              horizontal: 16,
                            ),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16.0),
                              borderSide: BorderSide.none,
                            ),
                            enabledBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16.0),
                              borderSide: BorderSide.none,
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16.0),
                              borderSide: const BorderSide(
                                color: BDDesign.colorMutedBlue,
                                width: 1.5,
                              ),
                            ),
                          ),
                          onSubmitted: _searchModels,
                          onChanged: _onSearchChanged,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Padding(
                          padding: const EdgeInsets.only(left: 4, bottom: 8),
                          child: Text(
                            textLocalize('recall_search_mode'),
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: isDark
                                  ? Colors.white.withValues(alpha: 0.58)
                                  : BDDesign.colorMutedBlue,
                            ),
                          ),
                        ),
                      ),
                      Row(
                        children: [
                          Expanded(
                            child: _buildSearchModeChip(
                              isDark: isDark,
                              label: textLocalize('recall_local_rag'),
                              icon: Icons.privacy_tip_rounded,
                              mode: _RecallSearchMode.local,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: _buildSearchModeChip(
                              isDark: isDark,
                              label: textLocalize('recall_cloud_rag'),
                              icon: Icons.cloud_rounded,
                              mode: _RecallSearchMode.cloud,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      BDPanelCard(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 14,
                          vertical: 12,
                        ),
                        child: Row(
                          children: [
                            Icon(
                              _isLocalIndexing
                                  ? Icons.memory_rounded
                                  : Icons.privacy_tip_rounded,
                              size: 18,
                              color: isDark
                                  ? BDDesign.colorPaperWhite
                                  : BDDesign.colorInkBlack,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                _searchMode == _RecallSearchMode.local
                                    ? (_isLocalIndexing
                                          ? textLocalize(
                                              'recall_local_indexing',
                                            )
                                          : '${textLocalize('recall_local_ready')} · ${textLocalize('recall_local_scope')}')
                                    : textLocalize('recall_cloud_scope'),
                                style: TextStyle(
                                  fontSize: 12.5,
                                  color: isDark
                                      ? Colors.white.withValues(alpha: 0.72)
                                      : BDDesign.colorMutedBlue,
                                  height: 1.35,
                                ),
                              ),
                            ),
                            if (_searchMode == _RecallSearchMode.local &&
                                _indexStats != null &&
                                !_isLocalIndexing)
                              BDStatusPill(
                                label:
                                    '${_indexStats!.rebuiltItems}/${_indexStats!.totalItems}',
                                icon: Icons.storage_rounded,
                                color: BDDesign.colorMutedBlue,
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                if (_processingTasks.isNotEmpty)
                  _buildProcessingSection(theme, isDark, textColor),
                if (_isLoading)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 96.0),
                    child: Center(
                      child: TDLoading(
                        size: TDLoadingSize.large,
                        icon: TDLoadingIcon.circle,
                      ),
                    ),
                  )
                else if (_models.isEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 16.0),
                    child: _searchController.text.trim().isEmpty
                        ? _buildEmptyState(theme, isDark)
                        : _buildSearchEmptyState(theme, isDark),
                  )
                else
                  _buildModelGrid(theme, isDark),
              ],
            ),
          ),
        ),
      ),
    );
  }

  /// 构建 Processing 任务区域（可展开收起）
  Widget _buildProcessingSection(
    TDThemeData theme,
    bool isDark,
    Color textColor,
  ) {
    final hintTextColor = isDark ? const Color(0xFF888888) : theme.fontGyColor3;

    return BDPanelCard(
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          InkWell(
            onTap: () {
              setState(() {
                _isProcessingExpanded = !_isProcessingExpanded;
              });
            },
            borderRadius: BDDesign.radiusLarge,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
              child: Row(
                children: [
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: BDDesign.colorMutedBlue.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Center(
                      child: SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            BDDesign.colorMutedBlue,
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          textLocalize('status_processing'),
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: textColor,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '这个场景还在重建，共 ${_processingTasks.length} 项任务正在推进。',
                          style: TextStyle(
                            fontSize: 12.5,
                            color: hintTextColor,
                            height: 1.35,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  BDStatusPill(
                    label: '${_processingTasks.length}',
                    icon: Icons.motion_photos_on_rounded,
                    color: BDDesign.colorMutedBlue,
                  ),
                  const SizedBox(width: 8),
                  AnimatedRotation(
                    turns: _isProcessingExpanded ? 0.5 : 0,
                    duration: BDMotion.durationFast,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: isDark
                          ? Colors.white.withValues(alpha: 0.56)
                          : BDDesign.colorMutedBlue,
                    ),
                  ),
                ],
              ),
            ),
          ),
          AnimatedCrossFade(
            firstChild: const SizedBox.shrink(),
            secondChild: Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Column(
                children: _processingTasks.asMap().entries.map((entry) {
                  final index = entry.key;
                  final task = entry.value;
                  return _buildProcessingTaskItem(
                    task,
                    theme,
                    isDark,
                    textColor,
                    hintTextColor,
                    isFirst: index == 0,
                    isLast: index == _processingTasks.length - 1,
                  );
                }).toList(),
              ),
            ),
            crossFadeState: _isProcessingExpanded
                ? CrossFadeState.showSecond
                : CrossFadeState.showFirst,
            duration: BDMotion.durationNormal,
          ),
        ],
      ),
    );
  }

  /// 构建 processing 任务项
  Widget _buildProcessingTaskItem(
    Map<String, dynamic> task,
    TDThemeData theme,
    bool isDark,
    Color textColor,
    Color hintTextColor, {
    required bool isFirst,
    required bool isLast,
  }) {
    final taskId = task['id'].toString();
    final sceneId = task['scene_id']?.toString() ?? 'Unknown';
    final displayName = task['display_name']?.toString();
    final allLogs = _taskAllLogs[taskId] ?? [];
    final latestLog = allLogs.isNotEmpty ? allLogs.last : null;
    final isExpanded = _expandedTaskLogs.contains(taskId);

    return Container(
      margin: EdgeInsets.fromLTRB(12, isFirst ? 1 : 4, 12, isLast ? 12 : 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark
            ? darkInput.withValues(alpha: 0.86)
            : BDDesign.colorMutedBlueLight.withValues(alpha: 0.38),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: isDark
              ? Colors.white.withValues(alpha: 0.05)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: BDDesign.colorMutedBlue.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Center(
                  child: SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        BDDesign.colorMutedBlue,
                      ),
                    ),
                  ),
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
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: textColor,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      latestLog ?? textLocalize('status_processing'),
                      style: TextStyle(
                        fontSize: 12.5,
                        color: hintTextColor,
                        height: 1.35,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
              if (allLogs.length > 1)
                IconButton(
                  icon: AnimatedRotation(
                    turns: isExpanded ? 0.5 : 0,
                    duration: BDMotion.durationFast,
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: isDark
                          ? Colors.white.withValues(alpha: 0.56)
                          : BDDesign.colorMutedBlue,
                      size: 20,
                    ),
                  ),
                  onPressed: () {
                    setState(() {
                      if (isExpanded) {
                        _expandedTaskLogs.remove(taskId);
                      } else {
                        _expandedTaskLogs.add(taskId);
                      }
                    });
                  },
                ),
            ],
          ),
          if (allLogs.length > 1)
            AnimatedCrossFade(
              firstChild: const SizedBox.shrink(),
              secondChild: Container(
                margin: const EdgeInsets.only(top: 10),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: isDark
                      ? const Color(0xFF1A1A20).withValues(alpha: 0.94)
                      : Colors.white.withValues(alpha: 0.7),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: allLogs.reversed
                      .map(
                        (log) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 2),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                margin: const EdgeInsets.only(top: 6, right: 8),
                                width: 5,
                                height: 5,
                                decoration: BoxDecoration(
                                  color: BDDesign.colorMutedBlue.withValues(
                                    alpha: isDark ? 0.72 : 0.55,
                                  ),
                                  shape: BoxShape.circle,
                                ),
                              ),
                              Expanded(
                                child: Text(
                                  log,
                                  style: TextStyle(
                                    fontSize: 11.5,
                                    color: isDark
                                        ? Colors.white.withValues(alpha: 0.7)
                                        : theme.fontGyColor2,
                                    height: 1.35,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      )
                      .toList(),
                ),
              ),
              crossFadeState: isExpanded
                  ? CrossFadeState.showSecond
                  : CrossFadeState.showFirst,
              duration: BDMotion.durationFast,
            ),
        ],
      ),
    );
  }

  Future<void> _searchModels(String query) async {
    if (query.trim().isEmpty) {
      if (!mounted) return;
      setState(() {
        _models = List<Map<String, dynamic>>.from(_allModels);
        _isLoading = false;
      });
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final results = _searchMode == _RecallSearchMode.local
          ? await _localRagIndex.search(query)
          : await _searchModelsFromCloud(query);
      if (!mounted) return;
      setState(() {
        _models = results;
        _isLoading = false;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        TDToast.showText(
          '${textLocalize("recall_error_search")}$e',
          context: context,
        );
      }
    }
  }

  Future<List<Map<String, dynamic>>> _searchModelsFromCloud(
    String query,
  ) async {
    final response = await Supabase.instance.client.functions.invoke(
      'search-models',
      body: {'query': query},
    );

    final data = response.data;
    if (data is Map && data['success'] == true) {
      return List<Map<String, dynamic>>.from(data['results'] ?? []);
    }

    final errMsg = (data is Map) ? (data['error'] ?? textLocalize('recall_unknown_error')) : textLocalize('recall_server_error');
    throw Exception(errMsg);
  }

  void _onSearchChanged(String value) {
    setState(() {});
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 180), () {
      _searchModels(value);
    });
  }

  Widget _buildSearchModeChip({
    required bool isDark,
    required String label,
    required IconData icon,
    required _RecallSearchMode mode,
  }) {
    final selected = _searchMode == mode;
    return GestureDetector(
      onTap: () {
        if (_searchMode == mode) return;
        setState(() {
          _searchMode = mode;
        });
        final keyword = _searchController.text.trim();
        if (keyword.isNotEmpty) {
          _searchModels(keyword);
        }
      },
      child: AnimatedContainer(
        duration: BDMotion.durationFast,
        curve: Curves.easeOutCubic,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: selected
              ? BDDesign.colorMutedBlue.withValues(alpha: isDark ? 0.22 : 0.12)
              : (isDark ? darkCard : BDDesign.colorPaperWhite),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: selected
                ? BDDesign.colorMutedBlue
                : (isDark
                      ? Colors.white.withValues(alpha: 0.08)
                      : BDDesign.colorMutedBlue.withValues(alpha: 0.18)),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 16,
              color: selected
                  ? BDDesign.colorMutedBlue
                  : (isDark
                        ? Colors.white.withValues(alpha: 0.72)
                        : BDDesign.colorMutedBlue),
            ),
            const SizedBox(width: 8),
            Flexible(
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w700,
                  color: selected
                      ? (isDark
                            ? BDDesign.colorPaperWhite
                            : BDDesign.colorInkBlack)
                      : (isDark
                            ? Colors.white.withValues(alpha: 0.78)
                            : BDDesign.colorInkBlack),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState(TDThemeData theme, bool isDark) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final iconColor = isDark
        ? const Color(0xFFEEEEEE)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32.0),
          border: Border.all(
            color: isDark ? darkBorder : theme.whiteColor1,
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
            TDImage(
              assetUrl: 'assets/sprites/empty_state.png',
              width: 120,
              height: 120,
              errorWidget: Icon(
                TDIcons.time_filled,
                size: 80,
                color: iconColor,
              ),
            ),
            const SizedBox(height: 24),
            TDText(
              textLocalize("home_page"),
              font: theme.fontTitleLarge,
              textColor: textColor,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              textLocalize("recall_empty_title"),
              font: theme.fontBodyMedium,
              textColor: hintTextColor,
            ),
            const SizedBox(height: 40),
            TDButton(
              text: textLocalize("recall_open_demo"),
              iconWidget: Icon(
                TDIcons.view_module,
                color: Colors.white,
                size: 20,
              ),
              type: TDButtonType.fill,
              theme: TDButtonTheme.primary,
              shape: TDButtonShape.round,
              size: TDButtonSize.large,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => WebGLViewerPage(
                      sceneId: textLocalize("recall_demo_title"),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchEmptyState(TDThemeData theme, bool isDark) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32.0),
          border: Border.all(
            color: isDark ? darkBorder : theme.whiteColor1,
            width: 1,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.travel_explore_rounded,
              size: 56,
              color: isDark
                  ? Colors.white.withValues(alpha: 0.8)
                  : BDDesign.colorMutedBlue,
            ),
            const SizedBox(height: 18),
            TDText(
              textLocalize('recall_local_rag'),
              font: theme.fontTitleLarge,
              textColor: textColor,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              _searchMode == _RecallSearchMode.local
                  ? textLocalize('recall_local_empty')
                  : textLocalize('recall_cloud_empty'),
              font: theme.fontBodyMedium,
              textColor: hintTextColor,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelGrid(TDThemeData theme, bool isDark) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;

    // If it's search results and has matched_frames, use ListView
    bool isSearchWithFrames =
        _models.isNotEmpty && _models.first.containsKey('matched_frames');

    if (isSearchWithFrames) {
      return ListView.builder(
        padding: const EdgeInsets.fromLTRB(16.0, 6.0, 16.0, 16.0),
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        itemCount: _models.length,
        itemBuilder: (context, index) {
          final model = _models[index];
          final sceneId = model['scene_id'] ?? 'Unknown Scene';
          final desc = model['description'] ?? textLocalize('recall_no_desc');
          final similarity = model['similarity'] as double?;
          final userId = model['user_id'] ?? '';
          final matchedFrames = model['matched_frames'] as List<dynamic>? ?? [];

          return TweenAnimationBuilder<double>(
            tween: Tween(begin: 0.0, end: 1.0),
            duration:
                BDMotion.durationNormal +
                Duration(milliseconds: (index * 50).clamp(0, 400)),
            curve: BDMotion.curveEnter,
            builder: (context, value, child) {
              return Transform.translate(
                offset: Offset(0, 20 * (1 - value)),
                child: Opacity(opacity: value, child: child),
              );
            },
            child: Container(
              margin: const EdgeInsets.only(bottom: 16.0),
              decoration: BoxDecoration(
                color: isDark ? darkCard : BDDesign.colorPaperWhite,
                borderRadius: BDDesign.radiusLarge,
                boxShadow: isDark ? [] : [BDDesign.shadowLight],
                border: Border.all(
                  color: isDark ? const Color(0xFF2A2A30) : Colors.transparent,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Top header: Model Info
                  GestureDetector(
                    onTap: () {
                      _navigateToViewer(model, null);
                    },
                    onLongPress: () => _showModelActions(model),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                TDText(
                                  sceneId,
                                  font: theme.fontTitleMedium,
                                  fontWeight: FontWeight.w600,
                                  maxLines: 1,
                                  textColor: textColor,
                                ),
                                const SizedBox(height: 4),
                                TDText(
                                  desc,
                                  font: theme.fontBodySmall,
                                  textColor: hintTextColor,
                                  maxLines: 2,
                                ),
                              ],
                            ),
                          ),
                          if (similarity != null)
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: theme.brandColor4.withAlpha(220),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: TDText(
                                '${(similarity * 100).toStringAsFixed(1)}%',
                                font: theme.fontBodySmall,
                                textColor: isDark
                                    ? const Color(0xFFFFFFFF)
                                    : Colors.white,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                  // Horizontal list of frames
                  if (matchedFrames.isNotEmpty)
                    SizedBox(
                      height: 120,
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16.0,
                        ).copyWith(bottom: 16.0),
                        itemCount: matchedFrames.length,
                        itemBuilder: (context, frameIndex) {
                          final frame = matchedFrames[frameIndex];
                          final imageName = frame['image_name'];
                          final transformMatrix = frame['transform_matrix'];
                          final frameSim = frame['similarity'] as double?;

                          final imageUrl = Supabase.instance.client.storage
                              .from('braindance-assets')
                              .getPublicUrl(
                                '$userId/$sceneId/output/images/$imageName',
                              );

                          return GestureDetector(
                            onTap: () {
                              _navigateToViewer(model, transformMatrix);
                            },
                            child: Container(
                              width: 140,
                              margin: const EdgeInsets.only(right: 12.0),
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8.0),
                                color: isDark ? darkInput : theme.grayColor3,
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(8.0),
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    Image.network(
                                      imageUrl,
                                      fit: BoxFit.cover,
                                      loadingBuilder:
                                          (context, child, loadingProgress) {
                                            if (loadingProgress == null) {
                                              return child;
                                            }
                                            return Center(
                                              child: CircularProgressIndicator(
                                                value:
                                                    loadingProgress
                                                            .expectedTotalBytes !=
                                                        null
                                                    ? loadingProgress
                                                              .cumulativeBytesLoaded /
                                                          loadingProgress
                                                              .expectedTotalBytes!
                                                    : null,
                                              ),
                                            );
                                          },
                                      errorBuilder:
                                          (context, error, stackTrace) {
                                            return const Center(
                                              child: Icon(
                                                Icons.broken_image,
                                                color: Colors.grey,
                                              ),
                                            );
                                          },
                                    ),
                                    if (frameSim != null)
                                      Positioned(
                                        bottom: 4,
                                        left: 4,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 4,
                                            vertical: 2,
                                          ),
                                          decoration: BoxDecoration(
                                            color: Colors.black.withAlpha(100),
                                            borderRadius: BorderRadius.circular(
                                              4,
                                            ),
                                          ),
                                          child: Text(
                                            '${(frameSim * 100).toStringAsFixed(1)}%',
                                            style: const TextStyle(
                                              color: Colors.white,
                                              fontSize: 10,
                                            ),
                                          ),
                                        ),
                                      ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                ],
              ),
            ),
          );
        },
      );
    }

    return GridView.builder(
      padding: const EdgeInsets.only(
        left: 16.0,
        right: 16.0,
        top: 14.0,
        bottom: 16.0,
      ),
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 16.0,
        mainAxisSpacing: 16.0,
        childAspectRatio: 0.85,
      ),
      itemCount: _models.length,
      itemBuilder: (context, index) {
        final model = _models[index];
        final sceneId = model['scene_id'] ?? 'Unknown Scene';
        final desc = model['description'] ?? textLocalize("recall_no_desc");
        final similarity = model['similarity'] as double?;

        return TweenAnimationBuilder<double>(
          tween: Tween(begin: 0.0, end: 1.0),
          duration:
              BDMotion.durationNormal +
              Duration(milliseconds: (index * 50).clamp(0, 400)),
          curve: BDMotion.curveEnter,
          builder: (context, value, child) {
            return Transform.translate(
              offset: Offset(0, 20 * (1 - value)),
              child: Opacity(opacity: value, child: child),
            );
          },
          child: GestureDetector(
            onTap: () {
              _navigateToViewer(model, null);
            },
            onLongPress: () => _showModelActions(model),
            child: Container(
              decoration: BoxDecoration(
                color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
                borderRadius: BorderRadius.circular(28.0),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withAlpha(20),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            color: isDark ? darkInput : theme.grayColor3,
                            borderRadius: const BorderRadius.vertical(
                              top: Radius.circular(28.0),
                            ),
                          ),
                          clipBehavior: Clip.hardEdge,
                          child:
                              model['preview_img_path'] != null &&
                                  model['preview_img_path']
                                      .toString()
                                      .isNotEmpty
                              ? Image.network(
                                  model['preview_img_path'],
                                  fit: BoxFit.cover,
                                  errorBuilder: (context, error, stackTrace) =>
                                      _buildModelMockCover(
                                        isDark: isDark,
                                        theme: theme,
                                      ),
                                  loadingBuilder:
                                      (context, child, loadingProgress) {
                                        if (loadingProgress == null) {
                                          return child;
                                        }
                                        return const Center(
                                          child: CircularProgressIndicator(),
                                        );
                                      },
                                )
                              : _buildModelMockCover(
                                  isDark: isDark,
                                  theme: theme,
                                ),
                        ),
                        if (similarity != null)
                          Positioned(
                            top: 8,
                            right: 8,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 6,
                                vertical: 2,
                              ),
                              decoration: BoxDecoration(
                                color: theme.brandColor4.withAlpha(220),
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: TDText(
                                '${(similarity * 100).toStringAsFixed(1)}%',
                                font: theme.fontBodyExtraSmall,
                                textColor: isDark
                                    ? const Color(0xFFFFFFFF)
                                    : Colors.white,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        TDText(
                          sceneId,
                          font: theme.fontTitleMedium,
                          fontWeight: FontWeight.w600,
                          maxLines: 1,
                          textColor: textColor,
                        ),
                        const SizedBox(height: 4),
                        TDText(
                          desc,
                          font: theme.fontBodySmall,
                          textColor: hintTextColor,
                          maxLines: 2,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildModelMockCover({
    required bool isDark,
    required TDThemeData theme,
  }) {
    final accent = isDark ? const Color(0xFF7AA2FF) : BDDesign.colorMutedBlue;

    return Container(
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1A1E27) : const Color(0xFFF6F8FC),
        border: Border.all(
          color: isDark ? Colors.white.withAlpha(18) : accent.withAlpha(35),
        ),
      ),
      child: Center(
        child: Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: isDark
                ? Colors.white.withAlpha(6)
                : Colors.white.withAlpha(190),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: isDark ? Colors.white.withAlpha(18) : accent.withAlpha(28),
            ),
          ),
          child: Icon(
            Icons.auto_awesome_mosaic_rounded,
            size: 28,
            color: accent.withAlpha(210),
          ),
        ),
      ),
    );
  }

  void _navigateToViewer(Map<String, dynamic> model, dynamic transformMatrix) {
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty
        ? _toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? _toPosesUrl(plyPath) : null;
    final sceneId = model['scene_id'] ?? 'Unknown Scene';

    // 如果传入的 matrix 为空，尝试从模型元数据中获取智能初始视角
    if (transformMatrix == null && model['meta_info'] != null) {
      if (model['meta_info'] is Map &&
          model['meta_info']['initial_camera_pose'] != null) {
        transformMatrix = model['meta_info']['initial_camera_pose'];
      }
    }

    // Convert transformMatrix if not null to List<double>
    List<double>? initialPose;
    if (transformMatrix != null && transformMatrix is List) {
      initialPose = transformMatrix.map((e) => (e as num).toDouble()).toList();
    }

    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) =>
            WebGLViewerPage(
              initialModelUrl: modelUrl,
              posesUrl: posesUrl,
              sceneId: sceneId,
              initialPose: initialPose,
            ),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(
            opacity: animation,
            child: ScaleTransition(
              scale: Tween<double>(begin: 0.95, end: 1.0).animate(
                CurvedAnimation(parent: animation, curve: Curves.easeOutCubic),
              ),
              child: child,
            ),
          );
        },
      ),
    );
  }

  Future<void> _shareModelToCommunity(Map<String, dynamic> model) async {
    final draft = await showCommunityComposerSheet(
      context,
      models: [_modelToCommunityOption(model)],
      initialModelId: model['id']?.toString(),
    );

    if (draft == null) {
      return;
    }

    await CommunityRepository().createPost(draft);
    if (!mounted) {
      return;
    }
    TDToast.showText(context: context, textLocalize('recall_published'));
  }

  Future<void> _showModelActions(Map<String, dynamic> model) async {
    final selectedAction = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) {
        final isDark = Theme.of(context).brightness == Brightness.dark;
        final textColor = isDark
            ? BDDesign.colorPaperWhite
            : BDDesign.colorInkBlack;
        final hintColor = isDark
            ? Colors.white.withValues(alpha: 0.62)
            : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
        final sceneId = model['scene_id']?.toString() ?? textLocalize('recall_unnamed_model');
        final desc =
            model['description']?.toString() ?? textLocalize("recall_no_desc");

        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
          child: BDPanelCard(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
            child: SafeArea(
              top: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    sceneId,
                    style: TextStyle(
                      color: textColor,
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    desc,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: hintColor, height: 1.35),
                  ),
                  const SizedBox(height: 16),
                  ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: const Icon(Icons.public_rounded),
                    title: Text(textLocalize('recall_share_to_community')),
                    subtitle: Text(textLocalize('recall_share_subtitle')),
                    onTap: () => Navigator.pop(context, 'share'),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );

    if (!mounted) {
      return;
    }

    if (selectedAction == 'share') {
      await _shareModelToCommunity(model);
    }
  }

  CommunityModelOption _modelToCommunityOption(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString() ?? '';
    final preview = model['preview_img_path']?.toString();
    return CommunityModelOption(
      id: model['id']?.toString() ?? model['scene_id']?.toString() ?? 'model',
      sceneId: model['scene_id']?.toString() ?? textLocalize('recall_unnamed_model'),
      description: model['description']?.toString() ?? '',
      modelUrl: plyPath.isEmpty
          ? './models/scene_auto_sync_raw.ply'
          : _toPublicUrl(plyPath),
      posesUrl: _toPosesUrl(plyPath),
      coverUrl: preview,
    );
  }
}

class _RecallMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color? accent;

  const _RecallMetric({required this.label, required this.value, this.accent});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isDark
                ? Colors.white.withValues(alpha: 0.58)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color:
                accent ??
                (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack),
          ),
        ),
      ],
    );
  }
}
