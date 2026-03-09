import 'dart:async';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/app_config.dart';
import 'webgl_viewer.dart';
import 'task_list.dart';
import 'time_peeling.dart';

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
  String _currentFolder = 'root'; // 'root', 'in_progress', 'completed'
  List<Map<String, dynamic>> _models = [];
  List<Map<String, dynamic>> _processingTasks = [];
  Map<String, List<String>> _taskAllLogs = {}; // taskId -> all log msgs
  Set<String> _expandedTaskLogs = {}; // 展开的任务ID集合
  bool _isLoading = true;
  bool _isProcessingExpanded = true;
  final TextEditingController _searchController = TextEditingController();
  RealtimeChannel? _realtimeChannel;

  @override
  void initState() {
    super.initState();
    _fetchModels();
    _fetchProcessingTasks();
    _setupRealtimeListener();
  }

  @override
  void dispose() {
    _searchController.dispose();
    _realtimeChannel?.unsubscribe();
    super.dispose();
  }

  /// 设置 Realtime 监听 processing_tasks 表的变化
  void _setupRealtimeListener() {
    _realtimeChannel = Supabase.instance.client.channel('public:processing_tasks:recall');
    
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
    final String? status = newData['status']?.toString() ?? oldData['status']?.toString();
    
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
            'id, scene_id, description, ply_path, preview_img_path, meta_info, created_at, space_id, capture_id, captured_at',
          )
          .order('created_at', ascending: false);

      if (mounted) {
        setState(() {
          _models = List<Map<String, dynamic>>.from(response);
          if (_models.isEmpty) {
            _models.add({
              'id': 'local_demo',
              'scene_id': textLocalize('recall_demo_title'),
              'description': textLocalize('recall_demo_desc'),
              'ply_path': '',
            });
          }
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _models = [
            {
              'id': 'local_demo',
              'scene_id': textLocalize('recall_demo_title'),
              'description': textLocalize('recall_demo_desc'),
              'ply_path': '',
            },
          ];
          _isLoading = false;
        });
        TDToast.showText(
          textLocalize('recall_error_offline'),
          context: context,
        );
      }
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
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final iconColor = isDark
        ? const Color(0xFFEEEEEE)
        : const Color(0xFF333333);
    return Scaffold(
      backgroundColor: isDark ? darkBg : theme.grayColor1,
      appBar: AppBar(
        backgroundColor: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        elevation: 0,
        scrolledUnderElevation: 0,
        surfaceTintColor: Colors.transparent,
        centerTitle: true,
        title: TDText(
          textLocalize("home_page"),
          font: theme.fontHeadlineSmall,
          fontWeight: FontWeight.w600,
          textColor: textColor,
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.history_toggle_off, color: iconColor),
            tooltip: '时光剥离',
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const TimePeelingPage()),
              );
            },
          ),
          IconButton(
            icon: Icon(Icons.task_alt, color: iconColor),
            tooltip: textLocalize("task_list_title"),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const TaskListPage()),
              );
            },
          ),
          IconButton(
            icon: AnimatedRotation(
              turns: _isLoading ? 1 : 0,
              duration: const Duration(milliseconds: 600),
              child: Icon(Icons.refresh, color: iconColor),
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
        child: Stack(
          alignment: Alignment.center,
          children: [
            Column(
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 15.0,
                    vertical: 10.0,
                  ),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.transparent,
                      borderRadius: BorderRadius.circular(32.0),
                    ),
                    child: TextField(
                      controller: _searchController,
                      style: TextStyle(
                        color: isDark
                            ? const Color(0xFFFFFFFF)
                            : const Color(0xFF333333),
                        fontSize: 16,
                      ),
                      decoration: InputDecoration(
                        hintText: textLocalize("recall_search_hint"),
                        hintStyle: TextStyle(
                          color: isDark
                              ? const Color(0xFF888888)
                              : theme.fontGyColor3,
                          fontSize: 16,
                        ),
                        prefixIcon: Icon(
                          Icons.search_rounded,
                          color: isDark
                              ? const Color(0xFF888888)
                              : theme.fontGyColor3,
                        ),
                        filled: true,
                        fillColor: Colors.transparent,
                        contentPadding: const EdgeInsets.symmetric(
                          vertical: 14,
                          horizontal: 20,
                        ),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(32.0),
                          borderSide: BorderSide.none,
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(32.0),
                          borderSide: BorderSide.none,
                        ),
                        focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(32.0),
                          borderSide: isDark
                              ? const BorderSide(
                                  color: Color(0xFF4582FF),
                                  width: 1.5,
                                )
                              : BorderSide(
                                  color: theme.brandColor7,
                                  width: 1.5,
                                ),
                        ),
                      ),
                      onSubmitted: (value) => _searchModels(value),
                      onChanged: (value) {
                        if (value.isEmpty) _searchModels('');
                      },
                    ),
                  ),
                ),
                // Processing 任务区域
                if (_processingTasks.isNotEmpty) _buildProcessingSection(theme, isDark, textColor),
                Expanded(
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      if (!_isLoading && _models.isEmpty)
                        Padding(
                          padding: const EdgeInsets.only(
                            bottom: 108.0,
                          ), // 留出 main.dart 的 BottomNavigationBar 和间距高度 (90 height + 18 padding)
                          child: _buildEmptyState(theme, isDark),
                        ),
                      if (_isLoading)
                        const Center(
                          child: TDLoading(
                            size: TDLoadingSize.large,
                            icon: TDLoadingIcon.circle,
                          ),
                        )
                      else if (_models.isNotEmpty)
                        _buildModelGrid(theme, isDark),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// 构建 Processing 任务区域（可展开收起）
  Widget _buildProcessingSection(TDThemeData theme, bool isDark, Color textColor) {
    final hintTextColor = isDark ? const Color(0xFF888888) : theme.fontGyColor3;
    
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark ? const Color(0xFF333333) : theme.grayColor3,
          width: 1,
        ),
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
            onTap: () {
              setState(() {
                _isProcessingExpanded = !_isProcessingExpanded;
              });
            },
            borderRadius: BorderRadius.circular(16),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  // 旋转加载图标
                  SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      '${textLocalize('status_processing')} (${_processingTasks.length})',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: textColor,
                      ),
                    ),
                  ),
                  AnimatedRotation(
                    turns: _isProcessingExpanded ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // 任务列表（可展开/收起）
          AnimatedCrossFade(
            firstChild: const SizedBox.shrink(),
            secondChild: Column(
              children: _processingTasks.map((task) => _buildProcessingTaskItem(task, theme, isDark, textColor, hintTextColor)).toList(),
            ),
            crossFadeState: _isProcessingExpanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
            duration: const Duration(milliseconds: 200),
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
    Color hintTextColor,
  ) {
    final taskId = task['id'].toString();
    final sceneId = task['scene_id']?.toString() ?? 'Unknown';
    final displayName = task['display_name']?.toString();
    final allLogs = _taskAllLogs[taskId] ?? [];
    final latestLog = allLogs.isNotEmpty ? allLogs.last : null;
    final isExpanded = _expandedTaskLogs.contains(taskId);
    
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
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: Colors.blue.withAlpha(20),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Center(
                  child: SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
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
                        fontWeight: FontWeight.w500,
                        color: textColor,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      latestLog ?? textLocalize('status_processing'),
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
              // 展开/收起日志按钮
              if (allLogs.length > 1)
                IconButton(
                  icon: AnimatedRotation(
                    turns: isExpanded ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(
                      Icons.keyboard_arrow_down,
                      color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
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
          // 展开的日志列表
          if (allLogs.length > 1)
            AnimatedCrossFade(
              firstChild: const SizedBox.shrink(),
              secondChild: Container(
                margin: const EdgeInsets.only(top: 8),
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: isDark ? const Color(0xFF1A1A20) : theme.grayColor2,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: allLogs.reversed.map((log) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          margin: const EdgeInsets.only(top: 6, right: 8),
                          width: 4,
                          height: 4,
                          decoration: BoxDecoration(
                            color: isDark ? const Color(0xFF666666) : theme.fontGyColor4,
                            shape: BoxShape.circle,
                          ),
                        ),
                        Expanded(
                          child: Text(
                            log,
                            style: TextStyle(
                              fontSize: 11,
                              color: isDark ? const Color(0xFFAAAAAA) : theme.fontGyColor3,
                            ),
                          ),
                        ),
                      ],
                    ),
                  )).toList(),
                ),
              ),
              crossFadeState: isExpanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
              duration: const Duration(milliseconds: 200),
            ),
        ],
      ),
    );
  }

  Future<void> _searchModels(String query) async {
    if (query.trim().isEmpty) {
      setState(() {
        _isLoading = true;
      });
      _fetchModels();
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final response = await Supabase.instance.client.functions.invoke(
        'search-models',
        body: {'query': query},
      );

      final data = response.data;
      if (data is Map && data['success'] == true) {
        if (mounted) {
          setState(() {
            _models = List<Map<String, dynamic>>.from(data['results'] ?? []);
            _isLoading = false;
          });
        }
      } else {
        final errMsg = (data is Map) ? (data['error'] ?? '未知错误') : '服务器返回异常';
        throw Exception(errMsg);
      }
    } on FunctionException catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        // 从 details 里提取 Edge Function 返回的真实错误信息
        String errMsg;
        final details = e.details;
        if (details is Map && details['error'] != null) {
          errMsg = details['error'].toString();
        } else if (details is String && details.isNotEmpty) {
          errMsg = details;
        } else {
          errMsg = 'HTTP ${e.status}';
        }
        TDToast.showText(
          '${textLocalize("recall_error_search")}$errMsg',
          context: context,
        );
      }
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

  Widget _buildModelGrid(TDThemeData theme, bool isDark) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;

    // If it's search results and has matched_frames, use ListView
    bool isSearchWithFrames = _models.isNotEmpty && _models.first.containsKey('matched_frames');

    if (isSearchWithFrames) {
      return ListView.builder(
        padding: const EdgeInsets.all(16.0),
        itemCount: _models.length,
        itemBuilder: (context, index) {
          final model = _models[index];
          final sceneId = model['scene_id'] ?? 'Unknown Scene';
          final desc = model['description'] ?? '没有描述信息';
          final similarity = model['similarity'] as double?;
          final userId = model['user_id'] ?? '';
          final matchedFrames = model['matched_frames'] as List<dynamic>? ?? [];

          return Container(
            margin: const EdgeInsets.only(bottom: 16.0),
            decoration: BoxDecoration(
              color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
              borderRadius: BorderRadius.circular(theme.radiusLarge),
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
                // Top header: Model Info
                GestureDetector(
                  onTap: () {
                    _navigateToViewer(model, null);
                  },
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              TDText(sceneId, font: theme.fontTitleMedium, fontWeight: FontWeight.w600, maxLines: 1, textColor: textColor),
                              const SizedBox(height: 4),
                              TDText(desc, font: theme.fontBodySmall, textColor: hintTextColor, maxLines: 2),
                            ],
                          ),
                        ),
                        if (similarity != null)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: theme.brandColor4.withAlpha(220),
                              borderRadius: BorderRadius.circular(6)
                            ),
                            child: TDText('${(similarity * 100).toStringAsFixed(1)}%', font: theme.fontBodySmall, textColor: isDark ? const Color(0xFFFFFFFF) : Colors.white),
                          ),
                        if (model['space_id'] != null)
                          IconButton(
                            icon: const Icon(Icons.timeline),
                            tooltip: '时光剥离',
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) => TimePeelingPage(
                                    initialSpaceId: model['space_id']?.toString(),
                                    preferredCaptureId: model['capture_id']?.toString(),
                                  ),
                                ),
                              );
                            },
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
                      padding: const EdgeInsets.symmetric(horizontal: 16.0).copyWith(bottom: 16.0),
                      itemCount: matchedFrames.length,
                      itemBuilder: (context, frameIndex) {
                        final frame = matchedFrames[frameIndex];
                        final imageName = frame['image_name'];
                        final transformMatrix = frame['transform_matrix'];
                        final frameSim = frame['similarity'] as double?;

                        final imageUrl = "https://kntcynswgrmgbbgntkiv.supabase.co/storage/v1/object/public/braindance-assets/$userId/$sceneId/output/images/$imageName";

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
                              image: DecorationImage(
                                image: NetworkImage(imageUrl),
                                fit: BoxFit.cover,
                              ),
                            ),
                            child: frameSim != null ? Stack(
                              children: [
                                Positioned(
                                  bottom: 4,
                                  left: 4,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                                    decoration: BoxDecoration(
                                      color: Colors.black.withAlpha(100),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(
                                      '${(frameSim * 100).toStringAsFixed(1)}%',
                                      style: const TextStyle(color: Colors.white, fontSize: 10),
                                    ),
                                  ),
                                ),
                              ],
                            ) : null,
                          ),
                        );
                      },
                    ),
                  ),
              ],
            ),
          );
        },
      );
    }

    return GridView.builder(
      padding: const EdgeInsets.only(
        left: 16.0,
        right: 16.0,
        top: 4.0,
        bottom: 16.0,
      ),
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
          duration: Duration(milliseconds: 400 + (index * 100).clamp(0, 600)),
          curve: Curves.easeOutCubic,
          builder: (context, value, child) {
            return Transform.translate(
              offset: Offset(0, 50 * (1 - value)),
              child: Opacity(opacity: value, child: child),
            );
          },
          child: GestureDetector(
            onTap: () {
              _navigateToViewer(model, null);
            },
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
                          child: model['preview_img_path'] != null && model['preview_img_path'].toString().isNotEmpty
                              ? Image.network(
                                  model['preview_img_path'],
                                  fit: BoxFit.cover,
                                  errorBuilder: (context, error, stackTrace) =>
                                      Center(child: Icon(Icons.view_in_ar, size: 64, color: theme.brandColor7.withAlpha(200))),
                                  loadingBuilder: (context, child, loadingProgress) {
                                    if (loadingProgress == null) return child;
                                    return const Center(child: CircularProgressIndicator());
                                  },
                                )
                              : Center(
                                  child: Icon(
                                    Icons.view_in_ar,
                                    size: 64,
                                    color: theme.brandColor7.withAlpha(200),
                                  ),
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

  void _navigateToViewer(Map<String, dynamic> model, dynamic transformMatrix) {
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty ? _toPublicUrl(plyPath) : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? _toPosesUrl(plyPath) : null;
    final sceneId = model['scene_id'] ?? 'Unknown Scene';

    // 如果传入的 matrix 为空，尝试从模型元数据中获取智能初始视角
    if (transformMatrix == null && model['meta_info'] != null) {
      if (model['meta_info'] is Map && model['meta_info']['initial_camera_pose'] != null) {
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
        pageBuilder: (context, animation, secondaryAnimation) => WebGLViewerPage(
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
}
