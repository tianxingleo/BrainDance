import 'dart:async';
import 'dart:convert';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../configs/app_config.dart';
import 'webgl_viewer.dart';
import 'task_list.dart';

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
  List<Map<String, dynamic>> _models = [];
  bool _isLoading = true;
  final TextEditingController _searchController = TextEditingController();

  // 任务状态监听
  Timer? _taskStatusTimer;
  Set<String> _notifiedCompletedTasks = {}; // 已通知过的 completed 任务ID
  Set<String> _notifiedFailedTasks = {}; // 已通知过的 failed 任务ID
  OverlayEntry? _notificationOverlay;
  int _completedCount = 0;
  int _failedCount = 0;

  // 本地缓存 key
  static const String _kNotifiedCompletedTasks = 'notified_completed_tasks';
  static const String _kNotifiedFailedTasks = 'notified_failed_tasks';

  @override
  void initState() {
    super.initState();
    _loadNotifiedTasksFromCache(); // 先加载本地缓存
    _fetchModels();
    _startTaskStatusMonitoring();
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

  @override
  void dispose() {
    _searchController.dispose();
    _taskStatusTimer?.cancel();
    _hideNotification();
    super.dispose();
  }

  /// 启动任务状态监听（每5秒检查一次）
  void _startTaskStatusMonitoring() {
    _fetchTaskStatuses(); // 立即获取一次
    _taskStatusTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      _fetchTaskStatuses();
    });
  }

  /// 获取任务状态并检测变化
  Future<void> _fetchTaskStatuses() async {
    try {
      // 调试阶段：不检查登录状态
      final response = await Supabase.instance.client
          .from('processing_tasks')
          .select('id, status, scene_id, display_name')
          .order('created_at', ascending: false);

      if (!mounted) return;

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
        _completedCount = newlyCompletedIds.length;
        _failedCount = newlyFailedIds.length;

        // 将新任务ID加入已通知集合
        _notifiedCompletedTasks.addAll(newlyCompletedIds);
        _notifiedFailedTasks.addAll(newlyFailedIds);

        // 保存到本地缓存
        _saveNotifiedTasksToCache();

        // 显示通知
        _showTaskNotification();
      }
    } catch (e) {
      // 静默失败
    }
  }

  /// 显示任务状态变化通知（类似 Edge 浏览器下载提示）
  void _showTaskNotification() {
    _hideNotification(); // 先隐藏之前的

    _notificationOverlay = OverlayEntry(
      builder: (context) => _TaskNotificationWidget(
        completedCount: _completedCount,
        failedCount: _failedCount,
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

    Overlay.of(context).insert(_notificationOverlay!);

    // 5秒后自动隐藏
    Future.delayed(const Duration(seconds: 5), () {
      _hideNotification();
    });
  }

  /// 隐藏通知
  void _hideNotification() {
    _notificationOverlay?.remove();
    _notificationOverlay = null;
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
            'id, scene_id, description, ply_path, preview_img_path, meta_info, created_at',
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
      message = '${widget.completedCount} ${textLocalize('task_completed')}，${widget.failedCount} ${textLocalize('task_failed')}';
      icon = Icons.info;
      iconColor = Colors.orange;
    } else if (hasCompleted) {
      message = '${widget.completedCount} ${textLocalize('task_notification_completed')}';
      icon = Icons.check_circle;
      iconColor = Colors.green;
    } else if (hasFailed) {
      message = '${widget.failedCount} ${textLocalize('task_notification_failed')}';
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
                  margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
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
                      color: isDark ? const Color(0xFF3A3A40) : const Color(0xFFE0E0E0),
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
                              color: isDark ? Colors.white : const Color(0xFF333333),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Icon(
                          Icons.keyboard_arrow_right,
                          color: isDark ? const Color(0xFF888888) : const Color(0xFF999999),
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
