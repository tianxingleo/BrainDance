// ignore_for_file: invalid_use_of_protected_member
part of '../recall.dart';

extension _RecallPageDataSync on _RecallPageState {
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
    if (!_isTabActive) {
      _shouldRefreshProcessingOnResume = true;
      return;
    }

    final newData = payload.newRecord;
    final oldData = payload.oldRecord;
    final taskId = (newData['id'] ?? oldData['id'])?.toString();
    final String? status =
        newData['status']?.toString() ?? oldData['status']?.toString();

    if (taskId == null) return;

    if (status == 'processing') {
      // 更新或添加 processing 任务
      final rawLogs = newData['logs'];
      final logsJson = rawLogs is List<dynamic> ? rawLogs : null;
      final allLogs = _parseAllLogMsgs(logsJson);
      if (mounted) {
        _refreshState(() {
          // 移除旧版本（如果存在）
          _processingTasks.removeWhere((t) => t['id'].toString() == taskId);
          // 添加更新后的任务
          _processingTasks.add(Map<String, dynamic>.from(newData));
          if (allLogs.isNotEmpty) {
            _taskAllLogs[taskId] = allLogs;
          }
        });
      }
      _processDualChainMilestones(taskId, newData, allLogs);
    } else if (status != 'processing' && oldData['status'] == 'processing') {
      // 任务从 processing 变为其他状态，移除
      // 终态前再扫一次最新 logs，避免错过最后一次写入的里程碑
      final rawLogs = newData['logs'];
      final finalLogs = _parseAllLogMsgs(logsJson);
      _processDualChainMilestones(taskId, newData, finalLogs);
      if (mounted) {
        _refreshState(() {
          _processingTasks.removeWhere((t) => t['id'].toString() == taskId);
          _taskAllLogs.remove(taskId);
          _expandedTaskLogs.remove(taskId);
        });
      }
      _taskMilestones.remove(taskId);
    }
  }

  /// 扫描任务的最新 logs，判定新出现的双链里程碑并驱动模型刷新。
  /// 仅对 task_type == 'video_dual_chain' 生效。
  void _processDualChainMilestones(
    String taskId,
    Map<String, dynamic>? task,
    List<String> allLogs,
  ) {
    if (task == null) return;
    if (task['task_type']?.toString() != 'video_dual_chain') return;
    if (allLogs.isEmpty) return;

    final known = _taskMilestones.putIfAbsent(taskId, () => <DualChainMilestone>{});
    final fresh = detectNewMilestones(
      allLogs: allLogs,
      knownMilestones: known,
    );
    if (fresh.isEmpty) return;
    known.addAll(fresh);

    final sceneId = task['scene_id']?.toString();

    if (fresh.contains(DualChainMilestone.fastReady)) {
      // 快链产物已 upsert 进 model_assets，刷新一次即可看到卡片。
      unawaited(_fetchModels(
        preserveExistingDataOnError: true,
        showErrorToast: false,
      ));
    }

    if (fresh.contains(DualChainMilestone.slowReady) && sceneId != null) {
      // 慢链覆盖了 Storage 上同一个 ply_path，
      // 需要丢弃 viewer 在 appDocDir 下基于 URL 派生的旧缓存。
      unawaited(_invalidateViewerCacheForScene(sceneId).then((_) async {
        if (!mounted) return;
        await _fetchModels(
          preserveExistingDataOnError: true,
          showErrorToast: false,
        );
      }));
    }
  }

  /// 根据 scene_id 反查 model_assets.ply_path，清掉 webgl_viewer 的本地缓存。
  Future<void> _invalidateViewerCacheForScene(String sceneId) async {
    try {
      final asset = await Supabase.instance.client
          .from('model_assets')
          .select('ply_path')
          .eq('scene_id', sceneId)
          .maybeSingle();
      final plyPath = asset?['ply_path']?.toString() ?? '';
      if (plyPath.isEmpty) return;
      await invalidateViewerCacheForUrl(_toPublicUrl(plyPath));
    } catch (_) {
      // 缓存清理是 best-effort，失败不影响主流程
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

        _refreshState(() {
          _processingTasks = List<Map<String, dynamic>>.from(response);
          _taskAllLogs = logMap;
        });
        _updateOverviewProvider();
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

  /// 将可能包含旧服务器地址的完整 Storage URL 转换为当前配置下的 URL。
  /// 如果已经是相对路径则直接生成公开 URL。
  String _normalizeStorageUrl(String raw) {
    const marker = '/storage/v1/object/public/braindance-assets/';
    final idx = raw.indexOf(marker);
    if (idx >= 0) {
      final relativePath = raw.substring(idx + marker.length);
      return _toPublicUrl(relativePath);
    }
    // 不含完整 URL 标记，当作相对路径处理
    if (!raw.startsWith('http')) {
      return _toPublicUrl(raw);
    }
    return raw;
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

  Future<bool> _fetchModels({
    bool preserveExistingDataOnError = false,
    bool showErrorToast = true,
  }) async {
    try {
      List<Map<String, dynamic>> models;
      try {
        final response = await Supabase.instance.client
            .from('model_assets')
            .select(
              'id, scene_id, user_id, display_name, description, objects, tags, ply_path, preview_img_path, meta_info, created_at, is_official',
            )
            .order('created_at', ascending: false);
        models = List<Map<String, dynamic>>.from(response);
      } catch (_) {
        // is_official column may not exist yet — fall back without it
        final response = await Supabase.instance.client
            .from('model_assets')
            .select(
              'id, scene_id, user_id, display_name, description, objects, tags, ply_path, preview_img_path, meta_info, created_at',
            )
            .order('created_at', ascending: false);
        models = List<Map<String, dynamic>>.from(response);
      }

      // Merge display_name from processing_tasks as fallback for rows missing it
      try {
        final sceneIds = models
            .map((m) => m['scene_id']?.toString())
            .where((s) => s != null)
            .toList();
        if (sceneIds.isNotEmpty) {
          final tasksResp = await Supabase.instance.client
              .from('processing_tasks')
              .select('scene_id, display_name')
              .inFilter('scene_id', sceneIds);
          final tasksList = List<Map<String, dynamic>>.from(tasksResp);
          final displayNameMap = <String, String>{};
          for (final t in tasksList) {
            final dn = t['display_name']?.toString() ?? '';
            if (dn.isNotEmpty) {
              displayNameMap[t['scene_id'].toString()] = dn;
            }
          }
          for (final m in models) {
            final sid = m['scene_id']?.toString();
            final existingDn = m['display_name']?.toString() ?? '';
            if (sid != null &&
                existingDn.isEmpty &&
                displayNameMap.containsKey(sid)) {
              m['display_name'] = displayNameMap[sid];
            }
          }
        }
      } catch (_) {
        // display_name 获取失败不影响主流程
      }

      // 修正 preview_img_path：数据库中可能存储了旧服务器的完整 URL，
      // 需要提取相对路径后用当前 Supabase 配置重新生成。
      for (final m in models) {
        final raw = m['preview_img_path']?.toString() ?? '';
        if (raw.isNotEmpty) {
          m['preview_img_path'] = _normalizeStorageUrl(raw);
        }
        materializePreviewWebpPath(m, normalize: _normalizeStorageUrl);
      }

      try {
        final scanner = const LocalModelScanner();
        final localModels = await scanner.scanDownloadedModels();
        models.addAll(localModels);
      } catch (_) {
        // best-effort: local models don't block cloud display
      }

      if (models.isEmpty) {
        models.add(_buildDemoModel());
      }

      if (mounted) {
        final ownModelSignature = _buildModelSignature(
          _extractOwnModels(models),
        );
        _refreshState(() {
          _allModels = models;
          _models = models;
          _didFinishInitialModelLoad = true;
          _isLoading = false;
          _lastOwnModelSignature = ownModelSignature;
        });
        _updateOverviewProvider();
      }
      _searchCache.clear();
      _lastSearchKey = null;
      await _syncLocalIndex(models);
      return true;
    } catch (e) {
      if (preserveExistingDataOnError) {
        if (mounted) {
          _refreshState(() {
            _isLoading = false;
          });
        }
        return false;
      }

      // Offline: scan for locally downloaded models
      List<Map<String, dynamic>> fallbackModels;
      try {
        final scanner = const LocalModelScanner();
        final localModels = await scanner.scanDownloadedModels();
        if (localModels.isNotEmpty) {
          fallbackModels = localModels;
        } else {
          fallbackModels = [_buildDemoModel()];
        }
      } catch (_) {
        fallbackModels = [_buildDemoModel()];
      }

      if (mounted) {
        final ownModelSignature = _buildModelSignature(
          _extractOwnModels(fallbackModels),
        );
        _refreshState(() {
          _allModels = fallbackModels;
          _models = fallbackModels;
          _didFinishInitialModelLoad = true;
          _isLoading = false;
          _lastOwnModelSignature = ownModelSignature;
        });
        _updateOverviewProvider();
        if (showErrorToast) {
          debugPrint('[RecallDataSync] offline error: $e');
          showAppToast(context, textLocalize('recall_error_offline'));
        }
      }
      _searchCache.clear();
      _lastSearchKey = null;
      await _syncLocalIndex(fallbackModels);
      return false;
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
      _refreshState(() {
        _isLocalIndexing = true;
      });
    }

    try {
      final stats = await _localRagIndex.syncModels(models);
      if (!mounted) return;
      _refreshState(() {
        _indexStats = stats;
        _isLocalIndexing = false;
      });
    } catch (_) {
      if (!mounted) return;
      _refreshState(() {
        _isLocalIndexing = false;
      });
    }
  }

  int _recentModelCount({int days = 7}) {
    final now = DateTime.now();
    return _allModels.where((model) {
      final rawCreatedAt = model['created_at']?.toString();
      if (rawCreatedAt == null || rawCreatedAt.isEmpty) {
        return false;
      }
      final createdAt = DateTime.tryParse(rawCreatedAt);
      if (createdAt == null) {
        return false;
      }
      return now.difference(createdAt.toLocal()).inDays < days;
    }).length;
  }

  void _updateOverviewProvider() {
    ref.read(overviewStatsProvider.notifier).state = {
      'allModelCount': _allModels.length,
      'processingTaskCount': _processingTasks.length,
      'ragCount': _indexStats?.totalItems ?? _allModels.length,
      'recentCount': _recentModelCount(),
    };
    ref.read(overviewLocalIndexingProvider.notifier).state = _isLocalIndexing;
  }

  /// 按模型名称分组，每组内按 created_at 降序排列（Time Peeling）
  Map<String, List<Map<String, dynamic>>> _groupModelsByName(
    List<Map<String, dynamic>> models,
  ) {
    final groups = <String, List<Map<String, dynamic>>>{};
    for (final model in models) {
      final name = _modelDisplayName(model, fallback: 'Unknown');
      groups.putIfAbsent(name, () => []).add(model);
    }
    for (final list in groups.values) {
      list.sort((a, b) {
        final ta =
            DateTime.tryParse(a['created_at']?.toString() ?? '') ?? DateTime(0);
        final tb =
            DateTime.tryParse(b['created_at']?.toString() ?? '') ?? DateTime(0);
        return tb.compareTo(ta);
      });
    }
    return groups;
  }

  String _modelDisplayName(
    Map<String, dynamic> model, {
    String fallback = 'Unknown Scene',
  }) {
    final displayName = model['display_name']?.toString().trim() ?? '';
    if (displayName.isNotEmpty) {
      return displayName;
    }

    final tags = model['tags'];
    if (tags is List) {
      for (final tag in tags) {
        final value = tag?.toString().trim() ?? '';
        if (value.isNotEmpty) {
          return value;
        }
      }
    }

    final sceneId = model['scene_id']?.toString().trim() ?? '';
    if (sceneId.isNotEmpty) {
      return sceneId;
    }

    return fallback;
  }
}
