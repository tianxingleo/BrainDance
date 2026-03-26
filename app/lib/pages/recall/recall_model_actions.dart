part of '../recall.dart';

extension _RecallPageModelActions on _RecallPageState {
  String _modelKey(Map<String, dynamic> model) {
    return model['id']?.toString() ??
        model['scene_id']?.toString() ??
        model.hashCode.toString();
  }

  GlobalKey _modelCardKeyFor(Map<String, dynamic> model) {
    final key = _modelKey(model);
    return _modelCardKeys.putIfAbsent(key, GlobalKey.new);
  }

  bool _isSameModel(Map<String, dynamic>? left, Map<String, dynamic>? right) {
    if (left == null || right == null) {
      return false;
    }
    return _modelKey(left) == _modelKey(right);
  }

  void _navigateToViewer(Map<String, dynamic> model, dynamic transformMatrix) {
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty
        ? toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? _toPosesUrl(plyPath) : null;
    final sceneId = _modelDisplayName(model);
    String? initialPoseId;

    if (transformMatrix is Map) {
      initialPoseId = transformMatrix['image_name']?.toString();
      transformMatrix = transformMatrix['transform_matrix'];
    }

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

    unawaited(
      openViewer(
        context,
        initialModelUrl: modelUrl,
        posesUrl: posesUrl,
        sceneId: sceneId,
        initialPose: initialPose,
        initialPoseId: initialPoseId,
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

  Future<String> _getLocalModelSizeLabel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path'] as String? ?? '';
    if (plyPath.isEmpty) {
      return '';
    }

    try {
      final modelUrl = _toPublicUrl(plyPath);
      if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
        return '';
      }

      final encodedUrl = Uri.encodeFull(Uri.decodeFull(modelUrl));
      final uri = Uri.parse(encodedUrl);
      final sanitizedFileName = uri.path
          .replaceAll('/', '_')
          .replaceAll('\\', '_');
      final dir = await getApplicationDocumentsDirectory();
      final localFile = File('${dir.path}/$sanitizedFileName');
      if (!await localFile.exists()) {
        return '';
      }

      final sizeBytes = await localFile.length();
      if (sizeBytes <= 0) {
        return '';
      }

      final sizeMb = sizeBytes / 1024 / 1024;
      return '${sizeMb.toStringAsFixed(sizeMb >= 100 ? 0 : 1)}MB';
    } catch (_) {
      return '';
    }
  }

  Future<void> _showModelActions(
    Map<String, dynamic> model, {
    bool imageOnly = false,
  }) async {
    final cardKey = _modelCardKeyFor(model);
    final renderBox = cardKey.currentContext?.findRenderObject() as RenderBox?;
    final overlayRenderBox =
        _actionOverlayStackKey.currentContext?.findRenderObject() as RenderBox?;
    if (renderBox == null || overlayRenderBox == null) return;
    final offset = renderBox.localToGlobal(
      Offset.zero,
      ancestor: overlayRenderBox,
    );
    final rect = offset & renderBox.size;
    final sizeLabel = await _getLocalModelSizeLabel(model);
    if (!mounted) {
      return;
    }
    setState(() {
      _activeModelAction = {
        ...model,
        if (sizeLabel.isNotEmpty) '_local_size_label': sizeLabel,
        '_imageOnly': imageOnly,
      };
      _activeModelActionRect = rect;
    });
  }

  Future<void> _showModelDetails(Map<String, dynamic> model) async {
    final sceneId = model['scene_id']?.toString();
    final sizeLabel = await _getLocalModelSizeLabel(model);

    // 从 processing_tasks 表获取详细信息
    Map<String, dynamic>? taskInfo;
    if (sceneId != null) {
      try {
        final resp = await Supabase.instance.client
            .from('processing_tasks')
            .select('created_at, updated_at, task_type, quality_score')
            .eq('scene_id', sceneId)
            .limit(1)
            .maybeSingle();
        taskInfo = resp;
      } catch (_) {}
    }

    if (!mounted) return;

    final displayName = _modelDisplayName(
      model,
      fallback: textLocalize('recall_unnamed_model'),
    );

    // 格式化日期
    String formatDate(String? raw) {
      if (raw == null || raw.isEmpty) {
        return textLocalize('recall_detail_unknown');
      }
      final dt = DateTime.tryParse(raw);
      if (dt == null) return raw;
      final local = dt.toLocal();
      return '${local.year}-${local.month.toString().padLeft(2, '0')}-${local.day.toString().padLeft(2, '0')}  ${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}';
    }

    final createdAt = formatDate(
      taskInfo?['created_at']?.toString() ?? model['created_at']?.toString(),
    );
    final updatedAt = formatDate(taskInfo?['updated_at']?.toString());
    final taskType =
        taskInfo?['task_type']?.toString() ??
        textLocalize('recall_detail_unknown');
    final qualityScore = taskInfo?['quality_score'];

    await showRecallModelDetailSheet(
      context,
      displayName: displayName,
      createdAt: createdAt,
      updatedAt: updatedAt,
      taskType: taskType,
      qualityScore: qualityScore,
      sizeLabel: sizeLabel,
      qualityScoreTrailing: qualityScore != null
          ? _buildScoreBar(
              (qualityScore as num).toDouble(),
              AppConfig.isNightMode,
            )
          : null,
    );
  }

  Widget _buildScoreBar(double score, bool isDark) {
    final ratio = (score / 100).clamp(0.0, 1.0);
    final color = ratio >= 0.7
        ? const Color(0xFF4CAF50)
        : ratio >= 0.4
        ? const Color(0xFFFFA726)
        : const Color(0xFFEF5350);
    return SizedBox(
      width: 60,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(3),
        child: LinearProgressIndicator(
          value: ratio,
          minHeight: 6,
          backgroundColor: isDark
              ? Colors.white.withAlpha(20)
              : Colors.black.withAlpha(15),
          valueColor: AlwaysStoppedAnimation<Color>(color),
        ),
      ),
    );
  }

  Future<void> _renameModel(Map<String, dynamic> model) async {
    final sceneId = model['scene_id']?.toString();
    if (sceneId == null) return;

    final currentName = _modelDisplayName(model, fallback: sceneId);
    final newName = await showDialog<String>(
      context: context,
      builder: (_) => RecallRenameModelDialog(initialName: currentName),
    );

    if (newName == null || newName.isEmpty || !mounted) return;

    try {
      await Supabase.instance.client
          .from('processing_tasks')
          .update({'display_name': newName})
          .eq('scene_id', sceneId)
          .select();

      // 立即更新本地数据
      if (mounted) {
        TDToast.showText(
          textLocalize('recall_rename_success'),
          context: context,
        );
        final targetKey = _modelKey(model);
        setState(() {
          for (final m in _allModels) {
            if (_modelKey(m) == targetKey) {
              m['display_name'] = newName;
            }
          }
          for (final m in _models) {
            if (_modelKey(m) == targetKey) {
              m['display_name'] = newName;
            }
          }
          if (_activeModelAction != null &&
              _modelKey(_activeModelAction!) == targetKey) {
            _activeModelAction = {
              ..._activeModelAction!,
              'display_name': newName,
            };
          }
        });
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText(
          '${textLocalize('recall_rename_fail')}: $e',
          context: context,
        );
      }
    }
  }

  Future<void> _downloadRecallModel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path']?.toString() ?? '';
    if (plyPath.isEmpty) {
      if (mounted) {
        TDToast.showText(
          textLocalize('recall_download_model_unavailable'),
          context: context,
        );
      }
      return;
    }

    final modelUrl = _toPublicUrl(plyPath);
    if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
      if (mounted) {
        TDToast.showText(
          textLocalize('recall_download_model_unavailable'),
          context: context,
        );
      }
      return;
    }

    try {
      final targetPath = await _buildRecallDownloadTargetPath(modelUrl, model);
      if (targetPath.isEmpty) {
        if (mounted) {
          TDToast.showText(
            textLocalize('recall_download_model_fail'),
            context: context,
          );
        }
        return;
      }

      if (mounted) {
        TDToast.showText(
          textLocalize('recall_download_model_start'),
          context: context,
        );
      }

      await Dio().download(
        modelUrl,
        targetPath,
        deleteOnError: true,
        options: Options(
          responseType: ResponseType.stream,
          followRedirects: true,
          receiveTimeout: const Duration(minutes: 30),
          sendTimeout: const Duration(minutes: 2),
        ),
      );

      if (mounted) {
        TDToast.showText(
          '${textLocalize('recall_download_model_success')}: ${path.basename(targetPath)}',
          context: context,
        );
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText(
          '${textLocalize('recall_download_model_fail')}: $e',
          context: context,
        );
      }
    }
  }

  Future<String> _buildRecallDownloadTargetPath(
    String modelUrl,
    Map<String, dynamic> model,
  ) async {
    var baseDir = await DirFinder.downloadsDir();
    if (baseDir.isEmpty) {
      baseDir = await DirFinder.documentsDir();
    }
    if (baseDir.isEmpty) {
      baseDir = (await getApplicationDocumentsDirectory()).path;
    }
    final ensured = await DirSystem.ensureDir(baseDir);
    if (!ensured) {
      return '';
    }

    final uri = Uri.tryParse(Uri.encodeFull(Uri.decodeFull(modelUrl)));
    final urlFileName = uri == null ? '' : path.basename(uri.path);
    final ext = path.extension(urlFileName).isNotEmpty
        ? path.extension(urlFileName)
        : '.ply';
    final displayName = _sanitizeExportFileName(
      _modelDisplayName(model, fallback: 'recall_model'),
    );
    final candidateName = displayName.isEmpty
        ? (urlFileName.isNotEmpty ? urlFileName : 'recall_model$ext')
        : '$displayName$ext';

    return _dedupeExportPath(path.join(baseDir, candidateName));
  }

  Future<String> _dedupeExportPath(String targetPath) async {
    if (!await File(targetPath).exists()) {
      return targetPath;
    }

    final dir = path.dirname(targetPath);
    final ext = path.extension(targetPath);
    final name = path.basenameWithoutExtension(targetPath);
    for (var index = 2; index <= 999; index++) {
      final nextPath = path.join(dir, '$name($index)$ext');
      if (!await File(nextPath).exists()) {
        return nextPath;
      }
    }
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    return path.join(dir, '${name}_$timestamp$ext');
  }

  String _sanitizeExportFileName(String raw) {
    return raw
        .replaceAll(RegExp(r'[<>:"/\\|?*]'), '_')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  Future<List<String>> _listStorageFilesRecursively(
    String bucket,
    String folderPath,
  ) async {
    final normalizedFolder = folderPath.trim().replaceAll('\\', '/');
    if (normalizedFolder.isEmpty) {
      return const <String>[];
    }

    final files = <String>[];
    final pendingFolders = <String>[normalizedFolder];
    final storage = Supabase.instance.client.storage.from(bucket);

    while (pendingFolders.isNotEmpty) {
      final currentFolder = pendingFolders.removeLast();
      final entries = await storage.list(path: currentFolder);
      for (final entry in entries) {
        final entryName = entry.name.trim();
        if (entryName.isEmpty) {
          continue;
        }
        final childPath = '$currentFolder/$entryName';
        final entryId = entry.id?.trim() ?? '';
        if (entryId.isNotEmpty) {
          files.add(childPath);
        } else {
          pendingFolders.add(childPath);
        }
      }
    }

    return files;
  }

  String? _sceneFolderPathForModel(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString().trim() ?? '';
    if (plyPath.isNotEmpty) {
      final normalizedPath = plyPath.replaceAll('\\', '/');
      final marker = '/output/';
      final markerIndex = normalizedPath.indexOf(marker);
      if (markerIndex > 0) {
        return normalizedPath.substring(0, markerIndex);
      }
      final lastSlash = normalizedPath.lastIndexOf('/');
      if (lastSlash > 0) {
        return normalizedPath.substring(0, lastSlash);
      }
    }

    final userId = model['user_id']?.toString().trim() ?? '';
    final sceneId = model['scene_id']?.toString().trim() ?? '';
    if (userId.isEmpty || sceneId.isEmpty) {
      return null;
    }
    return '$userId/$sceneId';
  }

  Future<void> _deleteCloudModel(Map<String, dynamic> model) async {
    final modelId = model['id']?.toString().trim() ?? '';
    final currentUser = Supabase.instance.client.auth.currentUser;
    final currentUserId = currentUser == null ? '' : currentUser.id.trim();
    final modelUserId = model['user_id']?.toString().trim() ?? '';
    if (modelId.isEmpty) {
      if (mounted) {
        TDToast.showText('云端模型缺少 id，无法删除', context: context);
      }
      return;
    }
    if (currentUserId.isEmpty || modelUserId != currentUserId) {
      if (mounted) {
        TDToast.showText('只能删除当前账号自己的云端模型', context: context);
      }
      return;
    }

    final targetKey = _modelKey(model);
    final targetSceneFolder = _sceneFolderPathForModel(model);
    final plyPath = model['ply_path']?.toString().trim() ?? '';

    try {
      if (targetSceneFolder != null && targetSceneFolder.isNotEmpty) {
        final storageFiles = await _listStorageFilesRecursively(
          'braindance-assets',
          targetSceneFolder,
        );
        if (storageFiles.isNotEmpty) {
          await Supabase.instance.client.storage
              .from('braindance-assets')
              .remove(storageFiles);
        }
      }

      await Supabase.instance.client
          .from('model_assets')
          .delete()
          .eq('id', modelId)
          .eq('user_id', currentUserId);

      await _localRagIndex.deleteByModelId(modelId);

      if (plyPath.isNotEmpty) {
        final modelUrl = _toPublicUrl(plyPath);
        downloadEventBus.add(
          ModelDownloadEvent(url: modelUrl, progress: 0.0, isDeleted: true),
        );
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _allModels.removeWhere((item) => _modelKey(item) == targetKey);
        _models.removeWhere((item) => _modelKey(item) == targetKey);
        if (_activeModelAction != null &&
            _modelKey(_activeModelAction!) == targetKey) {
          _activeModelAction = null;
          _activeModelActionRect = null;
        }
        if (_allModels.isEmpty) {
          final demo = _buildDemoModel();
          _allModels = [demo];
          _models = [demo];
        } else if (_models.isEmpty) {
          _models = List<Map<String, dynamic>>.from(_allModels);
        }
        _lastOwnModelSignature = _buildModelSignature(
          _extractOwnModels(_allModels),
        );
      });
      _updateOverviewProvider();

      TDToast.showText('云端模型删除成功', context: context);
    } catch (e) {
      if (mounted) {
        TDToast.showText('云端模型删除失败：$e', context: context);
      }
    }
  }

  Future<void> _dismissModelActions() async {
    if (_activeModelAction == null && _activeModelActionRect == null) {
      return;
    }

    final overlayState = _overlayKey.currentState;
    if (overlayState != null) {
      await overlayState.hide();
    }

    if (mounted) {
      setState(() {
        _activeModelAction = null;
        _activeModelActionRect = null;
      });
    }
  }

  CommunityModelOption _modelToCommunityOption(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString() ?? '';
    final preview = model['preview_img_path']?.toString();
    return CommunityModelOption(
      id: model['id']?.toString() ?? model['scene_id']?.toString() ?? 'model',
      sceneId: _modelDisplayName(
        model,
        fallback: textLocalize('recall_unnamed_model'),
      ),
      description: model['description']?.toString() ?? '',
      modelUrl: plyPath.isEmpty
          ? './models/scene_auto_sync_raw.ply'
          : _toPublicUrl(plyPath),
      posesUrl: _toPosesUrl(plyPath),
      coverUrl: preview,
    );
  }
}
