// ignore_for_file: invalid_use_of_protected_member
part of '../recall.dart';

extension _RecallPageModelActions on _RecallPageState {
  void _setViewerOpeningState(bool isOpening, {String? label}) {
    _refreshState(() {
      _isOpeningViewer = isOpening;
      _openingViewerLabel = isOpening ? label : null;
    });
  }

  /// Compute the local cache path for a model URL (same logic used by WebGL viewer).
  Future<File> _localCacheFileForUrl(String url) async {
    final encodedUrl = Uri.encodeFull(Uri.decodeFull(url));
    final uri = Uri.parse(encodedUrl);
    final sanitized = uri.path.replaceAll('/', '_').replaceAll('\\', '_');
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$sanitized');
  }

  /// Save a .meta.json sidecar so the offline scanner can reconstruct model info.
  Future<void> _saveModelMetaSidecar({
    required String modelUrl,
    String? previewUrl,
    String? displayName,
    String? description,
    bool isLocalOnly = false,
    File? localFile,
  }) async {
    try {
      final targetFile = localFile ?? await _localCacheFileForUrl(modelUrl);
      final metaFile = File('${targetFile.path}.meta.json');
      // Don't overwrite existing metadata for local-only models
      if (isLocalOnly && await metaFile.exists()) return;
      final meta = {
        if (previewUrl != null && previewUrl.isNotEmpty)
          'preview_img_path': previewUrl,
        if (displayName != null && displayName.isNotEmpty)
          'display_name': displayName,
        if (description != null && description.isNotEmpty)
          'description': description,
      };
      if (meta.isNotEmpty) {
        await metaFile.writeAsString(
          JsonEncoder.withIndent(null).convert(meta),
        );
      }
    } catch (_) {}
  }

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
    if (_isOpeningViewer) {
      return;
    }

    final plyPath = model['ply_path'] as String? ?? '';
    final isLocalOnly = model['_is_local_only'] == true;
    // For local-only models, pass the file path directly instead of converting to a Supabase URL
    final modelUrl = plyPath.isNotEmpty
        ? (isLocalOnly ? plyPath : _toPublicUrl(plyPath))
        : '';
    final posesUrl = (plyPath.isNotEmpty && !isLocalOnly)
        ? _toPosesUrl(plyPath)
        : null;
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
      initialPose = transformMatrix
          .map((e) => (e is num) ? e.toDouble() : 0.0)
          .toList();
    }

    final needsSidecar = !isLocalOnly && modelUrl.isNotEmpty;
    final previewUrl =
        needsSidecar ? model['preview_img_path']?.toString() : null;

    _setViewerOpeningState(true, label: sceneId);

    unawaited(() async {
      try {
        await openViewer(
          context,
          initialModelUrl: modelUrl,
          posesUrl: posesUrl,
          sceneId: sceneId,
          initialPose: initialPose,
          initialPoseId: initialPoseId,
        );
      } catch (e) {
        debugPrint('[RecallModelActions] open viewer error: $e');
        if (mounted) {
          showAppToast(context, '打开模型失败，请稍后重试');
        }
      } finally {
        await Future<void>.delayed(const Duration(milliseconds: 220));
        if (mounted) {
          _setViewerOpeningState(false);

          // Write sidecar and cache thumbnail before rescanning
          if (needsSidecar) {
            await _saveModelMetaSidecar(
              modelUrl: modelUrl,
              previewUrl: previewUrl,
              displayName: sceneId,
              description: model['description']?.toString() ?? '',
            );
            if (previewUrl != null && previewUrl.isNotEmpty) {
              try {
                await ThumbnailCache().getPath(previewUrl);
              } catch (_) {}
            }
          }

          unawaited(_fetchModels(
            preserveExistingDataOnError: true,
            showErrorToast: false,
          ));
        }
      }
    }());
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
    ref.read(myPostsRefreshSignal.notifier).state++;
    showAppToast(context, textLocalize('recall_published'));
  }

  Future<String> _getLocalModelSizeLabel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path'] as String? ?? '';
    if (plyPath.isEmpty) {
      return '';
    }

    try {
      // For local-only models, check the file path directly
      if (model['_is_local_only'] == true) {
        final localFile = File(plyPath);
        if (!await localFile.exists()) return '';
        final sizeBytes = await localFile.length();
        if (sizeBytes <= 0) return '';
        final sizeMb = sizeBytes / 1024 / 1024;
        return '${sizeMb.toStringAsFixed(sizeMb >= 100 ? 0 : 1)}MB';
      }

      final modelUrl = _toPublicUrl(plyPath);
      if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
        return '';
      }

      final localFile = await _localCacheFileForUrl(modelUrl);
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
    _refreshState(() {
      _activeModelAction = {
        ...model,
        if (sizeLabel.isNotEmpty) '_local_size_label': sizeLabel,
        '_is_own_model': _isOwnModel(model),
        '_imageOnly': imageOnly,
      };
      _activeModelActionRect = rect;
    });
  }

  Future<void> _showModelDetails(Map<String, dynamic> model) async {
    final sceneId = model['scene_id']?.toString();
    final sizeLabel = await _getLocalModelSizeLabel(model);
    final isLocalOnly = model['_is_local_only'] == true;

    // 从 processing_tasks 表获取详细信息 (skip for local-only models)
    Map<String, dynamic>? taskInfo;
    if (sceneId != null && !isLocalOnly) {
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
      // Update display_name on model_assets (source of truth)
      final updateResult = await Supabase.instance.client
          .from('model_assets')
          .update({'display_name': newName})
          .eq('scene_id', sceneId)
          .select('id');

      if (updateResult.isEmpty) {
        if (mounted) {
          debugPrint(
            '[RecallModelActions] rename update returned empty for scene_id=$sceneId',
          );
          showAppToast(context, textLocalize('recall_rename_fail'));
        }
        return;
      }

      // Also update processing_tasks so the name appears in task list
      try {
        await Supabase.instance.client
            .from('processing_tasks')
            .update({'display_name': newName})
            .eq('scene_id', sceneId);
      } catch (_) {
        // processing_tasks rows may not exist for all models
      }

      // 立即更新本地数据
      if (mounted) {
        showAppToast(context, textLocalize('recall_rename_success'));
        final targetKey = _modelKey(model);
        _refreshState(() {
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
        debugPrint('[RecallModelActions] rename error: $e');
        showAppToast(context, textLocalize('recall_rename_fail'));
      }
    }
  }

  bool _isOwnModel(Map<String, dynamic> model) {
    final currentUserId =
        Supabase.instance.client.auth.currentUser?.id.trim() ?? '';
    if (currentUserId.isEmpty) return false;
    return (model['user_id']?.toString().trim() ?? '') == currentUserId;
  }

  Future<void> _deleteLocalModel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path']?.toString() ?? '';
    if (plyPath.isEmpty) return;

    try {
      if (model['_is_local_only'] == true) {
        final localFile = File(plyPath);
        if (await localFile.exists()) {
          await localFile.delete();
          // Also clean up the .meta.json sidecar
          final metaFile = File('${localFile.path}.meta.json');
          if (await metaFile.exists()) {
            await metaFile.delete();
          }
        }
        if (mounted) {
          final targetKey = _modelKey(model);
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
          });
          _updateOverviewProvider();
          showAppToast(context, textLocalize('recall_delete_local_success'));
        }
        return;
      }

      final modelUrl = _toPublicUrl(plyPath);
      if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
        return;
      }
      final localFile = await _localCacheFileForUrl(modelUrl);
      if (await localFile.exists()) {
        await localFile.delete();
        if (mounted) {
          showAppToast(context, textLocalize('recall_delete_local_success'));
          unawaited(_fetchModels(
            preserveExistingDataOnError: true,
            showErrorToast: false,
          ));
        }
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[RecallModelActions] delete local model error: $e');
      }
    }
  }

  Future<void> _downloadRecallModel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path']?.toString() ?? '';
    if (plyPath.isEmpty) {
      if (mounted) {
        showAppToast(
          context,
          textLocalize('recall_download_model_unavailable'),
        );
      }
      return;
    }

    final modelUrl = _toPublicUrl(plyPath);
    if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
      if (mounted) {
        showAppToast(
          context,
          textLocalize('recall_download_model_unavailable'),
        );
      }
      return;
    }

    try {
      final targetPath = await _buildRecallDownloadTargetPath(modelUrl, model);
      if (targetPath.isEmpty) {
        if (mounted) {
          showAppToast(context, textLocalize('recall_download_model_fail'));
        }
        return;
      }

      if (mounted) {
        showAppToast(context, textLocalize('recall_download_model_start'));
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
        showAppToast(
          context,
          '${textLocalize('recall_download_model_success')}: ${path.basename(targetPath)}',
        );
      }

      // Save metadata sidecar next to the downloaded file
      unawaited(
        _saveModelMetaSidecar(
          modelUrl: modelUrl,
          previewUrl: model['preview_img_path']?.toString(),
          displayName: _modelDisplayName(model, fallback: ''),
          description: model['description']?.toString() ?? '',
          localFile: File(targetPath),
        ),
      );

      // Cache the preview thumbnail locally so offline scanning can find it
      unawaited(() async {
        final previewUrl = model['preview_img_path']?.toString() ?? '';
        if (previewUrl.isNotEmpty) {
          try {
            await ThumbnailCache().getPath(previewUrl);
          } catch (_) {}
        }
      }());
    } catch (e) {
      if (mounted) {
        debugPrint('[RecallModelActions] download error: $e');
        showAppToast(context, textLocalize('recall_download_model_fail'));
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
        showAppToast(context, textLocalize('cloud_model_missing_id'));
      }
      return;
    }
    if (currentUserId.isEmpty || modelUserId != currentUserId) {
      if (mounted) {
        showAppToast(context, textLocalize('cloud_model_no_permission'));
      }
      return;
    }

    final targetKey = _modelKey(model);
    final targetSceneFolder = _sceneFolderPathForModel(model);
    final plyPath = model['ply_path']?.toString().trim() ?? '';

    try {
      final deleteResult = await Supabase.instance.client
          .from('model_assets')
          .delete()
          .eq('id', modelId)
          .eq('user_id', currentUserId)
          .select('id');

      if (deleteResult.isEmpty) {
        if (mounted) {
          showAppToast(context, textLocalize('cloud_model_delete_fail'));
        }
        return;
      }

      if (targetSceneFolder != null && targetSceneFolder.isNotEmpty) {
        try {
          final storageFiles = await _listStorageFilesRecursively(
            'braindance-assets',
            targetSceneFolder,
          );
          if (storageFiles.isNotEmpty) {
            await Supabase.instance.client.storage
                .from('braindance-assets')
                .remove(storageFiles);
          }
        } catch (_) {
          debugPrint(
            '[RecallModelActions] storage cleanup failed for: $targetSceneFolder',
          );
        }
      }

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

      _refreshState(() {
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

      showAppToast(context, textLocalize('cloud_model_delete_success'));
    } catch (e) {
      if (mounted) {
        debugPrint('[RecallModelActions] delete cloud model error: $e');
        showAppToast(context, textLocalize('cloud_model_delete_fail'));
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
      _refreshState(() {
        _activeModelAction = null;
        _activeModelActionRect = null;
      });
    }
  }

  CommunityModelOption _modelToCommunityOption(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString() ?? '';
    final preview = resolvePreviewImagePaths(
      model,
      normalize: _normalizeStorageUrl,
    );
    return CommunityModelOption(
      id: model['id']?.toString() ?? model['scene_id']?.toString() ?? 'model',
      sceneId: _modelDisplayName(
        model,
        fallback: textLocalize('recall_unnamed_model'),
      ),
      description: model['description']?.toString() ?? '',
      modelUrl: plyPath.isEmpty ? '' : _toPublicUrl(plyPath),
      posesUrl: _toPosesUrl(plyPath),
      coverUrl: preview.primary,
      coverFallbackUrl: preview.fallback,
    );
  }
}
