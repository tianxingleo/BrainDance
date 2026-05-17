part of '../recall.dart';

extension _RecallPageLocalAi on _RecallPageState {
  String get _defaultModelDownloadUrl => SupabaseConfig.localModelUrl;

  void _initRecallPageState() {
    _localModelUrlController = TextEditingController(
      text: _defaultModelDownloadUrl,
    );
    _localModelUrlController.addListener(_handleLocalModelUrlChanged);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _bootstrapPage();
    });
  }

  void _disposeRecallPageState() {
    _modelPollingTimer?.cancel();
    _agentElapsedTimer?.cancel();
    _searchController.dispose();
    _localModelPathController.dispose();
    _localModelUrlController.removeListener(_handleLocalModelUrlChanged);
    _localModelUrlController.dispose();
    _realtimeChannel?.unsubscribe();
    unawaited(_disposeLocalQnaModel());
  }

  void _handleRecallPageDependenciesChanged() {
    // ignore: deprecated_member_use
    final isTabActive = TickerMode.of(context);
    if (_isTabActive == isTabActive) {
      return;
    }
    _isTabActive = isTabActive;
    if (_isTabActive && _shouldRefreshProcessingOnResume) {
      _shouldRefreshProcessingOnResume = false;
      unawaited(_fetchProcessingTasks());
    }
    _syncModelPollingState();
  }

  void _bootstrapPage() {
    if (!mounted || _didBootstrap) {
      return;
    }
    _didBootstrap = true;
    unawaited(_restoreLocalModelPath());
    unawaited(_loadLocalModelCatalog());
    unawaited(_fetchModels());
    unawaited(_fetchProcessingTasks());
    _setupRealtimeListener();
    _syncModelPollingState();
  }

  void _handleLocalModelUrlChanged() {
    final currentUrl = _localModelUrlController.text.trim();
    final matchedItem = _findCatalogItemByUrl(currentUrl);
    final nextSelectedUrl = matchedItem?.downloadUrl ?? currentUrl;
    final downloadedPath = _downloadedLocalModelPathsByUrl[currentUrl];
    if (_selectedLocalModelUrl == nextSelectedUrl &&
        (downloadedPath == null ||
            _localModelPathController.text.trim() == downloadedPath)) {
      return;
    }
    if (!mounted) {
      return;
    }
    setState(() {
      _selectedLocalModelUrl = nextSelectedUrl.isEmpty ? null : nextSelectedUrl;
      if (downloadedPath != null) {
        _localModelPathController.text = downloadedPath;
      }
    });
  }

  void _syncModelPollingState() {
    if (!_isTabActive) {
      _modelPollingTimer?.cancel();
      _modelPollingTimer = null;
      return;
    }
    _modelPollingTimer ??= Timer.periodic(
      const Duration(seconds: 5),
      (_) => unawaited(_pollModelUpdates()),
    );
  }

  List<Map<String, dynamic>> _extractOwnModels(
    List<Map<String, dynamic>> models,
  ) {
    final currentUserId = Supabase.instance.client.auth.currentUser?.id;
    if (currentUserId == null || currentUserId.isEmpty) {
      return const <Map<String, dynamic>>[];
    }
    return models
        .where((model) => model['user_id']?.toString() == currentUserId)
        .map((model) => Map<String, dynamic>.from(model))
        .toList();
  }

  String _buildModelSignature(List<Map<String, dynamic>> models) {
    if (models.isEmpty) {
      return '';
    }
    final parts = models.map((model) {
      final id = model['id']?.toString() ?? '';
      final sceneId = model['scene_id']?.toString() ?? '';
      final createdAt = model['created_at']?.toString() ?? '';
      return '$id|$sceneId|$createdAt';
    }).toList()..sort();
    return parts.join('||');
  }

  Future<String> _fetchRemoteOwnModelSignature() async {
    final currentUserId = Supabase.instance.client.auth.currentUser?.id;
    if (currentUserId == null || currentUserId.isEmpty) {
      return '';
    }
    final response = await Supabase.instance.client
        .from('model_assets')
        .select('id, scene_id, created_at')
        .eq('user_id', currentUserId)
        .order('created_at', ascending: false);
    final ownModels = List<Map<String, dynamic>>.from(response);
    return _buildModelSignature(ownModels);
  }

  Future<void> _refreshModelsForCurrentState({
    bool showLoadingIndicator = true,
  }) async {
    final query = _searchController.text.trim();
    if (!mounted) return;
    if (showLoadingIndicator) {
      setState(() {
        _isLoading = true;
      });
    }
    final results = await Future.wait([
      _fetchModels(
        preserveExistingDataOnError: !showLoadingIndicator,
        showErrorToast: showLoadingIndicator,
      ),
      _fetchProcessingTasks(),
    ]);
    final didRefreshModels = results.first as bool;
    if (!mounted || !didRefreshModels || query.isEmpty) {
      return;
    }
    await _searchModels(query);
  }

  Future<void> _pollModelUpdates() async {
    if (!mounted ||
        !_didFinishInitialModelLoad ||
        !_isTabActive ||
        _isLoading ||
        _isModelPollingInFlight ||
        _activeModelAction != null) {
      return;
    }

    _isModelPollingInFlight = true;
    try {
      final remoteSignature = await _fetchRemoteOwnModelSignature();
      if (!mounted) {
        return;
      }
      if (remoteSignature == _lastOwnModelSignature) {
        return;
      }
      await _refreshModelsForCurrentState(showLoadingIndicator: false);
    } catch (_) {
      // Ignore polling failures and retry on the next cycle.
    } finally {
      _isModelPollingInFlight = false;
    }
  }

  Future<void> _restoreLocalModelPath() async {
    final prefs = await SharedPreferences.getInstance();
    final savedPath = prefs
        .getString(_RecallPageState._localModelPathPrefKey)
        ?.trim();
    final savedUrl = prefs
        .getString(_RecallPageState._localModelUrlPrefKey)
        ?.trim();
    final effectiveUrl = (savedUrl == null || savedUrl.isEmpty)
        ? _defaultModelDownloadUrl
        : savedUrl;
    final defaultPath = await _getPrivateModelPathForUrl(effectiveUrl);
    final effectivePath = (savedPath == null || savedPath.isEmpty)
        ? defaultPath
        : savedPath;
    final hasLocalFile = await File(effectivePath).exists();
    if (!mounted) return;
    setState(() {
      _localModelPathController.text = effectivePath;
      _localModelUrlController.text = effectiveUrl;
      _selectedLocalModelUrl = effectiveUrl;
      if (hasLocalFile) {
        _downloadedLocalModelPathsByUrl[effectiveUrl] = effectivePath;
      }
    });
  }

  Future<void> _loadLocalModelCatalog() async {
    final catalog = await _localModelCatalogService.fetchCatalog();
    final downloadedPaths = await _collectDownloadedModelPaths(catalog);
    final currentUrl = _localModelUrlController.text.trim();
    final savedUrl = currentUrl.isEmpty ? _defaultModelDownloadUrl : currentUrl;
    final matchedItem = _findCatalogItemByUrl(savedUrl, catalog);
    final preferredItem =
        matchedItem ??
        catalog.cast<LocalModelCatalogItem?>().firstWhere(
          (item) => item?.isRecommended == true,
          orElse: () => catalog.isEmpty ? null : catalog.first,
        );

    if (!mounted) {
      return;
    }

    setState(() {
      _localModelCatalog = catalog;
      _downloadedLocalModelPathsByUrl
        ..clear()
        ..addAll(downloadedPaths);
      if (matchedItem != null) {
        _selectedLocalModelUrl = matchedItem.downloadUrl;
      } else if (_selectedLocalModelUrl == null && preferredItem != null) {
        _selectedLocalModelUrl = preferredItem.downloadUrl;
      }
    });

    if (currentUrl.isEmpty && preferredItem != null) {
      await _selectCatalogModel(preferredItem.downloadUrl, persist: false);
    }
  }

  Future<void> _persistLocalModelPath(String modelPath) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_RecallPageState._localModelPathPrefKey, modelPath);
  }

  Future<void> _persistLocalModelUrl(String modelUrl) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_RecallPageState._localModelUrlPrefKey, modelUrl);
  }

  Future<String> _getPrivateModelPathForUrl(String? modelUrl) async {
    final dir = await getApplicationDocumentsDirectory();
    var fileName = _RecallPageState._defaultModelFileName;
    final trimmedUrl = modelUrl?.trim() ?? '';
    if (trimmedUrl.isNotEmpty) {
      final uri = Uri.tryParse(trimmedUrl);
      final segments = (uri?.pathSegments ?? const <String>[])
          .where((segment) => segment.isNotEmpty)
          .toList();
      if (segments.isNotEmpty) {
        fileName = segments.last;
      }
    }
    return path.join(dir.path, fileName);
  }

  LocalModelCatalogItem? _findCatalogItemByUrl(
    String? modelUrl, [
    List<LocalModelCatalogItem>? source,
  ]) {
    if (modelUrl == null || modelUrl.isEmpty) {
      return null;
    }
    final catalog = source ?? _localModelCatalog;
    for (final item in catalog) {
      if (item.downloadUrl == modelUrl) {
        return item;
      }
    }
    return null;
  }

  Future<Map<String, String>> _collectDownloadedModelPaths(
    List<LocalModelCatalogItem> catalog,
  ) async {
    final downloaded = <String, String>{};
    for (final item in catalog) {
      final modelPath = await _getPrivateModelPathForUrl(item.downloadUrl);
      if (await File(modelPath).exists()) {
        downloaded[item.downloadUrl] = modelPath;
      }
    }
    final currentUrl = _localModelUrlController.text.trim();
    final currentPath = _localModelPathController.text.trim();
    if (currentUrl.isNotEmpty &&
        currentPath.isNotEmpty &&
        await File(currentPath).exists()) {
      downloaded[currentUrl] = currentPath;
    }
    return downloaded;
  }

  Future<void> _selectCatalogModel(
    String? modelUrl, {
    bool persist = true,
  }) async {
    if (modelUrl == null || modelUrl.isEmpty) {
      return;
    }
    final modelPath =
        _downloadedLocalModelPathsByUrl[modelUrl] ??
        await _getPrivateModelPathForUrl(modelUrl);
    if (!mounted) {
      return;
    }
    setState(() {
      _selectedLocalModelUrl = modelUrl;
      _localModelUrlController.text = modelUrl;
      _localModelPathController.text = modelPath;
    });
    if (persist) {
      await _persistLocalModelUrl(modelUrl);
      await _persistLocalModelPath(modelPath);
    }
  }

  Future<void> _downloadModelToPrivateDir() async {
    final modelUrl = _localModelUrlController.text.trim();
    if (modelUrl.isEmpty) {
      showAppToast(context, textLocalize('local_model_fill_url'));
      return;
    }

    final modelPath = await _getPrivateModelPathForUrl(modelUrl);
    setState(() {
      _isModelDownloading = true;
      _modelDownloadProgress = 0;
      _modelDownloadedBytes = 0;
      _modelDownloadTotalBytes = null;
      _localAnswerStatus = '正在下载模型到应用私有目录...';
      _localModelPathController.text = modelPath;
    });

    try {
      await _persistLocalModelUrl(modelUrl);
      await _persistLocalModelPath(modelPath);

      await Dio().download(
        modelUrl,
        modelPath,
        deleteOnError: true,
        options: Options(
          responseType: ResponseType.stream,
          followRedirects: true,
          receiveTimeout: const Duration(minutes: 30),
          sendTimeout: const Duration(minutes: 2),
        ),
        onReceiveProgress: (received, total) {
          if (!mounted) return;
          setState(() {
            _modelDownloadedBytes = received;
            _modelDownloadTotalBytes = total > 0 ? total : null;
            _modelDownloadProgress = total > 0 ? received / total : null;
          });
        },
      );

      final fileSize = await File(modelPath).length();
      if (!mounted) return;
      setState(() {
        _isModelDownloading = false;
        _modelDownloadProgress = 1;
        _modelDownloadedBytes = fileSize;
        _modelDownloadTotalBytes = fileSize;
        _selectedLocalModelUrl = modelUrl;
        _downloadedLocalModelPathsByUrl[modelUrl] = modelPath;
        _localAnswerStatus =
            '模型下载完成：${(fileSize / 1024 / 1024).toStringAsFixed(1)} MB';
      });
      showAppToast(context, textLocalize('local_model_download_success'));
    } catch (e) {
      if (!mounted) return;
      debugPrint('[RecallLocalAI] model download error: $e');
      setState(() {
        _isModelDownloading = false;
        _modelDownloadProgress = null;
        _modelDownloadedBytes = 0;
        _modelDownloadTotalBytes = null;
        _localAnswerStatus = textLocalize('local_model_download_fail');
      });
      showAppToast(context, textLocalize('local_model_download_fail'));
    }
  }

  Future<void> _disposeLocalQnaModel() async {
    await _llamaStreamSubscription?.cancel();
    _llamaStreamSubscription = null;

    final model = _localQnaModel;
    _localQnaModel = null;
    if (model != null) {
      try {
        await model.dispose();
      } catch (_) {
        // 忽略释放阶段异常，避免影响页面退出
      }
    }
  }

  Future<void> _loadLocalQnaModel() async {
    final modelPath = _localModelPathController.text.trim();
    final modelUrl = _localModelUrlController.text.trim();
    final selectedCatalogItem = _findCatalogItemByUrl(modelUrl);
    final modelLabel =
        selectedCatalogItem?.name ??
        path.basename(
          modelPath.isEmpty
              ? _RecallPageState._defaultModelFileName
              : modelPath,
        );
    if (modelPath.isEmpty) {
      showAppToast(context, textLocalize('local_model_fill_path'));
      return;
    }

    setState(() {
      _isLocalModelLoading = true;
      _isLocalModelReady = false;
      _localAnswer = '';
      _localAnswerStatus = '正在加载端侧模型：$modelLabel...';
    });

    try {
      await _disposeLocalQnaModel();
      await _persistLocalModelUrl(modelUrl);
      await _persistLocalModelPath(modelPath);

      final modelFile = File(modelPath);
      final exists = await modelFile.exists();
      if (!exists) {
        throw Exception('model file not found: $modelPath');
      }
      final modelSize = await modelFile.length();
      if (modelSize < 100 * 1024 * 1024) {
        throw Exception(
          'model file too small: ${(modelSize / 1024 / 1024).toStringAsFixed(1)} MB',
        );
      }

      final llama = LlamaEngine(LlamaBackend());
      var backendSummary = 'CPU';
      late final ModelParams mobileProfile;
      mobileProfile = const ModelParams(
        contextSize: 1024,
        gpuLayers: 12,
        preferredBackend: GpuBackend.vulkan,
        numberOfThreads: 4,
        numberOfThreadsBatch: 4,
        batchSize: 64,
        microBatchSize: 32,
      );
      try {
        await llama.loadModel(modelPath, modelParams: mobileProfile);
        final backendName = await llama.getBackendName();
        backendSummary = '$backendName (GPU 优先)';
      } catch (_) {
        await llama.dispose();
        final fallbackLlama = LlamaEngine(LlamaBackend());
        await fallbackLlama.loadModel(
          modelPath,
          modelParams: const ModelParams(
            contextSize: 768,
            gpuLayers: 0,
            preferredBackend: GpuBackend.cpu,
            numberOfThreads: 3,
            numberOfThreadsBatch: 3,
            batchSize: 64,
            microBatchSize: 32,
          ),
        );
        final backendName = await fallbackLlama.getBackendName();
        if (!mounted) {
          await fallbackLlama.dispose();
          return;
        }
        await fallbackLlama.setNativeLogLevel(LlamaLogLevel.info);
        setState(() {
          _localQnaModel = fallbackLlama;
          _isLocalModelLoading = false;
          _isLocalModelReady = true;
          _activeLocalModelUrl = modelUrl.isEmpty ? null : modelUrl;
          _localAnswerStatus =
              '$modelLabel 已加载，当前后端：$backendName（已从 GPU 回退到 CPU），模型大小：${(modelSize / 1024 / 1024).toStringAsFixed(1)} MB';
        });
        return;
      }
      await llama.setNativeLogLevel(LlamaLogLevel.info);

      if (!mounted) {
        await llama.dispose();
        return;
      }

      setState(() {
        _localQnaModel = llama;
        _isLocalModelLoading = false;
        _isLocalModelReady = true;
        _activeLocalModelUrl = modelUrl.isEmpty ? null : modelUrl;
        _localAnswerStatus =
            '$modelLabel 已加载，当前后端：$backendSummary，模型大小：${(modelSize / 1024 / 1024).toStringAsFixed(1)} MB';
      });
    } catch (e) {
      await _disposeLocalQnaModel();
      if (!mounted) {
        return;
      }
      setState(() {
        _isLocalModelLoading = false;
        _isLocalModelReady = false;
        _activeLocalModelUrl = null;
        _localAnswerStatus = textLocalize('local_model_load_fail');
      });
      debugPrint('[RecallLocalAI] model load error: $e');
      showAppToast(context, textLocalize('local_model_load_fail'));
    }
  }

  static const String _kSystemPrompt =
      '你是 BrainDance 的本地记忆问答助手。'
      '你只能根据 retrieval 提供的证据回答，不要猜测。'
      '规则：'
      '1. hit_count > 0 时，必须回答具体内容，不能只说有记录。'
      '2. hit_count == 0 时，只能回答‘暂无相关记录’。'
      '3. 部分命中时，只能回答证据覆盖到的部分，对未命中部分明确说‘暂无相关记录’或‘未见相关记录’。'
      '4. 输出必须是自然语言短句，最多两句。不要输出 JSON、代码块、列表或键值对。'
      '5. 不复述问题，不解释规则，不说‘根据给定证据’。'
      '6. 不要输出<think>或任何思考链。';

  Future<void> _askLocalQuestion({String? question}) async {
    final userQuestion = (question ?? '').trim();
    if (userQuestion.isEmpty) {
      showAppToast(context, textLocalize('local_model_fill_question'));
      return;
    }
    if (_localQnaModel == null || !_isLocalModelReady) {
      showAppToast(context, textLocalize('local_model_load_first'));
      return;
    }

    // 1. 构建紧凑的 retrieval payload，尽量减少端侧上下文占用
    final retrieval = await _buildRetrievalPayload(userQuestion);
    final userPayload = jsonEncode({
      'question': userQuestion,
      'retrieval': retrieval,
    });

    // 2. 构建 ChatML 格式 Prompt
    final prompt =
        '<|im_start|>system\n$_kSystemPrompt<|im_end|>\n'
        '<|im_start|>user\n$userPayload<|im_end|>\n'
        '<|im_start|>assistant\n'
        '请直接给出最终回答，且只输出一句到两句。';

    setState(() {
      _localAnswer = '';
      _localReasoning = '';
      // 预览上下文现在展示构建好的 JSON，方便调试
      const encoder = JsonEncoder.withIndent('  ');
      _localContextPreview = encoder.convert(retrieval);
      _localAnswerStatus = '正在根据本地记忆片段生成回答...';
    });

    try {
      var streamedAnswer = '';
      var lockedAnswer = false;
      _localQnaModel!.cancelGeneration();
      await _llamaStreamSubscription?.cancel();
      _llamaStreamSubscription = _localQnaModel!
          .generate(
            prompt,
            params: const GenerationParams(
              maxTokens: 96,
              temp: 0.0,
              topK: 1,
              topP: 1.0,
              penalty: 1.02,
              stopSequences: ['<|im_end|>', '<|endoftext|>'], // Qwen3 停止符
            ),
          )
          .listen(
            (token) {
              if (!mounted) {
                return;
              }
              if (token.isEmpty) {
                return;
              }
              final nextRaw = streamedAnswer + token;
              final parsedOutput = _parseLocalModelOutput(nextRaw);
              final nextReasoning = parsedOutput.reasoning;
              final nextAnswer = _truncateLocalAnswer(parsedOutput.answer);
              if (lockedAnswer) {
                if (nextAnswer != _localAnswer ||
                    nextReasoning != _localReasoning) {
                  _localQnaModel!.cancelGeneration();
                }
                return;
              }
              streamedAnswer = nextRaw;
              lockedAnswer =
                  nextAnswer.trim().isNotEmpty && _shouldLockAnswer(nextAnswer);
              setState(() {
                _localReasoning = nextReasoning;
                _localAnswer = nextAnswer;
              });
              if (lockedAnswer) {
                _localQnaModel!.cancelGeneration();
              }
            },
            onError: (Object error) {
              if (!mounted) {
                return;
              }
              setState(() {
                _localAnswerStatus = '端侧问答失败：$error';
              });
            },
            onDone: () {
              if (!mounted) {
                return;
              }
              setState(() {
                if (_localAnswer.trim().isEmpty) {
                  _localAnswer = '我不知道';
                }
                _localAnswerStatus = '端侧回答完成';
              });
            },
            cancelOnError: true,
          );
    } catch (e) {
      if (!mounted) {
        return;
      }
      debugPrint('[RecallLocalAI] Q&A error: $e');
      setState(() {
        _localAnswerStatus = textLocalize('local_model_qa_fail');
      });
      showAppToast(context, textLocalize('local_model_qa_fail'));
    }
  }

  Future<Map<String, dynamic>> _buildRetrievalPayload(String question) async {
    List<Map<String, dynamic>> matches = [];
    try {
      // 本地语义扩展，弥补 HashingEmbedder 的语义缺失
      final expandedQuery = _expandQuery(question);
      matches = await _localRagIndex.search(
        expandedQuery,
        limit: 3,
        minScore: 0.08,
      );
    } catch (_) {
      matches = const [];
    }

    if (matches.isEmpty) {
      return {
        'evidence': const [],
        'hit_count': 0,
        'intent': 'unknown',
        'answerability_hint': 'no_hit',
      };
    }

    final evidence = matches
        .map((item) {
          final metaInfo = _toMap(item['meta_info']);
          final tags = _joinList(item['tags']);
          final objects = _joinList(item['objects']);
          final summary = _summarizeEvidence(metaInfo);

          return {
            'id': item['id']?.toString() ?? '',
            'score': (item['similarity'] as num?)?.toDouble() ?? 0.0,
            'object': objects,
            'tags': tags,
            'summary': summary,
            'scene_id': item['scene_id']?.toString() ?? '',
          };
        })
        .take(3)
        .toList();

    return {
      'evidence': evidence,
      'hit_count': evidence.length,
      // 本地暂无 Intent 识别模型，先默认为 unknown 或根据是否有结果判断
      'intent': evidence.isEmpty ? 'unknown' : 'object_lookup',
      'answerability_hint': evidence.isEmpty ? 'no_hit' : 'hit',
    };
  }

  String _expandQuery(String query) {
    if (query.trim().isEmpty) return query;
    var expanded = query;

    // 关键领域词汇扩展 - 对齐 ai_engine 的 SEMANTIC_QUERY_EXPANSIONS
    const semanticExpansions = {
      "理工": [
        "算法",
        "算法导论",
        "数学",
        "高等数学",
        "教材",
        "词典",
        "电脑",
        "笔记本电脑",
        "显示器",
        "白板",
      ],
      "计算机": ["电脑", "笔记本电脑", "显示器", "机械键盘", "办公桌", "白板"],
      "学习": ["教材", "词典", "地球仪", "白板", "办公桌", "笔记本电脑"],
      "书房": ["办公桌", "椅子", "书架", "电脑", "书"],
    };

    // 对象查找扩展 - 对齐 ai_engine 的 OBJECT_LOOKUP_PHRASE_EXPANSIONS
    const objectExpansions = {
      "洛天依": ["洛天依", "手办", "展台"],
      "学习摆件": ["学习相关", "地球仪", "手办"],
      "桌面设备": ["显示器", "笔记本电脑", "键盘", "办公桌"],
    };

    // 简单包含匹配
    semanticExpansions.forEach((key, values) {
      if (query.contains(key)) {
        expanded += ' ${values.join(' ')}';
      }
    });

    objectExpansions.forEach((key, values) {
      if (query.contains(key)) {
        expanded += ' ${values.join(' ')}';
      }
    });

    return expanded;
  }

  String _sanitizeLocalAnswer(String raw) {
    final original = raw.trim();
    var cleaned = raw;
    const cutMarkers = ['【说明】', '\n问题：', '\n用户：', '\n系统：'];
    for (final marker in cutMarkers) {
      final index = cleaned.indexOf(marker);
      if (index >= 0) {
        cleaned = cleaned.substring(0, index);
      }
    }

    final answerIndex = cleaned.lastIndexOf('答案：');
    if (answerIndex >= 0) {
      cleaned = cleaned.substring(answerIndex + 3);
    }

    cleaned = cleaned.trim();

    if (cleaned.isEmpty && original.isNotEmpty) {
      return original;
    }

    if (cleaned.length >= 8) {
      final half = cleaned.length ~/ 2;
      final first = cleaned.substring(0, half).trim();
      final second = cleaned.substring(half).trim();
      if (first.isNotEmpty && first == second) {
        cleaned = first;
      }
    }

    return cleaned;
  }

  _ParsedLocalModelOutput _parseLocalModelOutput(String raw) {
    final normalized = raw.replaceAll('\r\n', '\n');
    final thinkStart = normalized.indexOf('<think>');
    if (thinkStart < 0) {
      return _ParsedLocalModelOutput(
        reasoning: '',
        answer: _sanitizeLocalAnswer(_stripDanglingThinkTag(normalized)),
      );
    }

    final reasoningStart = thinkStart + '<think>'.length;
    final thinkEnd = normalized.indexOf('</think>', reasoningStart);
    if (thinkEnd < 0) {
      return _ParsedLocalModelOutput(
        reasoning: normalized.substring(reasoningStart).trim(),
        answer: '',
      );
    }

    final reasoning = normalized.substring(reasoningStart, thinkEnd).trim();
    final answer = normalized.substring(thinkEnd + '</think>'.length);
    return _ParsedLocalModelOutput(
      reasoning: reasoning,
      answer: _sanitizeLocalAnswer(_stripDanglingThinkTag(answer)),
    );
  }

  String _stripDanglingThinkTag(String value) {
    var cleaned = value;
    const danglingPrefixes = ['<think>', '<thin', '<thi', '<th', '<t', '<'];
    for (final prefix in danglingPrefixes) {
      if (cleaned.trimLeft().startsWith(prefix)) {
        final index = cleaned.indexOf(prefix);
        cleaned = index >= 0 ? cleaned.substring(0, index) : cleaned;
        break;
      }
    }
    return cleaned.replaceAll('</think>', '');
  }

  bool _shouldLockAnswer(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return false;
    }
    if (trimmed.length >= 8 &&
        (trimmed.endsWith('。') ||
            trimmed.endsWith('！') ||
            trimmed.endsWith('？') ||
            trimmed.endsWith('.') ||
            trimmed.endsWith('!') ||
            trimmed.endsWith('?'))) {
      return true;
    }
    return false;
  }

  String _truncateLocalAnswer(String value) {
    var cleaned = _sanitizeLocalAnswer(_stripDanglingThinkTag(value));
    final newlineIndex = cleaned.indexOf('\n');
    if (newlineIndex >= 0) {
      cleaned = cleaned.substring(0, newlineIndex);
    }
    final sentenceParts = cleaned.split(RegExp(r'[。！？!?]'));
    if (sentenceParts.isEmpty) {
      return cleaned.trim();
    }
    final first = sentenceParts.first.trim();
    if (first.isNotEmpty) {
      return first.endsWith('。') ||
              first.endsWith('！') ||
              first.endsWith('？') ||
              first.endsWith('!') ||
              first.endsWith('?')
          ? first
          : '$first。';
    }
    return cleaned.trim();
  }

  String _summarizeEvidence(Map<String, dynamic> metaInfo) {
    final parts = _collectStrings(metaInfo)
        .take(4)
        .map((item) => item.trim())
        .where((item) => item.isNotEmpty)
        .toList();
    if (parts.isEmpty) {
      return '';
    }
    final summary = parts.join('；');
    return summary.length > 120 ? summary.substring(0, 120) : summary;
  }

  String _joinList(dynamic rawList) {
    if (rawList is! List) {
      return '';
    }
    return rawList
        .map((item) => item.toString().trim())
        .where((item) => item.isNotEmpty)
        .join('、');
  }

  Map<String, dynamic> _toMap(dynamic value) {
    if (value is Map<String, dynamic>) {
      return value;
    }
    if (value is Map) {
      return value.map((key, item) => MapEntry(key.toString(), item));
    }
    return const <String, dynamic>{};
  }

  List<String> _collectStrings(dynamic value) {
    if (value == null) {
      return const [];
    }
    if (value is String) {
      final trimmed = value.trim();
      return trimmed.isEmpty ? const [] : [trimmed];
    }
    if (value is num || value is bool) {
      return [value.toString()];
    }
    if (value is List) {
      return value.expand(_collectStrings).toList();
    }
    if (value is Map) {
      return value.values.expand(_collectStrings).toList();
    }
    return const [];
  }
}
