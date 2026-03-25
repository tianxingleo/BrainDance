import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/atom-one-dark.dart';
import 'package:flutter_highlight/themes/atom-one-light.dart';
import '../configs/app_config.dart';
import '../configs/supabase_config.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../extra_func/dir_and_file.dart';
import '../main.dart'
    show
        overviewStatsProvider,
        overviewLocalIndexingProvider,
        pageIndexProvider;
import '../configs/motion_tokens.dart';
import '../services/agent_recall_service.dart';
import '../services/local_rag_index.dart';
import '../services/local_model_catalog_service.dart';
import '../services/download_event_bus.dart';
import '../services/viewer_navigation.dart';
import '../widgets/bd_surfaces.dart';
import 'community/composer_sheet.dart';
import 'community/models.dart';
import 'community/repository.dart';
import 'recall/empty_states.dart';
import 'recall/model_grid.dart';
import 'recall/model_detail_sheet.dart';
import 'recall/processing_section.dart';
import 'recall/rename_model_dialog.dart';
import 'recall/search_header_section.dart';
import 'recall/search_mode.dart';

class RecallPage extends ConsumerStatefulWidget {
  const RecallPage({super.key});

  @override
  ConsumerState<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends ConsumerState<RecallPage> {
  static const String _defaultModelFileName = 'qwen3-1.7b.gguf';
  static const String _localModelPathPrefKey = 'recall.local_llm_model_path';
  static const String _localModelUrlPrefKey = 'recall.local_llm_model_url';
  final LocalModelCatalogService _localModelCatalogService =
      const LocalModelCatalogService();

  List<Map<String, dynamic>> _models = [];
  List<Map<String, dynamic>> _allModels = [];
  List<Map<String, dynamic>> _processingTasks = [];
  Map<String, List<String>> _taskAllLogs = {}; // taskId -> all log msgs
  final Set<String> _expandedTaskLogs = {}; // 展开的任务ID集合
  bool _isLoading = true;
  bool _isLocalIndexing = false;
  bool _isProcessingExpanded = false;
  final TextEditingController _searchController = TextEditingController();
  final TextEditingController _localModelPathController =
      TextEditingController();
  late final TextEditingController _localModelUrlController;
  final LocalRagIndexService _localRagIndex = LocalRagIndexService();
  List<LocalModelCatalogItem> _localModelCatalog = const [];
  final Map<String, String> _downloadedLocalModelPathsByUrl = {};
  String? _selectedLocalModelUrl;
  String? _activeLocalModelUrl;
  RealtimeChannel? _realtimeChannel;
  Timer? _modelPollingTimer;
  LocalRagIndexStats? _indexStats;
  RecallSearchMode _searchMode = RecallSearchMode.cloud;
  LlamaEngine? _localQnaModel;
  StreamSubscription<dynamic>? _llamaStreamSubscription;
  String _localAnswer = '';
  String _localReasoning = '';
  String _localAnswerStatus = 'Qwen3-1.7B 端侧模型未加载';
  String _localContextPreview = '';
  bool _isLocalModelLoading = false;
  bool _isLocalModelReady = false;
  bool _isModelDownloading = false;
  double? _modelDownloadProgress;
  int _modelDownloadedBytes = 0;
  int? _modelDownloadTotalBytes;
  final Map<String, GlobalKey> _modelCardKeys = {};
  final Map<String, _RecallSearchCacheEntry> _searchCache = {};
  final GlobalKey _actionOverlayStackKey = GlobalKey();
  final GlobalKey<RecallModelActionOverlayState> _overlayKey = GlobalKey();
  Map<String, dynamic>? _activeModelAction;
  Rect? _activeModelActionRect;
  bool _didBootstrap = false;
  bool _didFinishInitialModelLoad = false;
  bool _isTabActive = true;
  bool _shouldRefreshProcessingOnResume = false;
  bool _isModelPollingInFlight = false;
  int _searchRequestId = 0;
  String? _lastSearchKey;
  String _lastOwnModelSignature = '';
  bool _isAgentSearching = false;
  AgentRecallResponse? _agentResult;
  ChatMessage? _agentChatMessage;
  final ScrollController _recallScrollController = ScrollController();
  StreamSubscription<String>? _agentStreamSubscription;

  void _stopAgentSearch() {
    if (_isAgentSearching) {
      _agentStreamSubscription?.cancel();
      setState(() {
        _isAgentSearching = false;
        _agentChatMessage?.addStep(
          AgentStep(type: 'error', content: '🚫 用户已强行中断 Agent'),
        );
      });
    }
  }

  String get _defaultModelDownloadUrl => SupabaseConfig.localModelUrl;

  @override
  void initState() {
    super.initState();
    _localModelUrlController = TextEditingController(
      text: _defaultModelDownloadUrl,
    );
    _localModelUrlController.addListener(_handleLocalModelUrlChanged);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _bootstrapPage();
    });
  }

  @override
  void dispose() {
    _modelPollingTimer?.cancel();
    _searchController.dispose();
    _localModelPathController.dispose();
    _localModelUrlController.removeListener(_handleLocalModelUrlChanged);
    _localModelUrlController.dispose();
    _realtimeChannel?.unsubscribe();
    unawaited(_disposeLocalQnaModel());
    super.dispose();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
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
    final savedPath = prefs.getString(_localModelPathPrefKey)?.trim();
    final savedUrl = prefs.getString(_localModelUrlPrefKey)?.trim();
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
    await prefs.setString(_localModelPathPrefKey, modelPath);
  }

  Future<void> _persistLocalModelUrl(String modelUrl) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_localModelUrlPrefKey, modelUrl);
  }

  Future<String> _getPrivateModelPathForUrl(String? modelUrl) async {
    final dir = await getApplicationDocumentsDirectory();
    var fileName = _defaultModelFileName;
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
      TDToast.showText(context: context, '请先填写模型下载链接');
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
      TDToast.showText(context: context, '模型已下载到应用私有目录');
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isModelDownloading = false;
        _modelDownloadProgress = null;
        _modelDownloadedBytes = 0;
        _modelDownloadTotalBytes = null;
        _localAnswerStatus = '模型下载失败：$e';
      });
      TDToast.showText(context: context, '模型下载失败：$e');
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
        path.basename(modelPath.isEmpty ? _defaultModelFileName : modelPath);
    if (modelPath.isEmpty) {
      TDToast.showText(context: context, '请先填写 GGUF 模型路径');
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
        throw Exception('模型文件不存在：$modelPath');
      }
      final modelSize = await modelFile.length();
      if (modelSize < 100 * 1024 * 1024) {
        throw Exception(
          '模型文件体积异常：${(modelSize / 1024 / 1024).toStringAsFixed(1)} MB，可能不是完整 GGUF',
        );
      }

      final llama = LlamaEngine(LlamaBackend());
      var backendSummary = 'CPU';
      try {
        await llama.loadModel(
          modelPath,
          modelParams: const ModelParams(
            contextSize: 2048,
            gpuLayers: 24,
            preferredBackend: GpuBackend.vulkan,
            numberOfThreads: 4,
            numberOfThreadsBatch: 4,
            batchSize: 256,
            microBatchSize: 256,
          ),
        );
        final backendName = await llama.getBackendName();
        backendSummary = '$backendName (GPU 优先)';
      } catch (_) {
        await llama.dispose();
        final fallbackLlama = LlamaEngine(LlamaBackend());
        await fallbackLlama.loadModel(
          modelPath,
          modelParams: const ModelParams(
            contextSize: 2048,
            gpuLayers: 0,
            preferredBackend: GpuBackend.cpu,
            numberOfThreads: 4,
            numberOfThreadsBatch: 4,
            batchSize: 256,
            microBatchSize: 256,
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
        _localAnswerStatus = '模型加载失败：$e';
      });
      TDToast.showText(context: context, '端侧模型加载失败：$e');
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
      '5. 不复述问题，不解释规则，不说‘根据给定证据’。';

  Future<void> _askLocalQuestion({String? question}) async {
    final userQuestion = (question ?? '').trim();
    if (userQuestion.isEmpty) {
      TDToast.showText(context: context, '请输入要提问的问题');
      return;
    }
    if (_localQnaModel == null || !_isLocalModelReady) {
      TDToast.showText(context: context, '请先加载本地 GGUF 模型');
      return;
    }

    // 1. 构建符合 Qwen3-1.7B-Instruct 格式的 retrieval payload
    final retrieval = await _buildRetrievalPayload(userQuestion);
    final userPayload = jsonEncode({
      'question': userQuestion,
      'retrieval': retrieval,
    });

    // 2. 构建 ChatML 格式 Prompt
    // 注意：微调后的模型对 System Prompt 和 JSON Payload 格式非常敏感
    final prompt =
        '<|im_start|>system\n$_kSystemPrompt<|im_end|>\n'
        '<|im_start|>user\n$userPayload<|im_end|>\n'
        '<|im_start|>assistant\n'
        '请直接给出最终回答；如果你仍然生成 <think> 思考链，系统会将其与正式回答分离，仅正式回答会作为最终结果展示。\n';

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
              maxTokens: 384, // 给 <think> + 正式回答留出更充足的联合预算，降低被截断概率
              temp: 0.1, // 接近 Greedy Search，减少幻觉
              topK: 20,
              topP: 0.1, // 进一步限制采样范围
              penalty: 1.05, // 对齐 Python 脚本的 repetition_penalty=1.05
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
              final nextAnswer = parsedOutput.answer;
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
      setState(() {
        _localAnswerStatus = '端侧问答失败：$e';
      });
      TDToast.showText(context: context, '端侧问答失败：$e');
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
      final fallbackModels = (_models.isNotEmpty ? _models : _allModels)
          .take(3)
          .map((item) => Map<String, dynamic>.from(item))
          .toList();
      matches = fallbackModels;
    }

    // 转换为微调模型预期的 evidence 格式
    final evidence = matches.map((item) {
      final metaInfo = _toMap(item['meta_info']);
      final tags = _joinList(item['tags']);
      final objects = _joinList(item['objects']);
      final summary = _collectStrings(
        metaInfo,
      ).take(6).map((t) => t.trim()).where((t) => t.isNotEmpty).join('；');

      return {
        'id': item['id']?.toString() ?? '',
        'created_at': item['created_at']?.toString() ?? '',
        'description': item['description']?.toString() ?? '',
        'tags': tags,
        'objects': objects,
        'summary': summary,
        'scene_id': item['scene_id']?.toString() ?? '',
      };
    }).toList();

    return {
      'evidence': evidence,
      'hit_count': evidence.length,
      // 本地暂无 Intent 识别模型，先默认为 unknown 或根据是否有结果判断
      'intent': evidence.isEmpty ? 'unknown' : 'object_lookup',
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
      final logsJson = newData['logs'] as List<dynamic>?;
      final allLogs = _parseAllLogMsgs(logsJson);
      if (mounted) {
        setState(() {
          // 移除旧版本（如果存在）
          _processingTasks.removeWhere((t) => t['id'].toString() == taskId);
          // 添加更新后的任务
          _processingTasks.add(Map<String, dynamic>.from(newData));
          if (allLogs.isNotEmpty) {
            _taskAllLogs[taskId] = allLogs;
          }
        });
      }
    } else if (status != 'processing' && oldData['status'] == 'processing') {
      // 任务从 processing 变为其他状态，移除
      if (mounted) {
        setState(() {
          _processingTasks.removeWhere((t) => t['id'].toString() == taskId);
          _taskAllLogs.remove(taskId);
          _expandedTaskLogs.remove(taskId);
        });
      }
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
      final response = await Supabase.instance.client
          .from('model_assets')
          .select(
            'id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info, created_at',
          )
          .order('created_at', ascending: false);

      final models = List<Map<String, dynamic>>.from(response);

      // 从 processing_tasks 获取 display_name 并合并
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
            final dn = t['display_name']?.toString();
            if (dn != null && dn.isNotEmpty) {
              displayNameMap[t['scene_id'].toString()] = dn;
            }
          }
          for (final m in models) {
            final sid = m['scene_id']?.toString();
            if (sid != null && displayNameMap.containsKey(sid)) {
              m['display_name'] = displayNameMap[sid];
            }
          }
        }
      } catch (_) {
        // display_name 获取失败不影响主流程
      }

      if (models.isEmpty) {
        models.add(_buildDemoModel());
      }

      if (mounted) {
        final ownModelSignature = _buildModelSignature(
          _extractOwnModels(models),
        );
        setState(() {
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
          setState(() {
            _isLoading = false;
          });
        }
        return false;
      }

      final demoModels = [_buildDemoModel()];
      if (mounted) {
        final ownModelSignature = _buildModelSignature(
          _extractOwnModels(demoModels),
        );
        setState(() {
          _allModels = demoModels;
          _models = demoModels;
          _didFinishInitialModelLoad = true;
          _isLoading = false;
          _lastOwnModelSignature = ownModelSignature;
        });
        _updateOverviewProvider();
        if (showErrorToast) {
          TDToast.showText(
            '${textLocalize('recall_error_offline')} [${SupabaseConfig.modeLabel}] $e',
            context: context,
          );
        }
      }
      _searchCache.clear();
      _lastSearchKey = null;
      await _syncLocalIndex(demoModels);
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
      body: Stack(
        key: _actionOverlayStackKey,
        children: [
          BDPageBackdrop(
            child: SafeArea(
              child: CustomScrollView(
                controller: _recallScrollController,
                cacheExtent: 1200,
                slivers: [
                  SliverToBoxAdapter(
                    child: Column(
                      children: [
                        BDPageHeader(
                          title: textLocalize("home_page"),
                          padding: const EdgeInsets.fromLTRB(20, 16, 20, 4),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
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
                                  unawaited(_refreshModelsForCurrentState());
                                },
                              ),
                            ],
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.fromLTRB(20, 2, 20, 8),
                          child: RecallSearchHeaderSection(
                            theme: theme,
                            isDark: isDark,
                            textColor: textColor,
                            darkInput: darkInput,
                            searchController: _searchController,
                            searchMode: _searchMode,
                            searchModeTitleBuilder: _searchModeTitle,
                            searchModeSubtitleBuilder: _searchModeSubtitle,
                            searchFieldHint: _searchFieldHint(),
                            onSubmit: _handleSearchSubmitted,
                            onChanged: _searchModels,
                            onClear: () {
                              _searchController.clear();
                              unawaited(_searchModels(''));
                            },
                            onTapSearchMode: _showSearchModeSheet,
                            isLocalModelReady: _isLocalModelReady,
                            isModelDownloading: _isModelDownloading,
                            isLocalModelLoading: _isLocalModelLoading,
                            modelDownloadProgress: _modelDownloadProgress,
                            modelDownloadedBytes: _modelDownloadedBytes,
                            modelDownloadTotalBytes: _modelDownloadTotalBytes,
                            localAnswer: _localAnswer,
                            localReasoning: _localReasoning,
                            localAnswerStatus: _localAnswerStatus,
                            localContextPreview: _localContextPreview,
                            defaultModelDownloadUrl: _defaultModelDownloadUrl,
                            localModelCatalog: _localModelCatalog,
                            selectedLocalModelUrl: _selectedLocalModelUrl,
                            activeLocalModelUrl: _activeLocalModelUrl,
                            downloadedLocalModelUrls:
                                _downloadedLocalModelPathsByUrl.keys.toSet(),
                            localModelUrlController: _localModelUrlController,
                            localModelPathController: _localModelPathController,
                            onSelectCatalogModel: (value) {
                              unawaited(_selectCatalogModel(value));
                            },
                            onDownloadModel: _downloadModelToPrivateDir,
                            onLoadModel: _loadLocalQnaModel,
                          ),
                        ),
                        if (_searchMode == RecallSearchMode.agent)
                          Padding(
                            padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
                            child: _buildAgentResultCard(isDark, textColor),
                          ),
                        if (_processingTasks.isNotEmpty)
                          RepaintBoundary(
                            child: RecallProcessingSection(
                              theme: theme,
                              isDark: isDark,
                              textColor: textColor,
                              darkInput: darkInput,
                              isExpanded: _isProcessingExpanded,
                              processingTasks: _processingTasks,
                              taskAllLogs: _taskAllLogs,
                              expandedTaskLogs: _expandedTaskLogs,
                              onToggleExpanded: () {
                                setState(() {
                                  _isProcessingExpanded =
                                      !_isProcessingExpanded;
                                });
                              },
                              onToggleTaskLogs: (taskId) {
                                setState(() {
                                  if (_expandedTaskLogs.contains(taskId)) {
                                    _expandedTaskLogs.remove(taskId);
                                  } else {
                                    _expandedTaskLogs.add(taskId);
                                  }
                                });
                              },
                            ),
                          ),
                      ],
                    ),
                  ),
                  if (_isLoading)
                    const SliverFillRemaining(
                      hasScrollBody: false,
                      child: Padding(
                        padding: EdgeInsets.symmetric(vertical: 96.0),
                        child: Center(
                          child: TDLoading(
                            size: TDLoadingSize.large,
                            icon: TDLoadingIcon.circle,
                          ),
                        ),
                      ),
                    )
                  else if (_models.isEmpty)
                    SliverFillRemaining(
                      hasScrollBody: false,
                      child: Padding(
                        padding: const EdgeInsets.only(top: 16.0),
                        child: _searchController.text.trim().isEmpty
                            ? RecallEmptyState(
                                theme: theme,
                                isDark: isDark,
                                darkCard: darkCard,
                                darkBorder: darkBorder,
                              )
                            : RecallSearchEmptyState(
                                theme: theme,
                                isDark: isDark,
                                darkCard: darkCard,
                                darkBorder: darkBorder,
                                searchMode: _searchMode,
                                searchModeTitleBuilder: _searchModeTitle,
                              ),
                      ),
                    )
                  else if (_models.isNotEmpty &&
                      _models.first.containsKey('matched_frames'))
                    RecallModelGrid(
                      theme: theme,
                      isDark: isDark,
                      darkCard: darkCard,
                      darkInput: darkInput,
                      models: _models,
                      activeModelAction: _activeModelAction,
                      modelCardKeyFor: _modelCardKeyFor,
                      isSameModel: _isSameModel,
                      onNavigateToViewer: _navigateToViewer,
                      toPublicUrl: _toPublicUrl,
                      onShowModelActions: (model, {bool imageOnly = false}) {
                        _showModelActions(model, imageOnly: imageOnly);
                      },
                    )
                  else
                    TimePeelingList(
                      theme: theme,
                      isDark: isDark,
                      darkCard: darkCard,
                      darkInput: darkInput,
                      groupedModels: _groupModelsByName(_models),
                      activeModelAction: _activeModelAction,
                      modelCardKeyFor: _modelCardKeyFor,
                      isSameModel: _isSameModel,
                      onNavigateToViewer: _navigateToViewer,
                      onShowModelActions: (model, {bool imageOnly = false}) {
                        _showModelActions(model, imageOnly: imageOnly);
                      },
                      onAddNewTask: (name) {
                        ref.read(pageIndexProvider.notifier).state = 1;
                      },
                    ),
                  const SliverToBoxAdapter(child: SizedBox(height: 96)),
                ],
              ),
            ),
          ),
          if (_activeModelAction != null && _activeModelActionRect != null)
            RecallModelActionOverlay(
              key: _overlayKey,
              theme: theme,
              isDark: isDark,
              darkCard: darkCard,
              darkInput: darkInput,
              model: _activeModelAction!,
              rect: _activeModelActionRect!,
              toPublicUrl: _toPublicUrl,
              onDismiss: _dismissModelActions,
              onNavigateToViewer: _navigateToViewer,
              onShowModelDetails: _showModelDetails,
              onDownloadModel: _downloadRecallModel,
              onShareModelToCommunity: _shareModelToCommunity,
              onRenameModel: _renameModel,
              onDeleteCloudModel: _deleteCloudModel,
            ),
        ],
      ),
    );
  }

  Future<void> _searchModels(String query) async {
    final normalizedQuery = query.trim();
    if (normalizedQuery.isEmpty) {
      _lastSearchKey = null;
      if (!mounted) return;
      setState(() {
        _models = List<Map<String, dynamic>>.from(_allModels);
        if (_searchMode == RecallSearchMode.localAi) {
          _localAnswer = '';
          _localReasoning = '';
          _localContextPreview = '';
        }
        if (_searchMode == RecallSearchMode.agent) {
          _agentResult = null;
          _agentChatMessage = null;
        }
        _isLoading = false;
      });
      return;
    }

    // Agent 模式不做即时列表检索，仅在 submit 时调用 _askAgentRecall
    if (_searchMode == RecallSearchMode.agent) {
      return;
    }

    final cacheKey = '${_searchMode.name}:$normalizedQuery';
    final now = DateTime.now();
    final cached = _searchCache[cacheKey];
    if (cached != null &&
        now.difference(cached.createdAt) < const Duration(minutes: 2)) {
      _lastSearchKey = cacheKey;
      if (!mounted) return;
      setState(() {
        _models = cached.results
            .map((item) => Map<String, dynamic>.from(item))
            .toList();
        _isLoading = false;
      });
      return;
    }

    if (_lastSearchKey == cacheKey && !_isLoading) {
      return;
    }

    final requestId = ++_searchRequestId;
    _lastSearchKey = cacheKey;
    setState(() {
      _isLoading = true;
    });

    try {
      final results = _usesLocalIndex(_searchMode)
          ? await _localRagIndex.search(normalizedQuery)
          : await _searchModelsFromCloud(normalizedQuery);
      if (!mounted || requestId != _searchRequestId) return;
      _searchCache[cacheKey] = _RecallSearchCacheEntry(
        createdAt: now,
        results: results
            .map((item) => Map<String, dynamic>.from(item))
            .toList(),
      );
      if (_searchCache.length > 24) {
        final oldestKey = _searchCache.entries.reduce((left, right) {
          return left.value.createdAt.isBefore(right.value.createdAt)
              ? left
              : right;
        }).key;
        _searchCache.remove(oldestKey);
      }
      setState(() {
        _models = results;
        _isLoading = false;
      });
    } catch (e) {
      if (mounted && requestId == _searchRequestId) {
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

    final errMsg = (data is Map)
        ? (data['error'] ?? textLocalize('recall_unknown_error'))
        : textLocalize('recall_server_error');
    throw Exception(errMsg);
  }

  Future<void> _askAgentRecall(String query) async {
    if (query.isEmpty) return;
    setState(() {
      _isAgentSearching = true;
      _agentResult = null;
      _agentChatMessage = ChatMessage(isUser: false);
    });

    _agentStreamSubscription?.cancel();

    void fallback() async {
      if (!mounted) return;
      setState(() {
        _isAgentSearching = true;
      });
      try {
        final result = await AgentRecallService().query(query);
        if (!mounted) return;
        setState(() {
          _agentResult = result;
          _agentChatMessage!.finalAnswer = result.answer;
          _isAgentSearching = false;
        });

        if (_recallScrollController.hasClients) {
          _recallScrollController.animateTo(
            _recallScrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        }
      } catch (ex) {
        if (!mounted) return;
        setState(() {
          _isAgentSearching = false;
        });
        TDToast.showText('Agent 检索失败：$ex', context: context);
      }
    }

    try {
      final stream = AgentRecallService().queryStream(query);
      _agentStreamSubscription = stream.listen(
        (chunk) {
          if (!mounted) return;
          if (chunk.isEmpty) return;

          try {
            final data = jsonDecode(chunk);
            final event = data['event']?.toString() ?? '';
            final payload = data['data'];

            if (event == 'plan' && payload is Map) {
              final title = payload['title']?.toString() ?? '';
              final stepsStr = (payload['steps'] as List?)?.join('\n') ?? '';
              _agentChatMessage!.addStep(
                AgentStep(type: 'thought', content: 'Plan: $title\n$stepsStr'),
              );
            } else if (event == 'thinking' || event == 'thought') {
              final content = payload is Map ? payload['content']?.toString() ?? '' : payload?.toString() ?? '';
              _agentChatMessage!.addStep(
                AgentStep(type: 'thought', content: content),
              );
            } else if (event == 'tool_call' && payload is Map) {
              final argsStr = payload['args'] is Map
                  ? jsonEncode(payload['args'])
                  : payload['args']?.toString() ?? '';
              _agentChatMessage!.addStep(
                AgentStep(
                  type: 'tool_call',
                  toolName: payload['name']?.toString() ?? '',
                  content: argsStr,
                ),
              );
            } else if (event == 'tool_result' && payload is Map) {
              final name = payload['name']?.toString() ?? '';
              var lastTool = _agentChatMessage!.steps.lastWhere(
                (s) => s.type == 'tool_call' && s.toolName == name,
                orElse: () => AgentStep(
                  type: 'tool_call',
                  toolName: name,
                  content: '',
                ),
              );
              lastTool.isCompleted = true;
            } else if (event == 'message' && payload is Map) {
              _agentChatMessage!.finalAnswer += payload['delta']?.toString() ?? '';
            } else if (event == 'error' && payload is Map) {
              _agentChatMessage!.addStep(
                AgentStep(
                  type: 'error',
                  content: payload['message']?.toString() ?? 'Unknown error',
                ),
              );
            } else if (event == 'done') {
              if (payload != null && payload is Map) {
                setState(() {
                  _agentResult = AgentRecallResponse.fromJson(Map<String, dynamic>.from(payload));
                });
              } else if (data['result'] != null && data['result'] is Map) {
                // 回退兼容旧格式
                setState(() {
                  _agentResult = AgentRecallResponse.fromJson(Map<String, dynamic>.from(data['result']));
                });
              }
              setState(() {
                _isAgentSearching = false;
              });
            }

            if (_recallScrollController.hasClients) {
              _recallScrollController.animateTo(
                _recallScrollController.position.maxScrollExtent,
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeOut,
              );
            }
          } catch (e) {
            debugPrint('Error parsing chunk: $e');
          }
        },
        onError: (e) {
          if (!mounted) return;
          setState(() {
            _isAgentSearching = false;
          });
          TDToast.showText('Agent 检索流式失败 (尝试回退)：$e', context: context);
          fallback();
        },
        onDone: () {
          if (mounted) {
            setState(() {
              _isAgentSearching = false;
            });
          }
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isAgentSearching = false;
      });
      TDToast.showText('Agent 流启动失败 (尝试回退)：$e', context: context);
      fallback();
    }
  }

  void _openAgentRecallResult(AgentRecallResponse result) {
    final openScene = result.actions
        .where((a) => a.type == 'open_scene')
        .cast<AgentAction?>()
        .firstOrNull;
    final flyToPose = result.actions
        .where((a) => a.type == 'fly_to_pose')
        .cast<AgentAction?>()
        .firstOrNull;

    if (openScene == null || openScene.ply == null || openScene.ply!.isEmpty) {
      TDToast.showText('缺少 open_scene.ply，无法打开 Viewer', context: context);
      return;
    }

    // ply 可能是 Storage 相对路径，需转为公开 URL 才能下载
    final rawPly = openScene.ply!;
    final modelUrl =
        rawPly.startsWith('http://') || rawPly.startsWith('https://')
        ? rawPly
        : toPublicUrl(rawPly);
    final posesUrlResolved =
        openScene.poses != null &&
            openScene.poses!.isNotEmpty &&
            !openScene.poses!.startsWith('http')
        ? toPublicUrl(openScene.poses!)
        : openScene.poses ?? toPosesUrl(rawPly);

    unawaited(
      openViewer(
        context,
        initialModelUrl: modelUrl,
        posesUrl: posesUrlResolved,
        sceneId: openScene.sceneId,
        initialPose: flyToPose?.matrix,
        initialPoseId: flyToPose?.imageName,
      ),
    );
  }

  Future<void> _handleSearchSubmitted(String value) async {
    final query = value.trim();
    if (_searchMode == RecallSearchMode.localAi) {
      await _searchModels(query);
      if (query.isNotEmpty) {
        await _askLocalQuestion(question: query);
      }
      return;
    }
    if (_searchMode == RecallSearchMode.agent) {
      if (query.isNotEmpty) {
        await _askAgentRecall(query);
      }
      return;
    }
    await _searchModels(query);
  }

  bool _usesLocalIndex(RecallSearchMode mode) {
    return mode == RecallSearchMode.local || mode == RecallSearchMode.localAi;
  }

  String _searchModeTitle(RecallSearchMode mode) {
    switch (mode) {
      case RecallSearchMode.cloud:
        return textLocalize('recall_cloud_rag');
      case RecallSearchMode.local:
        return textLocalize('recall_local_rag');
      case RecallSearchMode.localAi:
        return textLocalize('recall_local_ai_rag');
      case RecallSearchMode.agent:
        return 'Agent 检索';
    }
  }

  String _searchModeSubtitle(RecallSearchMode mode) {
    switch (mode) {
      case RecallSearchMode.cloud:
        return textLocalize('recall_cloud_scope');
      case RecallSearchMode.local:
        if (_isLocalIndexing) {
          return textLocalize('recall_local_indexing');
        }
        final base =
            '${textLocalize('recall_local_ready')} · ${textLocalize('recall_local_scope')}';
        if (_indexStats == null) {
          return base;
        }
        return '$base · ${_indexStats!.rebuiltItems}/${_indexStats!.totalItems}';
      case RecallSearchMode.localAi:
        return textLocalize('recall_local_ai_scope');
      case RecallSearchMode.agent:
        return '空间检索 Agent · 直接带你去看';
    }
  }

  String _searchFieldHint() {
    if (_searchMode == RecallSearchMode.localAi) {
      return textLocalize('recall_local_ai_hint');
    }
    if (_searchMode == RecallSearchMode.agent) {
      return '输入空间问题，例如"厨房在哪里"';
    }
    return textLocalize('recall_search_hint');
  }

  void _setSearchMode(RecallSearchMode mode) {
    if (_searchMode == mode) {
      return;
    }
    setState(() {
      _searchMode = mode;
      if (mode != RecallSearchMode.localAi) {
        _localAnswer = '';
        _localReasoning = '';
        _localContextPreview = '';
      }
      if (mode != RecallSearchMode.agent) {
        _agentResult = null;
        _agentChatMessage = null;
      }
    });
    final keyword = _searchController.text.trim();
    if (keyword.isNotEmpty) {
      unawaited(_searchModels(keyword));
    }
  }

  Future<void> _showSearchModeSheet() async {
    final selected = await showModalBottomSheet<RecallSearchMode>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return RecallSearchModeSheet(
          selectedMode: _searchMode,
          titleBuilder: _searchModeTitle,
          subtitleBuilder: _searchModeSubtitle,
          darkInput: darkInput,
          onSelect: (mode) => Navigator.pop(context, mode),
        );
      },
    );

    if (selected != null) {
      _setSearchMode(selected);
    }
  }

  Widget _buildAgentResultCard(bool isDark, Color textColor) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    if (_agentChatMessage == null &&
        _agentResult == null &&
        !_isAgentSearching) {
      return BDPanelCard(
        padding: const EdgeInsets.all(16),
        child: Text(
          '输入问题后按回车，Agent 将为你检索空间并定位视角。',
          style: TextStyle(color: hintColor, fontSize: 13),
        ),
      );
    }

    final hasActions =
        _agentResult != null &&
        _agentResult!.actions.any((a) => a.type == 'open_scene');

    return BDPanelCard(
      padding: const EdgeInsets.all(16),
      child: ListenableBuilder(
        listenable: _agentChatMessage!,
        builder: (context, _) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.travel_explore_rounded,
                    size: 18,
                    color: BDDesign.colorMutedBlue,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'Agent',
                    style: TextStyle(
                      color: textColor,
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  if (_isAgentSearching) ...[
                    const SizedBox(width: 12),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    const Spacer(),
                    SizedBox(
                      height: 24,
                      child: OutlinedButton.icon(
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.red,
                          side: const BorderSide(color: Colors.red, width: 1),
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                        ),
                        onPressed: _stopAgentSearch,
                        icon: const Icon(Icons.stop_circle_outlined, size: 14),
                        label: const Text('停止', style: TextStyle(fontSize: 12)),
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 10),

              ..._agentChatMessage!.steps.map((step) {
                return ListenableBuilder(
                  listenable: step,
                  builder: (context, _) {
                    if (step.type == 'tool_call') {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 8.0),
                        child: Theme(
                          data: Theme.of(
                            context,
                          ).copyWith(dividerColor: Colors.transparent),
                          child: _AgentStepTile(
                            step: step,
                            isDark: isDark,
                            textColor: textColor,
                          ),
                        ),
                      );
                    } else if (step.type == 'thought') {
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4.0),
                        child: Text(
                          '🤔 思考中: ${step.content}',
                          style: TextStyle(color: hintColor, fontSize: 13),
                        ),
                      );
                    } else if (step.type == 'error') {
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4.0),
                        child: Row(
                          children: [
                            const Icon(
                              Icons.error_outline,
                              color: Colors.red,
                              size: 16,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                step.content,
                                style: const TextStyle(
                                  color: Colors.red,
                                  fontSize: 13,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    }
                    return const SizedBox.shrink();
                  },
                );
              }),

              if (_agentChatMessage!.finalAnswer.isNotEmpty) ...[
                const SizedBox(height: 8),
                MarkdownBody(
                  data: _agentChatMessage!.finalAnswer,
                  styleSheet: MarkdownStyleSheet(
                    p: TextStyle(color: textColor, fontSize: 14, height: 1.5),
                  ),
                ),
              ],

              if (_agentResult?.evidence != null) ...[
                const SizedBox(height: 10),
                Text(
                  '场景：${_agentResult!.evidence!.sceneId}  ·  相似度：${(_agentResult!.evidence!.similarity * 100).toStringAsFixed(1)}%',
                  style: TextStyle(color: hintColor, fontSize: 12),
                ),
              ],

              if (hasActions) ...[
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  height: 40,
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: BDDesign.colorMutedBlue,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      elevation: 0,
                    ),
                    icon: const Icon(Icons.open_in_new_rounded, size: 16),
                    label: const Text('打开场景', style: TextStyle(fontSize: 14)),
                    onPressed: () => _openAgentRecallResult(_agentResult!),
                  ),
                ),
              ],
            ],
          );
        },
      ),
    );
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

class _RecallSearchCacheEntry {
  const _RecallSearchCacheEntry({
    required this.createdAt,
    required this.results,
  });

  final DateTime createdAt;
  final List<Map<String, dynamic>> results;
}

class _ParsedLocalModelOutput {
  const _ParsedLocalModelOutput({
    required this.reasoning,
    required this.answer,
  });

  final String reasoning;
  final String answer;
}

class _AgentStepTile extends StatefulWidget {
  final AgentStep step;
  final bool isDark;
  final Color textColor;

  const _AgentStepTile({
    required this.step,
    required this.isDark,
    required this.textColor,
  });

  @override
  State<_AgentStepTile> createState() => _AgentStepTileState();
}

class _AgentStepTileState extends State<_AgentStepTile>
    with AutomaticKeepAliveClientMixin {
  final ExpansionTileController _controller = ExpansionTileController();
  bool _wasCompleted = false;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _wasCompleted = widget.step.isCompleted;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!widget.step.isCompleted && mounted) {
        _controller.expand();
      }
    });

    widget.step.addListener(_handleStepChange);
  }

  @override
  void dispose() {
    widget.step.removeListener(_handleStepChange);
    super.dispose();
  }

  void _handleStepChange() {
    if (widget.step.isCompleted && !_wasCompleted) {
      _wasCompleted = true;
      Future.delayed(const Duration(milliseconds: 500), () {
        if (mounted) {
          _controller.collapse();
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return ExpansionTile(
      controller: _controller,
      tilePadding: EdgeInsets.zero,
      minTileHeight: 0,
      leading: widget.step.isCompleted
          ? const Icon(Icons.check_circle, color: Colors.green, size: 20)
          : const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
      title: Text(
        'Using ${widget.step.toolName}...',
        style: TextStyle(fontSize: 13, color: widget.textColor),
      ),
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(8),
          margin: const EdgeInsets.only(top: 4),
          decoration: BoxDecoration(
            color: widget.isDark ? const Color(0xFF1E1E1E) : Colors.grey[100],
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: widget.isDark ? Colors.white10 : Colors.black12,
            ),
          ),
          child: HighlightView(
            widget.step.content,
            language: 'json',
            theme: widget.isDark ? atomOneDarkTheme : atomOneLightTheme,
            padding: const EdgeInsets.all(4),
            textStyle: const TextStyle(fontFamily: 'monospace', fontSize: 12),
          ),
        ),
      ],
    );
  }
}
