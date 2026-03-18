import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../configs/app_config.dart';
import '../configs/supabase_config.dart';
import '../configs/motion_tokens.dart';
import '../services/local_rag_index.dart';
import '../services/download_event_bus.dart';
import '../widgets/bd_surfaces.dart';
import 'community.dart';
import 'recall/local_ai_panel.dart';
import 'recall/model_grid.dart';
import 'recall/overview_card.dart';
import 'recall/processing_section.dart';
import 'settings.dart';
import 'webgl_viewer.dart';
import 'task_list.dart';
import 'recall/top_summary_card.dart';
import 'recall/model_card.dart';

enum _RecallSearchMode { local, cloud }

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
  static const String _defaultModelFileName = 'qwen3-1.7b.gguf';
  static const String _defaultModelDownloadUrl =
      'https://hf-mirror.com/jc-builds/Qwen3-1.7B-Q4_K_M-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true';
  static const String _localModelPathPrefKey = 'recall.local_llm_model_path';
  static const String _localModelUrlPrefKey = 'recall.local_llm_model_url';

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
  final TextEditingController _localModelUrlController = TextEditingController(
    text: _defaultModelDownloadUrl,
  );
  final TextEditingController _localQuestionController =
      TextEditingController();
  final LocalRagIndexService _localRagIndex = LocalRagIndexService();
  RealtimeChannel? _realtimeChannel;
  Timer? _searchDebounce;
  LocalRagIndexStats? _indexStats;
  _RecallSearchMode _searchMode = _RecallSearchMode.local;
  LlamaEngine? _localQnaModel;
  StreamSubscription<dynamic>? _llamaStreamSubscription;
  String _localAnswer = '';
  String _localAnswerStatus = 'Qwen3-1.7B 端侧模型未加载';
  String _localContextPreview = '';
  bool _isLocalModelLoading = false;
  bool _isLocalModelReady = false;
  bool _isLocalAnswering = false;
  bool _isModelDownloading = false;
  bool _isLocalAiPanelOpen = false;
  double? _modelDownloadProgress;
  int _modelDownloadedBytes = 0;
  int? _modelDownloadTotalBytes;
  final Map<String, GlobalKey> _modelCardKeys = {};
  Map<String, dynamic>? _activeModelAction;
  Rect? _activeModelActionRect;

  @override
  void initState() {
    super.initState();
    _restoreLocalModelPath();
    _fetchModels();
    _fetchProcessingTasks();
    _setupRealtimeListener();
  }

  @override
  void dispose() {
    _searchDebounce?.cancel();
    _searchController.dispose();
    _localModelPathController.dispose();
    _localModelUrlController.dispose();
    _localQuestionController.dispose();
    _realtimeChannel?.unsubscribe();
    unawaited(_disposeLocalQnaModel());
    super.dispose();
  }

  Future<void> _restoreLocalModelPath() async {
    final defaultPath = await _getDefaultPrivateModelPath();
    final prefs = await SharedPreferences.getInstance();
    final savedPath = prefs.getString(_localModelPathPrefKey)?.trim();
    final savedUrl = prefs.getString(_localModelUrlPrefKey)?.trim();
    if (!mounted) return;
    setState(() {
      _localModelPathController.text = (savedPath == null || savedPath.isEmpty)
          ? defaultPath
          : savedPath;
      if (savedUrl != null && savedUrl.isNotEmpty) {
        _localModelUrlController.text = savedUrl;
      }
    });
  }

  Future<void> _persistLocalModelPath(String modelPath) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_localModelPathPrefKey, modelPath);
  }

  Future<void> _persistLocalModelUrl(String modelUrl) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_localModelUrlPrefKey, modelUrl);
  }

  Future<String> _getDefaultPrivateModelPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return '${dir.path}/$_defaultModelFileName';
  }

  Future<void> _downloadModelToPrivateDir() async {
    final modelUrl = _localModelUrlController.text.trim();
    if (modelUrl.isEmpty) {
      TDToast.showText(context: context, '请先填写模型下载链接');
      return;
    }

    final modelPath = await _getDefaultPrivateModelPath();
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
    if (modelPath.isEmpty) {
      TDToast.showText(context: context, '请先填写 GGUF 模型路径');
      return;
    }

    setState(() {
      _isLocalModelLoading = true;
      _isLocalModelReady = false;
      _isLocalAnswering = false;
      _localAnswer = '';
      _localAnswerStatus = '正在加载 Qwen3-1.7B 端侧模型...';
    });

    try {
      await _disposeLocalQnaModel();
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
          _localAnswerStatus =
              'Qwen3-1.7B 已加载，当前后端：$backendName（已从 GPU 回退到 CPU），模型大小：${(modelSize / 1024 / 1024).toStringAsFixed(1)} MB';
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
        _localAnswerStatus =
            'Qwen3-1.7B 已加载，当前后端：$backendSummary，模型大小：${(modelSize / 1024 / 1024).toStringAsFixed(1)} MB';
      });
    } catch (e) {
      await _disposeLocalQnaModel();
      if (!mounted) {
        return;
      }
      setState(() {
        _isLocalModelLoading = false;
        _isLocalModelReady = false;
        _isLocalAnswering = false;
        _localAnswerStatus = '模型加载失败：$e';
      });
      TDToast.showText(context: context, '端侧模型加载失败：$e');
    }
  }

  Future<void> _askLocalQuestion() async {
    final userQuestion = _localQuestionController.text.trim();
    if (userQuestion.isEmpty) {
      TDToast.showText(context: context, '请输入要提问的问题');
      return;
    }
    if (_localQnaModel == null || !_isLocalModelReady) {
      TDToast.showText(context: context, '请先加载本地 GGUF 模型');
      return;
    }

    final memoryContext = await _buildMemoryContext(userQuestion);
    var streamedAnswer = '';
    var lockedAnswer = false;
    final prompt =
        '''
请根据下面的记忆片段，直接回答问题。
如果记忆片段没有明确答案，再简短回答不知道。
不要解释规则，不要重复题目。

记忆片段：
$memoryContext

问题：
$userQuestion

回答：
''';

    setState(() {
      _localAnswer = '';
      _localContextPreview = memoryContext;
      _isLocalAnswering = true;
      _localAnswerStatus = '正在根据本地记忆片段生成回答...';
    });

    try {
      _localQnaModel!.cancelGeneration();
      await _llamaStreamSubscription?.cancel();
      _llamaStreamSubscription = _localQnaModel!
          .generate(
            prompt,
            params: const GenerationParams(
              maxTokens: 64,
              temp: 0.1,
              topK: 20,
              topP: 0.85,
              penalty: 1.12,
              stopSequences: ['【说明】', '\n答案：', '\n问题：', '\n记忆片段：'],
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
              final nextAnswer = _sanitizeLocalAnswer(streamedAnswer + token);
              if (lockedAnswer) {
                if (nextAnswer != streamedAnswer) {
                  _localQnaModel!.cancelGeneration();
                }
                return;
              }
              final shouldStop = nextAnswer == streamedAnswer;
              if (nextAnswer.isNotEmpty) {
                streamedAnswer = nextAnswer;
              }
              lockedAnswer = _shouldLockAnswer(streamedAnswer);
              setState(() {
                _localAnswer = streamedAnswer;
              });
              if (shouldStop || lockedAnswer) {
                _localQnaModel!.cancelGeneration();
              }
            },
            onError: (Object error) {
              if (!mounted) {
                return;
              }
              setState(() {
                _isLocalAnswering = false;
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
                _isLocalAnswering = false;
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
        _isLocalAnswering = false;
        _localAnswerStatus = '端侧问答失败：$e';
      });
      TDToast.showText(context: context, '端侧问答失败：$e');
    }
  }

  Future<String> _buildMemoryContext(String question) async {
    List<Map<String, dynamic>> matches = [];
    try {
      matches = await _localRagIndex.search(question, limit: 3, minScore: 0.08);
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

    if (matches.isEmpty) {
      return '暂无可用记忆片段。';
    }

    return matches
        .asMap()
        .entries
        .map((entry) {
          return _formatMemorySnippet(entry.key + 1, entry.value);
        })
        .join('\n\n');
  }

  String _formatMemorySnippet(int index, Map<String, dynamic> model) {
    final metaInfo = _toMap(model['meta_info']);
    final tags = _joinList(model['tags']);
    final objects = _joinList(model['objects']);
    final summary = _collectStrings(metaInfo)
        .take(6)
        .map((item) => item.trim())
        .where((item) => item.isNotEmpty)
        .join('；');

    final parts = <String>[
      '片段$index',
      '场景：${model['scene_id']?.toString() ?? '未知场景'}',
      '描述：${model['description']?.toString() ?? '暂无描述'}',
      if (tags.isNotEmpty) '标签：$tags',
      if (objects.isNotEmpty) '对象：$objects',
      if (summary.isNotEmpty) '摘要：$summary',
    ];

    return parts.join('\n');
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
        children: [
          BDPageBackdrop(
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
                          IconButton(
                            icon: Icon(
                              Icons.settings_rounded,
                              color: isDark
                                  ? BDDesign.colorPaperWhite
                                  : BDDesign.colorInkBlack,
                            ),
                            tooltip: textLocalize("settings"),
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) => const SettingsPage(),
                                ),
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                    RecallOverviewCard(
                      isDark: isDark,
                      textColor: textColor,
                      recentCount: _recentModelCount(),
                      allModelCount: _allModels.length,
                      processingTaskCount: _processingTasks.length,
                      ragCount: _indexStats?.totalItems ?? _allModels.length,
                      isLocalIndexing: _isLocalIndexing,
                      onOpenTasks: () {
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
                                suffixIcon: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    _buildSearchModeMenuButton(isDark),
                                    if (_searchController.text
                                        .trim()
                                        .isNotEmpty)
                                      IconButton(
                                        onPressed: () {
                                          _searchController.clear();
                                          _searchDebounce?.cancel();
                                          _searchModels('');
                                          setState(() {});
                                        },
                                        icon: Icon(
                                          Icons.close_rounded,
                                          color: isDark
                                              ? Colors.white.withValues(
                                                  alpha: 0.5,
                                                )
                                              : BDDesign.colorMutedBlue,
                                        ),
                                      ),
                                  ],
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
                          const SizedBox(height: 10),
                          RecallLocalAiPanel(
                            theme: theme,
                            isDark: isDark,
                            textColor: textColor,
                            darkInput: darkInput,
                            isLocalAiPanelOpen: _isLocalAiPanelOpen,
                            isLocalModelReady: _isLocalModelReady,
                            isModelDownloading: _isModelDownloading,
                            isLocalModelLoading: _isLocalModelLoading,
                            isLocalAnswering: _isLocalAnswering,
                            modelDownloadProgress: _modelDownloadProgress,
                            modelDownloadedBytes: _modelDownloadedBytes,
                            modelDownloadTotalBytes: _modelDownloadTotalBytes,
                            localAnswer: _localAnswer,
                            localAnswerStatus: _localAnswerStatus,
                            localContextPreview: _localContextPreview,
                            defaultModelDownloadUrl: _defaultModelDownloadUrl,
                            localModelUrlController: _localModelUrlController,
                            localModelPathController: _localModelPathController,
                            localQuestionController: _localQuestionController,
                            onToggleOpen: () {
                              setState(() {
                                _isLocalAiPanelOpen = !_isLocalAiPanelOpen;
                              });
                            },
                            onDownloadModel: _downloadModelToPrivateDir,
                            onLoadModel: _loadLocalQnaModel,
                            onAskQuestion: _askLocalQuestion,
                          ),
                        ],
                      ),
                    ),
                    if (_processingTasks.isNotEmpty)
                      RecallProcessingSection(
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
                            _isProcessingExpanded = !_isProcessingExpanded;
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
                        onShowModelActions: (model) {
                          unawaited(_showModelActions(model));
                        },
                      ),
                  ],
                ),
              ),
            ),
          ),
          if (_activeModelAction != null && _activeModelActionRect != null)
            RecallModelActionOverlay(
              theme: theme,
              isDark: isDark,
              darkCard: darkCard,
              darkInput: darkInput,
              model: _activeModelAction!,
              rect: _activeModelActionRect!,
              toPublicUrl: _toPublicUrl,
              onDismiss: _dismissModelActions,
              onNavigateToViewer: _navigateToViewer,
              onShareModelToCommunity: _shareModelToCommunity,
              onRenameModel: _renameModel,
              onDeleteLocalModel: _deleteLocalModel,
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

    final errMsg = (data is Map)
        ? (data['error'] ?? textLocalize('recall_unknown_error'))
        : textLocalize('recall_server_error');
    throw Exception(errMsg);
  }

  void _onSearchChanged(String value) {
    setState(() {});
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 180), () {
      _searchModels(value);
    });
  }

  void _setSearchMode(_RecallSearchMode mode) {
    if (_searchMode == mode) {
      return;
    }
    setState(() {
      _searchMode = mode;
    });
    final keyword = _searchController.text.trim();
    if (keyword.isNotEmpty) {
      _searchModels(keyword);
    }
  }

  Widget _buildSearchModeMenuButton(bool isDark) {
    final selectedLocal = _searchMode == _RecallSearchMode.local;
    final icon = selectedLocal
        ? Icons.privacy_tip_rounded
        : Icons.cloud_rounded;
    final foreground = isDark
        ? Colors.white.withValues(alpha: 0.82)
        : BDDesign.colorInkBlack;

    return IconButton(
      tooltip: textLocalize('recall_search_mode'),
      padding: EdgeInsets.zero,
      constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
      onPressed: _showSearchModeSheet,
      icon: Container(
        width: 32,
        height: 32,
        decoration: BoxDecoration(
          color: isDark
              ? Colors.white.withValues(alpha: 0.06)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: isDark
                ? Colors.white.withValues(alpha: 0.08)
                : BDDesign.colorMutedBlue.withValues(alpha: 0.18),
          ),
        ),
        child: Icon(icon, size: 16, color: foreground),
      ),
    );
  }

  Future<void> _showSearchModeSheet() async {
    final selected = await showModalBottomSheet<_RecallSearchMode>(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) {
        final isDark = AppConfig.isNightMode;
        final textColor = isDark
            ? BDDesign.colorPaperWhite
            : BDDesign.colorInkBlack;
        final hintColor = isDark
            ? Colors.white.withValues(alpha: 0.62)
            : BDDesign.colorMutedBlue;

        Widget modeTile({
          required _RecallSearchMode mode,
          required IconData icon,
          required String title,
          required String subtitle,
        }) {
          final selected = _searchMode == mode;
          return InkWell(
            borderRadius: BorderRadius.circular(18),
            onTap: () => Navigator.pop(context, mode),
            child: AnimatedContainer(
              duration: BDMotion.durationFast,
              curve: Curves.easeOutCubic,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: selected
                    ? BDDesign.colorMutedBlue.withValues(
                        alpha: isDark ? 0.22 : 0.10,
                      )
                    : (isDark ? darkInput : const Color(0xFFF6F8FC)),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: selected
                      ? BDDesign.colorMutedBlue
                      : (isDark
                            ? Colors.white.withValues(alpha: 0.08)
                            : BDDesign.colorMutedBlue.withValues(alpha: 0.14)),
                ),
              ),
              child: Row(
                children: [
                  Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                      color: selected
                          ? BDDesign.colorMutedBlue.withValues(alpha: 0.18)
                          : (isDark
                                ? Colors.white.withValues(alpha: 0.05)
                                : Colors.white),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Icon(icon, color: textColor, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          title,
                          style: TextStyle(
                            color: textColor,
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          subtitle,
                          style: TextStyle(
                            color: hintColor,
                            fontSize: 12.5,
                            height: 1.35,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (selected)
                    const Icon(
                      Icons.check_circle_rounded,
                      color: BDDesign.colorMutedBlue,
                    ),
                ],
              ),
            ),
          );
        }

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
                    textLocalize('recall_search_mode'),
                    style: TextStyle(
                      color: textColor,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    '选择当前搜索栏要优先使用的检索方式。',
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 12.5,
                      height: 1.35,
                    ),
                  ),
                  const SizedBox(height: 16),
                  modeTile(
                    mode: _RecallSearchMode.local,
                    icon: Icons.privacy_tip_rounded,
                    title: textLocalize('recall_local_rag'),
                    subtitle: textLocalize('recall_local_scope'),
                  ),
                  const SizedBox(height: 10),
                  modeTile(
                    mode: _RecallSearchMode.cloud,
                    icon: Icons.cloud_rounded,
                    title: textLocalize('recall_cloud_rag'),
                    subtitle: textLocalize('recall_cloud_scope'),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );

    if (selected != null) {
      _setSearchMode(selected);
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
        ? _toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? _toPosesUrl(plyPath) : null;
    final sceneId =
        model['display_name']?.toString() ??
        model['scene_id'] ??
        'Unknown Scene';

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

  Future<String> _getLocalModelSizeLabel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path'] as String? ?? '';
    if (plyPath.isEmpty) return '';

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

      int totalBytes = 0;
      final localFile = File('${dir.path}/$sanitizedFileName');
      final tmpFile = File('${dir.path}/$sanitizedFileName.tmp');
      final metaFile = File('${dir.path}/$sanitizedFileName.meta');

      if (await localFile.exists()) totalBytes += await localFile.length();
      if (await tmpFile.exists()) totalBytes += await tmpFile.length();
      if (await metaFile.exists()) totalBytes += await metaFile.length();

      if (totalBytes == 0) return '';
      if (totalBytes < 1024 * 1024) {
        return '(${(totalBytes / 1024).toStringAsFixed(0)}KB)';
      }
      return '(${(totalBytes / (1024 * 1024)).toStringAsFixed(1)}MB)';
    } catch (_) {
      return '';
    }
  }

  Future<void> _showModelActions(Map<String, dynamic> model) async {
    final sizeLabel = await _getLocalModelSizeLabel(model);

    if (!mounted) return;

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
        final sceneId =
            model['display_name']?.toString() ??
            model['scene_id']?.toString() ??
            textLocalize('recall_unnamed_model');
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
                    leading: const Icon(Icons.edit_rounded),
                    title: Text(textLocalize('recall_rename_model')),
                    subtitle: Text(textLocalize('recall_rename_subtitle')),
                    onTap: () => Navigator.pop(context, 'rename'),
                  ),
                  ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: const Icon(Icons.delete_outline_rounded),
                    title: Text(
                      '${textLocalize('recall_delete_local')}$sizeLabel',
                    ),
                    subtitle: Text(
                      textLocalize('recall_delete_local_subtitle'),
                    ),
                    onTap: () => Navigator.pop(context, 'delete_local'),
                  ),
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

    switch (selectedAction) {
      case 'rename':
        await _renameModel(model);
        break;
      case 'delete_local':
        await _deleteLocalModel(model);
        break;
      case 'share':
        await _shareModelToCommunity(model);
        break;
    }
  }

  Future<void> _renameModel(Map<String, dynamic> model) async {
    final sceneId = model['scene_id']?.toString();
    if (sceneId == null) return;

    final currentName = model['display_name']?.toString() ?? sceneId;
    final newName = await showDialog<String>(
      context: context,
      builder: (_) => _RenameModelDialog(initialName: currentName),
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
        setState(() {
          for (final m in _allModels) {
            if (m['scene_id'] == sceneId) {
              m['display_name'] = newName;
            }
          }
          for (final m in _models) {
            if (m['scene_id'] == sceneId) {
              m['display_name'] = newName;
            }
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

  Future<void> _deleteLocalModel(Map<String, dynamic> model) async {
    final plyPath = model['ply_path'] as String? ?? '';
    if (plyPath.isEmpty) {
      if (mounted) {
        TDToast.showText(
          textLocalize('recall_delete_local_none'),
          context: context,
        );
      }
      return;
    }

    try {
      final modelUrl = _toPublicUrl(plyPath);
      if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
        if (mounted) {
          TDToast.showText(
            textLocalize('recall_delete_local_none'),
            context: context,
          );
        }
        return;
      }

      final encodedUrl = Uri.encodeFull(Uri.decodeFull(modelUrl));
      final uri = Uri.parse(encodedUrl);
      final sanitizedFileName = uri.path
          .replaceAll('/', '_')
          .replaceAll('\\', '_');
      final dir = await getApplicationDocumentsDirectory();

      final localFile = File('${dir.path}/$sanitizedFileName');
      final tmpFile = File('${dir.path}/$sanitizedFileName.tmp');
      final metaFile = File('${dir.path}/$sanitizedFileName.meta');

      bool deleted = false;
      if (await localFile.exists()) {
        await localFile.delete();
        deleted = true;
      }
      if (await tmpFile.exists()) {
        await tmpFile.delete();
        deleted = true;
      }
      if (await metaFile.exists()) {
        await metaFile.delete();
        deleted = true;
      }

      if (mounted) {
        if (deleted) {
          // 通知下载状态徽章重置为"未下载"
          downloadEventBus.add(
            ModelDownloadEvent(url: modelUrl, progress: 0.0, isDeleted: true),
          );
          TDToast.showText(
            textLocalize('recall_delete_local_success'),
            context: context,
          );
        } else {
          TDToast.showText(
            textLocalize('recall_delete_local_none'),
            context: context,
          );
        }
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText(
          '${textLocalize('recall_delete_local_fail')}: $e',
          context: context,
        );
      }
    }
  }

  void _dismissModelActions() {
    if (_activeModelAction == null && _activeModelActionRect == null) {
      return;
    }
    setState(() {
      _activeModelAction = null;
      _activeModelActionRect = null;
    });
  }

  CommunityModelOption _modelToCommunityOption(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString() ?? '';
    final preview = model['preview_img_path']?.toString();
    return CommunityModelOption(
      id: model['id']?.toString() ?? model['scene_id']?.toString() ?? 'model',
      sceneId:
          model['display_name']?.toString() ??
          model['scene_id']?.toString() ??
          textLocalize('recall_unnamed_model'),
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

/// 独立的重命名对话框组件，避免 StatefulBuilder + showDialog 导致的
/// `_dependents.isEmpty` 断言错误。
class _RenameModelDialog extends StatefulWidget {
  final String initialName;
  const _RenameModelDialog({required this.initialName});

  @override
  State<_RenameModelDialog> createState() => _RenameModelDialogState();
}

class _RenameModelDialogState extends State<_RenameModelDialog> {
  static final _invalidChars = RegExp(r'[/\\:*?"<>|]');
  late final TextEditingController _controller;
  String? _errorText;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialName);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(textLocalize('recall_rename_model')),
      content: TextField(
        controller: _controller,
        autofocus: true,
        decoration: InputDecoration(
          hintText: textLocalize('recall_rename_hint'),
          errorText: _errorText,
        ),
        onChanged: (value) {
          setState(() {
            _errorText = _invalidChars.hasMatch(value)
                ? textLocalize('recall_rename_invalid')
                : null;
          });
        },
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text(textLocalize('gen_cancel')),
        ),
        TextButton(
          onPressed: () {
            final text = _controller.text.trim();
            if (text.isEmpty || _invalidChars.hasMatch(text)) return;
            Navigator.pop(context, text);
          },
          child: Text(textLocalize('gen_button')),
        ),
      ],
    );
  }
}
