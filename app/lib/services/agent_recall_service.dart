import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/supabase_config.dart';

// ── Data models ──────────────────────────────────────────────

class AgentStep extends ChangeNotifier {
  final String type; // 'thought', 'tool_call', 'tool_result', 'error'
  final String? toolName;
  String content;
  bool _isCompleted;

  AgentStep({
    required this.type,
    this.toolName,
    required this.content,
    bool isCompleted = false,
  }) : _isCompleted = isCompleted;

  bool get isCompleted => _isCompleted;
  set isCompleted(bool value) {
    if (_isCompleted != value) {
      _isCompleted = value;
      notifyListeners();
    }
  }

  void updateContent(String newContent) {
    content = newContent;
    notifyListeners();
  }
}

class ChatMessage extends ChangeNotifier {
  final bool isUser;
  String _finalAnswer;
  String _liveStatus;
  final List<String> summaries;
  final List<AgentStep> steps;
  bool _isProcessCollapsed;

  ChatMessage({
    required this.isUser,
    String finalAnswer = '',
    String liveStatus = '',
    List<String>? summaries,
    List<AgentStep>? steps,
    bool isProcessCollapsed = false,
  }) : _finalAnswer = finalAnswer,
       _liveStatus = liveStatus,
       summaries = summaries ?? [],
       steps = steps ?? [],
       _isProcessCollapsed = isProcessCollapsed;

  String get finalAnswer => _finalAnswer;
  set finalAnswer(String value) {
    _finalAnswer = value;
    notifyListeners();
  }

  String get liveStatus => _liveStatus;
  set liveStatus(String value) {
    _liveStatus = value;
    notifyListeners();
  }

  bool get isProcessCollapsed => _isProcessCollapsed;
  set isProcessCollapsed(bool value) {
    if (_isProcessCollapsed != value) {
      _isProcessCollapsed = value;
      notifyListeners();
    }
  }

  void addStep(AgentStep step) {
    steps.add(step);
    notifyListeners();
  }

  void addSummary(String summary) {
    final trimmed = summary.trim();
    if (trimmed.isEmpty) return;
    if (summaries.isNotEmpty && summaries.last == trimmed) return;
    summaries.add(trimmed);
    notifyListeners();
  }

  void clearSteps() {
    steps.clear();
    notifyListeners();
  }
}

class AgentRecallResponse {
  final String mode;
  final String answer;
  final AgentEvidence? evidence;
  final List<AgentAction> actions;
  final List<AgentCandidate> candidates;
  final String? selectedCandidateReason;
  final Map<String, dynamic>? assetContext;
  final Map<String, dynamic>? compareContext;
  final Map<String, dynamic>? collectionContext;
  final Map<String, dynamic>? creativeContext;
  final Map<String, dynamic>? memoryGraphContext;
  final AgentSessionState? sessionState;
  final AgentFollowUp? followUp;
  final String? conversationSummary;

  AgentRecallResponse({
    required this.mode,
    required this.answer,
    required this.evidence,
    required this.actions,
    this.candidates = const [],
    this.selectedCandidateReason,
    this.assetContext,
    this.compareContext,
    this.collectionContext,
    this.creativeContext,
    this.memoryGraphContext,
    this.sessionState,
    this.followUp,
    this.conversationSummary,
  });

  factory AgentRecallResponse.fromJson(Map<String, dynamic> json) {
    final rawCandidates =
        (json['top_candidates'] as List?) ??
        (json['candidates'] as List?) ??
        const [];
    final rawEvidence = json['evidence'];
    final evidenceMap = rawEvidence is Map
        ? Map<String, dynamic>.from(rawEvidence)
        : null;
    return AgentRecallResponse(
      mode: json['mode']?.toString() ?? 'spatial_search',
      answer: json['answer']?.toString() ?? '',
      evidence:
          evidenceMap == null ||
              (!evidenceMap.containsKey('sceneId') &&
                  !evidenceMap.containsKey('similarity'))
          ? null
          : AgentEvidence.fromJson(evidenceMap),
      actions: ((json['actions'] as List?) ?? [])
          .map(
            (item) =>
                AgentAction.fromJson(Map<String, dynamic>.from(item as Map)),
          )
          .toList(),
      candidates: rawCandidates
          .map(
            (item) =>
                AgentCandidate.fromJson(Map<String, dynamic>.from(item as Map)),
          )
          .toList(),
      selectedCandidateReason: json['selected_candidate_reason']?.toString(),
      assetContext: json['asset_context'] is Map
          ? Map<String, dynamic>.from(json['asset_context'] as Map)
          : null,
      compareContext: json['compare_context'] is Map
          ? Map<String, dynamic>.from(json['compare_context'] as Map)
          : null,
      collectionContext: json['collection_context'] is Map
          ? Map<String, dynamic>.from(json['collection_context'] as Map)
          : null,
      creativeContext: json['creative_context'] is Map
          ? Map<String, dynamic>.from(json['creative_context'] as Map)
          : null,
      memoryGraphContext: json['memory_graph_context'] is Map
          ? Map<String, dynamic>.from(json['memory_graph_context'] as Map)
          : null,
      sessionState: json['session_state'] is Map
          ? AgentSessionState.fromJson(
              Map<String, dynamic>.from(json['session_state'] as Map),
            )
          : null,
      followUp: json['follow_up'] is Map
          ? AgentFollowUp.fromJson(
              Map<String, dynamic>.from(json['follow_up'] as Map),
            )
          : null,
      conversationSummary: json['conversation_summary']?.toString(),
    );
  }
}

class AgentSessionState {
  final String? lastMode;
  final List<String>? lastSelectedModelIds;
  final List<AgentCandidateRef>? lastCandidateRefs;
  final AgentOperationPreview? lastOperationPreview;

  AgentSessionState({
    this.lastMode,
    this.lastSelectedModelIds,
    this.lastCandidateRefs,
    this.lastOperationPreview,
  });

  factory AgentSessionState.fromJson(Map<String, dynamic> json) {
    return AgentSessionState(
      lastMode: json['lastMode']?.toString(),
      lastSelectedModelIds: (json['lastSelectedModelIds'] as List?)
          ?.map((item) => item.toString())
          .toList(),
      lastCandidateRefs: (json['lastCandidateRefs'] as List?)
          ?.map(
            (item) => AgentCandidateRef.fromJson(
              Map<String, dynamic>.from(item as Map),
            ),
          )
          .toList(),
      lastOperationPreview: json['lastOperationPreview'] is Map
          ? AgentOperationPreview.fromJson(
              Map<String, dynamic>.from(json['lastOperationPreview'] as Map),
            )
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      if (lastMode != null) 'lastMode': lastMode,
      if (lastSelectedModelIds != null)
        'lastSelectedModelIds': lastSelectedModelIds,
      if (lastCandidateRefs != null)
        'lastCandidateRefs': lastCandidateRefs!.map((item) => item.toJson()).toList(),
      if (lastOperationPreview != null)
        'lastOperationPreview': lastOperationPreview!.toJson(),
    };
  }
}

class AgentCandidateRef {
  final int index;
  final String sceneId;
  final String modelId;
  final String description;

  AgentCandidateRef({
    required this.index,
    required this.sceneId,
    required this.modelId,
    required this.description,
  });

  factory AgentCandidateRef.fromJson(Map<String, dynamic> json) {
    return AgentCandidateRef(
      index: (json['index'] as num?)?.toInt() ?? 0,
      sceneId: json['sceneId']?.toString() ?? '',
      modelId: json['modelId']?.toString() ?? '',
      description: json['description']?.toString() ?? '',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'index': index,
      'sceneId': sceneId,
      'modelId': modelId,
      'description': description,
    };
  }
}

class AgentOperationPreview {
  final String toolName;
  final int affectedCount;

  AgentOperationPreview({
    required this.toolName,
    required this.affectedCount,
  });

  factory AgentOperationPreview.fromJson(Map<String, dynamic> json) {
    return AgentOperationPreview(
      toolName: json['toolName']?.toString() ?? '',
      affectedCount: (json['affectedCount'] as num?)?.toInt() ?? 0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'toolName': toolName,
      'affectedCount': affectedCount,
    };
  }
}

class AgentFollowUp {
  final String status;
  final String kind;
  final String message;
  final String? inputPlaceholder;
  final List<String> suggestedReplies;

  AgentFollowUp({
    required this.status,
    required this.kind,
    required this.message,
    this.inputPlaceholder,
    this.suggestedReplies = const [],
  });

  bool get isWaitingUserInput => status == 'waiting_user_input';

  factory AgentFollowUp.fromJson(Map<String, dynamic> json) {
    return AgentFollowUp(
      status: json['status']?.toString() ?? 'idle',
      kind: json['kind']?.toString() ?? 'general',
      message: json['message']?.toString() ?? '',
      inputPlaceholder: json['input_placeholder']?.toString(),
      suggestedReplies: ((json['suggested_replies'] as List?) ?? [])
          .map((item) => item.toString())
          .where((item) => item.trim().isNotEmpty)
          .toList(),
    );
  }
}

class AgentCandidate {
  final String sceneId;
  final String modelId;
  final double score;
  final String description;
  final String? poseImageId;

  AgentCandidate({
    required this.sceneId,
    required this.modelId,
    required this.score,
    required this.description,
    this.poseImageId,
  });

  factory AgentCandidate.fromJson(Map<String, dynamic> json) {
    return AgentCandidate(
      sceneId: json['scene_id']?.toString() ?? '',
      modelId: json['model_id']?.toString() ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      description: json['description']?.toString() ?? '',
      poseImageId: json['pose_image_id']?.toString(),
    );
  }
}

class AgentEvidence {
  final String sceneId;
  final double similarity;
  final List<AgentMatchedFrame> matchedFrames;

  AgentEvidence({
    required this.sceneId,
    required this.similarity,
    required this.matchedFrames,
  });

  factory AgentEvidence.fromJson(Map<String, dynamic> json) {
    return AgentEvidence(
      sceneId: json['sceneId']?.toString() ?? '',
      similarity: (json['similarity'] as num?)?.toDouble() ?? 0,
      matchedFrames: ((json['matchedFrames'] as List?) ?? [])
          .map(
            (item) => AgentMatchedFrame.fromJson(
              Map<String, dynamic>.from(item as Map),
            ),
          )
          .toList(),
    );
  }
}

class AgentMatchedFrame {
  final String imageName;
  final double similarity;
  final List<double>? transformMatrix;

  AgentMatchedFrame({
    required this.imageName,
    required this.similarity,
    required this.transformMatrix,
  });

  factory AgentMatchedFrame.fromJson(Map<String, dynamic> json) {
    final raw = json['transformMatrix'] ?? json['transform_matrix'];
    return AgentMatchedFrame(
      imageName:
          json['imageName']?.toString() ?? json['image_name']?.toString() ?? '',
      similarity: (json['similarity'] as num?)?.toDouble() ?? 0,
      transformMatrix: _flattenNumericList(raw),
    );
  }
}

class AgentAction {
  final String type;
  final String sceneId;
  final String? modelId;
  final String? ply;
  final String? poses;
  final String? imageName;
  final List<double>? matrix;

  AgentAction({
    required this.type,
    required this.sceneId,
    this.modelId,
    this.ply,
    this.poses,
    this.imageName,
    this.matrix,
  });

  factory AgentAction.fromJson(Map<String, dynamic> json) {
    final payload = json['payload'] is Map
        ? Map<String, dynamic>.from(json['payload'] as Map)
        : const <String, dynamic>{};
    final rawMatrix = payload['matrix'] ?? json['matrix'];
    return AgentAction(
      type: json['type']?.toString() ?? '',
      sceneId:
          payload['sceneId']?.toString() ?? json['sceneId']?.toString() ?? '',
      modelId: payload['modelId']?.toString() ?? json['modelId']?.toString(),
      ply: payload['ply']?.toString() ?? json['ply']?.toString(),
      poses: payload['poses']?.toString() ?? json['poses']?.toString(),
      imageName:
          payload['imageId']?.toString() ??
          payload['imageName']?.toString() ??
          json['imageId']?.toString() ??
          json['imageName']?.toString(),
      matrix: _flattenNumericList(rawMatrix),
    );
  }
}

List<double>? _flattenNumericList(Object? value) {
  if (value is! List) {
    return null;
  }

  final flattened = <double>[];

  void collect(List<dynamic> input) {
    for (final item in input) {
      if (item is num) {
        flattened.add(item.toDouble());
      } else if (item is List) {
        collect(item);
      }
    }
  }

  collect(value);
  return flattened.isEmpty ? null : flattened;
}

// ── Service ──────────────────────────────────────────────────

class AgentRecallService {
  final SupabaseClient _client = Supabase.instance.client;

  Stream<String> queryStream(
    String query, {
    List<String>? selectedModelIds,
    String executionMode = 'preview',
    String? currentSceneId,
    String? currentModelId,
    String? currentMode,
    List<String>? candidateSceneIds,
    String? sessionId,
    String? conversationSummary,
    AgentSessionState? sessionState,
  }) async* {
    final trimmedQuery = query.trim();
    if (trimmedQuery.isEmpty) {
      throw Exception('查询语句不能为空');
    }

    final baseUrl = SupabaseConfig.url.trim();
    if (baseUrl.isEmpty) {
      throw Exception('未配置 SUPABASE_URL，无法建立 Agent 流式连接');
    }

    // 构造 Edge Function URL
    final endpoint = baseUrl.endsWith('/')
        ? '${baseUrl}functions/v1/agent-recall'
        : '$baseUrl/functions/v1/agent-recall';

    final dio = Dio();
    // 配置超时 (根据需要调整)
    dio.options.connectTimeout = const Duration(seconds: 10);
    dio.options.receiveTimeout = const Duration(seconds: 300); // 长连接需较长超时

    try {
      final session = _client.auth.currentSession;
      final token = session?.accessToken;
      final apiKey = SupabaseConfig.apiKey;

      final response = await dio.post<ResponseBody>(
        endpoint,
        queryParameters: {'stream': '1'},
        options: Options(
          responseType: ResponseType.stream,
          headers: {
            'Authorization': token != null ? 'Bearer $token' : null,
            'apikey': apiKey,
            'Accept': 'application/x-ndjson',
            'Content-Type': 'application/json',
          },
        ),
        data: {
          'query': trimmedQuery,
          if (selectedModelIds != null) 'selectedModelIds': selectedModelIds,
          'executionMode': executionMode,
          if (currentSceneId != null) 'currentSceneId': currentSceneId,
          if (currentModelId != null) 'currentModelId': currentModelId,
          if (currentMode != null) 'currentMode': currentMode,
          if (candidateSceneIds != null) 'candidateSceneIds': candidateSceneIds,
          if (sessionId != null) 'sessionId': sessionId,
          if (conversationSummary != null)
            'conversationSummary': conversationSummary,
          if (sessionState != null) 'sessionState': sessionState.toJson(),
        },
      );

      final stream = response.data?.stream;
      if (stream == null) {
        throw Exception('Response stream is null');
      }

      await for (final line
          in stream
              .cast<List<int>>()
              .transform(utf8.decoder)
              .transform(const LineSplitter())) {
        final trimmed = line.trim();
        if (trimmed.isNotEmpty) {
          yield trimmed;
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('Agent streaming error: $e');
      }

      try {
        final result = await this.query(
          trimmedQuery,
          selectedModelIds: selectedModelIds,
          executionMode: executionMode,
          currentSceneId: currentSceneId,
          currentModelId: currentModelId,
          currentMode: currentMode,
          candidateSceneIds: candidateSceneIds,
          sessionId: sessionId,
          conversationSummary: conversationSummary,
          sessionState: sessionState,
        );
        yield jsonEncode({'event': 'done', 'data': _encodeResponse(result)});
      } catch (fallbackError) {
        yield jsonEncode({
          'event': 'error',
          'data': _normalizeInvokeError(fallbackError),
        });
      }
    } finally {
      dio.close();
    }
  }

  Future<AgentRecallResponse> query(
    String query, {
    List<String>? selectedModelIds,
    String executionMode = 'preview',
    String? currentSceneId,
    String? currentModelId,
    String? currentMode,
    List<String>? candidateSceneIds,
    String? sessionId,
    String? conversationSummary,
    AgentSessionState? sessionState,
  }) async {
    final trimmedQuery = query.trim();
    if (trimmedQuery.isEmpty) {
      throw Exception('查询语句不能为空');
    }

    try {
      final response = await _client.functions.invoke(
        'agent-recall',
        body: {
          'query': trimmedQuery,
          if (selectedModelIds != null) 'selectedModelIds': selectedModelIds,
          'executionMode': executionMode,
          if (currentSceneId != null) 'currentSceneId': currentSceneId,
          if (currentModelId != null) 'currentModelId': currentModelId,
          if (currentMode != null) 'currentMode': currentMode,
          if (candidateSceneIds != null) 'candidateSceneIds': candidateSceneIds,
          if (sessionId != null) 'sessionId': sessionId,
          if (conversationSummary != null)
            'conversationSummary': conversationSummary,
          if (sessionState != null) 'sessionState': sessionState.toJson(),
        },
      );

      final data = _decodeInvokeData(response.data);
      if (data is! Map) {
        throw Exception('agent-recall 返回格式错误');
      }

      if (data['error'] != null) {
        throw Exception(data['error'].toString());
      }

      return AgentRecallResponse.fromJson(Map<String, dynamic>.from(data));
    } catch (e) {
      throw Exception(_normalizeInvokeError(e));
    }
  }

  Map<String, dynamic> _encodeResponse(AgentRecallResponse response) {
    return {
      'mode': response.mode,
      'answer': response.answer,
      'evidence': response.evidence == null
          ? null
          : {
              'sceneId': response.evidence!.sceneId,
              'similarity': response.evidence!.similarity,
              'matchedFrames': response.evidence!.matchedFrames
                  .map(
                    (frame) => {
                      'imageName': frame.imageName,
                      'similarity': frame.similarity,
                      'transformMatrix': frame.transformMatrix,
                    },
                  )
                  .toList(),
            },
      'actions': response.actions
          .map(
            (action) => {
              'type': action.type,
              'payload': {
                'sceneId': action.sceneId,
                if (action.modelId != null) 'modelId': action.modelId,
                if (action.ply != null) 'ply': action.ply,
                if (action.poses != null) 'poses': action.poses,
                if (action.imageName != null) 'imageId': action.imageName,
                if (action.matrix != null) 'matrix': action.matrix,
              },
            },
          )
          .toList(),
      'top_candidates': response.candidates
          .map(
            (candidate) => {
              'scene_id': candidate.sceneId,
              'model_id': candidate.modelId,
              'score': candidate.score,
              'description': candidate.description,
              'pose_image_id': candidate.poseImageId,
            },
          )
          .toList(),
      'selected_candidate_reason': response.selectedCandidateReason,
      'asset_context': response.assetContext,
      'compare_context': response.compareContext,
      'collection_context': response.collectionContext,
      'creative_context': response.creativeContext,
      'memory_graph_context': response.memoryGraphContext,
      'session_state': response.sessionState?.toJson(),
      'conversation_summary': response.conversationSummary,
      'follow_up': response.followUp == null
          ? null
          : {
              'status': response.followUp!.status,
              'kind': response.followUp!.kind,
              'message': response.followUp!.message,
              'input_placeholder': response.followUp!.inputPlaceholder,
              'suggested_replies': response.followUp!.suggestedReplies,
            },
    };
  }

  Object? _decodeInvokeData(Object? data) {
    if (data is String) {
      final trimmed = data.trim();
      if (trimmed.isEmpty) {
        return null;
      }

      try {
        return jsonDecode(trimmed);
      } catch (_) {
        return trimmed;
      }
    }

    return data;
  }

  String _normalizeDioError(DioException error) {
    final status = error.response?.statusCode;
    final data = _decodeInvokeData(error.response?.data);
    if (data is Map && data['error'] != null) {
      return data['error'].toString();
    }
    if (data is Map && data['message'] != null) {
      return data['message'].toString();
    }
    if (data is String && data.trim().isNotEmpty) {
      return data.trim();
    }
    if (status == 503) {
      return 'agent-recall 当前不可用（HTTP 503）。这通常表示 Edge Function worker 启动失败，或上游模型网关暂时不可用。请优先检查 Supabase Edge Function 日志。';
    }
    if (status == 502 || status == 504) {
      return 'agent-recall 上游服务响应异常，请检查 Edge Function 日志和模型网关配置';
    }
    if (status != null) {
      return 'agent-recall 调用失败（HTTP $status）';
    }
    return error.message ?? error.toString();
  }

  String _normalizeInvokeError(Object error) {
    if (error is DioException) {
      return _normalizeDioError(error);
    }

    if (error is FunctionException) {
      final detail = _decodeInvokeData(error.details);
      if (detail is Map && detail['error'] != null) {
        return detail['error'].toString();
      }
      if (detail is Map && detail['message'] != null) {
        return detail['message'].toString();
      }
      if (detail is String && detail.isNotEmpty) {
        return detail;
      }
      if (error.reasonPhrase != null && error.reasonPhrase!.isNotEmpty) {
        return error.reasonPhrase!;
      }
      if (error.status == 503) {
        return 'agent-recall 当前不可用（HTTP 503）。这通常表示 Edge Function worker 启动失败，或上游模型网关暂时不可用。请优先检查 Supabase Edge Function 日志。';
      }
      if (error.status == 502 || error.status == 504) {
        return 'agent-recall 上游服务响应异常，请检查 Edge Function 日志和模型网关配置';
      }
      return 'agent-recall 调用失败（HTTP ${error.status}）';
    }

    return error.toString();
  }
}
