import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/app_config.dart';
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

  String get compactTitle {
    final trimmed = content.trim();
    if (type == 'tool_call') {
      return toolName?.trim().isNotEmpty == true ? toolName!.trim() : '未命名工具';
    }
    if (trimmed.isEmpty) {
      switch (type) {
        case 'status':
          return '状态更新';
        case 'thought':
          return '思考';
        case 'error':
          return '执行异常';
        default:
          return '步骤';
      }
    }
    final firstLine = trimmed.split('\n').first.trim();
    return firstLine.isEmpty ? trimmed : firstLine;
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
  final List<AgentToolTrace> toolTrace;
  final Map<String, dynamic>? responseResolution;
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
    this.toolTrace = const [],
    this.responseResolution,
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
      toolTrace: ((json['tool_trace'] as List?) ?? [])
          .map(
            (item) =>
                AgentToolTrace.fromJson(Map<String, dynamic>.from(item as Map)),
          )
          .toList(),
      responseResolution: json['response_resolution'] is Map
          ? Map<String, dynamic>.from(json['response_resolution'] as Map)
          : null,
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
        'lastCandidateRefs': lastCandidateRefs!
            .map((item) => item.toJson())
            .toList(),
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
  final List<String>? modelIds;
  final Map<String, dynamic>? args;

  AgentOperationPreview({
    required this.toolName,
    required this.affectedCount,
    this.modelIds,
    this.args,
  });

  factory AgentOperationPreview.fromJson(Map<String, dynamic> json) {
    return AgentOperationPreview(
      toolName: json['toolName']?.toString() ?? '',
      affectedCount: (json['affectedCount'] as num?)?.toInt() ?? 0,
      modelIds: (json['modelIds'] as List?)
          ?.map((item) => item.toString())
          .where((item) => item.trim().isNotEmpty)
          .toList(),
      args: json['args'] is Map
          ? Map<String, dynamic>.from(json['args'] as Map)
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'toolName': toolName,
      'affectedCount': affectedCount,
      if (modelIds != null) 'modelIds': modelIds,
      if (args != null) 'args': args,
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

class AgentToolTrace {
  final String toolName;
  final Map<String, dynamic> args;
  final String resultSummary;

  AgentToolTrace({
    required this.toolName,
    required this.args,
    required this.resultSummary,
  });

  factory AgentToolTrace.fromJson(Map<String, dynamic> json) {
    return AgentToolTrace(
      toolName:
          json['toolName']?.toString() ?? json['tool_name']?.toString() ?? '',
      args: json['args'] is Map
          ? Map<String, dynamic>.from(json['args'] as Map)
          : const <String, dynamic>{},
      resultSummary:
          json['resultSummary']?.toString() ??
          json['result_summary']?.toString() ??
          '',
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
      throw Exception(textLocalize('agent_error_empty_query'));
    }

    final endpoint = SupabaseConfig.edgeFunctionUrl('agent-recall');
    if (endpoint.isEmpty) {
      throw Exception(textLocalize('agent_error_missing_supabase_url'));
    }

    final dio = Dio();
    // 配置超时 (根据需要调整)
    dio.options.connectTimeout = const Duration(seconds: 20);
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
            'Accept': 'text/event-stream, application/x-ndjson',
            'Cache-Control': 'no-cache',
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

      final eventBuffer = StringBuffer();
      await for (final chunk in stream.cast<List<int>>().transform(
        utf8.decoder,
      )) {
        if (chunk.isEmpty) {
          continue;
        }
        eventBuffer.write(chunk);
        final rawBuffer = eventBuffer.toString();
        final parsed = _drainStreamingEvents(rawBuffer);
        if (parsed.remaining != rawBuffer.length) {
          eventBuffer
            ..clear()
            ..write(rawBuffer.substring(parsed.remaining));
        }
        for (final event in parsed.events) {
          yield event;
        }
      }

      final tail = eventBuffer.toString().trim();
      if (tail.isNotEmpty) {
        for (final event in _parseEventChunk(tail)) {
          yield event;
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
      throw Exception(textLocalize('agent_error_empty_query'));
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
        throw Exception(textLocalize('agent_error_bad_response'));
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
      'tool_trace': response.toolTrace
          .map(
            (trace) => {
              'tool_name': trace.toolName,
              'args': trace.args,
              'result_summary': trace.resultSummary,
            },
          )
          .toList(),
      'response_resolution': response.responseResolution,
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
    if (error.type == DioExceptionType.connectionTimeout ||
        error.type == DioExceptionType.connectionError ||
        error.type == DioExceptionType.sendTimeout) {
      return SupabaseConfig.buildConnectionHelp(
        'agent-recall',
        endpoint: SupabaseConfig.edgeFunctionUrl('agent-recall'),
      );
    }

    final status = error.response?.statusCode;
    final data = _decodeInvokeData(error.response?.data);
    if (data is Map && data['error'] != null) {
      return _normalizeRawErrorText(data['error'].toString());
    }
    if (data is Map && data['message'] != null) {
      return _normalizeRawErrorText(data['message'].toString());
    }
    if (data is String && data.trim().isNotEmpty) {
      return _normalizeRawErrorText(data.trim());
    }
    if (status == 503) {
      return textLocalize('agent_error_unavailable');
    }
    if (status == 502 || status == 504) {
      return textLocalize('agent_error_upstream');
    }
    if (status != null) {
      return textLocalize(
        'agent_error_http',
      ).replaceAll('{status}', status.toString());
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
        return _normalizeRawErrorText(detail['error'].toString());
      }
      if (detail is Map && detail['message'] != null) {
        return _normalizeRawErrorText(detail['message'].toString());
      }
      if (detail is String && detail.isNotEmpty) {
        return _normalizeRawErrorText(detail);
      }
      if (error.reasonPhrase != null && error.reasonPhrase!.isNotEmpty) {
        return _normalizeRawErrorText(error.reasonPhrase!);
      }
      if (error.status == 503) {
        return textLocalize('agent_error_unavailable');
      }
      if (error.status == 502 || error.status == 504) {
        return textLocalize('agent_error_upstream');
      }
      return textLocalize(
        'agent_error_http',
      ).replaceAll('{status}', error.status.toString());
    }

    return _normalizeRawErrorText(error.toString());
  }

  String _normalizeRawErrorText(String message) {
    final trimmed = message.trim();
    if (trimmed.isEmpty) {
      return trimmed;
    }

    final normalized = trimmed.toLowerCase();
    if (normalized.contains('an invalid response was received from the upstream server')) {
      return textLocalize('agent_error_upstream');
    }
    if (normalized.contains('upstream connect error')) {
      return textLocalize('agent_error_upstream');
    }

    return trimmed;
  }

  _ParsedStreamEvents _drainStreamingEvents(String raw) {
    final events = <String>[];
    var cursor = 0;

    while (cursor < raw.length) {
      final sseBoundary = raw.indexOf('\n\n', cursor);
      final lineBoundary = raw.indexOf('\n', cursor);
      final looksLikeSse =
          raw.startsWith('event:', cursor) || raw.startsWith('data:', cursor);

      if (looksLikeSse) {
        if (sseBoundary == -1) {
          break;
        }
        final chunk = raw.substring(cursor, sseBoundary).trim();
        cursor = sseBoundary + 2;
        events.addAll(_parseEventChunk(chunk));
        continue;
      }

      if (lineBoundary == -1) {
        break;
      }

      final chunk = raw.substring(cursor, lineBoundary).trim();
      cursor = lineBoundary + 1;
      if (chunk.isEmpty) {
        continue;
      }
      events.addAll(_parseEventChunk(chunk));
    }

    return _ParsedStreamEvents(events: events, remaining: cursor);
  }

  List<String> _parseEventChunk(String chunk) {
    final trimmed = chunk.trim();
    if (trimmed.isEmpty) {
      return const [];
    }

    if (trimmed.startsWith('{')) {
      return [trimmed];
    }

    String eventName = 'message';
    final dataLines = <String>[];
    for (final line in const LineSplitter().convert(trimmed)) {
      final normalized = line.trimRight();
      if (normalized.isEmpty || normalized.startsWith(':')) {
        continue;
      }
      if (normalized.startsWith('event:')) {
        eventName = normalized.substring(6).trim();
        continue;
      }
      if (normalized.startsWith('data:')) {
        dataLines.add(normalized.substring(5).trimLeft());
      }
    }

    if (dataLines.isEmpty) {
      return const [];
    }

    final dataText = dataLines.join('\n').trim();
    if (dataText.isEmpty) {
      return const [];
    }

    try {
      final decoded = jsonDecode(dataText);
      return [
        jsonEncode({'event': eventName, 'data': decoded}),
      ];
    } catch (_) {
      return [
        jsonEncode({'event': eventName, 'data': dataText}),
      ];
    }
  }
}

class _ParsedStreamEvents {
  final List<String> events;
  final int remaining;

  const _ParsedStreamEvents({required this.events, required this.remaining});
}
