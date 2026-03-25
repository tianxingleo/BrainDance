import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

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
  final List<AgentStep> steps;

  ChatMessage({
    required this.isUser,
    String finalAnswer = '',
    List<AgentStep>? steps,
  }) : _finalAnswer = finalAnswer,
       steps = steps ?? [];

  String get finalAnswer => _finalAnswer;
  set finalAnswer(String value) {
    _finalAnswer = value;
    notifyListeners();
  }

  void addStep(AgentStep step) {
    steps.add(step);
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

  Stream<String> queryStream(String query) async* {
    try {
      final result = await this.query(query);
      yield jsonEncode({'event': 'done', 'data': _encodeResponse(result)});
    } catch (e) {
      yield jsonEncode({'event': 'error', 'data': _normalizeInvokeError(e)});
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

  String _normalizeInvokeError(Object error) {
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
      if (error.status == 502 || error.status == 504) {
        return 'agent-recall 上游服务响应异常，请检查 Edge Function 日志和模型网关配置';
      }
      return 'agent-recall 调用失败（HTTP ${error.status}）';
    }

    return error.toString();
  }
}
