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
    final rawCandidates = (json['top_candidates'] as List?) ?? (json['candidates'] as List?) ?? const [];
    final rawEvidence = json['evidence'];
    final evidenceMap = rawEvidence is Map ? Map<String, dynamic>.from(rawEvidence as Map) : null;
    return AgentRecallResponse(
      mode: json['mode']?.toString() ?? 'spatial_search',
      answer: json['answer']?.toString() ?? '',
      evidence: evidenceMap == null || (!evidenceMap.containsKey('sceneId') && !evidenceMap.containsKey('similarity'))
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
    final raw = json['transformMatrix'];
    return AgentMatchedFrame(
      imageName: json['imageName']?.toString() ?? '',
      similarity: (json['similarity'] as num?)?.toDouble() ?? 0,
      transformMatrix: raw is List
          ? raw.map((e) => (e as num).toDouble()).toList()
          : null,
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
    final rawMatrix = json['matrix'];
    return AgentAction(
      type: json['type']?.toString() ?? '',
      sceneId: json['sceneId']?.toString() ?? '',
      modelId: json['modelId']?.toString(),
      ply: json['ply']?.toString(),
      poses: json['poses']?.toString(),
      imageName: json['imageName']?.toString(),
      matrix: rawMatrix is List
          ? rawMatrix.map((e) => (e as num).toDouble()).toList()
          : null,
    );
  }
}

// ── Service ──────────────────────────────────────────────────

class AgentRecallService {
  final SupabaseClient _client = Supabase.instance.client;

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
    final response = await _client.functions.invoke(
      'agent-recall',
      body: {
        'query': query,
        if (selectedModelIds != null) 'selectedModelIds': selectedModelIds,
        'executionMode': executionMode,
        if (currentSceneId != null) 'currentSceneId': currentSceneId,
        if (currentModelId != null) 'currentModelId': currentModelId,
        if (currentMode != null) 'currentMode': currentMode,
        if (candidateSceneIds != null) 'candidateSceneIds': candidateSceneIds,
        if (sessionId != null) 'sessionId': sessionId,
        if (conversationSummary != null) 'conversationSummary': conversationSummary,
      },
    );

    final data = response.data;
    if (data is! Map) {
      throw Exception('agent-recall 返回格式错误');
    }

    if (data['error'] != null) {
      throw Exception(data['error'].toString());
    }

    return AgentRecallResponse.fromJson(Map<String, dynamic>.from(data));
  }
}
    final response = await _client.functions.invoke(
      'agent-recall',
      body: {
        'query': query,
        if (selectedModelIds != null) 'selectedModelIds': selectedModelIds,
        'executionMode': executionMode,
        if (currentSceneId != null) 'currentSceneId': currentSceneId,
        if (currentModelId != null) 'currentModelId': currentModelId,
        if (currentMode != null) 'currentMode': currentMode,
        if (candidateSceneIds != null) 'candidateSceneIds': candidateSceneIds,
        if (sessionId != null) 'sessionId': sessionId,
        if (conversationSummary != null) 'conversationSummary': conversationSummary,
      },
    );

    final data = response.data;
    if (data is! Map) {
      throw Exception('agent-recall 返回格式错误');
    }

    if (data['error'] != null) {
      throw Exception(data['error'].toString());
    }

    return AgentRecallResponse.fromJson(Map<String, dynamic>.from(data));
  }
}
