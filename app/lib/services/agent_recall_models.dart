import 'package:flutter/foundation.dart';

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

class AgentConversationEntry {
  AgentConversationEntry({
    required this.userQuery,
    required this.timestamp,
    required this.agentMessage,
    this.agentResult,
    this.elapsed,
  });

  final String userQuery;
  final DateTime timestamp;
  final ChatMessage agentMessage;
  AgentRecallResponse? agentResult;
  Duration? elapsed;
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
    var candidates = rawCandidates
        .map(
          (item) =>
              AgentCandidate.fromJson(Map<String, dynamic>.from(item as Map)),
        )
        .toList();

    if (candidates.isEmpty) {
      final evidenceSceneId = evidenceMap?['sceneId']?.toString() ?? '';
      final evidenceModelId = evidenceMap?['modelId']?.toString() ?? '';
      final evidenceSimilarity =
          (evidenceMap?['similarity'] as num?)?.toDouble() ?? 0.0;
      final evidenceDesc = evidenceMap?['description']?.toString() ?? '';
      final evidenceTags =
          (evidenceMap?['tags'] as List?)?.map((e) => e.toString()).toList() ??
              const <String>[];

      final actions = ((json['actions'] as List?) ?? [])
          .map((a) => a is Map ? Map<String, dynamic>.from(a) : null)
          .whereType<Map<String, dynamic>>()
          .toList();
      final openScene = actions.firstWhere(
        (a) => a['type'] == 'open_scene',
        orElse: () => const <String, dynamic>{},
      );
      final payload = openScene['payload'] is Map
          ? Map<String, dynamic>.from(openScene['payload'] as Map)
          : const <String, dynamic>{};

      final synthSceneId = evidenceSceneId.isNotEmpty
          ? evidenceSceneId
          : payload['sceneId']?.toString() ?? '';
      final synthModelId = evidenceModelId.isNotEmpty
          ? evidenceModelId
          : payload['modelId']?.toString() ?? '';
      final synthPly = payload['ply']?.toString();

      if (synthSceneId.isNotEmpty && evidenceSimilarity > 0) {
        candidates = [
          AgentCandidate(
            sceneId: synthSceneId,
            modelId: synthModelId,
            score: evidenceSimilarity,
            description: evidenceDesc,
            tags: evidenceTags,
            plyPath: synthPly,
          ),
        ];
      }
    }

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
      candidates: candidates,
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
  final String? displayName;
  final List<String> tags;
  final String? previewImgPath;
  final String? createdAt;
  final String? plyPath;

  AgentCandidate({
    required this.sceneId,
    required this.modelId,
    required this.score,
    required this.description,
    this.poseImageId,
    this.displayName,
    this.tags = const [],
    this.previewImgPath,
    this.createdAt,
    this.plyPath,
  });

  factory AgentCandidate.fromJson(Map<String, dynamic> json) {
    return AgentCandidate(
      sceneId: json['scene_id']?.toString() ?? '',
      modelId: json['model_id']?.toString() ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      description: json['description']?.toString() ?? '',
      poseImageId: json['pose_image_id']?.toString(),
      displayName: json['display_name']?.toString(),
      tags: (json['tags'] as List?)
              ?.map((e) => e.toString())
              .toList() ??
          const [],
      previewImgPath: json['preview_img_path']?.toString(),
      createdAt: json['created_at']?.toString(),
      plyPath: json['ply_path']?.toString(),
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
      transformMatrix: flattenNumericList(raw),
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
      matrix: flattenNumericList(rawMatrix),
    );
  }
}

/// 递归展开嵌套数值列表为扁平 `List<double>`。
List<double>? flattenNumericList(Object? value) {
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
