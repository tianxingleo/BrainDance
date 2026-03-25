import 'package:supabase_flutter/supabase_flutter.dart';

// ── Data models ──────────────────────────────────────────────

class AgentRecallResponse {
  final String answer;
  final AgentEvidence? evidence;
  final List<AgentAction> actions;

  AgentRecallResponse({
    required this.answer,
    required this.evidence,
    required this.actions,
  });

  factory AgentRecallResponse.fromJson(Map<String, dynamic> json) {
    return AgentRecallResponse(
      answer: json['answer']?.toString() ?? '',
      evidence: json['evidence'] == null
          ? null
          : AgentEvidence.fromJson(
              Map<String, dynamic>.from(json['evidence'] as Map),
            ),
      actions: ((json['actions'] as List?) ?? [])
          .map(
            (item) =>
                AgentAction.fromJson(Map<String, dynamic>.from(item as Map)),
          )
          .toList(),
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

  Future<AgentRecallResponse> query(String query) async {
    final response = await _client.functions.invoke(
      'agent-recall',
      body: {'query': query},
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
