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

  Stream<String> queryStream(String query) async* {
    // 调用 Supabase 边缘函数
    try {
      final response = await _client.functions.invoke(
        'agent-recall',
        body: {'query': query},
      );

      if (response.status != 200) {
        yield jsonEncode({
          'event': 'error',
          'data': 'API Invoke Error: ${response.status}',
        });
        return;
      }

      // 注意：如果函数返回的是流式响应，需确认 invoke 的处理方式
      // 这里的原始代码似乎是想手动处理流式响应，但 supabase_flutter 的 invoke 默认返回全部数据
      // 如果业务确实需要流式，通常需要更复杂的 http 处理或函数支持
      yield response.data.toString();
    } catch (e) {
      if (e is FunctionException) {
        yield jsonEncode({
          'event': 'error',
          'data': e.details ?? e.status.toString(),
        });
      } else {
        yield jsonEncode({'event': 'error', 'data': 'Function Error: $e'});
      }
    }
  }

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
