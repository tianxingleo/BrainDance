import 'dart:convert';

class AgentConversation {
  final String id;
  String title;
  final DateTime createdAt;
  DateTime updatedAt;
  String? sessionId;
  String? conversationSummary;
  Map<String, dynamic>? sessionStateJson;
  Map<String, dynamic>? shortTermMemory;

  AgentConversation({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.updatedAt,
    this.sessionId,
    this.conversationSummary,
    this.sessionStateJson,
    this.shortTermMemory,
  });

  factory AgentConversation.fromMap(Map<String, dynamic> map) {
    return AgentConversation(
      id: map['id'] as String,
      title: map['title'] as String? ?? '',
      createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at'] as int),
      updatedAt: DateTime.fromMillisecondsSinceEpoch(map['updated_at'] as int),
      sessionId: map['session_id'] as String?,
      conversationSummary: map['conversation_summary'] as String?,
      sessionStateJson: map['session_state_json'] != null
          ? jsonDecode(map['session_state_json'] as String)
              as Map<String, dynamic>
          : null,
      shortTermMemory: map['short_term_memory_json'] != null
          ? jsonDecode(map['short_term_memory_json'] as String)
              as Map<String, dynamic>
          : null,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'created_at': createdAt.millisecondsSinceEpoch,
      'updated_at': updatedAt.millisecondsSinceEpoch,
      'session_id': sessionId,
      'conversation_summary': conversationSummary,
      'session_state_json':
          sessionStateJson != null ? jsonEncode(sessionStateJson) : null,
      'short_term_memory_json':
          shortTermMemory != null ? jsonEncode(shortTermMemory) : null,
    };
  }
}

class AgentMessageRecord {
  final int? id;
  final String conversationId;
  final bool isUser;
  final String content;
  final String? finalAnswer;
  final DateTime timestamp;
  final String? agentResultJson;
  final int? elapsedMs;

  AgentMessageRecord({
    this.id,
    required this.conversationId,
    required this.isUser,
    required this.content,
    this.finalAnswer,
    required this.timestamp,
    this.agentResultJson,
    this.elapsedMs,
  });

  factory AgentMessageRecord.fromMap(Map<String, dynamic> map) {
    return AgentMessageRecord(
      id: map['id'] as int?,
      conversationId: map['conversation_id'] as String,
      isUser: (map['is_user'] as int) == 1,
      content: map['content'] as String? ?? '',
      finalAnswer: map['final_answer'] as String?,
      timestamp: DateTime.fromMillisecondsSinceEpoch(map['timestamp'] as int),
      agentResultJson: map['agent_result_json'] as String?,
      elapsedMs: map['elapsed_ms'] as int?,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      if (id != null) 'id': id,
      'conversation_id': conversationId,
      'is_user': isUser ? 1 : 0,
      'content': content,
      'final_answer': finalAnswer,
      'timestamp': timestamp.millisecondsSinceEpoch,
      'agent_result_json': agentResultJson,
      'elapsed_ms': elapsedMs,
    };
  }
}
