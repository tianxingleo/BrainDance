part of '../recall.dart';

extension _RecallPageAgentRuntime on _RecallPageState {
  void _resetAgentUiState({bool preserveSession = true}) {
    _agentStreamSubscription?.cancel();
    _agentStreamSubscription = null;
    _agentElapsedTimer?.cancel();
    _agentElapsedTimer = null;
    _agentRunStartedAt = null;
    _agentRunFinishedAt = null;
    _agentLatestSubmittedQuery = null;
    _isAgentSearching = false;
    _agentResult = null;
    _agentChatMessage = null;

    if (!preserveSession) {
      _agentSessionId = null;
      _agentConversationSummary = null;
      _agentSessionState = null;
    }
  }

  void _startAgentRunTracking() {
    _agentRunStartedAt = DateTime.now();
    _agentRunFinishedAt = null;
    _ensureAgentRunTrackingTimer();
  }

  void _ensureAgentRunTrackingTimer() {
    _agentElapsedTimer?.cancel();
    _agentElapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted || !_isAgentSearching) {
        return;
      }
      setState(() {});
    });
  }

  void _finishAgentRunTracking() {
    _agentElapsedTimer?.cancel();
    _agentElapsedTimer = null;
    if (_agentRunStartedAt != null && _agentRunFinishedAt == null) {
      _agentRunFinishedAt = DateTime.now();
    }
  }

  Duration? get _agentElapsedDuration {
    final startedAt = _agentRunStartedAt;
    if (startedAt == null) {
      return null;
    }
    final endedAt =
        (_isAgentSearching ? null : _agentRunFinishedAt) ?? DateTime.now();
    return endedAt.difference(startedAt);
  }

  String _formatAgentElapsed(Duration duration) {
    final hours = duration.inHours;
    final minutes = duration.inMinutes.remainder(60);
    final seconds = duration.inSeconds.remainder(60);
    if (hours > 0) {
      return '$hours:${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
    }
    if (duration.inMinutes > 0) {
      return '${duration.inMinutes}:${seconds.toString().padLeft(2, '0')}';
    }
    return '${duration.inSeconds}s';
  }

  String? get _agentElapsedLabel {
    final duration = _agentElapsedDuration;
    if (duration == null) {
      return null;
    }
    return _formatAgentElapsed(duration);
  }

  void _stopAgentSearch() {
    if (_isAgentSearching) {
      _agentStreamSubscription?.cancel();
      setState(() {
        _isAgentSearching = false;
        _finishAgentRunTracking();
        _agentChatMessage?.isProcessCollapsed = false;
        _agentChatMessage?.liveStatus = textLocalize('agent_status_stopped');
        _agentChatMessage?.addSummary(
          textLocalize('agent_status_stopped_summary'),
        );
        _agentChatMessage?.addStep(
          AgentStep(
            type: 'error',
            content: textLocalize('agent_status_force_interrupted'),
          ),
        );
      });
    }
  }

  void _updateAgentLiveStatus(
    String summary, {
    String? detail,
    bool persistSummary = true,
  }) {
    final message = [
      summary.trim(),
      if (detail != null && detail.trim().isNotEmpty) detail.trim(),
    ].join(' · ');
    if (_agentChatMessage == null || message.isEmpty) {
      return;
    }
    _agentChatMessage!.liveStatus = message;
    if (persistSummary) {
      _agentChatMessage!.addSummary(message);
    }
  }

  void _pushAgentStatusStep(
    String summary, {
    String? detail,
    bool isCompleted = true,
  }) {
    if (_agentChatMessage == null) {
      return;
    }
    final normalizedSummary = summary.trim();
    final normalizedDetail = detail?.trim();
    final content = [
      normalizedSummary,
      if (normalizedDetail != null && normalizedDetail.isNotEmpty)
        normalizedDetail,
    ].join('\n');
    if (content.isEmpty) {
      return;
    }

    final steps = _agentChatMessage!.steps;
    if (steps.isNotEmpty) {
      final last = steps.last;
      if (last.type == 'status' && last.content.trim() == content.trim()) {
        last.isCompleted = isCompleted;
        return;
      }
    }

    _agentChatMessage!.addStep(
      AgentStep(type: 'status', content: content, isCompleted: isCompleted),
    );
  }

  AgentStep? _findLastToolStep(String name) {
    for (var index = _agentChatMessage!.steps.length - 1; index >= 0; index--) {
      final step = _agentChatMessage!.steps[index];
      if (step.type == 'tool_call' && step.toolName == name) {
        return step;
      }
    }
    return null;
  }

  String _formatToolStepContent({
    required Map<String, dynamic> args,
    String? resultSummary,
  }) {
    final sections = <String>[
      const JsonEncoder.withIndent('  ').convert(args.isEmpty ? {} : args),
    ];
    final normalizedSummary = resultSummary?.trim();
    if (normalizedSummary != null && normalizedSummary.isNotEmpty) {
      sections.add('结果摘要:\n$normalizedSummary');
    }
    return sections.join('\n\n');
  }

  void _syncToolTraceSteps(List<AgentToolTrace> traces) {
    if (_agentChatMessage == null || traces.isEmpty) {
      return;
    }
    for (final trace in traces) {
      final content = _formatToolStepContent(
        args: trace.args,
        resultSummary: trace.resultSummary,
      );
      final existing = _findLastToolStep(trace.toolName);
      if (existing != null) {
        existing.updateContent(content);
        existing.isCompleted = true;
        continue;
      }
      _agentChatMessage!.addStep(
        AgentStep(
          type: 'tool_call',
          toolName: trace.toolName,
          content: content,
          isCompleted: true,
        ),
      );
    }
  }

  void _completeAgentRun({bool collapseProcess = true}) {
    if (_agentChatMessage == null) {
      return;
    }
    _finishAgentRunTracking();
    _agentChatMessage!.isProcessCollapsed = collapseProcess;
    final followUp = _agentResult?.followUp;
    if (followUp != null && followUp.message.trim().isNotEmpty) {
      _agentChatMessage!.liveStatus = followUp.message.trim();
    } else if (_agentChatMessage!.finalAnswer.isNotEmpty) {
      _agentChatMessage!.liveStatus = textLocalize(
        'agent_status_final_answer_ready',
      );
    }
  }

  void _ensureAgentSessionId() {
    _agentSessionId ??= 'recall-agent-${DateTime.now().millisecondsSinceEpoch}';
  }

  void _rememberAgentResponse(String query, AgentRecallResponse response) {
    _agentSessionState = response.sessionState;
    _agentConversationSummary = _mergeAgentConversationSummary(
      previous: _agentConversationSummary,
      query: query,
      response: response,
    );
    _ensureAgentSessionId();
  }

  String _mergeAgentConversationSummary({
    required String? previous,
    required String query,
    required AgentRecallResponse response,
  }) {
    final currentSummary = response.conversationSummary?.trim();
    final merged = <String>[
      if (previous != null && previous.trim().isNotEmpty) previous.trim(),
      if (currentSummary != null && currentSummary.isNotEmpty)
        currentSummary
      else
        '用户：${query.trim()} | Agent：${response.answer.trim()}',
    ];
    return merged
        .where((item) => item.trim().isNotEmpty)
        .join('\n')
        .split('\n')
        .where((item) => item.trim().isNotEmpty)
        .toList()
        .reversed
        .take(4)
        .toList()
        .reversed
        .join('\n');
  }

  void _consumeAgentEvent(Map<String, dynamic> data) {
    final event = data['event']?.toString() ?? '';
    final payload = data['data'];
    if (_agentChatMessage == null) {
      return;
    }

    if (event == 'status' && payload is Map) {
      final summary = payload['summary']?.toString() ?? '';
      final detail = payload['detail']?.toString();
      _updateAgentLiveStatus(summary, detail: detail);
      _pushAgentStatusStep(summary, detail: detail);
      return;
    }

    if (event == 'plan' && payload is Map) {
      final title = payload['title']?.toString() ?? '';
      final stepsStr = (payload['steps'] as List?)?.join('\n') ?? '';
      final content = 'Plan: $title\n$stepsStr'.trim();
      _updateAgentLiveStatus(
        title.isEmpty ? textLocalize('agent_status_plan_ready') : title,
      );
      _pushAgentStatusStep(
        title.isEmpty ? textLocalize('agent_status_plan_ready') : title,
        detail: stepsStr.isEmpty ? null : stepsStr,
      );
      _agentChatMessage!.addStep(AgentStep(type: 'thought', content: content));
      return;
    }

    if (event == 'thinking' || event == 'thought') {
      final content = payload is Map
          ? payload['content']?.toString() ?? ''
          : payload?.toString() ?? '';
      if (content.isEmpty) {
        return;
      }
      _updateAgentLiveStatus(content);
      _agentChatMessage!.addStep(AgentStep(type: 'thought', content: content));
      return;
    }

    if (event == 'tool_call' && payload is Map) {
      final toolName = payload['name']?.toString() ?? '';
      final args = payload['args'] is Map
          ? Map<String, dynamic>.from(payload['args'] as Map)
          : const <String, dynamic>{};
      final summary =
          payload['summary']?.toString() ??
          '${textLocalize('agent_status_tool_start')} ${toolName.isEmpty ? textLocalize('agent_status_tool_unnamed') : toolName}';
      _updateAgentLiveStatus(
        summary,
        detail: textLocalize('agent_status_waiting_tool_result'),
      );
      _pushAgentStatusStep(
        summary,
        detail: textLocalize('agent_status_waiting_tool_result'),
      );
      _agentChatMessage!.addStep(
        AgentStep(
          type: 'tool_call',
          toolName: toolName,
          content: _formatToolStepContent(args: args),
        ),
      );
      return;
    }

    if (event == 'tool_result' && payload is Map) {
      final name = payload['name']?.toString() ?? '';
      final summary =
          payload['summary']?.toString() ??
          textLocalize('agent_status_tool_result_ready');
      final lastTool = _findLastToolStep(name);
      if (lastTool != null) {
        final existing = lastTool.content.trim();
        lastTool.updateContent(
          existing.isEmpty ? '结果摘要:\n$summary' : '$existing\n\n结果摘要:\n$summary',
        );
        lastTool.isCompleted = true;
      } else {
        _agentChatMessage!.addStep(
          AgentStep(
            type: 'tool_call',
            toolName: name,
            content: _formatToolStepContent(
              args: const <String, dynamic>{},
              resultSummary: summary,
            ),
            isCompleted: true,
          ),
        );
      }
      _updateAgentLiveStatus(summary);
      _pushAgentStatusStep(summary);
      return;
    }

    if (event == 'message' && payload is Map) {
      final delta = payload['delta']?.toString() ?? '';
      if (delta.isEmpty) {
        return;
      }
      _agentChatMessage!.finalAnswer += delta;
      _agentChatMessage!.liveStatus = textLocalize(
        'agent_status_generating_final_answer',
      );
      return;
    }

    if (event == 'error') {
      String errorMsg = 'Unknown error';
      if (payload is Map) {
        errorMsg =
            payload['message']?.toString() ??
            payload['error']?.toString() ??
            'Unknown error';
      } else if (payload != null) {
        errorMsg = payload.toString();
      }
      _updateAgentLiveStatus(
        textLocalize('agent_status_execution_failed'),
        detail: errorMsg,
      );
      _pushAgentStatusStep(
        textLocalize('agent_status_execution_failed'),
        detail: errorMsg,
      );
      _agentChatMessage!.addStep(AgentStep(type: 'error', content: errorMsg));
      return;
    }

    if (event == 'done') {
      if (payload != null && payload is Map) {
        setState(() {
          _agentResult = AgentRecallResponse.fromJson(
            Map<String, dynamic>.from(payload),
          );
        });
        if (_agentResult != null) {
          _rememberAgentResponse(
            _agentLatestSubmittedQuery ?? _searchController.text.trim(),
            _agentResult!,
          );
          _syncToolTraceSteps(_agentResult!.toolTrace);
        }
        final answer = _agentResult?.answer ?? '';
        if (answer.isNotEmpty &&
            _agentChatMessage!.finalAnswer.trim().isEmpty) {
          _agentChatMessage!.finalAnswer = answer;
        }
      } else if (data['result'] != null && data['result'] is Map) {
        setState(() {
          _agentResult = AgentRecallResponse.fromJson(
            Map<String, dynamic>.from(data['result']),
          );
        });
        if (_agentResult != null) {
          _syncToolTraceSteps(_agentResult!.toolTrace);
        }
      }
      _completeAgentRun();
    }
  }
}
