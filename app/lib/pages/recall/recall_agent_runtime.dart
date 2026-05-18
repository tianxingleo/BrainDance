// ignore_for_file: invalid_use_of_protected_member
part of '../recall.dart';

extension _RecallPageAgentRuntime on _RecallPageState {
  void _resetAgentUiState({bool preserveSession = true}) {
    _agentStreamSubscription?.cancel();
    _agentStreamSubscription = null;
    _agentElapsedTimer?.cancel();
    _agentElapsedTimer = null;
    _agentBootstrapTimer?.cancel();
    _agentBootstrapTimer = null;
    _agentRunStartedAt = null;
    _agentRunFinishedAt = null;
    _agentFirstRemoteEventAt = null;
    _agentLatestSubmittedQuery = null;
    _isAgentSearching = false;
    _agentBootstrapStep = null;

    if (!preserveSession) {
      _agentSessionId = null;
      _agentConversationSummary = null;
      _agentSessionState = null;
      _agentConversationHistory.clear();
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
      _refreshState();
    });
  }

  void _finishAgentRunTracking() {
    _agentElapsedTimer?.cancel();
    _agentElapsedTimer = null;
    if (_agentRunStartedAt != null && _agentRunFinishedAt == null) {
      _agentRunFinishedAt = DateTime.now();
    }
    if (_agentConversationHistory.isNotEmpty &&
        _agentRunStartedAt != null &&
        _agentRunFinishedAt != null) {
      _agentConversationHistory.first.elapsed =
          _agentRunFinishedAt!.difference(_agentRunStartedAt!);
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
      _agentBootstrapTimer?.cancel();
      _agentBootstrapTimer = null;
      _refreshState(() {
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

  void _upsertAgentBootstrapStatus(
    String summary, {
    String? detail,
    bool isCompleted = false,
    bool persistSummary = true,
  }) {
    final normalizedSummary = summary.trim();
    final normalizedDetail = detail?.trim();
    final content = [
      normalizedSummary,
      if (normalizedDetail != null && normalizedDetail.isNotEmpty)
        normalizedDetail,
    ].join('\n');
    if (_agentChatMessage == null || content.isEmpty) {
      return;
    }

    if (_agentBootstrapStep == null ||
        !_agentChatMessage!.steps.contains(_agentBootstrapStep)) {
      final step = AgentStep(
        type: 'status',
        content: content,
        isCompleted: isCompleted,
      );
      _agentBootstrapStep = step;
      _agentChatMessage!.addStep(step);
    } else {
      _agentBootstrapStep!.updateContent(content);
      _agentBootstrapStep!.isCompleted = isCompleted;
    }

    _updateAgentLiveStatus(
      normalizedSummary,
      detail: normalizedDetail,
      persistSummary: persistSummary,
    );
  }

  void _startAgentBootstrapStatusUpdates() {
    _agentBootstrapTimer?.cancel();
    _agentFirstRemoteEventAt = null;
    _upsertAgentBootstrapStatus(
      '已提交请求，正在连接 Agent 服务',
      detail: '前端已发出流式请求，等待服务端建立返回通道',
      isCompleted: false,
    );

    const stages = <Map<String, String>>[
      {
        'summary': '已建立请求，等待首个流式事件',
        'detail': '如果网络正常，马上会看到 Agent 的阶段反馈',
      },
      {
        'summary': '正在等待 Agent 返回编排进度',
        'detail': '服务端可能正在装载上下文、分类意图或准备工具调用',
      },
      {
        'summary': 'Agent 仍在处理中',
        'detail': '首个详细阶段事件还没到达，先保持连接并持续等待',
      },
    ];

    var stageIndex = 0;
    _agentBootstrapTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      if (!mounted || !_isAgentSearching) {
        _agentBootstrapTimer?.cancel();
        _agentBootstrapTimer = null;
        return;
      }
      if (_agentFirstRemoteEventAt != null) {
        _agentBootstrapTimer?.cancel();
        _agentBootstrapTimer = null;
        return;
      }
      final stage = stages[stageIndex < stages.length
          ? stageIndex
          : stages.length - 1];
      _upsertAgentBootstrapStatus(
        stage['summary'] ?? '',
        detail: stage['detail'],
        isCompleted: false,
        persistSummary: false,
      );
      if (stageIndex < stages.length - 1) {
        stageIndex += 1;
      }
    });
  }

  void _markAgentRemoteEventReceived({
    String? summary,
    String? detail,
  }) {
    _agentFirstRemoteEventAt ??= DateTime.now();
    _agentBootstrapTimer?.cancel();
    _agentBootstrapTimer = null;

    if (_agentBootstrapStep != null) {
      if (summary != null && summary.trim().isNotEmpty) {
        _agentBootstrapStep!.updateContent([
          summary.trim(),
          if (detail != null && detail.trim().isNotEmpty) detail.trim(),
        ].join('\n'));
      }
      _agentBootstrapStep!.isCompleted = true;
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

    if (_agentBootstrapStep != null &&
        _agentBootstrapStep!.content.trim() == content.trim()) {
      _agentBootstrapStep!.isCompleted = isCompleted;
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

  String _mergeAgentAnswerDelta({
    required String current,
    required String incoming,
  }) {
    final normalizedIncoming = incoming.trimRight();
    if (normalizedIncoming.isEmpty) {
      return current;
    }
    if (current.isEmpty) {
      return normalizedIncoming;
    }

    // 兼容两类上游流式正文：
    // 1. 真正的增量片段，只包含这次新增内容。
    // 2. 累计片段，包含“截至当前为止的完整回答”。
    // 若直接统一 append，会把累计片段的前缀重复拼进最终回答。
    if (normalizedIncoming.startsWith(current)) {
      return normalizedIncoming;
    }
    if (current.endsWith(normalizedIncoming)) {
      return current;
    }
    if (normalizedIncoming.contains(current)) {
      return normalizedIncoming;
    }
    return '$current$normalizedIncoming';
  }

  void _consumeAgentEvent(Map<String, dynamic> data) {
    final event = data['event']?.toString() ?? '';
    final payload = data['data'];
    if (_agentChatMessage == null) {
      return;
    }

    if (event == 'ping') {
      _upsertAgentBootstrapStatus(
        '已建立流式连接',
        detail: '服务端已开始回传数据，等待 Agent 返回首个阶段状态',
        isCompleted: false,
        persistSummary: false,
      );
      return;
    }

    if (event.isNotEmpty) {
      String? summary;
      String? detail;
      if (payload is Map) {
        summary = payload['summary']?.toString();
        detail = payload['detail']?.toString();
      }
      _markAgentRemoteEventReceived(summary: summary, detail: detail);
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
      _agentChatMessage!.finalAnswer = _mergeAgentAnswerDelta(
        current: _agentChatMessage!.finalAnswer,
        incoming: delta,
      );
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
        _refreshState(() {
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
        _refreshState(() {
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

  Future<void> _fetchAgentGreeting() async {
    _ensureAgentSessionId();
    final greetingMessage = ChatMessage(isUser: false, liveStatus: '正在加载...');
    setState(() {
      _agentConversationHistory.insert(
        0,
        AgentConversationEntry(
          userQuery: '',
          timestamp: DateTime.now(),
          agentMessage: greetingMessage,
        ),
      );
    });

    try {
      final result = await AgentRecallService().query(
        '你好',
        sessionId: _agentSessionId,
      );
      if (!mounted) return;
      setState(() {
        _agentConversationHistory.first.agentResult = result;
        greetingMessage.finalAnswer = result.answer;
      });
      _rememberAgentResponse('你好', result);
    } catch (_) {
      if (!mounted) return;
      setState(() {
        greetingMessage.finalAnswer = '你好，我在。你可以直接告诉我想找的场景/物体、要比较的时间段，或者要整理的模型。';
      });
    }
  }
}
