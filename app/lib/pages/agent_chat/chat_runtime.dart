part of '../agent_chat.dart';

extension _AgentChatRuntime on _AgentChatPageState {
  void _ensureSessionId() {
    _agentSessionId ??= 'agent-chat-${DateTime.now().millisecondsSinceEpoch}';
  }

  String _resolveExecutionMode(String query) {
    final pending = _agentSessionState?.lastOperationPreview;
    if (pending == null) return 'preview';
    final isConfirm = RegExp(
      r'确认执行|正式写入|开始执行|执行刚才|执行上一次|确认写入',
    ).hasMatch(query.trim());
    return isConfirm ? 'execute' : 'preview';
  }

  Future<void> _askAgent(String query) async {
    _ensureSessionId();
    final executionMode = _resolveExecutionMode(query);
    _consumedEventKeys.clear();

    setState(() {
      _isSearching = true;
      _activeChatMessage = ChatMessage(
        isUser: false,
        liveStatus: '已提交请求，正在连接 Agent 服务',
      );
      _activeResult = null;
    });
    _scrollToBottom();
    _startRunTracking();
    _startBootstrapStatusUpdates();

    _streamSubscription?.cancel();

    var hasPersisted = false;

    void fallback() async {
      if (!mounted || hasPersisted) return;
      _bootstrapTimer?.cancel();
      _bootstrapTimer = null;
      setState(() => _isSearching = true);
      _ensureRunTrackingTimer();
      try {
        final result = await AgentRecallService().query(
          query,
          executionMode: executionMode,
          sessionId: _agentSessionId,
          conversationSummary: _agentConversationSummary,
          sessionState: _agentSessionState,
        );
        if (!mounted || hasPersisted) return;
        setState(() {
          _activeResult = result;
          _activeChatMessage!.finalAnswer = result.answer;
          _isSearching = false;
        });
        _finishRunTracking();
        _rememberResponse(query, result);
        _completeRun();
        hasPersisted = true;
        await _persistAgentResponse(query);
      } catch (ex) {
        if (!mounted) return;
        setState(() => _isSearching = false);
        _finishRunTracking();
        debugPrint('[AgentChat] fallback error: $ex');
        showAppToast(context, textLocalize('agent_search_failed'));
      }
    }

    try {
      final stream = AgentRecallService().queryStream(
        query,
        executionMode: executionMode,
        sessionId: _agentSessionId,
        conversationSummary: _agentConversationSummary,
        sessionState: _agentSessionState,
      );
      _streamSubscription = stream.listen(
        (chunk) {
          if (!mounted || chunk.isEmpty) return;
          try {
            final data = jsonDecode(chunk);
            if (data is Map) {
              _consumeEvent(Map<String, dynamic>.from(data));
              setState(() {});
              if (data['event']?.toString() == 'done' && !hasPersisted) {
                hasPersisted = true;
                setState(() => _isSearching = false);
                _finishRunTracking();
                unawaited(_persistAgentResponse(query));
              }
            }
          } catch (e) {
            debugPrint('[AgentChat] parse error: $e');
          }
        },
        onError: (e) {
          if (!mounted) return;
          setState(() => _isSearching = false);
          _bootstrapTimer?.cancel();
          debugPrint('[AgentChat] stream error: $e');
          if (!hasPersisted) {
            fallback();
          }
        },
        onDone: () {
          if (!mounted) return;
          setState(() => _isSearching = false);
          _finishRunTracking();
          _bootstrapTimer?.cancel();
          if (!hasPersisted &&
              _activeChatMessage?.finalAnswer.isNotEmpty == true) {
            hasPersisted = true;
            _completeRun();
            unawaited(_persistAgentResponse(query));
          }
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _isSearching = false);
      _bootstrapTimer?.cancel();
      debugPrint('[AgentChat] stream start error: $e');
      if (!hasPersisted) {
        fallback();
      }
    }
  }

  Future<void> _persistAgentResponse(String query) async {
    final conv = _currentConversation;
    if (conv == null) return;
    final answer =
        _activeChatMessage?.finalAnswer ?? _activeResult?.answer ?? '';
    final elapsed = _runStartedAt != null && _runFinishedAt != null
        ? _runFinishedAt!.difference(_runStartedAt!).inMilliseconds
        : null;

    final agentMsg = AgentMessageRecord(
      conversationId: conv.id,
      isUser: false,
      content: answer,
      finalAnswer: answer,
      timestamp: DateTime.now(),
      agentResultJson: _activeResult != null
          ? jsonEncode(_activeResultToJson())
          : null,
      elapsedMs: elapsed,
    );
    final msgId = await _db.insertMessage(agentMsg);
    setState(() {
      _messages.add(
        AgentMessageRecord(
          id: msgId,
          conversationId: agentMsg.conversationId,
          isUser: false,
          content: answer,
          finalAnswer: answer,
          timestamp: agentMsg.timestamp,
          agentResultJson: agentMsg.agentResultJson,
          elapsedMs: elapsed,
        ),
      );
      _activeChatMessage = null;
      _activeResult = null;
    });
    await _saveCurrentConversationState();
    _scrollToBottom();
  }

  Map<String, dynamic> _activeResultToJson() {
    final r = _activeResult;
    if (r == null) return {};
    return {
      'mode': r.mode,
      'answer': r.answer,
      'evidence': r.evidence == null
          ? null
          : {
              'sceneId': r.evidence!.sceneId,
              'similarity': r.evidence!.similarity,
              'matchedFrames': r.evidence!.matchedFrames
                  .map(
                    (frame) => {
                      'imageName': frame.imageName,
                      'similarity': frame.similarity,
                      'transformMatrix': frame.transformMatrix,
                    },
                  )
                  .toList(),
            },
      'actions': r.actions
          .map(
            (action) => {
              'type': action.type,
              if (action.title != null) 'title': action.title,
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
      'top_candidates': r.candidates
          .map(
            (candidate) => {
              'scene_id': candidate.sceneId,
              'model_id': candidate.modelId,
              'score': candidate.score,
              'display_name': candidate.displayName,
              'description': candidate.description,
              'pose_image_id': candidate.poseImageId,
              'tags': candidate.tags,
              'preview_img_path': candidate.previewImgPath,
              'preview_webp_path': candidate.previewWebpPath,
              'created_at': candidate.createdAt,
              'ply_path': candidate.plyPath,
            },
          )
          .toList(),
      'selected_candidate_reason': r.selectedCandidateReason,
      if (r.followUp != null)
        'follow_up': {
          'status': r.followUp!.status,
          'kind': r.followUp!.kind,
          'message': r.followUp!.message,
          'suggested_replies': r.followUp!.suggestedReplies,
        },
      if (r.assetContext != null) 'asset_context': r.assetContext,
      if (r.compareContext != null) 'compare_context': r.compareContext,
      if (r.collectionContext != null)
        'collection_context': r.collectionContext,
      if (r.creativeContext != null) 'creative_context': r.creativeContext,
      if (r.memoryGraphContext != null)
        'memory_graph_context': r.memoryGraphContext,
      if (r.responseResolution != null)
        'response_resolution': r.responseResolution,
      if (_activeChatMessage != null && _activeChatMessage!.steps.isNotEmpty)
        'steps': _activeChatMessage!.steps
            .where((s) => s.type == 'tool_call' || s.type == 'status')
            .map(
              (s) => {
                'type': s.type,
                'content': s.content,
                if (s.toolName != null) 'tool_name': s.toolName,
                'is_completed': s.isCompleted,
              },
            )
            .toList(),
    };
  }

  void _rememberResponse(String query, AgentRecallResponse response) {
    _agentSessionState = response.sessionState;
    _agentShortTermMemory = response.shortTermMemory;
    _agentConversationSummary = _mergeSummary(
      previous: _agentConversationSummary,
      query: query,
      response: response,
    );
    _ensureSessionId();
  }

  String _mergeSummary({
    required String? previous,
    required String query,
    required AgentRecallResponse response,
  }) {
    final current = response.conversationSummary?.trim();
    final merged = <String>[
      if (previous != null && previous.trim().isNotEmpty) previous.trim(),
      if (current != null && current.isNotEmpty)
        current
      else
        '用户：${query.trim()} | Agent：${response.answer.trim()}',
    ];
    return merged
        .where((s) => s.trim().isNotEmpty)
        .join('\n')
        .split('\n')
        .where((s) => s.trim().isNotEmpty)
        .toList()
        .reversed
        .take(4)
        .toList()
        .reversed
        .join('\n');
  }

  // ── Run tracking ──

  void _startRunTracking() {
    _runStartedAt = DateTime.now();
    _runFinishedAt = null;
    _ensureRunTrackingTimer();
  }

  void _ensureRunTrackingTimer() {
    _elapsedTimer?.cancel();
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted || !_isSearching) return;
      setState(() {});
    });
  }

  void _finishRunTracking() {
    _elapsedTimer?.cancel();
    _elapsedTimer = null;
    if (_runStartedAt != null && _runFinishedAt == null) {
      _runFinishedAt = DateTime.now();
    }
  }

  Duration? get _elapsedDuration {
    if (_runStartedAt == null) return null;
    final end = (_isSearching ? null : _runFinishedAt) ?? DateTime.now();
    return end.difference(_runStartedAt!);
  }

  String _formatElapsed(Duration d) {
    if (d.inHours > 0) {
      return '${d.inHours}:${(d.inMinutes % 60).toString().padLeft(2, '0')}:${(d.inSeconds % 60).toString().padLeft(2, '0')}';
    }
    if (d.inMinutes > 0) {
      return '${d.inMinutes}:${(d.inSeconds % 60).toString().padLeft(2, '0')}';
    }
    return '${d.inSeconds}s';
  }

  // ── Bootstrap status ──

  void _startBootstrapStatusUpdates() {
    _bootstrapTimer?.cancel();
    _firstRemoteEventAt = null;
    _upsertBootstrapStatus('已提交请求，正在连接 Agent 服务', isCompleted: false);
    const stages = ['已建立请求，等待首个流式事件', '正在等待 Agent 返回编排进度', 'Agent 仍在处理中'];
    var idx = 0;
    _bootstrapTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      if (!mounted || !_isSearching || _firstRemoteEventAt != null) {
        _bootstrapTimer?.cancel();
        _bootstrapTimer = null;
        return;
      }
      _upsertBootstrapStatus(
        stages[idx < stages.length ? idx : stages.length - 1],
        isCompleted: false,
      );
      if (idx < stages.length - 1) idx++;
    });
  }

  void _upsertBootstrapStatus(String summary, {bool isCompleted = false}) {
    if (_activeChatMessage == null) return;
    if (_bootstrapStep == null ||
        !_activeChatMessage!.steps.contains(_bootstrapStep)) {
      _bootstrapStep = AgentStep(
        type: 'status',
        content: summary,
        isCompleted: isCompleted,
      );
      _activeChatMessage!.addStep(_bootstrapStep!);
    } else {
      _bootstrapStep!.updateContent(summary);
      _bootstrapStep!.isCompleted = isCompleted;
    }
    _activeChatMessage!.liveStatus = summary;
  }

  void _completeRun() {
    if (_activeChatMessage == null) return;
    _activeChatMessage!.isProcessCollapsed = true;
    final followUp = _activeResult?.followUp;
    if (followUp != null && followUp.message.trim().isNotEmpty) {
      _activeChatMessage!.liveStatus = followUp.message.trim();
    } else if (_activeChatMessage!.finalAnswer.isNotEmpty) {
      _activeChatMessage!.liveStatus = '回答完成';
    }
  }

  // ── Event consumption ──

  void _consumeEvent(Map<String, dynamic> data) {
    final event = data['event']?.toString() ?? '';
    final payload = data['data'];
    if (_activeChatMessage == null) return;

    if (event == 'ping') {
      _upsertBootstrapStatus('已建立流式连接', isCompleted: false);
      return;
    }

    if (event.isNotEmpty && _firstRemoteEventAt == null) {
      _firstRemoteEventAt = DateTime.now();
      _bootstrapTimer?.cancel();
      _bootstrapStep?.isCompleted = true;
    }

    if (event == 'status' && payload is Map) {
      final summary = payload['summary']?.toString() ?? '';
      final key = 'status:$summary';
      if (_consumedEventKeys.contains(key)) return;
      _consumedEventKeys.add(key);
      _activeChatMessage!.liveStatus = summary;
      _activeChatMessage!.addStep(
        AgentStep(type: 'status', content: summary, isCompleted: true),
      );
      return;
    }

    if (event == 'plan' && payload is Map) {
      final title = payload['title']?.toString() ?? '计划就绪';
      _activeChatMessage!.liveStatus = title;
      _activeChatMessage!.addStep(AgentStep(type: 'thought', content: title));
      return;
    }

    if (event == 'thinking' || event == 'thought') {
      final content = payload is Map
          ? payload['content']?.toString() ?? ''
          : payload?.toString() ?? '';
      if (content.isEmpty) return;
      _activeChatMessage!.liveStatus = content;
      _activeChatMessage!.addStep(AgentStep(type: 'thought', content: content));
      return;
    }

    if (event == 'tool_call' && payload is Map) {
      final toolName = payload['name']?.toString() ?? '';
      final round = payload['round']?.toString() ?? '';
      final args = payload['args'] is Map
          ? Map<String, dynamic>.from(payload['args'] as Map)
          : const <String, dynamic>{};
      final key = 'tool_call:$toolName:$round:${args.keys.join(",")}';
      if (_consumedEventKeys.contains(key)) return;
      _consumedEventKeys.add(key);
      _activeChatMessage!.liveStatus = '调用工具: $toolName';
      _activeChatMessage!.addStep(
        AgentStep(
          type: 'tool_call',
          toolName: toolName,
          content: const JsonEncoder.withIndent('  ').convert(args),
        ),
      );
      return;
    }

    if (event == 'tool_result' && payload is Map) {
      final name = payload['name']?.toString() ?? '';
      final summary = payload['summary']?.toString() ?? '工具返回结果';
      final steps = _activeChatMessage!.steps;
      for (var i = steps.length - 1; i >= 0; i--) {
        if (steps[i].type == 'tool_call' && steps[i].toolName == name) {
          steps[i].updateContent('${steps[i].content}\n\n结果: $summary');
          steps[i].isCompleted = true;
          break;
        }
      }
      _activeChatMessage!.liveStatus = summary;
      return;
    }

    if (event == 'message' && payload is Map) {
      final delta = payload['delta']?.toString() ?? '';
      final streamed = payload['streamed'] == true;
      if (delta.isEmpty) {
        if (streamed && payload['done'] == true) {
          _activeChatMessage!.liveStatus = '回答完成';
        }
        return;
      }
      _activeChatMessage!.finalAnswer = streamed
          ? '${_activeChatMessage!.finalAnswer}$delta'
          : _mergeAnswerDelta(
              current: _activeChatMessage!.finalAnswer,
              incoming: delta,
            );
      _activeChatMessage!.liveStatus = '正在生成回答...';
      return;
    }

    if (event == 'error') {
      final msg = payload is Map
          ? payload['message']?.toString() ?? 'Unknown error'
          : payload?.toString() ?? 'Unknown error';
      _activeChatMessage!.addStep(AgentStep(type: 'error', content: msg));
      _activeChatMessage!.liveStatus = '执行出错';
      return;
    }

    if (event == 'done') {
      if (payload != null && payload is Map) {
        _activeResult = AgentRecallResponse.fromJson(
          _normalizeAgentDonePayload(payload),
        );
        _rememberResponse(
          _messages
              .lastWhere((m) => m.isUser, orElse: () => _messages.last)
              .content,
          _activeResult!,
        );
        final answer = _activeResult?.answer ?? '';
        if (answer.isNotEmpty &&
            _activeChatMessage!.finalAnswer.trim().isEmpty) {
          _activeChatMessage!.finalAnswer = answer;
        }
      }
      _completeRun();
    }
  }

  Map<String, dynamic> _normalizeAgentDonePayload(
    Map<dynamic, dynamic> payload,
  ) {
    final map = Map<String, dynamic>.from(payload);
    final nestedResult = map['result'];
    if (nestedResult is Map) {
      return Map<String, dynamic>.from(nestedResult);
    }
    final nestedData = map['data'];
    if (nestedData is Map) {
      final dataMap = Map<String, dynamic>.from(nestedData);
      if (dataMap.containsKey('answer') ||
          dataMap.containsKey('top_candidates') ||
          dataMap.containsKey('candidates')) {
        return dataMap;
      }
    }
    return map;
  }

  String _mergeAnswerDelta({
    required String current,
    required String incoming,
  }) {
    final normalized = incoming.trimRight();
    if (normalized.isEmpty) return current;
    if (current.isEmpty) return normalized;
    if (normalized.startsWith(current)) return normalized;
    if (current.endsWith(normalized)) return current;
    if (normalized.contains(current)) return normalized;
    if (current.contains(normalized)) return current;
    return '$current$normalized';
  }

  void _stopSearch() {
    if (!_isSearching) return;
    _streamSubscription?.cancel();
    _bootstrapTimer?.cancel();
    setState(() {
      _isSearching = false;
      _finishRunTracking();
      _activeChatMessage?.liveStatus = '已停止';
      _activeChatMessage?.isProcessCollapsed = true;
    });
  }

  Future<void> _fetchGreeting() async {
    if (_currentConversation == null) {
      await _createNewConversation();
    }
    _ensureSessionId();
    setState(() {
      _activeChatMessage = ChatMessage(isUser: false, liveStatus: '正在加载...');
    });

    try {
      final result = await AgentRecallService().query(
        '你好',
        sessionId: _agentSessionId,
      );
      if (!mounted) return;
      setState(() {
        _activeResult = result;
        _activeChatMessage!.finalAnswer = result.answer;
      });
      _rememberResponse('你好', result);
      await _persistAgentResponse('你好');
    } catch (_) {
      if (!mounted) return;
      final fallback = '你好，我在。你可以直接告诉我想找的场景/物体、要比较的时间段，或者要整理的模型。';
      setState(() {
        _activeChatMessage!.finalAnswer = fallback;
      });
      await _persistAgentResponse('你好');
    }
  }
}
