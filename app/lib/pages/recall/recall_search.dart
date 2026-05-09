part of '../recall.dart';

extension _RecallPageSearch on _RecallPageState {
  void _restoreRecallScrollOffset(double offset) {
    if (!mounted || !_recallScrollController.hasClients) {
      return;
    }
    final position = _recallScrollController.position;
    final clampedOffset = offset.clamp(
      position.minScrollExtent,
      position.maxScrollExtent,
    );
    if ((position.pixels - clampedOffset).abs() < 0.5) {
      return;
    }
    _recallScrollController.jumpTo(clampedOffset);
  }

  void _preserveRecallScrollOffset() {
    if (!_recallScrollController.hasClients) {
      return;
    }
    final offset = _recallScrollController.offset;

    // Agent 追问会立刻重建结果卡片，连续两帧恢复可避免视口被重排推到底部。
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _restoreRecallScrollOffset(offset);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _restoreRecallScrollOffset(offset);
      });
    });
  }

  Future<void> _submitAgentFollowUp(String query) async {
    final trimmedQuery = query.trim();
    if (trimmedQuery.isEmpty) {
      return;
    }
    FocusManager.instance.primaryFocus?.unfocus();
    _searchController.value = TextEditingValue(
      text: trimmedQuery,
      selection: TextSelection.collapsed(offset: trimmedQuery.length),
    );
    _preserveRecallScrollOffset();
    await _askAgentRecall(trimmedQuery);
  }

  Future<void> _searchModels(String query) async {
    final normalizedQuery = query.trim();
    if (normalizedQuery.isEmpty) {
      _lastSearchKey = null;
      if (_searchMode == RecallSearchMode.agent) {
        _resetAgentUiState();
      }
      if (!mounted) return;
      setState(() {
        _models = List<Map<String, dynamic>>.from(_allModels);
        if (_searchMode == RecallSearchMode.localAi) {
          _localAnswer = '';
          _localReasoning = '';
          _localContextPreview = '';
        }
        _isLoading = false;
      });
      return;
    }

    // Agent 模式不做即时列表检索，仅在 submit 时调用 _askAgentRecall
    if (_searchMode == RecallSearchMode.agent) {
      return;
    }

    final cacheKey = '${_searchMode.name}:$normalizedQuery';
    final now = DateTime.now();
    final cached = _searchCache[cacheKey];
    if (cached != null &&
        now.difference(cached.createdAt) < const Duration(minutes: 2)) {
      _lastSearchKey = cacheKey;
      if (!mounted) return;
      setState(() {
        _models = cached.results
            .map((item) => Map<String, dynamic>.from(item))
            .toList();
        _isLoading = false;
      });
      return;
    }

    if (_lastSearchKey == cacheKey && !_isLoading) {
      return;
    }

    final requestId = ++_searchRequestId;
    _lastSearchKey = cacheKey;
    setState(() {
      _isLoading = true;
    });

    try {
      final results = _usesLocalIndex(_searchMode)
          ? await _localRagIndex.search(normalizedQuery)
          : await _searchModelsFromCloud(normalizedQuery);
      if (!mounted || requestId != _searchRequestId) return;
      _searchCache[cacheKey] = _RecallSearchCacheEntry(
        createdAt: now,
        results: results
            .map((item) => Map<String, dynamic>.from(item))
            .toList(),
      );
      if (_searchCache.length > 24) {
        final oldestKey = _searchCache.entries.reduce((left, right) {
          return left.value.createdAt.isBefore(right.value.createdAt)
              ? left
              : right;
        }).key;
        _searchCache.remove(oldestKey);
      }
      setState(() {
        _models = results;
        _isLoading = false;
      });
    } catch (e) {
      if (mounted && requestId == _searchRequestId) {
        setState(() {
          _isLoading = false;
        });
        debugPrint('[RecallSearch] search error: $e');
        showAppToast(context, textLocalize("recall_error_search"));
      }
    }
  }

  Future<List<Map<String, dynamic>>> _searchModelsFromCloud(
    String query,
  ) async {
    final response = await Supabase.instance.client.functions.invoke(
      'search-models',
      body: {'query': query},
    );

    final data = response.data;
    if (data is Map && data['success'] == true) {
      return List<Map<String, dynamic>>.from(data['results'] ?? []);
    }

    final errMsg = (data is Map)
        ? (data['error'] ?? textLocalize('recall_unknown_error'))
        : textLocalize('recall_server_error');
    throw Exception(errMsg);
  }

  Future<void> _askAgentRecall(String query) async {
    final trimmedQuery = query.trim();
    if (trimmedQuery.isEmpty) return;
    _ensureAgentSessionId();
    _agentLatestSubmittedQuery = trimmedQuery;
    final executionMode = _resolveAgentExecutionMode(trimmedQuery);

    setState(() {
      _isAgentSearching = true;
      _agentResult = null;
      _agentChatMessage = ChatMessage(
        isUser: false,
        liveStatus: '已提交请求，正在连接 Agent 服务',
      );
    });
    _startAgentRunTracking();
    _startAgentBootstrapStatusUpdates();

    _agentStreamSubscription?.cancel();

    void fallback() async {
      if (!mounted) return;
      _agentBootstrapTimer?.cancel();
      _agentBootstrapTimer = null;
      setState(() {
        _isAgentSearching = true;
      });
      _ensureAgentRunTrackingTimer();
      try {
        final result = await AgentRecallService().query(
          trimmedQuery,
          executionMode: executionMode,
          sessionId: _agentSessionId,
          conversationSummary: _agentConversationSummary,
          sessionState: _agentSessionState,
        );
        if (!mounted) return;
        setState(() {
          _agentResult = result;
          _agentChatMessage!.finalAnswer = result.answer;
          _isAgentSearching = false;
          _finishAgentRunTracking();
        });
        _rememberAgentResponse(trimmedQuery, result);
        _agentChatMessage!.addSummary(
          textLocalize('agent_status_single_response_fallback'),
        );
        _completeAgentRun();
      } catch (ex) {
        if (!mounted) return;
        setState(() {
          _isAgentSearching = false;
          _finishAgentRunTracking();
        });
        debugPrint('[RecallSearch] agent search error: $ex');
        showAppToast(context, textLocalize('agent_search_failed'));
      }
    }

    try {
      _updateAgentLiveStatus(
        '已发起 Agent 请求',
        detail: '正在等待服务端建立流式返回通道',
      );
      final stream = AgentRecallService().queryStream(
        trimmedQuery,
        executionMode: executionMode,
        sessionId: _agentSessionId,
        conversationSummary: _agentConversationSummary,
        sessionState: _agentSessionState,
      );
      _agentStreamSubscription = stream.listen(
        (chunk) {
          if (!mounted) return;
          if (chunk.isEmpty) return;

          try {
            final data = jsonDecode(chunk);
            if (data is Map) {
              final eventData = Map<String, dynamic>.from(data);
              _consumeAgentEvent(eventData);
              setState(() {}); // 强制刷新 UI，体现最新状态
              if (eventData['event']?.toString() == 'done') {
                setState(() {
                  _isAgentSearching = false;
                  _finishAgentRunTracking();
                });
              }
            }
          } catch (e) {
            debugPrint('Error parsing chunk: $e');
          }
        },
        onError: (e) {
          if (!mounted) return;
          setState(() {
            _isAgentSearching = false;
          });
          _agentBootstrapTimer?.cancel();
          _agentBootstrapTimer = null;
          debugPrint('[RecallSearch] agent stream error: $e');
          _updateAgentLiveStatus(
            textLocalize('agent_status_stream_fallback'),
            detail: '$e',
          );
          showAppToast(context, textLocalize('agent_stream_failed'));
          fallback();
        },
        onDone: () {
          if (mounted) {
            setState(() {
              _isAgentSearching = false;
              _finishAgentRunTracking();
            });
            _agentBootstrapTimer?.cancel();
            _agentBootstrapTimer = null;
            if (_agentChatMessage?.finalAnswer.isNotEmpty == true) {
              _completeAgentRun();
            }
          }
        },
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isAgentSearching = false;
      });
      _agentBootstrapTimer?.cancel();
      _agentBootstrapTimer = null;
      debugPrint('[RecallSearch] agent stream start error: $e');
      _updateAgentLiveStatus(
        textLocalize('agent_status_stream_start_fallback'),
        detail: '$e',
      );
      showAppToast(context, textLocalize('agent_stream_start_failed'));
      fallback();
    }
  }

  void _openAgentRecallResult(AgentRecallResponse result) {
    final openScene = result.actions
        .where((a) => a.type == 'open_scene')
        .cast<AgentAction?>()
        .firstOrNull;
    final flyToPose = result.actions
        .where((a) => a.type == 'fly_to_pose')
        .cast<AgentAction?>()
        .firstOrNull;

    if (openScene == null || openScene.ply == null || openScene.ply!.isEmpty) {
      showAppToast(context, '缺少 open_scene.ply，无法打开 Viewer');
      return;
    }

    // ply 可能是 Storage 相对路径，需转为公开 URL 才能下载
    final rawPly = openScene.ply!;
    final modelUrl =
        rawPly.startsWith('http://') || rawPly.startsWith('https://')
        ? rawPly
        : toPublicUrl(rawPly);
    final posesUrlResolved =
        openScene.poses != null &&
            openScene.poses!.isNotEmpty &&
            !openScene.poses!.startsWith('http')
        ? toPublicUrl(openScene.poses!)
        : openScene.poses ?? toPosesUrl(rawPly);

    unawaited(
      openViewer(
        context,
        initialModelUrl: modelUrl,
        posesUrl: posesUrlResolved,
        sceneId: openScene.sceneId,
        initialPose: flyToPose?.matrix,
        initialPoseId: flyToPose?.imageName,
      ),
    );
  }

  Future<void> _handleSearchSubmitted(String value) async {
    final query = value.trim();
    if (_searchMode == RecallSearchMode.localAi) {
      await _searchModels(query);
      if (query.isNotEmpty) {
        await _askLocalQuestion(question: query);
      }
      return;
    }
    if (_searchMode == RecallSearchMode.agent) {
      if (query.isNotEmpty) {
        await _askAgentRecall(query);
      }
      return;
    }
    await _searchModels(query);
  }

  bool _usesLocalIndex(RecallSearchMode mode) {
    return mode == RecallSearchMode.local || mode == RecallSearchMode.localAi;
  }

  String _resolveAgentExecutionMode(String query) {
    final pendingPreview = _agentSessionState?.lastOperationPreview;
    if (pendingPreview == null) {
      return 'preview';
    }
    final normalizedQuery = query.trim();
    final isExecuteConfirmation = RegExp(
      r'确认执行|正式写入|开始执行|执行刚才|执行上一次|确认写入',
    ).hasMatch(normalizedQuery);
    return isExecuteConfirmation ? 'execute' : 'preview';
  }

  String _formatAgentModeLabel(String mode) {
    switch (mode.trim()) {
      case 'preview':
        return '预览';
      case 'execute':
        return '执行';
      default:
        return mode;
    }
  }

  String _searchModeTitle(RecallSearchMode mode) {
    switch (mode) {
      case RecallSearchMode.cloud:
        return textLocalize('recall_cloud_rag');
      case RecallSearchMode.local:
        return textLocalize('recall_local_rag');
      case RecallSearchMode.localAi:
        return textLocalize('recall_local_ai_rag');
      case RecallSearchMode.agent:
        return textLocalize('recall_agent_rag');
    }
  }

  String _searchModeSubtitle(RecallSearchMode mode) {
    switch (mode) {
      case RecallSearchMode.cloud:
        return textLocalize('recall_cloud_scope');
      case RecallSearchMode.local:
        if (_isLocalIndexing) {
          return textLocalize('recall_local_indexing');
        }
        final base =
            '${textLocalize('recall_local_ready')} · ${textLocalize('recall_local_scope')}';
        if (_indexStats == null) {
          return base;
        }
        return '$base · ${_indexStats!.rebuiltItems}/${_indexStats!.totalItems}';
      case RecallSearchMode.localAi:
        return textLocalize('recall_local_ai_scope');
      case RecallSearchMode.agent:
        return textLocalize('recall_agent_scope');
    }
  }

  String _searchFieldHint() {
    if (_searchMode == RecallSearchMode.localAi) {
      return textLocalize('recall_local_ai_hint');
    }
    if (_searchMode == RecallSearchMode.agent) {
      final followUp = _agentResult?.followUp;
      if (followUp != null &&
          followUp.isWaitingUserInput &&
          (followUp.inputPlaceholder?.trim().isNotEmpty ?? false)) {
        return followUp.inputPlaceholder!.trim();
      }
      return textLocalize('recall_agent_hint');
    }
    return textLocalize('recall_search_hint');
  }

  void _setSearchMode(RecallSearchMode mode) {
    if (_searchMode == mode) {
      return;
    }
    setState(() {
      _searchMode = mode;
      if (mode != RecallSearchMode.localAi) {
        _localAnswer = '';
        _localReasoning = '';
        _localContextPreview = '';
      }
      if (mode != RecallSearchMode.agent) {
        _resetAgentUiState(preserveSession: false);
      }
    });
    final keyword = _searchController.text.trim();
    if (keyword.isNotEmpty) {
      unawaited(_searchModels(keyword));
    }
  }

  Future<void> _showSearchModeSheet() async {
    final selected = await showModalBottomSheet<RecallSearchMode>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return RecallSearchModeSheet(
          selectedMode: _searchMode,
          titleBuilder: _searchModeTitle,
          subtitleBuilder: _searchModeSubtitle,
          darkInput: darkInput,
          onSelect: (mode) => Navigator.pop(context, mode),
        );
      },
    );

    if (selected != null) {
      _setSearchMode(selected);
    }
  }

  Widget _buildQuickPromptChip(String text, bool isDark) {
    return ActionChip(
      label: Text(
        text,
        style: TextStyle(
          fontSize: 12,
          color: isDark ? Colors.white70 : Colors.black87,
        ),
      ),
      backgroundColor: isDark
          ? Colors.white.withOpacity(0.05)
          : Colors.black.withOpacity(0.05),
      side: BorderSide.none,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      onPressed: () {
        unawaited(_submitAgentFollowUp(text));
      },
    );
  }

  Widget _buildAgentResultCard(bool isDark, Color textColor) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    final elapsedLabel = _agentElapsedLabel;
    final chatMessage = _agentChatMessage;

    if (chatMessage == null && _agentResult == null && !_isAgentSearching) {
      return BDPanelCard(
        padding: const EdgeInsets.all(16),
        child: Text(
          textLocalize('recall_agent_panel_hint'),
          style: TextStyle(color: hintColor, fontSize: 13),
        ),
      );
    }

    final hasActions =
        _agentResult != null &&
        _agentResult!.actions.any((a) => a.type == 'open_scene');
    final topCandidates = _agentResult?.candidates.take(3).toList() ?? [];
    final followUp = _agentResult?.followUp;

    if (chatMessage == null) {
      final fallbackAnswer = _agentResult?.answer.trim() ?? '';
      final fallbackStatus = followUp?.message.trim() ?? '';
      return BDPanelCard(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              textLocalize('recall_agent_rag'),
              style: TextStyle(
                color: textColor,
                fontSize: 14,
                fontWeight: FontWeight.w700,
              ),
            ),
            if (fallbackStatus.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                fallbackStatus,
                style: TextStyle(color: hintColor, fontSize: 12.5, height: 1.4),
              ),
            ],
            if (fallbackAnswer.isNotEmpty) ...[
              const SizedBox(height: 10),
              MarkdownBody(
                data: fallbackAnswer,
                builders: {'code': _CodeElementBuilder(isDark, context)},
              ),
            ],
            if (topCandidates.isNotEmpty) ...[
              const SizedBox(height: 10),
              for (final candidate in topCandidates)
                Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Text(
                    '${candidate.sceneId} · ${(candidate.score * 100).toStringAsFixed(1)}% · ${candidate.description}',
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 12,
                      height: 1.4,
                    ),
                  ),
                ),
            ],
            if (hasActions) ...[
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                height: 40,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: BDDesign.colorMutedBlue,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    elevation: 0,
                  ),
                  onPressed: () => _openAgentRecallResult(_agentResult!),
                  icon: const Icon(Icons.open_in_new_rounded, size: 16),
                  label: const Text('打开场景', style: TextStyle(fontSize: 14)),
                ),
              ),
            ],
          ],
        ),
      );
    }

    return BDPanelCard(
      padding: const EdgeInsets.all(16),
      child: ListenableBuilder(
        listenable: chatMessage,
        builder: (context, _) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.travel_explore_rounded,
                        size: 18,
                        color: BDDesign.colorMutedBlue,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _isAgentSearching &&
                                  _agentChatMessage?.finalAnswer.isEmpty ==
                                      true &&
                                  _agentElapsedDuration != null &&
                                  _agentElapsedDuration!.inSeconds > 3
                              ? [
                                  textLocalize('agent_status_warming_up'),
                                  textLocalize(
                                    'agent_status_reviewing_context',
                                  ),
                                  textLocalize('agent_status_deep_thinking'),
                                ][((_agentElapsedDuration!.inSeconds - 3) ~/
                                        3) %
                                    3]
                              : textLocalize('recall_agent_rag'),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: textColor,
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      if (_isAgentSearching) ...[
                        const SizedBox(width: 12),
                        const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ],
                    ],
                  ),
                  if (elapsedLabel != null || _isAgentSearching) ...[
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 10,
                      runSpacing: 8,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        if (elapsedLabel != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: isDark
                                  ? Colors.white.withValues(alpha: 0.06)
                                  : BDDesign.colorMutedBlue.withValues(
                                      alpha: 0.08,
                                    ),
                              borderRadius: BorderRadius.circular(999),
                              border: Border.all(
                                color: isDark
                                    ? Colors.white.withValues(alpha: 0.08)
                                    : BDDesign.colorMutedBlue.withValues(
                                        alpha: 0.16,
                                      ),
                              ),
                            ),
                            child: Text(
                              _isAgentSearching
                                  ? textLocalize(
                                      'agent_elapsed_running',
                                    ).replaceAll('{duration}', elapsedLabel)
                                  : textLocalize(
                                      'agent_elapsed_finished',
                                    ).replaceAll('{duration}', elapsedLabel),
                              style: TextStyle(
                                color: hintColor,
                                fontSize: 11.5,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        if (_isAgentSearching)
                          SizedBox(
                            height: 24,
                            child: OutlinedButton.icon(
                              style: OutlinedButton.styleFrom(
                                foregroundColor: Colors.red,
                                side: const BorderSide(
                                  color: Colors.red,
                                  width: 1,
                                ),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8,
                                ),
                              ),
                              onPressed: _stopAgentSearch,
                              icon: const Icon(
                                Icons.stop_circle_outlined,
                                size: 14,
                              ),
                              label: Text(
                                textLocalize('agent_action_stop'),
                                style: const TextStyle(fontSize: 12),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 10),

              if (_agentChatMessage!.liveStatus.isNotEmpty) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: isDark
                        ? const Color(0xFF141A24)
                        : const Color(0xFFF3F7FC),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: isDark
                          ? Colors.white10
                          : BDDesign.colorMutedBlue.withValues(alpha: 0.18),
                    ),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(
                        _isAgentSearching
                            ? Icons.auto_awesome_motion_rounded
                            : Icons.done_all_rounded,
                        size: 18,
                        color: BDDesign.colorMutedBlue,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _agentChatMessage!.liveStatus,
                              style: TextStyle(
                                color: textColor,
                                fontSize: 13,
                                height: 1.45,
                              ),
                            ),
                            if (elapsedLabel != null) ...[
                              const SizedBox(height: 4),
                              Text(
                                _isAgentSearching
                                    ? textLocalize(
                                        'agent_elapsed_running',
                                      ).replaceAll('{duration}', elapsedLabel)
                                    : textLocalize(
                                        'agent_elapsed_finished',
                                      ).replaceAll('{duration}', elapsedLabel),
                                style: TextStyle(
                                  color: hintColor,
                                  fontSize: 11.5,
                                  height: 1.3,
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
              ],

              if (_agentChatMessage!.steps.isNotEmpty)
                _AgentProcessPanel(
                  chatMessage: _agentChatMessage!,
                  isDark: isDark,
                  textColor: textColor,
                  hintColor: hintColor,
                  isSearching: _isAgentSearching,
                  onRetry: () {
                    final query = _searchController.text.trim();
                    if (query.isNotEmpty) {
                      unawaited(_submitAgentFollowUp(query));
                    }
                  },
                ),

              if (_agentChatMessage!.finalAnswer.isNotEmpty) ...[
                const SizedBox(height: 14),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: isDark
                        ? const Color(0xFF151B12)
                        : const Color(0xFFFFFEF6),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: BDDesign.colorFadedOlive.withValues(
                        alpha: isDark ? 0.24 : 0.18,
                      ),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '最终回答',
                        style: TextStyle(
                          color: textColor,
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 10),
                      _AnimatedMarkdownAnswer(
                        data: _agentChatMessage!.finalAnswer,
                        isDark: isDark,
                        textColor: textColor,
                        hintColor: hintColor,
                      ),
                    ],
                  ),
                ),
              ],

              if (followUp != null &&
                  (followUp.message.trim().isNotEmpty ||
                      followUp.suggestedReplies.isNotEmpty)) ...[
                const SizedBox(height: 12),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: isDark
                        ? const Color(0xFF161D28)
                        : const Color(0xFFF6F9FD),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: isDark
                          ? Colors.white12
                          : BDDesign.colorMutedBlue.withValues(alpha: 0.18),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        followUp.isWaitingUserInput ? '继续对话' : '下一步建议',
                        style: TextStyle(
                          color: textColor,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      if (followUp.message.trim().isNotEmpty) ...[
                        const SizedBox(height: 6),
                        Text(
                          followUp.message.trim(),
                          style: TextStyle(
                            color: hintColor,
                            fontSize: 12,
                            height: 1.45,
                          ),
                        ),
                      ],
                      if (followUp.suggestedReplies.isNotEmpty) ...[
                        const SizedBox(height: 10),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: followUp.suggestedReplies.map((reply) {
                            return ActionChip(
                              label: Text(
                                reply,
                                style: const TextStyle(fontSize: 12),
                              ),
                              onPressed: () {
                                unawaited(_submitAgentFollowUp(reply));
                              },
                            );
                          }).toList(),
                        ),
                      ],
                    ],
                  ),
                ),
              ],

              if (_agentResult?.mode != null) ...[
                const SizedBox(height: 8),
                Text(
                  '模式：${_formatAgentModeLabel(_agentResult!.mode)}',
                  style: TextStyle(color: hintColor, fontSize: 12),
                ),
              ],

              if (_agentResult?.selectedCandidateReason != null &&
                  _agentResult!.selectedCandidateReason!.isNotEmpty) ...[
                const SizedBox(height: 4),
                Text(
                  '选择理由：${_agentResult!.selectedCandidateReason!}',
                  style: TextStyle(color: hintColor, fontSize: 12, height: 1.4),
                ),
              ],

              if (_agentResult?.evidence != null) ...[
                const SizedBox(height: 10),
                Text(
                  '场景：${_agentResult!.evidence!.sceneId}  ·  相似度：${(_agentResult!.evidence!.similarity * 100).toStringAsFixed(1)}%',
                  style: TextStyle(color: hintColor, fontSize: 12),
                ),
              ],

              if (topCandidates.isNotEmpty) ...[
                const SizedBox(height: 12),
                Text(
                  '候选结果',
                  style: TextStyle(
                    color: textColor,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                for (final candidate in topCandidates)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Text(
                      '${candidate.sceneId} · ${(candidate.score * 100).toStringAsFixed(1)}% · ${candidate.description}',
                      style: TextStyle(
                        color: hintColor,
                        fontSize: 12,
                        height: 1.4,
                      ),
                    ),
                  ),
              ],

              if (hasActions) ...[
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  height: 40,
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: BDDesign.colorMutedBlue,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      elevation: 0,
                    ),
                    icon: const Icon(Icons.open_in_new_rounded, size: 16),
                    label: const Text('打开场景', style: TextStyle(fontSize: 14)),
                    onPressed: () => _openAgentRecallResult(_agentResult!),
                  ),
                ),
              ],
            ],
          );
        },
      ),
    );
  }

  Widget _buildEmptyState(TDThemeData theme, bool isDark) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final iconColor = isDark
        ? const Color(0xFFEEEEEE)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32.0),
          border: Border.all(
            color: isDark ? darkBorder : theme.whiteColor1,
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withAlpha(20),
              blurRadius: 20,
              spreadRadius: 5,
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TDImage(
              assetUrl: 'assets/sprites/empty_state.png',
              width: 120,
              height: 120,
              errorWidget: Icon(
                TDIcons.time_filled,
                size: 80,
                color: iconColor,
              ),
            ),
            const SizedBox(height: 24),
            TDText(
              textLocalize("home_page"),
              font: theme.fontTitleLarge,
              textColor: textColor,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              textLocalize("recall_empty_title"),
              font: theme.fontBodyMedium,
              textColor: hintTextColor,
            ),
            const SizedBox(height: 40),
            TDButton(
              text: textLocalize("recall_open_demo"),
              iconWidget: Icon(
                TDIcons.view_module,
                color: Colors.white,
                size: 20,
              ),
              type: TDButtonType.fill,
              theme: TDButtonTheme.primary,
              shape: TDButtonShape.round,
              size: TDButtonSize.large,
              onTap: () {
                unawaited(
                  openViewer(
                    context,
                    initialModelUrl: '',
                    sceneId: textLocalize("recall_demo_title"),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchEmptyState(TDThemeData theme, bool isDark) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32.0),
          border: Border.all(
            color: isDark ? darkBorder : theme.whiteColor1,
            width: 1,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.travel_explore_rounded,
              size: 56,
              color: isDark
                  ? Colors.white.withValues(alpha: 0.8)
                  : BDDesign.colorMutedBlue,
            ),
            const SizedBox(height: 18),
            TDText(
              _searchModeTitle(_searchMode),
              font: theme.fontTitleLarge,
              textColor: textColor,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              switch (_searchMode) {
                RecallSearchMode.local => textLocalize('recall_local_empty'),
                RecallSearchMode.cloud => textLocalize('recall_cloud_empty'),
                RecallSearchMode.localAi => textLocalize(
                  'recall_local_ai_empty',
                ),
                RecallSearchMode.agent => textLocalize('recall_agent_empty'),
              },
              font: theme.fontBodyMedium,
              textColor: hintTextColor,
            ),
          ],
        ),
      ),
    );
  }
}
