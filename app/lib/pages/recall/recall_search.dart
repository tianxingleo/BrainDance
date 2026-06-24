// ignore_for_file: invalid_use_of_protected_member
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

  /// 用本地已加载的 _allModels 补全云端搜索结果缺失的 display_name
  void _enrichDisplayNames(List<Map<String, dynamic>> results) {
    if (_allModels.isEmpty || results.isEmpty) return;

    // 构建 id -> display_name 和 scene_id -> display_name 的查找表
    final displayNameById = <String, String>{};
    final displayNameBySceneId = <String, String>{};
    for (final m in _allModels) {
      final dn = m['display_name']?.toString().trim() ?? '';
      if (dn.isEmpty) continue;
      final id = m['id']?.toString() ?? '';
      if (id.isNotEmpty) displayNameById[id] = dn;
      final sid = m['scene_id']?.toString() ?? '';
      if (sid.isNotEmpty) displayNameBySceneId[sid] = dn;
    }

    for (final row in results) {
      final existing = row['display_name']?.toString().trim() ?? '';
      if (existing.isNotEmpty) continue;
      final id = row['id']?.toString() ?? '';
      final sid = row['scene_id']?.toString() ?? '';
      final dn = displayNameById[id] ?? displayNameBySceneId[sid];
      if (dn != null) row['display_name'] = dn;
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
      final results = List<Map<String, dynamic>>.from(data['results'] ?? []);
      _enrichDisplayNames(results);
      return results;
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
      final newMessage = ChatMessage(
        isUser: false,
        liveStatus: '已提交请求，正在连接 Agent 服务',
      );
      _agentConversationHistory.insert(
        0,
        AgentConversationEntry(
          userQuery: trimmedQuery,
          timestamp: DateTime.now(),
          agentMessage: newMessage,
        ),
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
      _updateAgentLiveStatus('已发起 Agent 请求', detail: '正在等待服务端建立流式返回通道');
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

  VoidCallback? _buildAssetCardOnOpen(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString() ?? '';
    final sceneId = model['scene_id']?.toString() ?? '';
    if (plyPath.isEmpty || sceneId.isEmpty) return null;
    return () {
      final modelUrl =
          plyPath.startsWith('http://') || plyPath.startsWith('https://')
          ? plyPath
          : _toPublicUrl(plyPath);
      final posesUrl = _toPosesUrl(plyPath);
      unawaited(
        openViewer(
          context,
          initialModelUrl: modelUrl,
          posesUrl: posesUrl,
          sceneId: sceneId,
        ),
      );
    };
  }

  VoidCallback? _buildCandidateCardOnOpen(
    AgentCandidate candidate,
    int index,
    AgentRecallResponse? result,
  ) {
    if (index == 0 && result != null) {
      return () => _openAgentRecallResult(result);
    }
    final plyPath = candidate.plyPath ?? '';
    if (plyPath.isEmpty || candidate.sceneId.isEmpty) return null;
    return () {
      final modelUrl =
          plyPath.startsWith('http://') || plyPath.startsWith('https://')
          ? plyPath
          : _toPublicUrl(plyPath);
      final posesUrl = _toPosesUrl(plyPath);
      unawaited(
        openViewer(
          context,
          initialModelUrl: modelUrl,
          posesUrl: posesUrl,
          sceneId: candidate.sceneId,
        ),
      );
    };
  }

  Future<void> _handleSearchSubmitted(String value) async {
    final query = value.trim();
    if (_searchMode == RecallSearchMode.localAi) {
      _presetGenerationId++;
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
        _presetGenerationId++;
      }
      if (mode != RecallSearchMode.agent) {
        _resetAgentUiState(preserveSession: false);
      }
    });
    if (mode == RecallSearchMode.agent && _agentConversationHistory.isEmpty) {
      unawaited(_fetchAgentGreeting());
    }
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

  Widget _buildAgentConversationList(bool isDark, Color textColor) {
    if (_agentConversationHistory.isEmpty && !_isAgentSearching) {
      final hintColor = isDark
          ? Colors.white.withValues(alpha: 0.62)
          : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
      return BDPanelCard(
        padding: const EdgeInsets.all(16),
        child: Text(
          textLocalize('recall_agent_panel_hint'),
          style: TextStyle(color: hintColor, fontSize: 13),
        ),
      );
    }

    return Column(
      children: [
        for (int i = 0; i < _agentConversationHistory.length; i++) ...[
          if (_agentConversationHistory[i].userQuery.isNotEmpty)
            _buildUserBubble(
              _agentConversationHistory[i].userQuery,
              isDark,
              textColor,
            ),
          if (_agentConversationHistory[i].userQuery.isNotEmpty)
            const SizedBox(height: 8),
          if (i == 0)
            _buildAgentResultCard(isDark, textColor)
          else
            RepaintBoundary(
              child: _buildCompletedAgentCard(
                _agentConversationHistory[i],
                isDark,
                textColor,
              ),
            ),
          if (i < _agentConversationHistory.length - 1)
            const SizedBox(height: 16),
        ],
      ],
    );
  }

  Widget _buildUserBubble(String query, bool isDark, Color textColor) {
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.sizeOf(context).width * 0.75,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: isDark
              ? Colors.white.withValues(alpha: 0.08)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(16),
            topRight: Radius.circular(4),
            bottomLeft: Radius.circular(16),
            bottomRight: Radius.circular(16),
          ),
        ),
        child: Text(
          query,
          style: TextStyle(color: textColor, fontSize: 14, height: 1.4),
        ),
      ),
    );
  }

  Widget _buildCompletedAgentCard(
    AgentConversationEntry entry,
    bool isDark,
    Color textColor,
  ) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    final result = entry.agentResult;
    final answer = entry.agentMessage.finalAnswer.trim();
    final hasActions =
        result != null && result.actions.any((a) => a.type == 'open_scene');
    final topCandidates = result?.candidates.take(3).toList() ?? [];

    // For asset_metadata mode, extract models from assetContext
    final isAssetMode = result?.mode == 'asset_metadata';
    final assetModels = <Map<String, dynamic>>[];
    if (isAssetMode && result?.assetContext != null) {
      final ctx = result!.assetContext!;
      final bundle = ctx['bundle'] as List?;
      final list = ctx['list'] as List?;
      final source = bundle ?? list;
      if (source != null) {
        for (final item in source.take(5)) {
          if (item is Map) {
            final m = Map<String, dynamic>.from(item);
            final rawImg = m['preview_img_path']?.toString() ?? '';
            if (rawImg.isNotEmpty) {
              m['preview_img_path'] = _normalizeStorageUrl(rawImg);
            }
            materializePreviewWebpPath(m, normalize: _normalizeStorageUrl);
            assetModels.add(m);
          }
        }
      }
    }

    return BDPanelCard(
      padding: const EdgeInsets.all(16),
      child: Column(
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
                  textLocalize('recall_agent_rag'),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              if (entry.elapsed != null)
                Text(
                  _formatDuration(entry.elapsed!),
                  style: TextStyle(color: hintColor, fontSize: 11.5),
                ),
            ],
          ),
          if (answer.isNotEmpty) ...[
            const SizedBox(height: 10),
            _AnimatedMarkdownAnswer(
              data: answer,
              isDark: isDark,
              textColor: textColor,
              hintColor: hintColor,
            ),
          ],
          if (isAssetMode && assetModels.isNotEmpty) ...[
            const SizedBox(height: 12),
            for (final model in assetModels)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: AgentAssetCard(
                  displayName: model['display_name']?.toString(),
                  description: model['description']?.toString() ?? '',
                  tags:
                      (model['tags'] as List?)
                          ?.map((e) => e.toString())
                          .toList() ??
                      const [],
                  previewImgPath: model['preview_img_path']?.toString(),
                  previewWebpPath: readPreviewWebpPath(model),
                  isDark: isDark,
                  onOpen: _buildAssetCardOnOpen(model),
                ),
              ),
          ] else if (topCandidates.isNotEmpty) ...[
            const SizedBox(height: 12),
            for (int i = 0; i < topCandidates.length; i++)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: AgentAssetCard(
                  displayName:
                      topCandidates[i].displayName ??
                      (topCandidates[i].description.isNotEmpty
                          ? topCandidates[i].description
                          : topCandidates[i].sceneId),
                  description: topCandidates[i].description.isNotEmpty
                      ? topCandidates[i].description
                      : '场景 ${topCandidates[i].sceneId}',
                  tags: topCandidates[i].tags,
                  previewImgPath: topCandidates[i].previewImgPath,
                  previewWebpPath: topCandidates[i].previewWebpPath,
                  score: topCandidates[i].score,
                  isDark: isDark,
                  actionLabel: i == 0 ? '飞到视角' : '打开场景',
                  onOpen: _buildCandidateCardOnOpen(
                    topCandidates[i],
                    i,
                    result,
                  ),
                ),
              ),
          ],
          if (hasActions &&
              !(isAssetMode && assetModels.isNotEmpty) &&
              topCandidates.isEmpty) ...[
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
                onPressed: () => _openAgentRecallResult(result!),
                icon: const Icon(Icons.open_in_new_rounded, size: 16),
                label: const Text('打开场景', style: TextStyle(fontSize: 14)),
              ),
            ),
          ],
        ],
      ),
    );
  }

  String _formatDuration(Duration d) {
    if (d.inSeconds < 60) return '${d.inSeconds}s';
    return '${d.inMinutes}m ${d.inSeconds % 60}s';
  }

  Widget _buildAgentResultCard(bool isDark, Color textColor) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    final elapsedLabel = _agentElapsedLabel;
    final chatMessage = _agentChatMessage;

    if (chatMessage == null && _agentResult == null && !_isAgentSearching) {
      return const SizedBox.shrink();
    }

    final hasActions =
        _agentResult != null &&
        _agentResult!.actions.any((a) => a.type == 'open_scene');
    final topCandidates = _agentResult?.candidates.take(3).toList() ?? [];
    final followUp = _agentResult?.followUp;

    // For asset_metadata mode, extract models from assetContext
    final isActiveAssetMode = _agentResult?.mode == 'asset_metadata';
    final activeAssetModels = <Map<String, dynamic>>[];
    if (isActiveAssetMode && _agentResult?.assetContext != null) {
      final ctx = _agentResult!.assetContext!;
      final bundle = ctx['bundle'] as List?;
      final list = ctx['list'] as List?;
      final source = bundle ?? list;
      if (source != null) {
        for (final item in source.take(5)) {
          if (item is Map) {
            final m = Map<String, dynamic>.from(item);
            final rawImg = m['preview_img_path']?.toString() ?? '';
            if (rawImg.isNotEmpty) {
              m['preview_img_path'] = _normalizeStorageUrl(rawImg);
            }
            materializePreviewWebpPath(m, normalize: _normalizeStorageUrl);
            activeAssetModels.add(m);
          }
        }
      }
    }

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
            if (isActiveAssetMode && activeAssetModels.isNotEmpty) ...[
              const SizedBox(height: 12),
              for (final model in activeAssetModels)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: AgentAssetCard(
                    displayName: model['display_name']?.toString(),
                    description: model['description']?.toString() ?? '',
                    tags:
                        (model['tags'] as List?)
                            ?.map((e) => e.toString())
                            .toList() ??
                        const [],
                    previewImgPath: model['preview_img_path']?.toString(),
                    previewWebpPath: readPreviewWebpPath(model),
                    isDark: isDark,
                    onOpen: _buildAssetCardOnOpen(model),
                  ),
                ),
            ] else if (topCandidates.isNotEmpty) ...[
              const SizedBox(height: 12),
              for (int i = 0; i < topCandidates.length; i++)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: AgentAssetCard(
                    displayName:
                        topCandidates[i].displayName ??
                        (topCandidates[i].description.isNotEmpty
                            ? topCandidates[i].description
                            : topCandidates[i].sceneId),
                    description: topCandidates[i].description.isNotEmpty
                        ? topCandidates[i].description
                        : '场景 ${topCandidates[i].sceneId}',
                    tags: topCandidates[i].tags,
                    previewImgPath: topCandidates[i].previewImgPath,
                    previewWebpPath: topCandidates[i].previewWebpPath,
                    score: topCandidates[i].score,
                    isDark: isDark,
                    actionLabel: i == 0 ? '飞到视角' : '打开场景',
                    onOpen: _buildCandidateCardOnOpen(
                      topCandidates[i],
                      i,
                      _agentResult,
                    ),
                  ),
                ),
            ],
            if (hasActions &&
                !(isActiveAssetMode && activeAssetModels.isNotEmpty) &&
                topCandidates.isEmpty) ...[
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

              if (!isActiveAssetMode || activeAssetModels.isEmpty) ...[
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
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 12,
                      height: 1.4,
                    ),
                  ),
                ],

                if (_agentResult?.evidence != null) ...[
                  const SizedBox(height: 10),
                  Text(
                    '场景：${_agentResult!.evidence!.sceneId}  ·  相似度：${(_agentResult!.evidence!.similarity * 100).toStringAsFixed(1)}%',
                    style: TextStyle(color: hintColor, fontSize: 12),
                  ),
                ],
              ],

              if (isActiveAssetMode && activeAssetModels.isNotEmpty) ...[
                const SizedBox(height: 12),
                for (final model in activeAssetModels)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: AgentAssetCard(
                      displayName: model['display_name']?.toString(),
                      description: model['description']?.toString() ?? '',
                      tags:
                          (model['tags'] as List?)
                              ?.map((e) => e.toString())
                              .toList() ??
                          const [],
                      previewImgPath: model['preview_img_path']?.toString(),
                      previewWebpPath: readPreviewWebpPath(model),
                      isDark: isDark,
                      onOpen: _buildAssetCardOnOpen(model),
                    ),
                  ),
              ] else if (topCandidates.isNotEmpty) ...[
                const SizedBox(height: 12),
                for (int i = 0; i < topCandidates.length; i++)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: AgentAssetCard(
                      displayName:
                          topCandidates[i].displayName ??
                          (topCandidates[i].description.isNotEmpty
                              ? topCandidates[i].description
                              : topCandidates[i].sceneId),
                      description: topCandidates[i].description.isNotEmpty
                          ? topCandidates[i].description
                          : '场景 ${topCandidates[i].sceneId}',
                      tags: topCandidates[i].tags,
                      previewImgPath: topCandidates[i].previewImgPath,
                      previewWebpPath: topCandidates[i].previewWebpPath,
                      score: topCandidates[i].score,
                      isDark: isDark,
                      actionLabel: i == 0 ? '飞到视角' : '打开场景',
                      onOpen: _buildCandidateCardOnOpen(
                        topCandidates[i],
                        i,
                        _agentResult,
                      ),
                    ),
                  ),
              ],

              if (hasActions &&
                  !(isActiveAssetMode && activeAssetModels.isNotEmpty) &&
                  topCandidates.isEmpty) ...[
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

}
