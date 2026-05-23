part of '../agent_chat.dart';

extension _AgentChatView on _AgentChatPageState {
  Widget _buildAppBar(bool isDark) {
    final textColor = isDark ? Colors.white : BDDesign.colorInkBlack;
    final title = _currentConversation?.title.isNotEmpty == true
        ? _currentConversation!.title
        : textLocalize('agent');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        children: [
          Builder(
            builder: (ctx) => IconButton(
              icon: Icon(Icons.menu_rounded, color: textColor, size: 22),
              onPressed: () => Scaffold.of(ctx).openDrawer(),
            ),
          ),
          Expanded(
            child: Text(
              title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: textColor,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          IconButton(
            icon: Icon(Icons.add_comment_rounded, color: textColor, size: 22),
            onPressed: () async {
              await _createNewConversation();
              unawaited(_fetchGreeting());
            },
          ),
        ],
      ),
    );
  }

  Widget _buildChatBody(bool isDark) {
    final textColor = isDark ? Colors.white : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.6)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.8);

    if (_isLoadingHistory) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_currentConversation == null && _messages.isEmpty) {
      return _buildEmptyState(isDark, textColor, hintColor);
    }

    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 100),
      itemCount: _messages.length + (_activeChatMessage != null ? 1 : 0),
      itemBuilder: (context, index) {
        if (index < _messages.length) {
          final msg = _messages[index];
          if (msg.isUser) {
            return _buildUserBubble(msg.content, isDark, textColor);
          }
          return _buildAgentBubble(msg, isDark, textColor, hintColor);
        }
        return _buildActiveAgentBubble(isDark, textColor, hintColor);
      },
    );
  }

  Widget _buildEmptyState(bool isDark, Color textColor, Color hintColor) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.travel_explore_rounded, size: 64, color: hintColor),
            const SizedBox(height: 16),
            Text(
              textLocalize('agent_empty_title'),
              style: TextStyle(
                color: textColor,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              textLocalize('agent_empty_subtitle'),
              textAlign: TextAlign.center,
              style: TextStyle(color: hintColor, fontSize: 14),
            ),
            const SizedBox(height: 24),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                _buildSuggestionChip('找一下上周拍的街景', isDark),
                _buildSuggestionChip('比较两个场景的变化', isDark),
                _buildSuggestionChip('整理我的模型标签', isDark),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSuggestionChip(String text, bool isDark) {
    return ActionChip(
      label: Text(text, style: const TextStyle(fontSize: 13)),
      backgroundColor: isDark
          ? Colors.white.withValues(alpha: 0.06)
          : BDDesign.colorMutedBlue.withValues(alpha: 0.06),
      side: BorderSide(
        color: isDark
            ? Colors.white.withValues(alpha: 0.1)
            : BDDesign.colorMutedBlue.withValues(alpha: 0.15),
      ),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      onPressed: () => _submitQuery(text),
    );
  }

  Widget _buildUserBubble(String content, bool isDark, Color textColor) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Align(
        alignment: Alignment.centerRight,
        child: Container(
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.75,
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
            content,
            style: TextStyle(color: textColor, fontSize: 14, height: 1.4),
          ),
        ),
      ),
    );
  }

  Widget _buildAgentBubble(
    AgentMessageRecord msg,
    bool isDark,
    Color textColor,
    Color hintColor,
  ) {
    final answer = msg.finalAnswer ?? msg.content;
    AgentRecallResponse? result;
    Map<String, dynamic>? resultJson;
    if (msg.agentResultJson != null && msg.agentResultJson!.isNotEmpty) {
      try {
        resultJson = jsonDecode(msg.agentResultJson!) as Map<String, dynamic>;
        result = AgentRecallResponse.fromJson(resultJson);
      } catch (_) {}
    }

    ChatMessage? restoredChatMessage;
    final stepsJson = resultJson?['steps'];
    if (stepsJson is List && stepsJson.isNotEmpty && msg.id != null) {
      restoredChatMessage = _restoredChatMessages.putIfAbsent(msg.id!, () {
        return ChatMessage(
          isUser: false,
          finalAnswer: answer,
          isProcessCollapsed: true,
          steps: stepsJson
              .whereType<Map<String, dynamic>>()
              .map((s) => AgentStep(
                    type: s['type'] as String? ?? 'status',
                    content: s['content'] as String? ?? '',
                    toolName: s['tool_name'] as String?,
                    isCompleted: s['is_completed'] as bool? ?? true,
                  ))
              .toList(),
        );
      });
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Align(
        alignment: Alignment.centerLeft,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: isDark ? const Color(0xFF1A2030) : const Color(0xFFF8FAFD),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(4),
              topRight: Radius.circular(16),
              bottomLeft: Radius.circular(16),
              bottomRight: Radius.circular(16),
            ),
            border: Border.all(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.06)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.1),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (restoredChatMessage != null) ...[
                AgentProcessPanel(
                  chatMessage: restoredChatMessage,
                  isDark: isDark,
                  textColor: textColor,
                  hintColor: hintColor,
                  isSearching: false,
                  onRetry: () {},
                ),
                const SizedBox(height: 10),
              ],
              _buildAgentContent(
                answer: answer,
                result: result,
                isDark: isDark,
                textColor: textColor,
                hintColor: hintColor,
                elapsedMs: msg.elapsedMs,
                isActive: false,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAgentContent({
    required String answer,
    required AgentRecallResponse? result,
    required bool isDark,
    required Color textColor,
    required Color hintColor,
    int? elapsedMs,
    required bool isActive,
  }) {
    final hasActions = result != null &&
        result.actions.any((a) => a.type == 'open_scene');
    final topCandidates = result?.candidates.take(3).toList() ?? [];
    final followUp = result?.followUp;

    final isAssetMode = result?.mode == 'asset_metadata';
    final isTimeCompareMode =
        result?.mode == 'time_compare' && result?.compareData != null;
    final assetModels = <Map<String, dynamic>>[];
    if (isAssetMode && result?.assetContext != null) {
      final ctx = result!.assetContext!;
      final source = (ctx['bundle'] as List?) ?? (ctx['list'] as List?);
      if (source != null) {
        for (final item in source.take(5)) {
          if (item is Map) {
            assetModels.add(Map<String, dynamic>.from(item));
          }
        }
      }
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (answer.isNotEmpty) ...[
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
            child: isActive
                ? AnimatedMarkdownAnswer(
                    data: answer,
                    isDark: isDark,
                    textColor: textColor,
                    hintColor: hintColor,
                  )
                : MarkdownBody(
                    data: answer,
                    builders: {'code': CodeElementBuilder(isDark, context)},
                    styleSheet: MarkdownStyleSheet(
                      p: TextStyle(
                        color: textColor, fontSize: 14, height: 1.6),
                      code: TextStyle(
                        color: textColor,
                        fontSize: 12.5,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
          ),
        ],
        if (isTimeCompareMode) ...[
          _buildTimeCompareSection(
            data: result!.compareData!,
            result: result,
            isDark: isDark,
            textColor: textColor,
            hintColor: hintColor,
          ),
        ] else if (isAssetMode && assetModels.isNotEmpty) ...[
          const SizedBox(height: 12),
          for (final model in assetModels)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: AgentAssetCard(
                displayName: model['display_name']?.toString(),
                description: model['description']?.toString() ?? '',
                tags: (model['tags'] as List?)
                        ?.map((e) => e.toString())
                        .toList() ??
                    const [],
                previewImgPath: model['preview_img_path']?.toString(),
                isDark: isDark,
                onOpen: _buildAssetOnOpen(model),
              ),
            ),
        ] else if (topCandidates.isNotEmpty) ...[
          const SizedBox(height: 12),
          for (int i = 0; i < topCandidates.length; i++)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: AgentAssetCard(
                displayName: topCandidates[i].displayName ??
                    (topCandidates[i].description.isNotEmpty
                        ? topCandidates[i].description
                        : topCandidates[i].sceneId),
                description: topCandidates[i].description.isNotEmpty
                    ? topCandidates[i].description
                    : '场景 ${topCandidates[i].sceneId}',
                tags: topCandidates[i].tags,
                previewImgPath: topCandidates[i].previewImgPath,
                score: topCandidates[i].score,
                isDark: isDark,
                actionLabel: i == 0 ? '飞到视角' : '打开场景',
                onOpen: _buildCandidateOnOpen(topCandidates[i], i, result),
              ),
            ),
        ],
        if (hasActions &&
            !(isAssetMode && assetModels.isNotEmpty) &&
            !isTimeCompareMode &&
            topCandidates.isEmpty) ...[
          const SizedBox(height: 12),
          _buildOpenSceneButton(result!, isDark),
        ],
        if (result?.mode != null &&
            !(isAssetMode && assetModels.isNotEmpty)) ...[
          const SizedBox(height: 8),
          Text(
            '模式：${_formatModeLabel(result!.mode)}',
            style: TextStyle(color: hintColor, fontSize: 12),
          ),
        ],
        if (result?.evidence != null &&
            !(isAssetMode && assetModels.isNotEmpty) &&
            !isTimeCompareMode) ...[
          const SizedBox(height: 4),
          Text(
            '场景：${result!.evidence!.sceneId}  ·  相似度：${(result.evidence!.similarity * 100).toStringAsFixed(1)}%',
            style: TextStyle(color: hintColor, fontSize: 12),
          ),
        ],
        if (followUp != null &&
            (followUp.message.trim().isNotEmpty ||
                followUp.suggestedReplies.isNotEmpty)) ...[
          const SizedBox(height: 12),
          _buildRichFollowUp(followUp, isDark, textColor, hintColor),
        ],
        if (elapsedMs != null) ...[
          const SizedBox(height: 8),
          Text(
            _formatElapsed(Duration(milliseconds: elapsedMs)),
            style: TextStyle(color: hintColor, fontSize: 11),
          ),
        ],
      ],
    );
  }

  Widget _buildActiveAgentBubble(
      bool isDark, Color textColor, Color hintColor) {
    final msg = _activeChatMessage;
    if (msg == null) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Align(
        alignment: Alignment.centerLeft,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: isDark ? const Color(0xFF1A2030) : const Color(0xFFF8FAFD),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(4),
              topRight: Radius.circular(16),
              bottomLeft: Radius.circular(16),
              bottomRight: Radius.circular(16),
            ),
            border: Border.all(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.06)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.1),
            ),
          ),
          child: ListenableBuilder(
            listenable: msg,
            builder: (context, _) {
              final elapsed = _elapsedDuration;
              final elapsedLabel =
                  elapsed != null ? _formatElapsed(elapsed) : null;
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header with status
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
                          msg.liveStatus.isNotEmpty
                              ? msg.liveStatus
                              : textLocalize('agent'),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: textColor,
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      if (_isSearching)
                        const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                    ],
                  ),
                  if (elapsedLabel != null || _isSearching) ...[
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 10,
                      runSpacing: 8,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        if (elapsedLabel != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: isDark
                                  ? Colors.white.withValues(alpha: 0.06)
                                  : BDDesign.colorMutedBlue
                                      .withValues(alpha: 0.08),
                              borderRadius: BorderRadius.circular(999),
                              border: Border.all(
                                color: isDark
                                    ? Colors.white.withValues(alpha: 0.08)
                                    : BDDesign.colorMutedBlue
                                        .withValues(alpha: 0.16),
                              ),
                            ),
                            child: Text(
                              _isSearching
                                  ? '运行中 $elapsedLabel'
                                  : '完成 $elapsedLabel',
                              style: TextStyle(
                                color: hintColor,
                                fontSize: 11.5,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        if (_isSearching)
                          SizedBox(
                            height: 24,
                            child: OutlinedButton.icon(
                              style: OutlinedButton.styleFrom(
                                foregroundColor: Colors.red,
                                side: const BorderSide(
                                    color: Colors.red, width: 1),
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 8),
                              ),
                              onPressed: _stopSearch,
                              icon: const Icon(
                                  Icons.stop_circle_outlined, size: 14),
                              label: const Text('停止',
                                  style: TextStyle(fontSize: 12)),
                            ),
                          ),
                      ],
                    ),
                  ],
                  // Process panel
                  if (msg.steps.isNotEmpty) ...[
                    const SizedBox(height: 10),
                    AgentProcessPanel(
                      chatMessage: msg,
                      isDark: isDark,
                      textColor: textColor,
                      hintColor: hintColor,
                      isSearching: _isSearching,
                      onRetry: () {
                        final lastUserMsg = _messages.lastWhere(
                          (m) => m.isUser,
                          orElse: () => _messages.last,
                        );
                        _submitQuery(lastUserMsg.content);
                      },
                    ),
                  ],
                  // Final answer
                  if (msg.finalAnswer.isNotEmpty) ...[
                    const SizedBox(height: 14),
                    _buildAgentContent(
                      answer: msg.finalAnswer,
                      result: _activeResult,
                      isDark: isDark,
                      textColor: textColor,
                      hintColor: hintColor,
                      isActive: true,
                    ),
                  ],
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _buildRichFollowUp(
    AgentFollowUp followUp,
    bool isDark,
    Color textColor,
    Color hintColor,
  ) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF161D28) : const Color(0xFFF6F9FD),
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
              style: TextStyle(color: hintColor, fontSize: 12, height: 1.45),
            ),
          ],
          if (followUp.suggestedReplies.isNotEmpty) ...[
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: followUp.suggestedReplies.map((reply) {
                return ActionChip(
                  label: Text(reply, style: const TextStyle(fontSize: 12)),
                  onPressed: () => _submitQuery(reply),
                );
              }).toList(),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildOpenSceneButton(AgentRecallResponse result, bool isDark) {
    return SizedBox(
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
        onPressed: () => _openResult(result),
        icon: const Icon(Icons.open_in_new_rounded, size: 16),
        label: const Text('打开场景', style: TextStyle(fontSize: 14)),
      ),
    );
  }

  void _openResult(AgentRecallResponse result) {
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

  VoidCallback? _buildAssetOnOpen(Map<String, dynamic> model) {
    final plyPath = model['ply_path']?.toString() ?? '';
    final sceneId = model['scene_id']?.toString() ?? '';
    if (plyPath.isEmpty || sceneId.isEmpty) return null;
    return () {
      final modelUrl =
          plyPath.startsWith('http://') || plyPath.startsWith('https://')
              ? plyPath
              : toPublicUrl(plyPath);
      final posesUrl = toPosesUrl(plyPath);
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

  VoidCallback? _buildCandidateOnOpen(
    AgentCandidate candidate,
    int index,
    AgentRecallResponse? result,
  ) {
    if (index == 0 && result != null) {
      return () => _openResult(result);
    }
    final plyPath = candidate.plyPath ?? '';
    if (plyPath.isEmpty || candidate.sceneId.isEmpty) return null;
    return () {
      final modelUrl =
          plyPath.startsWith('http://') || plyPath.startsWith('https://')
              ? plyPath
              : toPublicUrl(plyPath);
      final posesUrl = toPosesUrl(plyPath);
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

  String _formatModeLabel(String mode) {
    return switch (mode) {
      'spatial_search' => '空间检索',
      'asset_metadata' => '资产元数据',
      'time_compare' => '时间对比',
      'collection' => '合集管理',
      'creative' => '创意生成',
      'memory_graph' => '记忆图谱',
      _ => mode,
    };
  }

  Widget _buildInputBar(bool isDark) {
    final bgColor = isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.9)
        : Colors.white.withValues(alpha: 0.95);
    final borderColor = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.12);
    final textColor = isDark ? Colors.white : BDDesign.colorInkBlack;

    return Container(
      padding: EdgeInsets.fromLTRB(
        16, 8, 16, MediaQuery.of(context).padding.bottom + 80),
      decoration: BoxDecoration(
        color: bgColor,
        border: Border(top: BorderSide(color: borderColor)),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _inputController,
              style: TextStyle(color: textColor, fontSize: 15),
              maxLines: 4,
              minLines: 1,
              textInputAction: TextInputAction.send,
              onSubmitted: _submitQuery,
              decoration: InputDecoration(
                hintText: _activeResult?.followUp?.inputPlaceholder ??
                    textLocalize('agent_input_hint'),
                hintStyle: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.4)
                      : Colors.black.withValues(alpha: 0.35),
                  fontSize: 15,
                ),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(24),
                  borderSide: BorderSide(color: borderColor),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(24),
                  borderSide: BorderSide(color: borderColor),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(24),
                  borderSide: BorderSide(
                    color: BDDesign.colorMutedBlue.withValues(alpha: 0.4),
                  ),
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 10),
                isDense: true,
              ),
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            onPressed: _isSearching
                ? null
                : () => _submitQuery(_inputController.text),
            icon: Icon(
              Icons.send_rounded,
              color: _isSearching
                  ? (isDark ? Colors.white24 : Colors.black26)
                  : BDDesign.colorMutedBlue,
            ),
          ),
        ],
      ),
    );
  }
}
