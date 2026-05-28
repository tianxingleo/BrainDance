// ignore_for_file: invalid_use_of_protected_member
part of '../recall.dart';

extension _RecallPageView on _RecallPageState {
  Widget _buildRecallPage(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final textColor = isDark ? const Color(0xFFFFFFFF) : BDDesign.colorInkBlack;

    ref.listen(recallScrollToTopSignal, (prev, next) {
      if (next > (prev ?? 0) && _recallScrollController.hasClients) {
        _recallScrollController.animateTo(
          0,
          duration: const Duration(milliseconds: 400),
          curve: Curves.easeOutCubic,
        );
      }
    });

    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.transparent,
      body: Stack(
        key: _actionOverlayStackKey,
        children: [
          BDPageBackdrop(
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.only(right: 8),
                child: ScrollbarTheme(
                  data: ScrollbarThemeData(
                    thumbColor: WidgetStateProperty.all(
                      isDark
                          ? Colors.white.withValues(alpha: 0.28)
                          : Colors.black.withValues(alpha: 0.18),
                    ),
                    trackColor: WidgetStateProperty.all(
                      isDark
                          ? Colors.white.withValues(alpha: 0.06)
                          : Colors.black.withValues(alpha: 0.04),
                    ),
                    thickness: const WidgetStatePropertyAll(5),
                    radius: const Radius.circular(2.5),
                  ),
                  child: Scrollbar(
                    controller: _recallScrollController,
                    child: CustomScrollView(
                    controller: _recallScrollController,
                    cacheExtent: 1200,
                    slivers: [
                      SliverToBoxAdapter(
                        child: Column(
                          children: [
                            BDPageHeader(
                              title: textLocalize("home_page"),
                              padding: const EdgeInsets.fromLTRB(20, 16, 20, 4),
                              trailing: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  IconButton(
                                    icon: AnimatedRotation(
                                      turns: _isLoading ? 1 : 0,
                                      duration: const Duration(
                                        milliseconds: 600,
                                      ),
                                      child: Icon(
                                        Icons.sync_rounded,
                                        color: isDark
                                            ? BDDesign.colorPaperWhite
                                            : BDDesign.colorInkBlack,
                                      ),
                                    ),
                                    tooltip: textLocalize("recall_refresh"),
                                    onPressed: () {
                                      unawaited(
                                        _refreshModelsForCurrentState(),
                                      );
                                    },
                                  ),
                                ],
                              ),
                            ),
                            Padding(
                              padding: const EdgeInsets.fromLTRB(20, 2, 20, 8),
                              child: RecallSearchHeaderSection(
                                theme: theme,
                                isDark: isDark,
                                textColor: textColor,
                                darkInput: darkInput,
                                searchController: _searchController,
                                searchMode: _searchMode,
                                searchModeTitleBuilder: _searchModeTitle,
                                searchModeSubtitleBuilder: _searchModeSubtitle,
                                searchFieldHint: _searchFieldHint(),
                                onSubmit: _handleSearchSubmitted,
                                onChanged: _searchModels,
                                onClear: () {
                                  _searchController.clear();
                                  unawaited(_searchModels(''));
                                },
                                onTapSearchMode: _showSearchModeSheet,
                                isLocalModelReady: _isLocalModelReady,
                                isModelDownloading: _isModelDownloading,
                                isLocalModelLoading: _isLocalModelLoading,
                                modelDownloadProgress: _modelDownloadProgress,
                                modelDownloadedBytes: _modelDownloadedBytes,
                                modelDownloadTotalBytes:
                                    _modelDownloadTotalBytes,
                                localAnswer: _localAnswer,
                                localReasoning: _localReasoning,
                                localAnswerStatus: _localAnswerStatus,
                                localContextPreview: _localContextPreview,
                                defaultModelDownloadUrl:
                                    _defaultModelDownloadUrl,
                                localModelCatalog: _localModelCatalog,
                                selectedLocalModelUrl: _selectedLocalModelUrl,
                                activeLocalModelUrl: _activeLocalModelUrl,
                                downloadedLocalModelUrls:
                                    _downloadedLocalModelPathsByUrl.keys
                                        .toSet(),
                                localModelUrlController:
                                    _localModelUrlController,
                                localModelPathController:
                                    _localModelPathController,
                                onSelectCatalogModel: (value) {
                                  unawaited(_selectCatalogModel(value));
                                },
                                onDownloadModel: _downloadModelToPrivateDir,
                                onLoadModel: _loadLocalQnaModel,
                              ),
                            ),
                            if (_searchMode == RecallSearchMode.agent &&
                                _agentConversationHistory.isEmpty &&
                                !_isAgentSearching)
                              Padding(
                                padding: const EdgeInsets.fromLTRB(
                                  20,
                                  0,
                                  20,
                                  12,
                                ),
                                child: SingleChildScrollView(
                                  scrollDirection: Axis.horizontal,
                                  child: Row(
                                    children: [
                                      _buildQuickPromptChip('总结刚才的场景', isDark),
                                      const SizedBox(width: 8),
                                      _buildQuickPromptChip('找一个会议室资产', isDark),
                                      const SizedBox(width: 8),
                                      _buildQuickPromptChip(
                                        '有什么推荐的模型？',
                                        isDark,
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            if (_searchMode == RecallSearchMode.agent)
                              Padding(
                                padding: const EdgeInsets.fromLTRB(
                                  20,
                                  0,
                                  20,
                                  8,
                                ),
                                child: _buildAgentConversationList(
                                  isDark,
                                  textColor,
                                ),
                              ),
                            if (_slowReadyNotices.isNotEmpty)
                              RepaintBoundary(
                                child: DualChainNoticeStrip(
                                  notices: List.unmodifiable(
                                    _slowReadyNotices,
                                  ),
                                  isDark: isDark,
                                  onRefresh: (notice) {
                                    unawaited(
                                      _refreshFromSlowReadyNotice(notice),
                                    );
                                  },
                                  onDismiss: _dismissSlowReadyNotice,
                                ),
                              ),
                            if (_processingTasks.isNotEmpty)
                              RepaintBoundary(
                                child: RecallProcessingSection(
                                  theme: theme,
                                  isDark: isDark,
                                  textColor: textColor,
                                  darkInput: darkInput,
                                  isExpanded: _isProcessingExpanded,
                                  processingTasks: _processingTasks,
                                  taskAllLogs: _taskAllLogs,
                                  expandedTaskLogs: _expandedTaskLogs,
                                  onToggleExpanded: () {
                                    setState(() {
                                      _isProcessingExpanded =
                                          !_isProcessingExpanded;
                                    });
                                  },
                                  onToggleTaskLogs: (taskId) {
                                    setState(() {
                                      if (_expandedTaskLogs.contains(taskId)) {
                                        _expandedTaskLogs.remove(taskId);
                                      } else {
                                        _expandedTaskLogs.add(taskId);
                                      }
                                    });
                                  },
                                ),
                              ),
                          ],
                        ),
                      ),
                      if (_searchMode == RecallSearchMode.agent)
                        const SliverToBoxAdapter(child: SizedBox(height: 96))
                      else if (_isLoading)
                        const SliverFillRemaining(
                          hasScrollBody: false,
                          child: SizedBox.shrink(),
                        )
                      else if (_models.isEmpty)
                        SliverFillRemaining(
                          hasScrollBody: false,
                          child: Padding(
                            padding: const EdgeInsets.only(top: 16.0),
                            child: _searchController.text.trim().isEmpty
                                ? RecallEmptyState(
                                    theme: theme,
                                    isDark: isDark,
                                    darkCard: darkCard,
                                    darkBorder: darkBorder,
                                  )
                                : RecallSearchEmptyState(
                                    theme: theme,
                                    isDark: isDark,
                                    darkCard: darkCard,
                                    darkBorder: darkBorder,
                                    searchMode: _searchMode,
                                    searchModeTitleBuilder: _searchModeTitle,
                                  ),
                          ),
                        )
                      else if (_models.isNotEmpty &&
                          _models.first.containsKey('matched_frames'))
                        RecallModelGrid(
                          theme: theme,
                          isDark: isDark,
                          darkCard: darkCard,
                          darkInput: darkInput,
                          models: _models,
                          activeModelAction: _activeModelAction,
                          modelCardKeyFor: _modelCardKeyFor,
                          isSameModel: _isSameModel,
                          onNavigateToViewer: _navigateToViewer,
                          toPublicUrl: _toPublicUrl,
                          previewSceneIds: _previewSceneIds,
                          onShowModelActions:
                              (model, {bool imageOnly = false}) {
                                _showModelActions(model, imageOnly: imageOnly);
                              },
                        )
                      else ...[
                        if (_regularModels.isNotEmpty) ...[
                          SliverToBoxAdapter(
                            child: Consumer(
                              builder: (context, watchRef, _) {
                                final isExpanded = watchRef.watch(
                                  recallRegularExpandedProvider,
                                );
                                return RecallModelSectionHeader(
                                  icon: Icons.folder_rounded,
                                  color: const Color(0xFF8899BB),
                                  title: textLocalize('recall_regular_models'),
                                  count: _regularModels.length,
                                  isExpanded: isExpanded,
                                  isDark: isDark,
                                  textColor: textColor,
                                  onToggle: () {
                                    ref
                                        .read(
                                          recallRegularExpandedProvider.notifier,
                                        )
                                        .update((s) => !s);
                                  },
                                );
                              },
                            ),
                          ),
                          if (ref.watch(recallRegularExpandedProvider))
                            TimePeelingList(
                              theme: theme,
                              isDark: isDark,
                              darkCard: darkCard,
                              darkInput: darkInput,
                              groupedModels: _groupModelsByName(_regularModels),
                              activeModelAction: _activeModelAction,
                              modelCardKeyFor: _modelCardKeyFor,
                              isSameModel: _isSameModel,
                              onNavigateToViewer: _navigateToViewer,
                              onShowModelActions:
                                  (model, {bool imageOnly = false}) {
                                    _showModelActions(model, imageOnly: imageOnly);
                                  },
                              onAddNewTask: (name) {
                                ref
                                        .read(pendingSubmitTitleProvider.notifier)
                                        .state =
                                    name;
                                ref.read(pageIndexProvider.notifier).state = 2;
                              },
                              previewSceneIds: _previewSceneIds,
                            ),
                        ],
                        if (_officialModels.isNotEmpty) ...[
                          SliverToBoxAdapter(
                            child: Consumer(
                              builder: (context, watchRef, _) {
                                final isExpanded = watchRef.watch(
                                  recallOfficialExpandedProvider,
                                );
                                return RecallModelSectionHeader(
                                  icon: Icons.verified_rounded,
                                  color: BDDesign.colorMutedBlue,
                                  title: textLocalize('recall_official_models'),
                                  count: _officialModels.length,
                                  isExpanded: isExpanded,
                                  isDark: isDark,
                                  textColor: textColor,
                                  onToggle: () {
                                    ref
                                        .read(
                                          recallOfficialExpandedProvider.notifier,
                                        )
                                        .update((s) => !s);
                                  },
                                );
                              },
                            ),
                          ),
                          if (ref.watch(recallOfficialExpandedProvider))
                            TimePeelingList(
                              theme: theme,
                              isDark: isDark,
                              darkCard: darkCard,
                              darkInput: darkInput,
                              groupedModels: _groupModelsByName(_officialModels),
                              activeModelAction: _activeModelAction,
                              modelCardKeyFor: _modelCardKeyFor,
                              isSameModel: _isSameModel,
                              onNavigateToViewer: _navigateToViewer,
                              onShowModelActions:
                                  (model, {bool imageOnly = false}) {
                                    _showModelActions(model, imageOnly: imageOnly);
                                  },
                              onAddNewTask: (name) {
                                ref
                                        .read(pendingSubmitTitleProvider.notifier)
                                        .state =
                                    name;
                                ref.read(pageIndexProvider.notifier).state = 2;
                              },
                              previewSceneIds: _previewSceneIds,
                            ),
                        ],
                        if (_localModels.isNotEmpty) ...[
                          SliverToBoxAdapter(
                            child: Consumer(
                              builder: (context, watchRef, _) {
                                final isExpanded = watchRef.watch(
                                  recallLocalExpandedProvider,
                                );
                                return RecallModelSectionHeader(
                                  icon: Icons.folder_open_rounded,
                                  color: const Color(0xFFCC9A5C),
                                  title: textLocalize('recall_local_models'),
                                  count: _localModels.length,
                                  isExpanded: isExpanded,
                                  isDark: isDark,
                                  textColor: textColor,
                                  onToggle: () {
                                    ref
                                        .read(
                                          recallLocalExpandedProvider.notifier,
                                        )
                                        .update((s) => !s);
                                  },
                                );
                              },
                            ),
                          ),
                          if (ref.watch(recallLocalExpandedProvider))
                            TimePeelingList(
                              theme: theme,
                              isDark: isDark,
                              darkCard: darkCard,
                              darkInput: darkInput,
                              groupedModels: _groupModelsByName(_localModels),
                              activeModelAction: _activeModelAction,
                              modelCardKeyFor: _modelCardKeyFor,
                              isSameModel: _isSameModel,
                              onNavigateToViewer: _navigateToViewer,
                              onShowModelActions:
                                  (model, {bool imageOnly = false}) {
                                    _showModelActions(model, imageOnly: imageOnly);
                                  },
                              onAddNewTask: (name) {
                                ref
                                        .read(pendingSubmitTitleProvider.notifier)
                                        .state =
                                    name;
                                ref.read(pageIndexProvider.notifier).state = 2;
                              },
                              previewSceneIds: _previewSceneIds,
                            ),
                        ],
                      ],
                      const SliverToBoxAdapter(child: SizedBox(height: 96)),
                    ],
                  ),
                ),
              ),
            ),
            ),
          ),
          if (_activeModelAction != null && _activeModelActionRect != null)
            RecallModelActionOverlay(
              key: _overlayKey,
              theme: theme,
              isDark: isDark,
              darkCard: darkCard,
              darkInput: darkInput,
              model: _activeModelAction!,
              rect: _activeModelActionRect!,
              isLocalCached:
                  (_activeModelAction!['_local_size_label']?.toString() ?? '')
                      .isNotEmpty,
              isOwnModel: _activeModelAction!['_is_own_model'] == true,
              onDismiss: _dismissModelActions,
              onNavigateToViewer: _navigateToViewer,
              onShowModelDetails: _showModelDetails,
              onDownloadModel: _downloadRecallModel,
              onDeleteLocalModel: _deleteLocalModel,
              onShareModelToCommunity: _shareModelToCommunity,
              onRenameModel: _renameModel,
              onDeleteCloudModel: _deleteCloudModel,
            ),
          if (_isOpeningViewer)
            _RecallViewerOpeningOverlay(
              isDark: isDark,
              label: _openingViewerLabel ?? '正在打开模型',
            ),
        ],
      ),
    );
  }
}
