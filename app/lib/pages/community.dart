import 'dart:async';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/community/detail.dart';
import 'package:braindance/pages/community/filtering.dart';
import 'package:braindance/pages/community/map_marker.dart';
import 'package:braindance/pages/community/map_page.dart';
import 'package:braindance/pages/community/models.dart';
import 'package:braindance/pages/community/repository.dart';
import 'package:braindance/pages/community/views.dart';
import 'package:braindance/services/network_service.dart';
import 'package:braindance/services/viewer_navigation.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:braindance/widgets/app_toast.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../main.dart' show myPostsRefreshSignal;

class CommunityPage extends ConsumerStatefulWidget {
  const CommunityPage({super.key});

  @override
  ConsumerState<CommunityPage> createState() => _CommunityPageState();
}

enum _CommunitySubPage { search, submit }

class _CommunityPageState extends ConsumerState<CommunityPage> {
  final CommunityRepository _repository = CommunityRepository();
  _CommunitySubPage? _currentPage;
  int _searchFocusTrigger = 0;

  List<CommunityPost> _posts = const [];
  List<CommunityModelOption> _shareableModels = const [];
  List<CommunityMapMarker> _mapMarkers = const [];
  bool _isLoading = true;
  bool _isOffline = false;
  Timer? _retryTimer;
  CommunityMapViewport _mapViewport = const CommunityMapViewport(
    latitude: 30.243,
    longitude: 120.150,
    zoom: 10,
  );

  // Explore tab — viewport + tag filter
  String? _exploreTag;

  // Search tab (探索)
  List<String> _searchHistory = const [];
  List<String> _recommendedKeywords = const [
    '街景',
    '建筑',
    '自然',
    '室内',
    '夜景',
    '人物',
    '旅行',
    '城市',
  ];
  static const _searchHistoryPrefKey = 'community_search_history';
  static const _maxSearchHistory = 10;

  // Submit tab — multi-model
  final List<CommunityModelOption> _selectedSubmitModels = [];
  late final TextEditingController _submitTitleController;
  late final TextEditingController _submitCaptionController;
  late final TextEditingController _submitPlaceController;
  late final TextEditingController _submitLatController;
  late final TextEditingController _submitLngController;
  bool _isSubmitting = false;

  @override
  void initState() {
    super.initState();
    _submitTitleController = TextEditingController();
    _submitCaptionController = TextEditingController();
    _submitPlaceController = TextEditingController();
    _submitLatController = TextEditingController();
    _submitLngController = TextEditingController();
    networkService.addListener(_onNetworkChanged);
    _loadCommunity();
    _loadSearchHistory();
  }

  @override
  void dispose() {
    networkService.removeListener(_onNetworkChanged);
    _retryTimer?.cancel();
    _submitTitleController.dispose();
    _submitCaptionController.dispose();
    _submitPlaceController.dispose();
    _submitLatController.dispose();
    _submitLngController.dispose();
    super.dispose();
  }

  Future<void> _loadSearchHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final history = prefs.getStringList(_searchHistoryPrefKey) ?? const [];
    if (!mounted) return;
    setState(() => _searchHistory = history);
  }

  Future<void> _persistSearchHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_searchHistoryPrefKey, _searchHistory);
  }

  String? _lastSearchQuery;

  void _addToSearchHistory(String query) {
    final q = query.trim();
    if (q.isEmpty) return;
    _lastSearchQuery = q;
    setState(() {
      _searchHistory.remove(q);
      _searchHistory.insert(0, q);
      if (_searchHistory.length > _maxSearchHistory) {
        _searchHistory = _searchHistory.sublist(0, _maxSearchHistory);
      }
    });
    _persistSearchHistory();
  }

  void _clearSearchHistory() {
    setState(() => _searchHistory = []);
    _persistSearchHistory();
  }

  void _onNetworkChanged() {
    if (!mounted) return;
    if (networkService.isConnected && _isOffline) {
      _loadCommunity();
    } else if (!networkService.isConnected && !_isOffline) {
      setState(() => _isOffline = true);
    }
  }

  Future<void> _loadCommunity() async {
    if (!networkService.isConnected) {
      setState(() {
        _isOffline = true;
        _isLoading = false;
      });
      _startRetry();
      return;
    }
    setState(() => _isLoading = true);
    try {
      final posts = await _repository.fetchPosts();
      final models = await _repository.fetchShareableModels();
      final markers = await _repository.fetchMapMarkers();
      if (!mounted) return;
      _retryTimer?.cancel();
      setState(() {
        _posts = posts;
        _shareableModels = models;
        _mapMarkers = markers;
        _isOffline = false;
        _isLoading = false;
      });
      _loadDraft();
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _isOffline = true;
        _isLoading = false;
      });
      _startRetry();
    }
  }

  void _startRetry() {
    _retryTimer?.cancel();
    _retryTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (!mounted || !_isOffline) {
        _retryTimer?.cancel();
        return;
      }
      if (_isLoading) return;
      _loadCommunity();
    });
  }

  Future<void> _loadDraft() async {
    final draft = await _repository.loadDraft();
    if (!mounted || draft.isEmpty) return;
    setState(() {
      _submitTitleController.text = draft.title;
      _submitCaptionController.text = draft.caption;
      _submitPlaceController.text = draft.placeName;
      _submitLatController.text = draft.latitude != 0
          ? draft.latitude.toStringAsFixed(3)
          : '';
      _submitLngController.text = draft.longitude != 0
          ? draft.longitude.toStringAsFixed(3)
          : '';
      _selectedSubmitModels.clear();
      for (final mid in draft.modelIds) {
        final model = _shareableModels.where((m) => m.id == mid);
        if (model.isNotEmpty) _selectedSubmitModels.add(model.first);
      }
    });
  }

  void _openDetail(CommunityPost post) {
    Navigator.push(
      context,
      PageRouteBuilder(
        transitionDuration: BDMotion.durationNormal,
        reverseTransitionDuration: BDMotion.durationNormal,
        opaque: true,
        pageBuilder: (_, __, ___) => CommunityDetailPage(post: post),
        transitionsBuilder: (_, animation, __, child) {
          return SlideTransition(
            position:
                Tween<Offset>(
                  begin: const Offset(1.0, 0.0),
                  end: Offset.zero,
                ).animate(
                  CurvedAnimation(
                    parent: animation,
                    curve: Curves.easeInOutCubic,
                  ),
                ),
            child: child,
          );
        },
      ),
    );
  }

  Future<void> _openMapPage() async {
    final result = await Navigator.push<CommunityMapViewport>(
      context,
      PageRouteBuilder(
        transitionDuration: BDMotion.durationNormal,
        reverseTransitionDuration: BDMotion.durationNormal,
        opaque: true,
        pageBuilder: (_, __, ___) => CommunityMapPage(
          initialViewport: _mapViewport,
          onMarkerTap: _openPostFromMarker,
          onMarkerLongPress: _openLocationHubFromMarker,
        ),
        transitionsBuilder: (_, animation, __, child) {
          return SlideTransition(
            position:
                Tween<Offset>(
                  begin: const Offset(0.0, 1.0),
                  end: Offset.zero,
                ).animate(
                  CurvedAnimation(
                    parent: animation,
                    curve: Curves.easeInOutCubic,
                  ),
                ),
            child: child,
          );
        },
      ),
    );
    if (!mounted || result == null) return;
    setState(() => _mapViewport = result);
  }

  Future<void> _openPostFromMarker(CommunityMapMarker marker) async {
    final cached = _posts.where((p) => p.id == marker.id);
    if (cached.isNotEmpty) {
      _openDetail(cached.first);
      return;
    }
    final post = await _repository.fetchPostById(marker.id);
    if (!mounted) return;
    if (post == null) {
      showAppToast(context, '帖子已不可用');
      return;
    }
    _openDetail(post);
  }

  void _openLocationHubFromMarker(CommunityMapMarker marker) {
    final peers = _posts.where((p) => p.placeName == marker.placeName).toList();
    if (peers.isNotEmpty) {
      _openLocationHub(peers.first);
      return;
    }
    _openPostFromMarker(marker);
  }

  void _openViewer(CommunityPost post) {
    openViewer(
      context,
      initialModelUrl: post.modelUrl,
      posesUrl: post.posesUrl,
      sceneId: post.modelName,
      initialMarkerArMode: true,
    );
  }

  void _openLocationHub(CommunityPost seedPost) {
    final peers = _posts
        .where((post) => post.placeName == seedPost.placeName)
        .toList();

    showModalBottomSheet<void>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) {
        final isDark = ctx.isDarkMode;
        final textColor = isDark
            ? BDDesign.colorPaperWhite
            : BDDesign.colorInkBlack;
        final hintColor = isDark
            ? Colors.white.withValues(alpha: 0.62)
            : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 24, 16, 24),
          child: BDPanelCard(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
            child: SafeArea(
              top: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              seedPost.placeName,
                              style: TextStyle(
                                color: textColor,
                                fontSize: 22,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              '这里收集了 ${peers.length} 个来自不同用户的空间记忆。',
                              style: TextStyle(color: hintColor, height: 1.4),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(ctx),
                        icon: Icon(Icons.close_rounded, color: textColor),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Flexible(
                    child: ListView.separated(
                      shrinkWrap: true,
                      itemCount: peers.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 10),
                      itemBuilder: (context, index) {
                        final post = peers[index];
                        return InkWell(
                          borderRadius: BDDesign.radiusLarge,
                          onTap: () {
                            Navigator.pop(ctx);
                            _openViewer(post);
                          },
                          child: CommunityLocationHubRow(post: post),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  void _toggleSubmitModel(CommunityModelOption model) {
    setState(() {
      if (_selectedSubmitModels.any((m) => m.id == model.id)) {
        _selectedSubmitModels.removeWhere((m) => m.id == model.id);
      } else {
        _selectedSubmitModels.add(model);
      }
    });
  }

  Future<void> _saveDraft() async {
    final lat = double.tryParse(_submitLatController.text.trim()) ?? 0;
    final lng = double.tryParse(_submitLngController.text.trim()) ?? 0;
    final draft = CommunityDraft(
      modelIds: _selectedSubmitModels.map((m) => m.id).toList(),
      title: _submitTitleController.text.trim(),
      caption: _submitCaptionController.text.trim(),
      placeName: _submitPlaceController.text.trim(),
      latitude: lat,
      longitude: lng,
    );
    await _repository.saveDraft(draft);
    if (!mounted) return;
    showAppToast(context, textLocalize('community_draft_saved'));
  }

  Future<bool> _submitPost() async {
    if (_selectedSubmitModels.isEmpty) {
      showAppToast(context, textLocalize('community_fill_all'));
      return false;
    }
    final lat = double.tryParse(_submitLatController.text.trim());
    final lng = double.tryParse(_submitLngController.text.trim());
    final title = _submitTitleController.text.trim();
    final caption = _submitCaptionController.text.trim();
    final place = _submitPlaceController.text.trim();

    if (title.isEmpty || caption.isEmpty) {
      showAppToast(context, textLocalize('community_fill_all'));
      return false;
    }

    setState(() => _isSubmitting = true);

    final effectivePlace =
        place.isEmpty ? textLocalize('community_no_location') : place;
    final hasBoth = lat != null && lng != null;
    final effectiveLat = hasBoth ? lat : 0.0;
    final effectiveLng = hasBoth ? lng : 0.0;

    final result = CommunityComposerResult(
      title: title,
      caption: caption,
      placeName: effectivePlace,
      latitude: effectiveLat,
      longitude: effectiveLng,
      models: _selectedSubmitModels,
      tags: _selectedSubmitModels
          .expand((m) => m.description.split(RegExp(r'[\s,，]+')))
          .where((t) => t.trim().length >= 2)
          .take(5)
          .toList(),
    );
    final created = await _repository.createPost(result);

    if (!mounted) return true;

    setState(() {
      _posts = [created, ..._posts];
      _isSubmitting = false;
      _selectedSubmitModels.clear();
      _submitTitleController.clear();
      _submitCaptionController.clear();
      _submitPlaceController.clear();
      _submitLatController.clear();
      _submitLngController.clear();
    });

    await _repository.clearDraft();
    ref.read(myPostsRefreshSignal.notifier).state++;
    showAppToast(context, textLocalize('community_joined'));
    return true;
  }

  /// Posts without location info — always shown at the end.
  List<CommunityPost> get _noLocationPosts =>
      _posts.where((p) => p.latitude == 0 && p.longitude == 0).toList();

  /// Posts with valid coordinates, filtered by viewport.
  List<CommunityPost> get _viewportPosts {
    final geoPosts =
        _posts.where((p) => p.latitude != 0 || p.longitude != 0).toList();
    return filterPostsByBounds(geoPosts, _mapViewport.bounds);
  }

  /// Final explore-tab list: viewport posts + tag filter, then no-location posts.
  List<CommunityPost> get _exploreFilteredPosts {
    var base = _viewportPosts;
    var noLoc = _noLocationPosts;
    final tag = _exploreTag;
    if (tag != null && tag.isNotEmpty) {
      base = filterPostsByTag(base, tag);
      final origin =
          _mapViewport.bounds?.center ??
          (latitude: _mapViewport.latitude,
              longitude: _mapViewport.longitude);
      base = filterPostsByRadius(
        base,
        origin,
        tagRadiusKmForZoom(_mapViewport.zoom),
      );
      noLoc = filterPostsByTag(noLoc, tag);
    }
    return [...base, ...noLoc];
  }

  void _onExploreToggleTag(String tag) {
    setState(() {
      if (_exploreTag == tag) {
        _exploreTag = null;
      } else {
        _exploreTag = tag;
      }
    });
  }

  void _onExploreClearFilters() {
    if (_exploreTag == null) return;
    setState(() => _exploreTag = null);
  }

  void _openSearch() {
    _searchFocusTrigger++;
    setState(() => _currentPage = _CommunitySubPage.search);
  }
  void _openSubmit() => setState(() => _currentPage = _CommunitySubPage.submit);
  void _goBack() {
    FocusManager.instance.primaryFocus?.unfocus();
    FocusScope.of(context).unfocus();
    setState(() => _currentPage = null);
  }

  Widget _buildFloatingHeader(bool isDark) {
    final headerBg = isDark ? const Color(0xFF1A1D21) : const Color(0xFFF2F4F8);
    final inputFill = isDark
        ? AppTheme.darkSurfaceElevated
        : const Color(0xFFE8ECF1);
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.48)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.65);
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;

    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: RepaintBoundary(
        child: Container(
        padding: EdgeInsets.fromLTRB(
          20,
          MediaQuery.paddingOf(context).top + 12,
          20,
          10,
        ),
        decoration: BoxDecoration(
          color: headerBg.withValues(alpha: 0.95),
          border: Border(
            bottom: BorderSide(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.06)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
              ),
            ),
          ),
          child: Row(
            children: [
              Text(
                textLocalize('community'),
                style: TextStyle(
                  color: textColor,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.4,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: GestureDetector(
                  onTap: _openSearch,
                  child: Container(
                    height: 38,
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    decoration: BoxDecoration(
                      color: inputFill,
                      borderRadius: BorderRadius.circular(19),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.search_rounded, color: hintColor, size: 18),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            (_lastSearchQuery != null &&
                                    _lastSearchQuery!.isNotEmpty)
                                ? _lastSearchQuery!
                                : textLocalize('community_search_placeholder'),
                            maxLines: 1,
                            style: TextStyle(
                              color: (_lastSearchQuery != null &&
                                      _lastSearchQuery!.isNotEmpty)
                                  ? textColor
                                  : hintColor,
                              fontSize: 13,
                              fontWeight: FontWeight.w400,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              TextButton.icon(
                onPressed: _openSubmit,
                icon: Icon(
                  Icons.edit_rounded,
                  size: 17,
                  color: isDark
                      ? BDDesign.colorPaperWhite
                      : BDDesign.colorInkBlack,
                ),
                label: Text(
                  textLocalize('community_tab_submit'),
                  style: TextStyle(
                    color: textColor,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 8,
                  ),
                  minimumSize: Size.zero,
                  visualDensity: VisualDensity.compact,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSearchOverlay() {
    return BDPageBackdrop(
      child: SafeArea(
        child: Stack(
          children: [
            CommunityExploreView(
              posts: _posts,
              searchHistory: _searchHistory,
              recommendedKeywords: _recommendedKeywords,
              onSearch: _addToSearchHistory,
              onClearHistory: _clearSearchHistory,
              onTapPost: (post) {
                FocusManager.instance.primaryFocus?.unfocus();
                _openDetail(post);
              },
              focusTrigger: _searchFocusTrigger,
              searchFieldLeftInset: 52,
            ),
            _buildBackButton(isDark: context.isDarkMode),
          ],
        ),
      ),
    );
  }

  Widget _buildSubmitOverlay() {
    return BDPageBackdrop(
      child: SafeArea(
        child: Stack(
          children: [
            CommunitySubmitView(
              shareableModels: _shareableModels,
              selectedModels: _selectedSubmitModels,
              onToggleModel: _toggleSubmitModel,
              titleController: _submitTitleController,
              captionController: _submitCaptionController,
              placeController: _submitPlaceController,
              latController: _submitLatController,
              lngController: _submitLngController,
              isSubmitting: _isSubmitting,
              onSubmit: () async {
                final ok = await _submitPost();
                if (ok && mounted) _goBack();
              },
              onSaveDraft: _saveDraft,
              searchFieldLeftInset: 52,
            ),
            _buildBackButton(isDark: context.isDarkMode),
          ],
        ),
      ),
    );
  }

  Widget _buildBackButton({required bool isDark}) {
    final iconColor = isDark ? Colors.white : Colors.black;
    return Positioned(
      left: 16,
      top: 8,
      child: IconButton(
        style: IconButton.styleFrom(
          backgroundColor: Colors.transparent,
          surfaceTintColor: Colors.transparent,
        ),
        onPressed: _goBack,
        icon: Icon(
          Icons.arrow_back_rounded,
          color: iconColor,
          size: 22,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final screenWidth = MediaQuery.sizeOf(context).width;
    final topSafe = MediaQuery.paddingOf(context).top;

    ref.listen(myPostsRefreshSignal, (prev, next) {
      if (prev != null && prev != next) _loadCommunity();
    });

    return PopScope(
      canPop: _currentPage == null,
      onPopInvokedWithResult: (didPop, result) {
        if (!didPop) _goBack();
      },
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        backgroundColor: Colors.transparent,
        body: Stack(
          children: [
            BDPageBackdrop(
              child: SafeArea(
                top: false,
                child: Stack(
                  children: [
                    if (_isLoading)
                      const Center(child: CircularProgressIndicator())
                    else if (_isOffline)
                      const _CommunityOfflineState()
                    else
                      RepaintBoundary(
                        child: Padding(
                          padding: EdgeInsets.only(top: topSafe + 60),
                          child: CommunityRecommendView(
                            posts: _exploreFilteredPosts,
                            totalPosts: _posts.length,
                            viewportPosts: _viewportPosts.length,
                            mapViewport: _mapViewport,
                            mapMarkers: _mapMarkers,
                            onOpenMap: _openMapPage,
                            onTapPost: _openDetail,
                            availableTags: rankTagsFromPosts(_viewportPosts),
                            selectedTag: _exploreTag,
                            onToggleTag: _onExploreToggleTag,
                            onClearFilters: _onExploreClearFilters,
                            tagRadiusKm: tagRadiusKmForZoom(_mapViewport.zoom),
                          ),
                        ),
                      ),
                    AnimatedPositioned(
                      duration: BDMotion.durationNormal,
                      curve: BDMotion.curveFluid,
                      left: _currentPage == _CommunitySubPage.search
                          ? 0
                          : screenWidth,
                      top: 0,
                      bottom: 0,
                      width: screenWidth,
                      child: RepaintBoundary(
                        child: ExcludeFocus(
                          excluding: _currentPage != _CommunitySubPage.search,
                          child: _buildSearchOverlay(),
                        ),
                      ),
                    ),
                    AnimatedPositioned(
                      duration: BDMotion.durationNormal,
                      curve: BDMotion.curveFluid,
                      left: _currentPage == _CommunitySubPage.submit
                          ? 0
                          : screenWidth,
                      top: 0,
                      bottom: 0,
                      width: screenWidth,
                      child: RepaintBoundary(
                        child: ExcludeFocus(
                          excluding: _currentPage != _CommunitySubPage.submit,
                          child: _buildSubmitOverlay(),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            if (_currentPage == null) _buildFloatingHeader(isDark),
          ],
        ),
      ),
    );
  }
}

class _CommunityOfflineState extends StatelessWidget {
  const _CommunityOfflineState();

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final topSafe = MediaQuery.paddingOf(context).top;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.48)
        : BDDesign.colorMutedBlue;

    return Padding(
      padding: EdgeInsets.only(top: topSafe + 100),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.wifi_off_rounded, size: 48, color: hintColor),
            const SizedBox(height: 16),
            Text(
              textLocalize('community_offline_title'),
              style: TextStyle(
                color: textColor,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              textLocalize('community_offline_hint'),
              style: TextStyle(color: hintColor, fontSize: 14, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }
}
