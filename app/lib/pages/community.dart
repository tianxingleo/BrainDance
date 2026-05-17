import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/community/detail.dart';
import 'package:braindance/pages/community/models.dart';
import 'package:braindance/pages/community/repository.dart';
import 'package:braindance/pages/community/views.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:braindance/widgets/bd_tab_switcher.dart';
import 'package:flutter/material.dart';
import 'package:braindance/widgets/app_toast.dart';

class CommunityPage extends StatefulWidget {
  const CommunityPage({super.key});

  @override
  State<CommunityPage> createState() => _CommunityPageState();
}

class _CommunityPageState extends State<CommunityPage>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  final CommunityRepository _repository = CommunityRepository();

  List<CommunityPost> _posts = const [];
  List<CommunityModelOption> _shareableModels = const [];
  String? _selectedPlaceName;
  bool _isLoading = true;
  int _tabIndex = 0;

  // Discover tab
  final Set<String> _selectedTags = {};

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
    _tabController = TabController(length: 3, vsync: this);
    _tabController.addListener(_onTabChanged);
    _submitTitleController = TextEditingController();
    _submitCaptionController = TextEditingController();
    _submitPlaceController = TextEditingController();
    _submitLatController = TextEditingController();
    _submitLngController = TextEditingController();
    _loadCommunity();
    _loadDraft();
  }

  void _onTabChanged() {
    if (_tabController.indexIsChanging) {
      setState(() => _tabIndex = _tabController.index);
    }
  }

  @override
  void dispose() {
    _tabController.removeListener(_onTabChanged);
    _tabController.dispose();
    _submitTitleController.dispose();
    _submitCaptionController.dispose();
    _submitPlaceController.dispose();
    _submitLatController.dispose();
    _submitLngController.dispose();
    super.dispose();
  }

  Future<void> _loadCommunity() async {
    setState(() => _isLoading = true);
    final posts = await _repository.fetchPosts();
    final models = await _repository.fetchShareableModels();
    if (!mounted) return;
    setState(() {
      _posts = posts;
      _shareableModels = models;
      _isLoading = false;
    });
  }

  Future<void> _loadDraft() async {
    final draft = await _repository.loadDraft();
    if (!mounted || draft.isEmpty) return;
    setState(() {
      _submitTitleController.text = draft.title;
      _submitCaptionController.text = draft.caption;
      _submitPlaceController.text = draft.placeName;
      _submitLatController.text =
          draft.latitude != 0 ? draft.latitude.toStringAsFixed(3) : '';
      _submitLngController.text =
          draft.longitude != 0 ? draft.longitude.toStringAsFixed(3) : '';
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
            position: Tween<Offset>(
              begin: const Offset(1.0, 0.0),
              end: Offset.zero,
            ).animate(
              CurvedAnimation(
                  parent: animation, curve: Curves.easeInOutCubic),
            ),
            child: child,
          );
        },
      ),
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
    final lat =
        double.tryParse(_submitLatController.text.trim()) ?? 0;
    final lng =
        double.tryParse(_submitLngController.text.trim()) ?? 0;
    final draft = CommunityDraft(
      modelIds:
          _selectedSubmitModels.map((m) => m.id).toList(),
      title: _submitTitleController.text.trim(),
      caption: _submitCaptionController.text.trim(),
      placeName: _submitPlaceController.text.trim(),
      latitude: lat,
      longitude: lng,
    );
    await _repository.saveDraft(draft);
    if (!mounted) return;
    showAppToast(context, '草稿已保存');
    _tabController.animateTo(2); // stay on submit tab
  }

  Future<void> _submitPost() async {
    if (_selectedSubmitModels.isEmpty) {
      showAppToast(context, textLocalize('community_fill_all'));
      return;
    }
    final lat =
        double.tryParse(_submitLatController.text.trim());
    final lng =
        double.tryParse(_submitLngController.text.trim());
    final title = _submitTitleController.text.trim();
    final caption = _submitCaptionController.text.trim();
    final place = _submitPlaceController.text.trim();

    if (lat == null ||
        lng == null ||
        title.isEmpty ||
        caption.isEmpty ||
        place.isEmpty) {
      showAppToast(context, textLocalize('community_fill_all'));
      return;
    }

    setState(() => _isSubmitting = true);

    final result = CommunityComposerResult(
      title: title,
      caption: caption,
      placeName: place,
      latitude: lat,
      longitude: lng,
      models: _selectedSubmitModels,
      tags: _selectedSubmitModels
          .expand((m) => m.description.split(RegExp(r'[\s,，]+')))
          .where((t) => t.trim().length >= 2)
          .take(5)
          .toList(),
    );
    final created = await _repository.createPost(result);

    if (!mounted) return;

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
    showAppToast(context, textLocalize('community_joined'));
  }

  List<CommunityPost> get _filteredPosts {
    if (_selectedTags.isEmpty) return _posts;
    final filtered = _posts
        .where((p) => p.tags.any((t) => _selectedTags.contains(t)))
        .toList();
    filtered.sort(
      (a, b) => b.tags
          .where((t) => _selectedTags.contains(t))
          .length
          .compareTo(
            a.tags.where((t) => _selectedTags.contains(t)).length,
          ),
    );
    return filtered;
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
                child: Text(
                  textLocalize('community'),
                  style: TextStyle(
                    color: isDark
                        ? BDDesign.colorPaperWhite
                        : BDDesign.colorInkBlack,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20),
                child: BDPanelCard(
                  padding: const EdgeInsets.all(6),
                  child: TabBar(
                    controller: _tabController,
                    dividerColor: Colors.transparent,
                    indicatorSize: TabBarIndicatorSize.tab,
                    indicator: BoxDecoration(
                      color: isDark
                          ? AppTheme.darkSurfaceElevated
                          : BDDesign.colorMutedBlue
                              .withValues(alpha: 0.12),
                      borderRadius: BDDesign.radiusLarge,
                    ),
                    labelColor: isDark
                        ? BDDesign.colorPaperWhite
                        : BDDesign.colorInkBlack,
                    unselectedLabelColor: isDark
                        ? Colors.white.withValues(alpha: 0.56)
                        : BDDesign.colorMutedBlue,
                    tabs: [
                      Tab(
                          text: textLocalize(
                              'community_tab_explore')),
                      Tab(
                          text: textLocalize(
                              'community_tab_discover')),
                      Tab(
                          text: textLocalize(
                              'community_tab_submit')),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Expanded(
                child: _isLoading
                    ? const Center(
                        child: CircularProgressIndicator())
                    : BDTabSwitcher(
                        index: _tabIndex,
                        children: [
                          CommunityExploreView(
                            posts: _posts,
                            selectedPlaceName: _selectedPlaceName,
                            onSelect: (placeName) => setState(
                                () => _selectedPlaceName = placeName),
                            onClearFilter: () => setState(
                                () => _selectedPlaceName = null),
                            onTapPost: _openDetail,
                          ),
                          CommunityDiscoverView(
                            posts: _filteredPosts,
                            selectedTags: _selectedTags,
                            onToggleTag: (tag) {
                              setState(() {
                                _selectedTags.contains(tag)
                                    ? _selectedTags
                                        .remove(tag)
                                    : _selectedTags.add(tag);
                              });
                            },
                            onTapPost: _openDetail,
                          ),
                          CommunitySubmitView(
                            shareableModels:
                                _shareableModels,
                            selectedModels:
                                _selectedSubmitModels,
                            onToggleModel:
                                _toggleSubmitModel,
                            titleController:
                                _submitTitleController,
                            captionController:
                                _submitCaptionController,
                            placeController:
                                _submitPlaceController,
                            latController:
                                _submitLatController,
                            lngController:
                                _submitLngController,
                            isSubmitting: _isSubmitting,
                            onSubmit: _submitPost,
                            onSaveDraft: _saveDraft,
                          ),
                        ],
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
