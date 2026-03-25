import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/community/composer_sheet.dart';
import 'package:braindance/pages/community/models.dart';
import 'package:braindance/pages/community/repository.dart';
import 'package:braindance/pages/community/views.dart';
import 'package:braindance/services/viewer_navigation.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

class CommunityPage extends StatefulWidget {
  const CommunityPage({super.key});

  @override
  State<CommunityPage> createState() => _CommunityPageState();
}

class _CommunityPageState extends State<CommunityPage>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  final PageController _feedController = PageController(viewportFraction: 0.96);
  final CommunityRepository _repository = CommunityRepository();

  List<CommunityPost> _posts = const [];
  List<CommunityModelOption> _shareableModels = const [];
  int _selectedMapIndex = 0;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _loadCommunity();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _feedController.dispose();
    super.dispose();
  }

  Future<void> _loadCommunity() async {
    setState(() {
      _isLoading = true;
    });

    final posts = await _repository.fetchPosts();
    final models = await _repository.fetchShareableModels();

    if (!mounted) {
      return;
    }

    setState(() {
      _posts = posts;
      _shareableModels = models;
      _selectedMapIndex = posts.isEmpty
          ? 0
          : _selectedMapIndex.clamp(0, posts.length - 1);
      _isLoading = false;
    });
  }

  Future<void> _openShareSheet() async {
    final draft = await showCommunityComposerSheet(
      context,
      models: _shareableModels,
    );

    if (draft == null) {
      return;
    }

    final createdPost = await _repository.createPost(draft);

    if (!mounted) {
      return;
    }

    setState(() {
      _posts = [createdPost, ..._posts];
      _selectedMapIndex = 0;
    });

    TDToast.showText(context: context, textLocalize('community_joined'));
  }

  void _openViewer(CommunityPost post) {
    openViewer(
      context,
      initialModelUrl: post.modelUrl,
      posesUrl: post.posesUrl,
      sceneId: post.modelName,
    );
  }

  void _openLocationHub(CommunityPost seedPost) {
    final peers = _posts
        .where((post) => post.placeName == seedPost.placeName)
        .toList(growable: false);

    showModalBottomSheet<void>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) {
        final isDark = context.isDarkMode;
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
                              '这里收集了 ${peers.length} 个来自不同用户的空间记忆，点进任何一个都可以直接进入 3D 模型。',
                              style: TextStyle(color: hintColor, height: 1.4),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: Icon(Icons.close_rounded, color: textColor),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Flexible(
                    child: ListView.separated(
                      shrinkWrap: true,
                      itemCount: peers.length,
                      separatorBuilder: (_, _) => const SizedBox(height: 10),
                      itemBuilder: (context, index) {
                        final post = peers[index];
                        return InkWell(
                          borderRadius: BDDesign.radiusLarge,
                          onTap: () {
                            Navigator.pop(context);
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

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final posts = _posts;
    final selectedPost = posts.isEmpty ? null : posts[_selectedMapIndex];

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: Column(
            children: [
              BDPageHeader(
                title: textLocalize('community'),
                subtitle: '把 3D 记忆变成一条可以下翻的世界流，也把地点重新组织成可以点击的空间索引。',
                trailing: IconButton(
                  onPressed: _openShareSheet,
                  icon: Icon(
                    Icons.add_location_alt_rounded,
                    color: isDark
                        ? BDDesign.colorPaperWhite
                        : BDDesign.colorInkBlack,
                  ),
                  tooltip: textLocalize('community_share_tooltip'),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Row(
                  children: [
                    Expanded(
                      child: CommunityMetricCard(
                        label: textLocalize('community_label_memories'),
                        value: '${posts.length}',
                        hint: textLocalize('community_label_memories_hint'),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: CommunityMetricCard(
                        label: textLocalize('community_label_nodes'),
                        value:
                            '${posts.map((post) => post.placeName).toSet().length}',
                        hint: textLocalize('community_label_nodes_hint'),
                      ),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 10),
                child: BDPanelCard(
                  padding: const EdgeInsets.all(6),
                  child: TabBar(
                    controller: _tabController,
                    dividerColor: Colors.transparent,
                    indicatorSize: TabBarIndicatorSize.tab,
                    indicator: BoxDecoration(
                      color: isDark
                          ? AppTheme.darkSurfaceElevated
                          : BDDesign.colorMutedBlue.withValues(alpha: 0.12),
                      borderRadius: BDDesign.radiusLarge,
                    ),
                    labelColor: isDark
                        ? BDDesign.colorPaperWhite
                        : BDDesign.colorInkBlack,
                    unselectedLabelColor: isDark
                        ? Colors.white.withValues(alpha: 0.56)
                        : BDDesign.colorMutedBlue,
                    tabs: [
                      Tab(text: textLocalize('community_tab_feed')),
                      Tab(text: textLocalize('community_tab_map')),
                    ],
                  ),
                ),
              ),
              Expanded(
                child: _isLoading
                    ? const Center(child: CircularProgressIndicator())
                    : TabBarView(
                        controller: _tabController,
                        children: [
                          CommunityFeedView(
                            posts: posts,
                            controller: _feedController,
                            onOpenViewer: _openViewer,
                            onOpenLocationHub: _openLocationHub,
                          ),
                          CommunityMapView(
                            posts: posts,
                            selectedIndex: _selectedMapIndex,
                            onSelect: (index) {
                              setState(() {
                                _selectedMapIndex = index;
                              });
                            },
                            onOpenViewer: _openViewer,
                            onOpenLocationHub: _openLocationHub,
                            selectedPost: selectedPost,
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
