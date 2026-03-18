import 'dart:math' as math;

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/webgl_viewer.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
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

    TDToast.showText(context: context, '记忆已加入社区流');
  }

  void _openViewer(CommunityPost post) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => WebGLViewerPage(
          initialModelUrl: post.modelUrl,
          posesUrl: post.posesUrl,
          sceneId: post.modelName,
        ),
      ),
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
                          child: _LocationHubRow(post: post),
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
                  tooltip: '分享记忆',
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Row(
                  children: [
                    Expanded(
                      child: _CommunityMetricCard(
                        label: '社区记忆',
                        value: '${posts.length}',
                        hint: '当前可浏览的空间贴文',
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _CommunityMetricCard(
                        label: '地点节点',
                        value:
                            '${posts.map((post) => post.placeName).toSet().length}',
                        hint: '地图上的可探索坐标',
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
                    tabs: const [
                      Tab(text: '流式浏览'),
                      Tab(text: '地图探索'),
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
                          _CommunityFeedView(
                            posts: posts,
                            controller: _feedController,
                            onOpenViewer: _openViewer,
                            onOpenLocationHub: _openLocationHub,
                          ),
                          _CommunityMapView(
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

Future<CommunityComposerResult?> showCommunityComposerSheet(
  BuildContext context, {
  required List<CommunityModelOption> models,
  String? initialModelId,
}) {
  return showModalBottomSheet<CommunityComposerResult>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) {
      return _CommunityComposerSheet(
        models: models,
        initialModelId: initialModelId,
      );
    },
  );
}

class _CommunityFeedView extends StatelessWidget {
  final List<CommunityPost> posts;
  final PageController controller;
  final ValueChanged<CommunityPost> onOpenViewer;
  final ValueChanged<CommunityPost> onOpenLocationHub;

  const _CommunityFeedView({
    required this.posts,
    required this.controller,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
  });

  @override
  Widget build(BuildContext context) {
    if (posts.isEmpty) {
      return const _CommunityEmptyState();
    }

    return PageView.builder(
      controller: controller,
      scrollDirection: Axis.vertical,
      itemCount: posts.length,
      padEnds: false,
      itemBuilder: (context, index) {
        final post = posts[index];
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
          child: _CommunityFeedCard(
            post: post,
            onOpenViewer: () => onOpenViewer(post),
            onOpenLocationHub: () => onOpenLocationHub(post),
          ),
        );
      },
    );
  }
}

class _CommunityMapView extends StatelessWidget {
  final List<CommunityPost> posts;
  final int selectedIndex;
  final ValueChanged<int> onSelect;
  final ValueChanged<CommunityPost> onOpenViewer;
  final ValueChanged<CommunityPost> onOpenLocationHub;
  final CommunityPost? selectedPost;

  const _CommunityMapView({
    required this.posts,
    required this.selectedIndex,
    required this.onSelect,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
    required this.selectedPost,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    if (posts.isEmpty) {
      return const _CommunityEmptyState();
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
      child: Column(
        children: [
          BDPanelCard(
            padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'World Memory Map',
                            style: TextStyle(
                              color: textColor,
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '点地图上的记忆节点，会展开同地点用户拍到的空间内容。',
                            style: TextStyle(color: hintColor, height: 1.35),
                          ),
                        ],
                      ),
                    ),
                    BDStatusPill(
                      label: '${posts.length} PINS',
                      icon: Icons.public_rounded,
                      color: BDDesign.colorMutedBlue,
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                LayoutBuilder(
                  builder: (context, constraints) {
                    final mapWidth = constraints.maxWidth;
                    final mapHeight = math.max(240.0, mapWidth * 0.58);
                    return SizedBox(
                      height: mapHeight,
                      child: Stack(
                        children: [
                          Positioned.fill(
                            child: DecoratedBox(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(26),
                                gradient: LinearGradient(
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                  colors: isDark
                                      ? const [
                                          Color(0xFF11161C),
                                          Color(0xFF1D2833),
                                          Color(0xFF11161C),
                                        ]
                                      : const [
                                          Color(0xFFEAF1F8),
                                          Color(0xFFDDE8F2),
                                          Color(0xFFF7FBFF),
                                        ],
                                ),
                              ),
                              child: CustomPaint(
                                painter: _WorldMapPainter(isDark: isDark),
                              ),
                            ),
                          ),
                          ...posts.asMap().entries.map((entry) {
                            final index = entry.key;
                            final post = entry.value;
                            final isSelected = index == selectedIndex;
                            final offset = _projectPoint(
                              post.latitude,
                              post.longitude,
                              mapWidth,
                              mapHeight,
                            );
                            return Positioned(
                              left: offset.dx - 16,
                              top: offset.dy - 16,
                              child: GestureDetector(
                                onTap: () => onSelect(index),
                                child: AnimatedContainer(
                                  duration: BDMotion.durationNormal,
                                  curve: BDMotion.curveFluid,
                                  width: isSelected ? 34 : 28,
                                  height: isSelected ? 34 : 28,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: isSelected
                                        ? const Color(0xFFE9654B)
                                        : const Color(0xFF2E7CF6),
                                    border: Border.all(
                                      color: Colors.white.withValues(
                                        alpha: 0.92,
                                      ),
                                      width: 3,
                                    ),
                                  ),
                                  child: Icon(
                                    Icons.location_on_rounded,
                                    color: Colors.white,
                                    size: isSelected ? 18 : 16,
                                  ),
                                ),
                              ),
                            );
                          }),
                        ],
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          if (selectedPost != null)
            _SelectedLocationCard(
              post: selectedPost!,
              relatedCount: posts
                  .where((post) => post.placeName == selectedPost!.placeName)
                  .length,
              onOpenViewer: () => onOpenViewer(selectedPost!),
              onOpenLocationHub: () => onOpenLocationHub(selectedPost!),
            ),
          const SizedBox(height: 14),
          SizedBox(
            height: 154,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: posts.length,
              separatorBuilder: (_, _) => const SizedBox(width: 12),
              itemBuilder: (context, index) {
                final post = posts[index];
                final isActive = index == selectedIndex;
                return SizedBox(
                  width: 244,
                  child: InkWell(
                    borderRadius: BDDesign.radiusLarge,
                    onTap: () => onSelect(index),
                    child: AnimatedContainer(
                      duration: BDMotion.durationNormal,
                      curve: BDMotion.curveFluid,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: isActive
                            ? (isDark
                                  ? AppTheme.darkSurfaceElevated
                                  : const Color(0xFFF7FAFD))
                            : (isDark
                                  ? AppTheme.darkSurface.withValues(alpha: 0.94)
                                  : Colors.white.withValues(alpha: 0.88)),
                        borderRadius: BDDesign.radiusLarge,
                        border: Border.all(
                          color: isActive
                              ? const Color(0xFF2E7CF6)
                              : (isDark
                                    ? Colors.white.withValues(alpha: 0.06)
                                    : BDDesign.colorMutedBlue.withValues(
                                        alpha: 0.08,
                                      )),
                        ),
                      ),
                      child: Row(
                        children: [
                          _CommunityThumbnail(
                            imageUrl: post.coverUrl,
                            height: 130,
                            width: 88,
                            icon: Icons.explore_rounded,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  post.placeName,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color: textColor,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  post.title,
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color: textColor.withValues(alpha: 0.86),
                                    height: 1.25,
                                  ),
                                ),
                                const Spacer(),
                                BDStatusPill(
                                  label: post.modelName,
                                  icon: Icons.view_in_ar_rounded,
                                  color: isActive
                                      ? const Color(0xFF2E7CF6)
                                      : BDDesign.colorMutedBlue,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Offset _projectPoint(
    double latitude,
    double longitude,
    double width,
    double height,
  ) {
    final dx = (longitude + 180) / 360 * width;
    final dy = (90 - latitude) / 180 * height;
    return Offset(dx.clamp(16.0, width - 16.0), dy.clamp(16.0, height - 16.0));
  }
}

class _CommunityFeedCard extends StatelessWidget {
  final CommunityPost post;
  final VoidCallback onOpenViewer;
  final VoidCallback onOpenLocationHub;

  const _CommunityFeedCard({
    required this.post,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.66)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.84);

    return BDPanelCard(
      borderRadius: BorderRadius.circular(30),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(30),
        child: Stack(
          fit: StackFit.expand,
          children: [
            _CommunityThumbnail(
              imageUrl: post.coverUrl,
              height: double.infinity,
              width: double.infinity,
              icon: Icons.slow_motion_video_rounded,
            ),
            DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withValues(alpha: 0.08),
                    Colors.black.withValues(alpha: 0.18),
                    Colors.black.withValues(alpha: 0.74),
                  ],
                ),
              ),
            ),
            Positioned(
              top: 18,
              left: 18,
              right: 18,
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  BDStatusPill(
                    label: post.modelName,
                    icon: Icons.view_in_ar_rounded,
                    color: const Color(0xFF78A6FF),
                  ),
                  BDStatusPill(
                    label: post.placeName,
                    icon: Icons.place_rounded,
                    color: const Color(0xFFEAC86B),
                  ),
                ],
              ),
            ),
            Positioned(
              left: 18,
              right: 18,
              bottom: 18,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    post.title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.w800,
                      height: 1.05,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    post.caption,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.88),
                      height: 1.45,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 14),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      _TranslucentTag(label: post.authorName),
                      _TranslucentTag(label: post.relativeTimeLabel),
                      ...post.tags
                          .take(3)
                          .map((tag) => _TranslucentTag(label: tag)),
                    ],
                  ),
                  const SizedBox(height: 18),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: onOpenViewer,
                          icon: const Icon(Icons.play_circle_fill_rounded),
                          label: const Text('进入记忆'),
                        ),
                      ),
                      const SizedBox(width: 10),
                      OutlinedButton.icon(
                        onPressed: onOpenLocationHub,
                        icon: Icon(
                          Icons.public_rounded,
                          color: isDark ? BDDesign.colorPaperWhite : textColor,
                        ),
                        label: Text(
                          '地点内容',
                          style: TextStyle(
                            color: isDark
                                ? BDDesign.colorPaperWhite
                                : textColor,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '向下继续刷，可以切到下一个地点的 3D 记忆。',
                    style: TextStyle(color: hintColor, fontSize: 12.5),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SelectedLocationCard extends StatelessWidget {
  final CommunityPost post;
  final int relatedCount;
  final VoidCallback onOpenViewer;
  final VoidCallback onOpenLocationHub;

  const _SelectedLocationCard({
    required this.post,
    required this.relatedCount,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        children: [
          Row(
            children: [
              _CommunityThumbnail(
                imageUrl: post.coverUrl,
                height: 118,
                width: 104,
                icon: Icons.terrain_rounded,
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      post.placeName,
                      style: TextStyle(
                        color: textColor,
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${post.latitude.toStringAsFixed(3)}, ${post.longitude.toStringAsFixed(3)}',
                      style: TextStyle(color: hintColor, fontSize: 12.5),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      post.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: textColor,
                        fontWeight: FontWeight.w700,
                        height: 1.2,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '这一地点已聚合 $relatedCount 条记忆，优先展示当前最热的 3D 模型。',
                      style: TextStyle(color: hintColor, height: 1.35),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: onOpenViewer,
                  icon: const Icon(Icons.travel_explore_rounded),
                  label: const Text('打开当前模型'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onOpenLocationHub,
                  icon: const Icon(Icons.map_rounded),
                  label: const Text('查看同地点内容'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _LocationHubRow extends StatelessWidget {
  final CommunityPost post;

  const _LocationHubRow({required this.post});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isDark
            ? AppTheme.darkSurfaceElevated.withValues(alpha: 0.94)
            : const Color(0xFFF6F8FB),
        borderRadius: BDDesign.radiusLarge,
        border: Border.all(
          color: isDark
              ? Colors.white.withValues(alpha: 0.06)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
        ),
      ),
      child: Row(
        children: [
          _CommunityThumbnail(
            imageUrl: post.coverUrl,
            height: 72,
            width: 92,
            icon: Icons.landscape_rounded,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  post.title,
                  style: TextStyle(
                    color: textColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${post.authorName} · ${post.modelName}',
                  style: TextStyle(color: hintColor, fontSize: 12.5),
                ),
                const SizedBox(height: 8),
                Text(
                  post.caption,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: textColor.withValues(alpha: 0.82),
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Icon(Icons.chevron_right_rounded, color: hintColor),
        ],
      ),
    );
  }
}

class _CommunityMetricCard extends StatelessWidget {
  final String label;
  final String value;
  final String hint;

  const _CommunityMetricCard({
    required this.label,
    required this.value,
    required this.hint,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.58)
                  : BDDesign.colorMutedBlue,
              fontWeight: FontWeight.w700,
              fontSize: 12.5,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            hint,
            style: TextStyle(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.54)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.86),
              height: 1.35,
              fontSize: 12.5,
            ),
          ),
        ],
      ),
    );
  }
}

class _CommunityThumbnail extends StatelessWidget {
  final String? imageUrl;
  final double height;
  final double width;
  final IconData icon;

  const _CommunityThumbnail({
    required this.imageUrl,
    required this.height,
    required this.width,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final fallback = Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF8BA8C5), Color(0xFF536C8B), Color(0xFF38485F)],
        ),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Center(
        child: Icon(
          icon,
          size: 34,
          color: Colors.white.withValues(alpha: 0.88),
        ),
      ),
    );

    final url = imageUrl;
    return SizedBox(
      height: height,
      width: width,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: url == null || url.isEmpty
            ? fallback
            : Image.network(
                url,
                fit: BoxFit.cover,
                errorBuilder: (_, _, _) => fallback,
                loadingBuilder: (context, child, loadingProgress) {
                  if (loadingProgress == null) {
                    return child;
                  }
                  return fallback;
                },
              ),
      ),
    );
  }
}

class _TranslucentTag extends StatelessWidget {
  final String label;

  const _TranslucentTag({required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: Colors.white.withValues(alpha: 0.88),
          fontWeight: FontWeight.w600,
          fontSize: 12.5,
        ),
      ),
    );
  }
}

class _CommunityEmptyState extends StatelessWidget {
  const _CommunityEmptyState();

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 6, 20, 104),
      child: BDPanelCard(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.public_off_rounded,
                size: 48,
                color: isDark
                    ? Colors.white.withValues(alpha: 0.48)
                    : BDDesign.colorMutedBlue,
              ),
              const SizedBox(height: 14),
              Text(
                '社区流暂时为空',
                style: TextStyle(
                  color: isDark
                      ? BDDesign.colorPaperWhite
                      : BDDesign.colorInkBlack,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                '先分享一个带地点的记忆模型，地图和流式浏览会同时出现。',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.62)
                      : BDDesign.colorMutedBlue.withValues(alpha: 0.88),
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _CommunityComposerSheet extends StatefulWidget {
  final List<CommunityModelOption> models;
  final String? initialModelId;

  const _CommunityComposerSheet({required this.models, this.initialModelId});

  @override
  State<_CommunityComposerSheet> createState() =>
      _CommunityComposerSheetState();
}

class _CommunityComposerSheetState extends State<_CommunityComposerSheet> {
  late final TextEditingController _titleController;
  late final TextEditingController _captionController;
  late final TextEditingController _placeController;
  late final TextEditingController _latitudeController;
  late final TextEditingController _longitudeController;
  CommunityModelOption? _selectedModel;
  bool _isSubmitting = false;

  static const _presets = <_LocationPreset>[
    _LocationPreset('西湖', 30.243, 120.150),
    _LocationPreset('外滩', 31.240, 121.490),
    _LocationPreset('东京塔', 35.659, 139.745),
    _LocationPreset('巴黎左岸', 48.853, 2.349),
    _LocationPreset('纽约中央公园', 40.782, -73.965),
  ];

  @override
  void initState() {
    super.initState();
    _selectedModel = widget.models.isEmpty
        ? null
        : widget.models.firstWhere(
            (model) => model.id == widget.initialModelId,
            orElse: () => widget.models.first,
          );
    _titleController = TextEditingController(
      text: _selectedModel == null
          ? ''
          : '我在 ${_selectedModel!.sceneId} 留下的一段记忆',
    );
    _captionController = TextEditingController();
    _placeController = TextEditingController(text: '西湖');
    _latitudeController = TextEditingController(text: '30.243');
    _longitudeController = TextEditingController(text: '120.150');
  }

  @override
  void dispose() {
    _titleController.dispose();
    _captionController.dispose();
    _placeController.dispose();
    _latitudeController.dispose();
    _longitudeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final mediaQuery = MediaQuery.of(context);
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Padding(
      padding: EdgeInsets.only(bottom: mediaQuery.viewInsets.bottom),
      child: DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.82,
        maxChildSize: 0.92,
        minChildSize: 0.62,
        builder: (context, scrollController) {
          return Padding(
            padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
            child: BDPanelCard(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
              child: SafeArea(
                top: false,
                child: SingleChildScrollView(
                  controller: scrollController,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  '分享你的地点记忆',
                                  style: TextStyle(
                                    color: textColor,
                                    fontSize: 22,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  '先选择一个你已有的模型，再给它补上地点和一句简短说明。',
                                  style: TextStyle(
                                    color: hintColor,
                                    height: 1.4,
                                  ),
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
                      const SizedBox(height: 18),
                      if (widget.models.isEmpty)
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: isDark
                                ? AppTheme.darkSurfaceElevated
                                : const Color(0xFFF7FAFD),
                            borderRadius: BDDesign.radiusLarge,
                          ),
                          child: Text(
                            '还没有可分享的模型。先在“过往回忆”里生成一个 3D 模型，再回来发布社区贴文。',
                            style: TextStyle(color: hintColor, height: 1.4),
                          ),
                        )
                      else
                        DropdownButtonFormField<CommunityModelOption>(
                          initialValue: _selectedModel,
                          decoration: const InputDecoration(
                            labelText: '选择要分享的模型',
                          ),
                          items: widget.models
                              .map(
                                (model) => DropdownMenuItem(
                                  value: model,
                                  child: Text(model.sceneId),
                                ),
                              )
                              .toList(),
                          onChanged: (value) {
                            setState(() {
                              _selectedModel = value;
                            });
                          },
                        ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _titleController,
                        decoration: const InputDecoration(labelText: '标题'),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _captionController,
                        minLines: 3,
                        maxLines: 5,
                        decoration: const InputDecoration(
                          labelText: '这段记忆想让别人看到什么',
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        '快速地点预设',
                        style: TextStyle(
                          color: textColor,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: _presets.map((preset) {
                          return ActionChip(
                            label: Text(preset.name),
                            onPressed: () {
                              setState(() {
                                _placeController.text = preset.name;
                                _latitudeController.text = preset.latitude
                                    .toStringAsFixed(3);
                                _longitudeController.text = preset.longitude
                                    .toStringAsFixed(3);
                              });
                            },
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _placeController,
                        decoration: const InputDecoration(labelText: '地点名称'),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _latitudeController,
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                    decimal: true,
                                    signed: true,
                                  ),
                              decoration: const InputDecoration(
                                labelText: '纬度',
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: TextField(
                              controller: _longitudeController,
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                    decimal: true,
                                    signed: true,
                                  ),
                              decoration: const InputDecoration(
                                labelText: '经度',
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 18),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: widget.models.isEmpty || _isSubmitting
                              ? null
                              : _submit,
                          icon: _isSubmitting
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Icon(Icons.send_rounded),
                          label: Text(_isSubmitting ? '发布中...' : '发布到社区'),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Future<void> _submit() async {
    final model = _selectedModel;
    final latitude = double.tryParse(_latitudeController.text.trim());
    final longitude = double.tryParse(_longitudeController.text.trim());
    final title = _titleController.text.trim();
    final caption = _captionController.text.trim();
    final place = _placeController.text.trim();

    if (model == null ||
        latitude == null ||
        longitude == null ||
        title.isEmpty ||
        caption.isEmpty ||
        place.isEmpty) {
      TDToast.showText(context: context, '请把模型、标题、文案和地点都填完整');
      return;
    }

    setState(() {
      _isSubmitting = true;
    });

    Navigator.pop(
      context,
      CommunityComposerResult(
        title: title,
        caption: caption,
        placeName: place,
        latitude: latitude,
        longitude: longitude,
        model: model,
      ),
    );
  }
}

class _WorldMapPainter extends CustomPainter {
  final bool isDark;

  const _WorldMapPainter({required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = (isDark ? Colors.white : BDDesign.colorMutedBlue).withValues(
        alpha: 0.08,
      )
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    for (var i = 1; i < 6; i++) {
      final dy = size.height * i / 6;
      canvas.drawLine(Offset(0, dy), Offset(size.width, dy), gridPaint);
    }

    final landPaint = Paint()
      ..color = isDark
          ? const Color(0xFFAFD0E9).withValues(alpha: 0.16)
          : const Color(0xFF8AA9C7).withValues(alpha: 0.28);

    void drawLand(List<Offset> points) {
      final path = Path()
        ..moveTo(points.first.dx * size.width, points.first.dy * size.height);
      for (final point in points.skip(1)) {
        path.lineTo(point.dx * size.width, point.dy * size.height);
      }
      path.close();
      canvas.drawPath(path, landPaint);
    }

    drawLand(const [
      Offset(0.08, 0.20),
      Offset(0.20, 0.13),
      Offset(0.28, 0.17),
      Offset(0.32, 0.28),
      Offset(0.25, 0.40),
      Offset(0.18, 0.38),
      Offset(0.13, 0.46),
      Offset(0.09, 0.35),
    ]);
    drawLand(const [
      Offset(0.42, 0.18),
      Offset(0.52, 0.13),
      Offset(0.62, 0.18),
      Offset(0.67, 0.30),
      Offset(0.61, 0.36),
      Offset(0.54, 0.32),
      Offset(0.49, 0.35),
      Offset(0.46, 0.26),
    ]);
    drawLand(const [
      Offset(0.67, 0.23),
      Offset(0.82, 0.19),
      Offset(0.92, 0.28),
      Offset(0.88, 0.42),
      Offset(0.78, 0.43),
      Offset(0.71, 0.36),
    ]);
  }

  @override
  bool shouldRepaint(covariant _WorldMapPainter oldDelegate) {
    return oldDelegate.isDark != isDark;
  }
}

class _LocationPreset {
  final String name;
  final double latitude;
  final double longitude;

  const _LocationPreset(this.name, this.latitude, this.longitude);
}

class CommunityComposerResult {
  final String title;
  final String caption;
  final String placeName;
  final double latitude;
  final double longitude;
  final CommunityModelOption model;

  const CommunityComposerResult({
    required this.title,
    required this.caption,
    required this.placeName,
    required this.latitude,
    required this.longitude,
    required this.model,
  });
}

class CommunityModelOption {
  final String id;
  final String sceneId;
  final String description;
  final String modelUrl;
  final String? posesUrl;
  final String? coverUrl;

  const CommunityModelOption({
    required this.id,
    required this.sceneId,
    required this.description,
    required this.modelUrl,
    required this.posesUrl,
    required this.coverUrl,
  });
}

class CommunityPost {
  final String id;
  final String title;
  final String caption;
  final String placeName;
  final double latitude;
  final double longitude;
  final String authorName;
  final String modelName;
  final String modelUrl;
  final String? posesUrl;
  final String? coverUrl;
  final DateTime createdAt;
  final List<String> tags;

  const CommunityPost({
    required this.id,
    required this.title,
    required this.caption,
    required this.placeName,
    required this.latitude,
    required this.longitude,
    required this.authorName,
    required this.modelName,
    required this.modelUrl,
    required this.posesUrl,
    required this.coverUrl,
    required this.createdAt,
    required this.tags,
  });

  String get relativeTimeLabel {
    final difference = DateTime.now().difference(createdAt);
    if (difference.inMinutes < 60) {
      return '${math.max(difference.inMinutes, 1)} 分钟前';
    }
    if (difference.inHours < 24) {
      return '${difference.inHours} 小时前';
    }
    return '${difference.inDays} 天前';
  }
}

class CommunityRepository {
  static final List<CommunityPost> _localDrafts = [];

  SupabaseClient get _client => Supabase.instance.client;

  Future<List<CommunityPost>> fetchPosts() async {
    try {
      final response = await _client
          .from('community_posts')
          .select('''
            id,
            title,
            caption,
            place_name,
            latitude,
            longitude,
            user_id,
            created_at,
            model_name,
            cover_image_url,
            model_assets (
              scene_id,
              description,
              ply_path,
              preview_img_path
            )
          ''')
          .order('created_at', ascending: false)
          .limit(24);

      final posts = response.map<CommunityPost>((raw) {
        final map = Map<String, dynamic>.from(raw);
        final model = map['model_assets'] is Map
            ? Map<String, dynamic>.from(map['model_assets'] as Map)
            : <String, dynamic>{};
        final modelUrl = _publicModelUrl(model['ply_path']?.toString() ?? '');
        final previewUrl = map['cover_image_url']?.toString().isNotEmpty == true
            ? map['cover_image_url']?.toString()
            : model['preview_img_path']?.toString();

        return CommunityPost(
          id: map['id'].toString(),
          title: map['title']?.toString() ?? '未命名记忆',
          caption:
              map['caption']?.toString() ??
              model['description']?.toString() ??
              '',
          placeName: map['place_name']?.toString() ?? '未标注地点',
          latitude: (map['latitude'] as num?)?.toDouble() ?? 0,
          longitude: (map['longitude'] as num?)?.toDouble() ?? 0,
          authorName: map['user_id']?.toString() ?? '匿名用户',
          modelName:
              map['model_name']?.toString() ??
              model['scene_id']?.toString() ??
              '3D 模型',
          modelUrl: modelUrl.isEmpty
              ? './models/scene_auto_sync_raw.ply'
              : modelUrl,
          posesUrl: _posesUrlFromPath(model['ply_path']?.toString()),
          coverUrl: previewUrl,
          createdAt:
              DateTime.tryParse(map['created_at']?.toString() ?? '') ??
              DateTime.now(),
          tags: _extractTags(
            model['description']?.toString(),
            map['place_name']?.toString(),
          ),
        );
      }).toList();

      final merged = [..._localDrafts, ...posts];
      if (merged.isNotEmpty) {
        return merged;
      }
    } catch (_) {}

    return [..._localDrafts, ..._demoPosts];
  }

  Future<List<CommunityModelOption>> fetchShareableModels() async {
    try {
      final userId = _client.auth.currentUser?.id;
      dynamic query = _client
          .from('model_assets')
          .select('id, scene_id, description, ply_path, preview_img_path')
          .order('created_at', ascending: false)
          .limit(20);
      if (userId != null && userId.isNotEmpty) {
        query = query.eq('user_id', userId);
      }

      final response = await query;
      final models = response.map<CommunityModelOption>((raw) {
        final map = Map<String, dynamic>.from(raw);
        final path = map['ply_path']?.toString() ?? '';
        final publicUrl = _publicModelUrl(path);
        return CommunityModelOption(
          id: map['id'].toString(),
          sceneId: map['scene_id']?.toString() ?? '未命名模型',
          description: map['description']?.toString() ?? '',
          modelUrl: publicUrl.isEmpty
              ? './models/scene_auto_sync_raw.ply'
              : publicUrl,
          posesUrl: _posesUrlFromPath(path),
          coverUrl: map['preview_img_path']?.toString(),
        );
      }).toList();

      if (models.isNotEmpty) {
        return models;
      }
    } catch (_) {}

    return _demoModels;
  }

  Future<CommunityPost> createPost(CommunityComposerResult draft) async {
    final optimistic = CommunityPost(
      id: 'local-${DateTime.now().microsecondsSinceEpoch}',
      title: draft.title,
      caption: draft.caption,
      placeName: draft.placeName,
      latitude: draft.latitude,
      longitude: draft.longitude,
      authorName: _client.auth.currentUser?.email ?? '我',
      modelName: draft.model.sceneId,
      modelUrl: draft.model.modelUrl,
      posesUrl: draft.model.posesUrl,
      coverUrl: draft.model.coverUrl,
      createdAt: DateTime.now(),
      tags: _extractTags(draft.model.description, draft.placeName),
    );

    try {
      await _client.from('community_posts').insert({
        'user_id': _client.auth.currentUser?.id ?? 'local-user',
        'model_asset_id': draft.model.id,
        'model_name': draft.model.sceneId,
        'title': draft.title,
        'caption': draft.caption,
        'place_name': draft.placeName,
        'latitude': draft.latitude,
        'longitude': draft.longitude,
        'cover_image_url': draft.model.coverUrl,
      });
      return optimistic;
    } catch (_) {
      _localDrafts.insert(0, optimistic);
      return optimistic;
    }
  }

  String _publicModelUrl(String storagePath) {
    if (storagePath.isEmpty) {
      return '';
    }
    if (storagePath.startsWith('http://') ||
        storagePath.startsWith('https://')) {
      return storagePath;
    }
    try {
      return _client.storage
          .from('braindance-assets')
          .getPublicUrl(storagePath);
    } catch (_) {
      return storagePath;
    }
  }

  String? _posesUrlFromPath(String? storagePath) {
    if (storagePath == null || storagePath.isEmpty) {
      return null;
    }
    final posesPath = storagePath.replaceAll(
      RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
      'webgl_poses.json',
    );
    if (posesPath == storagePath) {
      return null;
    }
    try {
      return _client.storage.from('braindance-assets').getPublicUrl(posesPath);
    } catch (_) {
      return null;
    }
  }

  List<String> _extractTags(String? description, String? placeName) {
    final words = <String>[];
    if (placeName != null && placeName.isNotEmpty) {
      words.add(placeName);
    }
    if (description != null && description.isNotEmpty) {
      final tokens = description
          .split(RegExp(r'[\s,，。.!！？]+'))
          .where((token) => token.trim().length >= 2)
          .take(2);
      words.addAll(tokens);
    }
    if (words.isEmpty) {
      words.add('记忆');
    }
    return words.take(3).toList();
  }
}

final List<CommunityModelOption> _demoModels = [
  CommunityModelOption(
    id: 'demo-1',
    sceneId: '西湖断桥晨雾',
    description: '湖面、柳树、桥面和低雾一起形成了安静的晨间空间。',
    modelUrl: './models/scene_auto_sync_raw.ply',
    posesUrl: null,
    coverUrl: null,
  ),
  CommunityModelOption(
    id: 'demo-2',
    sceneId: '东京塔夜色',
    description: '夜间城市灯光包围着塔体，适合做高对比空间浏览。',
    modelUrl: './models/scene_auto_sync_raw.ply',
    posesUrl: null,
    coverUrl: null,
  ),
];

final List<CommunityPost> _demoPosts = [
  CommunityPost(
    id: 'demo-post-1',
    title: '清晨刚亮时的断桥',
    caption: '薄雾和湖面的反光被一起留在模型里，适合从桥头慢慢推进看空间层次。',
    placeName: '杭州西湖',
    latitude: 30.258,
    longitude: 120.140,
    authorName: 'Lin',
    modelName: '西湖断桥晨雾',
    modelUrl: './models/scene_auto_sync_raw.ply',
    posesUrl: null,
    coverUrl: null,
    createdAt: DateTime.now().subtract(const Duration(hours: 2)),
    tags: const ['湖面', '晨雾', '桥'],
  ),
  CommunityPost(
    id: 'demo-post-2',
    title: '东京塔下的夜风',
    caption: '我把塔体、街道和人流一起收进去了，向下翻的时候会先看到塔身，再看到街区。',
    placeName: '东京塔',
    latitude: 35.659,
    longitude: 139.745,
    authorName: 'Aoi',
    modelName: '东京塔夜色',
    modelUrl: './models/scene_auto_sync_raw.ply',
    posesUrl: null,
    coverUrl: null,
    createdAt: DateTime.now().subtract(const Duration(hours: 6)),
    tags: const ['夜景', '城市', '塔体'],
  ),
];
