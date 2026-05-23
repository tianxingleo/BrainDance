import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:braindance/extra_func/theme_animation_notifier.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/widgets/bd_tab_switcher.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/pages/community/detail.dart';
import 'package:braindance/pages/community/models.dart';
import 'package:braindance/pages/community/repository.dart';
import 'package:braindance/pages/my/my_page_tabs.dart';
import 'package:braindance/pages/recall/overview_card.dart';
import 'package:braindance/pages/task_list.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../main.dart'
    show myPostsRefreshSignal, overviewStatsProvider, pageAnimatingProvider, pageIndexProvider;

class SettingsPage extends ConsumerStatefulWidget {
  const SettingsPage({super.key});

  @override
  ConsumerState<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends ConsumerState<SettingsPage>
    with TickerProviderStateMixin {
  late final TabController tabController;
  int _currentTabIndex = 0;
  final GlobalKey _themeSwitchKey = GlobalKey();
  final CommunityRepository _communityRepository = CommunityRepository();
  CommunityStats _communityStats = const CommunityStats();
  List<CommunityPost> _myPosts = const [];
  List<CommunityPost> _favoritePosts = const [];
  List<CommunityPost> _likedPosts = const [];
  CommunityDraft _communityDraft = const CommunityDraft();
  bool _isCommunityLoading = true;

  @override
  void initState() {
    super.initState();
    tabController = TabController(
      length: 3,
      vsync: this,
      animationDuration: const Duration(milliseconds: 200),
    );
    tabController.addListener(_handleTabChange);
    _loadCommunityAccount();
  }

  @override
  void dispose() {
    tabController.removeListener(_handleTabChange);
    tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final bottomInset = MediaQuery.paddingOf(context).bottom;
    final bottomContentPadding = bottomInset + 132.0;

    ref.listen(myPostsRefreshSignal, (prev, next) {
      if (prev != next) _loadCommunityAccount();
    });

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          bottom: false,
          child: SingleChildScrollView(
            padding: EdgeInsets.only(bottom: bottomContentPadding),
            child: Column(
              children: [
                BDPageHeader(
                  title: textLocalize('mine'),
                  subtitle: textLocalize('my_subtitle'),
                  trailing: GestureDetector(
                    onTap: () async {
                      final currentState = ref.read(themeAnimationProvider);

                      if (currentState.isAnimating) {
                        ref.read(themeAnimationProvider.notifier).toggleDirection(
                          themeAnimationFraction.value,
                        );
                        return;
                      }

                      final screenSize = MediaQuery.sizeOf(context);
                      final isDarkNow = AppConfig.isNightMode;
                      final mode = isDarkNow
                          ? ThemeTransitionMode.expandHole
                          : ThemeTransitionMode.shrinkClip;
                      final center = Offset(screenSize.width, 0);

                      final boundary =
                          themeAnimationKey.currentContext?.findRenderObject()
                              as RenderRepaintBoundary?;
                      if (boundary != null) {
                        try {
                          final dpr = MediaQuery.devicePixelRatioOf(context);
                          final image = await boundary.toImage(pixelRatio: dpr);
                          ref
                              .read(themeAnimationProvider.notifier)
                              .start(image, center, mode);
                        } catch (e) {
                          debugPrint('Theme transition error: $e');
                        }
                      }

                      WidgetsBinding.instance.addPostFrameCallback((_) {
                        SetConfig.setNightMode(!AppConfig.isNightMode, ref);
                        SetConfig.saveMsgToFile();
                      });
                    },
                    child: BDStatusPill(
                      key: _themeSwitchKey,
                      label: textLocalize(
                        isDark ? 'set_theme_night' : 'set_theme_day',
                      ),
                      icon: isDark
                          ? Icons.dark_mode_rounded
                          : Icons.wb_sunny_rounded,
                      color: textColor,
                    ),
                  ),
                ),

                Consumer(
                  builder: (context, ref, _) {
                    final stats = ref.watch(overviewStatsProvider);
                    return RecallOverviewCard(
                      isDark: isDark,
                      textColor: textColor,
                      recentCount: stats['recentCount'] ?? 0,
                      allModelCount: stats['allModelCount'] ?? 0,
                      processingTaskCount: stats['processingTaskCount'] ?? 0,
                      onOpenTasks: () {
                        Navigator.push(
                          context,
                          PageRouteBuilder(
                            transitionDuration: BDMotion.durationNormal,
                            reverseTransitionDuration: BDMotion.durationNormal,
                            opaque: true,
                            pageBuilder: (_, __, ___) => const TaskListPage(),
                            transitionsBuilder: (ctx, animation, __, child) {
                              final curved = animation.drive(
                                CurveTween(curve: Curves.easeInOutCubic),
                              );
                              return AnimatedBuilder(
                                animation: curved,
                                builder: (_, child) {
                                  final screenHeight = MediaQuery.sizeOf(ctx).height;
                                  return Transform.translate(
                                    offset: Offset(0, -(1.0 - curved.value) * screenHeight),
                                    child: child,
                                  );
                                },
                                child: child,
                              );
                            },
                          ),
                        ).then((_) {
                          FocusManager.instance.primaryFocus?.unfocus();
                        });
                      },
                    );
                  },
                ),
                const SizedBox(height: 14),
                _SettingsTabSwitch(controller: tabController),
                const SizedBox(height: 10),
                _buildTabContent(context, ref),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _handleTabChange() {
    final nextIndex = tabController.index;
    if (_currentTabIndex == nextIndex) {
      return;
    }
    setState(() {
      _currentTabIndex = nextIndex;
    });
  }

  Widget _buildTabContent(BuildContext context, WidgetRef ref) {
    return SizedBox(
      width: double.infinity,
      child: BDTabSwitcher(
        index: _currentTabIndex,
        children: [
          MyOverviewTab(
            userLabel: _userLabel,
            stats: _communityStats,
            isLoading: _isCommunityLoading,
            onOpenCommunity: _openCommunityTab,
            onRefresh: _loadCommunityAccount,
          ),
          MyCommunityTab(
            myPosts: _myPosts,
            favoritePosts: _favoritePosts,
            likedPosts: _likedPosts,
            draft: _communityDraft,
            isLoading: _isCommunityLoading,
            onOpenPost: _openPost,
            onDeletePost: _confirmDeletePost,
            onToggleVisibility: _togglePostVisibility,
            onContinueDraft: _openCommunityTab,
            onRefresh: _loadCommunityAccount,
          ),
          MySettingsTab(ref: ref),
        ],
      ),
    );
  }

  String get _userLabel {
    final user = Supabase.instance.client.auth.currentUser;
    if (user?.email?.isNotEmpty == true) {
      return user!.email!;
    }
    if (user?.id.isNotEmpty == true) {
      return user!.id;
    }
    return textLocalize('my_guest_user');
  }

  Future<void> _loadCommunityAccount() async {
    setState(() => _isCommunityLoading = true);
    final stats = await _communityRepository.fetchCommunityStats();
    final myPosts = await _communityRepository.fetchMyPosts();
    final favoritePosts = await _communityRepository.fetchFavoritePosts();
    final likedPosts = await _communityRepository.fetchLikedPosts();
    final draft = await _communityRepository.loadDraft();
    if (!mounted) return;
    setState(() {
      _communityStats = stats;
      _myPosts = myPosts;
      _favoritePosts = favoritePosts;
      _likedPosts = likedPosts;
      _communityDraft = draft;
      _isCommunityLoading = false;
    });
  }

  void _openCommunityTab() {
    ref.read(pageIndexProvider.notifier).state = 3;
  }

  void _openPost(CommunityPost post) {
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
                parent: animation,
                curve: Curves.easeInOutCubic,
              ),
            ),
            child: child,
          );
        },
      ),
    ).then((_) => _loadCommunityAccount());
  }

  Future<void> _togglePostVisibility(CommunityPost post) async {
    await _communityRepository.togglePostVisibility(post);
    await _loadCommunityAccount();
  }

  Future<void> _confirmDeletePost(CommunityPost post) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text(textLocalize('my_delete_post')),
          content: Text(textLocalize('my_delete_post_confirm')),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text(textLocalize('recall_delete_confirm_cancel')),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: Text(textLocalize('recall_delete_confirm_yes')),
            ),
          ],
        );
      },
    );
    if (confirmed != true) return;
    await _communityRepository.deletePost(post.id);
    await _loadCommunityAccount();
  }
}

class _SettingsTabSwitch extends StatelessWidget {
  final TabController controller;

  const _SettingsTabSwitch({required this.controller});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final navBackground = isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.55)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.52);
    final navBorder = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
    // Use slightly less shadow for this inner pill than the main floating bar
    final navShadow = Colors.black.withValues(alpha: isDark ? 0.22 : 0.05);

    final selectedBackground = isDark
        ? const Color(0xFFAEBAC7).withValues(alpha: 0.14)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.09);
    final selectedColor = isDark
        ? const Color(0xFFF4F7FA)
        : BDDesign.colorInkBlack;
    final unselectedColor = isDark
        ? const Color(0xFFB4BEC9)
        : const Color(0xFF9AA3AD);

    return Consumer(
      builder: (_, ref, child) {
        final skipBlur = ref.watch(pageAnimatingProvider);
        final content = Container(
          padding: const EdgeInsets.all(4.0),
          decoration: BoxDecoration(
            color: navBackground,
            borderRadius: BDDesign.radiusLarge,
            border: Border.all(color: navBorder, width: 1.0),
            boxShadow: [
              BoxShadow(
                color: navShadow,
                blurRadius: 28,
                offset: const Offset(0, 8),
              ),
            ],
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final tabWidth = constraints.maxWidth / 3;
                return Stack(
                  children: [
                    AnimatedBuilder(
                      animation: controller.animation!,
                      builder: (context, child) {
                        final double offset =
                            controller.animation!.value * tabWidth;
                        return Positioned(
                          left: offset,
                          width: tabWidth,
                          top: 0,
                          bottom: 0,
                          child: Container(
                            decoration: BoxDecoration(
                              color: selectedBackground,
                              borderRadius: BorderRadius.circular(24),
                            ),
                          ),
                        );
                      },
                    ),
                    Row(
                      children: [
                        _buildTabItem(
                          0,
                          textLocalize('my_tab_overview'),
                          selectedColor,
                          unselectedColor,
                        ),
                        _buildTabItem(
                          1,
                          textLocalize('my_tab_community'),
                          selectedColor,
                          unselectedColor,
                        ),
                        _buildTabItem(
                          2,
                          textLocalize('my_tab_settings'),
                          selectedColor,
                          unselectedColor,
                        ),
                      ],
                    ),
                  ],
                );
              },
            ),
          );
          return Container(
            margin: const EdgeInsets.symmetric(horizontal: 20),
            height: 56,
            child: ClipRRect(
              borderRadius: BDDesign.radiusLarge,
              child: skipBlur
                  ? content
                  : BackdropFilter(
                      filter: ui.ImageFilter.blur(
                        sigmaX: 24.0,
                        sigmaY: 24.0,
                      ),
                      child: content,
                    ),
            ),
          );
        },
      );
  }

  Widget _buildTabItem(
    int index,
    String label,
    Color selectedColor,
    Color unselectedColor,
  ) {
    return Expanded(
      child: GestureDetector(
        onTap: () => controller.animateTo(index),
        behavior: HitTestBehavior.opaque,
        child: Center(
          child: AnimatedBuilder(
            animation: controller.animation!,
            builder: (ctx, child) {
              final selected =
                  (controller.animation!.value - index).abs() < 0.5;
              return Text(
                label,
                style: TextStyle(
                  color: selected ? selectedColor : unselectedColor,
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
