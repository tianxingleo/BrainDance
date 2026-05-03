import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:braindance/extra_func/theme_animation_notifier.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/widgets/bd_tab_switcher.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/pages/recall/overview_card.dart';
import 'package:braindance/pages/settabs/settab1.dart';
import 'package:braindance/pages/settabs/settab3.dart';
import 'package:braindance/pages/task_list.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../main.dart' show overviewLocalIndexingProvider, overviewStatsProvider;

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

  @override
  void initState() {
    super.initState();
    tabController = TabController(
      length: 2,
      vsync: this,
      animationDuration: const Duration(milliseconds: 200),
    );
    tabController.addListener(_handleTabChange);
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
                  title: textLocalize('manage'),
                  trailing: GestureDetector(
                    onTap: () async {
                      // 1. Capture the UI before changing theme
                      final boundary = themeAnimationKey.currentContext
                          ?.findRenderObject() as RenderRepaintBoundary?;
                      if (boundary != null) {
                        try {
                          final image = await boundary.toImage(
                              pixelRatio: 1.0);

                          final RenderBox? buttonBox =
                              _themeSwitchKey.currentContext?.findRenderObject()
                                  as RenderBox?;

                          if (buttonBox != null) {
                            final offset = buttonBox.localToGlobal(Offset.zero);
                            final center = offset +
                                Offset(buttonBox.size.width / 2,
                                    buttonBox.size.height / 2);

                            ref
                                .read(themeAnimationProvider.notifier)
                                .startBase(image, center);
                          }
                        } catch (e) {
                          debugPrint('Theme transition error: $e');
                        }
                      }

                      SetConfig.setNightMode(!AppConfig.isNightMode, ref);
                      WidgetsBinding.instance.addPostFrameCallback((_) {
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
                      color: isDark
                          ? BDDesign.colorMutedBlueLight
                          : BDDesign.colorMutedBlue,
                    ),
                  ),
                ),

                Consumer(
                  builder: (context, ref, _) {
                    final stats = ref.watch(overviewStatsProvider);
                    final isIndexing = ref.watch(overviewLocalIndexingProvider);
                    return RecallOverviewCard(
                      isDark: isDark,
                      textColor: textColor,
                      recentCount: stats['recentCount'] ?? 0,
                      allModelCount: stats['allModelCount'] ?? 0,
                      processingTaskCount: stats['processingTaskCount'] ?? 0,
                      ragCount: stats['ragCount'] ?? 0,
                      isLocalIndexing: isIndexing,
                      onOpenTasks: () {
                        Navigator.push(
                          context,
                          PageRouteBuilder(
                            transitionDuration: const Duration(milliseconds: 320),
                            reverseTransitionDuration: const Duration(milliseconds: 320),
                            opaque: true,
                            pageBuilder: (_, __, ___) => const TaskListPage(),
                            transitionsBuilder: (_, animation, __, child) {
                              final curved = CurvedAnimation(
                                parent: animation,
                                curve: Curves.easeInOutCubic,
                              );
                              return SlideTransition(
                                position: Tween<Offset>(
                                  begin: const Offset(0, -1),
                                  end: Offset.zero,
                                ).animate(curved),
                                child: child,
                              );
                            },
                          ),
                        );
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
          setTab1(context, ref),
          setTab3(context),
        ],
      ),
    );
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

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      height: 56,
      child: ClipRRect(
        borderRadius: BDDesign.radiusLarge,
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 24.0, sigmaY: 24.0),
          child: Container(
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
                final tabWidth = constraints.maxWidth / 2;
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
                        _buildTabItem(0, textLocalize('set_tab1'),
                            selectedColor, unselectedColor),
                        _buildTabItem(1, textLocalize('set_tab3'),
                            selectedColor, unselectedColor),
                      ],
                    ),
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTabItem(int index, String label,
      Color selectedColor, Color unselectedColor) {
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
