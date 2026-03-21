import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/pages/recall/overview_card.dart';
import 'package:braindance/pages/settabs/settab1.dart';
import 'package:braindance/pages/settabs/settab3.dart';
import 'package:braindance/pages/task_list.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../main.dart' show overviewLocalIndexingProvider, overviewStatsProvider;

class SettingsPage extends ConsumerStatefulWidget {
  const SettingsPage({super.key});

  @override
  ConsumerState<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends ConsumerState<SettingsPage>
    with TickerProviderStateMixin {
  late final TabController tabController;
  final Set<int> _builtTabs = {0};
  int _currentTabIndex = 0;

  static const TextStyle tabTextStyle = TextStyle(
    fontSize: 16,
    fontFamily: AppConfig.fontFamily,
  );

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
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;

    final myTabBar = TDTabBar(
      tabs: [
        TDTab(text: textLocalize('set_tab1')),
        TDTab(text: textLocalize('set_tab3')),
      ],
      controller: tabController,
      showIndicator: true,
      indicatorPadding: const EdgeInsets.all(4),
      indicatorWidth: 24,
      indicatorHeight: 3,
      indicatorColor: BDDesign.colorMutedBlue,
      labelStyle: tabTextStyle.copyWith(
        fontWeight: FontWeight.w600,
        color: BDDesign.colorMutedBlue,
      ),
      unselectedLabelStyle: tabTextStyle.copyWith(
        fontWeight: FontWeight.w400,
        color: hintColor.withValues(alpha: 0.78),
      ),
    );

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.only(bottom: 24),
            child: Column(
              children: [
                BDPageHeader(
                  title: textLocalize('manage'),
                  trailing: GestureDetector(
                    onTap: () {
                      SetConfig.setNightMode(!AppConfig.isNightMode, ref);
                      SetConfig.saveMsgToFile();
                      onUpdate();
                    },
                    child: BDStatusPill(
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
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: BDPanelCard(
                    padding: const EdgeInsets.all(18),
                    child: Row(
                      children: [
                        Text(
                          textLocalize('set_label_lang'),
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: hintColor,
                          ),
                        ),
                        const Spacer(),
                        _LanguageToggleChip(
                          label: textLocalize('set_lang_zh'),
                          selected: AppConfig.langMap['locale'] == 'zh_CN',
                          onTap: () {
                            SetConfig.setLanguage('zh_CN', ref);
                            SetConfig.saveMsgToFile();
                            onUpdate();
                          },
                        ),
                        const SizedBox(width: 10),
                        _LanguageToggleChip(
                          label: textLocalize('set_lang_en'),
                          selected: AppConfig.langMap['locale'] == 'en_US',
                          onTap: () {
                            SetConfig.setLanguage('en_US', ref);
                            SetConfig.saveMsgToFile();
                            onUpdate();
                          },
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 14),
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
                          MaterialPageRoute(
                            builder: (_) => const TaskListPage(),
                          ),
                        );
                      },
                    );
                  },
                ),
                const SizedBox(height: 14),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: BDPanelCard(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 6,
                    ),
                    child: myTabBar,
                  ),
                ),
                const SizedBox(height: 10),
                Expanded(child: _buildTabContent(context, ref)),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void onUpdate() {
    if (mounted) {
      setState(() {});
    }
  }

  void _handleTabChange() {
    final nextIndex = tabController.index;
    if (_currentTabIndex == nextIndex) {
      return;
    }
    setState(() {
      _currentTabIndex = nextIndex;
      _builtTabs.add(nextIndex);
    });
  }

  Widget _buildTabContent(BuildContext context, WidgetRef ref) {
    return IndexedStack(
      index: _currentTabIndex,
      children: List<Widget>.generate(tabController.length, (index) {
        if (!_builtTabs.contains(index)) {
          return const SizedBox.shrink();
        }
        return KeyedSubtree(
          key: ValueKey<int>(index),
          child: switch (index) {
            0 => setTab1(ref),
            1 => setTab3(context),
            _ => const SizedBox.shrink(),
          },
        );
      }),
    );
  }
}

class _LanguageToggleChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _LanguageToggleChip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final selectedColor = isDark
        ? BDDesign.colorMutedBlueLight
        : BDDesign.colorMutedBlue;
    final borderColor = selected
        ? selectedColor.withValues(alpha: 0.22)
        : (isDark
              ? Colors.white.withValues(alpha: 0.08)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.10));

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(999),
        child: AnimatedContainer(
          duration: BDMotion.durationNormal,
          curve: BDMotion.curveFluid,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: selected
                ? selectedColor.withValues(alpha: 0.12)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: borderColor),
          ),
          child: Text(
            label,
            style: TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: selected
                  ? selectedColor
                  : (isDark
                        ? BDDesign.colorPaperWhite
                        : BDDesign.colorInkBlack),
            ),
          ),
        ),
      ),
    );
  }
}
