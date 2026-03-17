import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/settabs/settab1.dart';
import 'package:braindance/pages/settabs/settab2.dart';
import 'package:braindance/pages/settabs/settab3.dart';
import 'package:braindance/pages/settabs/settab4.dart';
import '../widgets/bd_surfaces.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.homeRef});
  final WidgetRef homeRef;

  @override
  State<SettingsPage> createState() => _SettingsPageState(); // 创建状态
}

class _SettingsPageState extends State<SettingsPage>
    with TickerProviderStateMixin {
  late final TabController tabController;
  late final ScrollController scrollController;
  static const TextStyle tabTextStyle = TextStyle(
    fontSize: 16,
    fontFamily: AppConfig.fontFamily,
  );
  // 选择器选中项索引;
  @override
  void initState() {
    //若 tab 数量有变，需同步修改 length
    super.initState();
    tabController = TabController(
      length: 4,
      vsync: this,
      animationDuration: Duration(milliseconds: 200),
    ); // 正确初始化
    tabController.addListener(onUpdate);
    scrollController = ScrollController();
  }

  @override
  void dispose() {
    tabController.removeListener(onUpdate);
    tabController.dispose(); // 在 dispose 中释放资源
    scrollController.dispose();
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

    final TDTabBar myTabBar = TDTabBar(
      tabs: [
        TDTab(text: textLocalize('set_tab1')),
        TDTab(text: textLocalize('set_tab2')),
        TDTab(text: textLocalize('set_tab3')),
        TDTab(text: textLocalize('set_tab4')),
      ],
      controller: tabController,
      showIndicator: true,
      indicatorPadding: const EdgeInsets.all(4.0),
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
            padding: const EdgeInsets.only(bottom: 96),
            child: Column(
              children: [
                BDPageHeader(
                  title: textLocalize("settings"),
                  subtitle: '调整账户、偏好和本地设备行为，保持整套界面一致。',
                  trailing: BDStatusPill(
                    label: isDark ? 'NIGHT' : 'DAY',
                    icon: isDark
                        ? Icons.dark_mode_rounded
                        : Icons.wb_sunny_rounded,
                    color: isDark
                        ? BDDesign.colorMutedBlueLight
                        : BDDesign.colorMutedBlue,
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: BDPanelCard(
                    padding: const EdgeInsets.all(18),
                    child: Row(
                      children: [
                        Expanded(
                          child: _SettingsMetric(
                            label: '界面',
                            value: isDark ? 'Night mode' : 'Paper mode',
                          ),
                        ),
                        Expanded(
                          child: _SettingsMetric(
                            label: '语言',
                            value: AppConfig.langMap['locale'] == 'en_US'
                                ? 'English'
                                : '简体中文',
                          ),
                        ),
                        Expanded(
                          child: _SettingsMetric(
                            label: '风格',
                            value: 'Archive UI',
                            accent: textColor,
                          ),
                        ),
                      ],
                    ),
                  ),
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
                Expanded(
                  child: AnimatedSwitcher(
                    duration: BDMotion.durationNormal,
                    switchInCurve: BDMotion.curveEnter,
                    switchOutCurve: BDMotion.curveExit,
                    child: TDTabBarView(
                      key: ValueKey<int>(tabController.index),
                      controller: tabController,
                      children: [
                        setTab1(onUpdate, widget.homeRef),
                        setTab2(onUpdate, context),
                        setTab3(context),
                        setTab4(context, scrollController),
                      ],
                    ),
                  ),
                ),
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
}

class _SettingsMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color? accent;

  const _SettingsMetric({
    required this.label,
    required this.value,
    this.accent,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final titleColor = isDark
        ? Colors.white.withValues(alpha: 0.6)
        : BDDesign.colorMutedBlue;
    final valueColor =
        accent ?? (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: titleColor,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: valueColor,
          ),
        ),
      ],
    );
  }
}
