import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/pages/settabs/settab1.dart';
import 'package:braindance/pages/settabs/settab2.dart';
import 'package:braindance/pages/settabs/settab3.dart';
import 'package:braindance/pages/settabs/settab4.dart';
import '../extra_func/dynamic_background.dart';

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
    scrollController = ScrollController();
  }

  @override
  void dispose() {
    tabController.dispose(); // 在 dispose 中释放资源
    scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
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
      indicatorColor: AppConfig.primaryColor,
      labelStyle: tabTextStyle.copyWith(
        fontWeight: FontWeight.w600,
        color: AppConfig.primaryColor,
      ),
      unselectedLabelStyle: tabTextStyle.copyWith(
        fontWeight: FontWeight.w400,
        color: TDTheme.of(context).fontGyColor3,
      ),
    );
    return Scaffold(
      backgroundColor: TDTheme.of(context).grayColor1,
      appBar: AppBar(
        title: TDText(
          textLocalize("settings"),
          font: TDTheme.of(context).fontTitleLarge,
          fontWeight: FontWeight.w600,
          textColor: TDTheme.of(context).fontGyColor1,
        ),
        backgroundColor: TDTheme.of(context).whiteColor1.withValues(alpha: 0.95),
        elevation: 0,
        centerTitle: true,
      ),
      extendBodyBehindAppBar: true,
      body: DynamicGradientBackground(
        child: SafeArea(
          child: Column(
            children: [
              Container(
                color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.6),
                child: myTabBar,
              ),
              Expanded(
                child: TDTabBarView(
                  controller: tabController,
                  children: [
                    setTab1(onUpdate, widget.homeRef),
                    setTab2(onUpdate, context),
                    setTab3(context),
                    setTab4(scrollController),
                  ],
                ),
              ),
            ],
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
