import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/pages/settabs/settab1.dart';
import 'package:braindance/pages/settabs/settab2.dart';
import 'package:braindance/pages/settabs/settab3.dart';
import 'package:braindance/pages/settabs/settab4.dart';

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
  late final AnimationController _bgAnimController;
  late final Animation<Alignment> _topAlignment;
  late final Animation<Alignment> _bottomAlignment;
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

    _bgAnimController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 40),
    )..repeat(reverse: true);

    _topAlignment = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topLeft, end: Alignment.topRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topRight, end: Alignment.bottomRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomRight, end: Alignment.bottomLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomLeft, end: Alignment.topLeft),
        weight: 1,
      ),
    ]).animate(CurvedAnimation(parent: _bgAnimController, curve: Curves.easeInOut));

    _bottomAlignment = TweenSequence<Alignment>([
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomRight, end: Alignment.bottomLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.bottomLeft, end: Alignment.topLeft),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topLeft, end: Alignment.topRight),
        weight: 1,
      ),
      TweenSequenceItem(
        tween: AlignmentTween(begin: Alignment.topRight, end: Alignment.bottomRight),
        weight: 1,
      ),
    ]).animate(CurvedAnimation(parent: _bgAnimController, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _bgAnimController.dispose();
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
      body: Stack(
        children: [
          // 动态渐变背景
          Positioned.fill(
            child: AnimatedBuilder(
              animation: _bgAnimController,
              builder: (context, child) {
                return Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: _topAlignment.value,
                      end: _bottomAlignment.value,
                      colors: [
                        TDTheme.of(context).brandColor4.withValues(alpha: 0.15),
                        AppConfig.primaryColor.withValues(alpha: 0.05),
                        TDTheme.of(context).grayColor1,
                        TDTheme.of(context).brandColor4.withValues(alpha: 0.05),
                      ],
                      stops: const [0.0, 0.4, 0.8, 1.0],
                    ),
                  ),
                );
              },
            ),
          ),
          SafeArea(
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
        ],
      ),
    );
  }

  void onUpdate() {
    if (mounted) {
      setState(() {});
    }
  }
}
