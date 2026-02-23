import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../configs/app_config.dart';

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> with TickerProviderStateMixin {
  late final AnimationController _bgAnimController;
  late final Animation<Alignment> _topAlignment;
  late final Animation<Alignment> _bottomAlignment;

  @override
  void initState() {
    super.initState();
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
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: TDTheme.of(context).grayColor1,
      appBar: AppBar(
        backgroundColor: TDTheme.of(context).whiteColor1.withValues(alpha: 0.95),
        title: Container(
          alignment: Alignment.centerLeft,
          child: TDText(
            textLocalize("home_page"),
            font: TDTheme.of(context).fontHeadlineSmall,
            fontWeight: FontWeight.w600,
            textColor: TDTheme.of(context).fontGyColor1,
          ),
        ),
        toolbarHeight: 60,
        elevation: 0,
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
                        TDTheme.of(context).brandColor4.withValues(alpha: 0.2),
                        AppConfig.primaryColor.withValues(alpha: 0.1),
                        TDTheme.of(context).grayColor1,
                        AppConfig.primaryColor.withValues(alpha: 0.05),
                      ],
                      stops: const [0.0, 0.4, 0.8, 1.0],
                    ),
                  ),
                );
              },
            ),
          ),
          Center(
            child: Container(
              width: MediaQuery.of(context).size.width * 0.85,
              padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
              decoration: BoxDecoration(
                color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.8),
                borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
                border: Border.all(
                  color: TDTheme.of(context).whiteColor1,
                  width: 1,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 20,
                    spreadRadius: 5,
                  )
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                    TDImage(
                      assetUrl: 'assets/sprites/empty_state.png',
                      width: 120,
                      height: 120,
                      errorWidget: Icon(
                        TDIcons.time_filled,
                        size: 80,
                        color: TDTheme.of(context).brandColor4,
                      ),
                    ),
                    const SizedBox(height: 24),
                    TDText(
                      textLocalize("home_page"),
                      font: TDTheme.of(context).fontTitleLarge,
                      textColor: TDTheme.of(context).fontGyColor1,
                      fontWeight: FontWeight.w600,
                    ),
                    const SizedBox(height: 8),
                    TDText(
                      "暂无回忆，去记录一些美好瞬间吧",
                      font: TDTheme.of(context).fontBodyMedium,
                      textColor: TDTheme.of(context).fontGyColor3,
                    ),
                    const SizedBox(height: 40),
                    TDButton(
                      text: "开始记录",
                      iconWidget: const Icon(TDIcons.camera, color: Colors.white, size: 20),
                      type: TDButtonType.fill,
                      theme: TDButtonTheme.primary,
                      shape: TDButtonShape.round,
                      size: TDButtonSize.large,
                      onTap: () {
                        // 预留跳转到记录页面的逻辑
                      },
                    ),
                  ],
                ),
              ),
          ),
        ],
      ),
    );
  }
}
