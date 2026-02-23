import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../configs/app_config.dart';
import '../extra_func/dynamic_background.dart';

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
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
      body: DynamicGradientBackground(
        child: Center(
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
      ),
    );
  }
}
