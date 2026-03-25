import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../webgl_viewer.dart';
import 'search_mode.dart';

class RecallEmptyState extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkBorder;

  const RecallEmptyState({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkBorder,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final iconColor = isDark
        ? const Color(0xFFEEEEEE)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32),
          border: Border.all(
            color: isDark ? darkBorder : theme.whiteColor1,
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withAlpha(20),
              blurRadius: 20,
              spreadRadius: 5,
            ),
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
                color: iconColor,
              ),
            ),
            const SizedBox(height: 24),
            TDText(
              textLocalize('home_page'),
              font: theme.fontTitleLarge,
              textColor: textColor,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              textLocalize('recall_empty_title'),
              font: theme.fontBodyMedium,
              textColor: hintTextColor,
            ),
            const SizedBox(height: 40),
            TDButton(
              text: textLocalize('recall_open_demo'),
              iconWidget: const Icon(
                TDIcons.view_module,
                color: Colors.white,
                size: 20,
              ),
              type: TDButtonType.fill,
              theme: TDButtonTheme.primary,
              shape: TDButtonShape.round,
              size: TDButtonSize.large,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => WebGLViewerPage(
                      sceneId: textLocalize('recall_demo_title'),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

class RecallSearchEmptyState extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkBorder;
  final RecallSearchMode searchMode;
  final String Function(RecallSearchMode mode) searchModeTitleBuilder;

  const RecallSearchEmptyState({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkBorder,
    required this.searchMode,
    required this.searchModeTitleBuilder,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
        decoration: BoxDecoration(
          color: isDark ? darkCard : theme.whiteColor1.withAlpha(200),
          borderRadius: BorderRadius.circular(32),
          border: Border.all(
            color: isDark ? darkBorder : theme.whiteColor1,
            width: 1,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.travel_explore_rounded,
              size: 56,
              color: isDark
                  ? Colors.white.withValues(alpha: 0.8)
                  : BDDesign.colorMutedBlue,
            ),
            const SizedBox(height: 18),
            TDText(
              searchModeTitleBuilder(searchMode),
              font: theme.fontTitleLarge,
              textColor: textColor,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              switch (searchMode) {
                RecallSearchMode.local => textLocalize('recall_local_empty'),
                RecallSearchMode.cloud => textLocalize('recall_cloud_empty'),
                RecallSearchMode.localAi => textLocalize(
                  'recall_local_ai_empty',
                ),
                RecallSearchMode.agent => '输入空间问题后点击搜索，Agent 将为你定位场景',
              },
              font: theme.fontBodyMedium,
              textColor: hintTextColor,
            ),
          ],
        ),
      ),
    );
  }
}
