import 'dart:ui' as ui;

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

Widget setTab1(BuildContext context, WidgetRef ref) {
  final isDark = Theme.of(context).brightness == Brightness.dark;

  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        BDPanelCard(
          padding: const EdgeInsets.all(18),
          child: _LanguageRow(isDark: isDark, ref: ref),
        ),
        if (SupabaseConfig.isAdminMode) ...[
          const SizedBox(height: 12),
          TDButton(
            text: textLocalize('set_admin_enabled'),
            type: TDButtonType.outline,
            theme: TDButtonTheme.primary,
            isBlock: true,
            shape: TDButtonShape.round,
            onTap: () {},
          ),
        ],
        const SizedBox(height: 12),
        BDPanelCard(
          padding: EdgeInsets.zero,
          child: _ClearCacheRow(context: context),
        ),
      ],
    ),
  );
}

class _LanguageRow extends StatelessWidget {
  final bool isDark;
  final WidgetRef ref;
  const _LanguageRow({required this.isDark, required this.ref});

  @override
  Widget build(BuildContext context) {
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final actionColor = isDark ? BDDesign.colorMutedBlueLight : BDDesign.colorMutedBlue;

    return Row(
      children: [
        Text(
          textLocalize('set_label_lang'),
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: textColor,
          ),
        ),
        const Spacer(),
        SizedBox(
          width: 160,
          child: _LanguageTabSwitch(ref: ref),
        ),
      ],
    );
  }
}

class _LanguageTabSwitch extends StatefulWidget {
  final WidgetRef ref;
  const _LanguageTabSwitch({required this.ref});

  @override
  State<_LanguageTabSwitch> createState() => _LanguageTabSwitchState();
}

class _LanguageTabSwitchState extends State<_LanguageTabSwitch>
    with SingleTickerProviderStateMixin {
  late final TabController _controller;

  @override
  void initState() {
    super.initState();
    final initialIndex = AppConfig.langMap['locale'] == 'en_US' ? 1 : 0;
    _controller = TabController(
      length: 2,
      vsync: this,
      initialIndex: initialIndex,
      animationDuration: const Duration(milliseconds: 200),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _switchTo(int index) {
    if (_controller.index == index) return;
    _controller.animateTo(index);
    final locale = index == 1 ? 'en_US' : 'zh_CN';
    SetConfig.setLanguage(locale, widget.ref);
    SetConfig.saveMsgToFile();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final navBackground = isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.55)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.52);
    final navBorder = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
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
      height: 40,
      child: ClipRRect(
        borderRadius: BDDesign.radiusLarge,
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 24.0, sigmaY: 24.0),
          child: Container(
            padding: const EdgeInsets.all(3.0),
            decoration: BoxDecoration(
              color: navBackground,
              borderRadius: BDDesign.radiusLarge,
              border: Border.all(color: navBorder, width: 1.0),
              boxShadow: [
                BoxShadow(
                  color: navShadow,
                  blurRadius: 16,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final tabWidth = constraints.maxWidth / 2;
                return Stack(
                  children: [
                    AnimatedBuilder(
                      animation: _controller.animation!,
                      builder: (context, child) {
                        final double offset =
                            _controller.animation!.value * tabWidth;
                        return Positioned(
                          left: offset,
                          width: tabWidth,
                          top: 0,
                          bottom: 0,
                          child: Container(
                            decoration: BoxDecoration(
                              color: selectedBackground,
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        );
                      },
                    ),
                    Row(
                      children: [
                        _buildTabItem(
                          0,
                          textLocalize('set_lang_zh'),
                          selectedColor,
                          unselectedColor,
                        ),
                        _buildTabItem(
                          1,
                          textLocalize('set_lang_en'),
                          selectedColor,
                          unselectedColor,
                        ),
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

  Widget _buildTabItem(
    int index,
    String label,
    Color selectedColor,
    Color unselectedColor,
  ) {
    return Expanded(
      child: GestureDetector(
        onTap: () => _switchTo(index),
        behavior: HitTestBehavior.opaque,
        child: Center(
          child: AnimatedBuilder(
            animation: _controller.animation!,
            builder: (ctx, child) {
              final selected =
                  (_controller.animation!.value - index).abs() < 0.5;
              return Text(
                label,
                style: TextStyle(
                  color: selected ? selectedColor : unselectedColor,
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}

class _ClearCacheRow extends StatelessWidget {
  final BuildContext context;
  const _ClearCacheRow({required this.context});

  @override
  Widget build(BuildContext ctx) {
    final isDark = Theme.of(ctx).brightness == Brightness.dark;
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final actionColor = isDark ? BDDesign.colorMutedBlueLight : BDDesign.colorMutedBlue;
    const radius = BorderRadius.all(Radius.circular(28));

    return Material(
      color: Colors.transparent,
      clipBehavior: Clip.antiAlias,
      borderRadius: radius,
      child: InkWell(
        onTap: () async {
          final messenger = ScaffoldMessenger.of(context);
          messenger.hideCurrentSnackBar();
          messenger.showSnackBar(
            SnackBar(content: Text(textLocalize('tip_cache'))),
          );
          await DirSystem.deleteDir(await DirFinder.cacheDir());
        },
        borderRadius: radius,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  textLocalize('set_cache'),
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: textColor),
                ),
              ),
              Icon(Icons.chevron_right_rounded, color: actionColor, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}
