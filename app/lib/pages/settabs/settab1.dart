import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

Widget setTab1(BuildContext context, WidgetRef ref) {
  final isDark = Theme.of(context).brightness == Brightness.dark;
  final hintColor = isDark
      ? Colors.white.withValues(alpha: 0.62)
      : BDDesign.colorMutedBlue;

  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        BDPanelCard(
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
                },
              ),
              const SizedBox(width: 10),
              _LanguageToggleChip(
                label: textLocalize('set_lang_en'),
                selected: AppConfig.langMap['locale'] == 'en_US',
                onTap: () {
                  SetConfig.setLanguage('en_US', ref);
                  SetConfig.saveMsgToFile();
                },
              ),
            ],
          ),
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
      ],
    ),
  );
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
                  : (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack),
            ),
          ),
        ),
      ),
    );
  }
}
