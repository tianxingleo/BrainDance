import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/widgets/bd_surfaces.dart';

Widget setTab1(VoidCallback onUpdate, WidgetRef homeRef) {
  final context = homeRef.context;

  void onChangeLanguage() {
    if (AppConfig.langMap['locale'] == 'en_US') {
      SetConfig.setLanguage('zh_CN', homeRef);
    } else {
      SetConfig.setLanguage('en_US', homeRef);
    }
    SetConfig.saveMsgToFile();
    onUpdate();
  }

  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: ListView(
      children: [
        Text(
          textLocalize('set_tab1_title'),
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: Theme.of(context).brightness == Brightness.dark
                ? BDDesign.colorPaperWhite
                : BDDesign.colorInkBlack,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          textLocalize('set_tab1_desc'),
          style: TextStyle(
            fontSize: 13,
            height: 1.45,
            color: Theme.of(context).brightness == Brightness.dark
                ? Colors.white.withValues(alpha: 0.62)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 14),
        BDPanelCard(
          child: ClipRRect(
            borderRadius: BDDesign.radiusLarge,
            child: TDCellGroup(
              cells: [
                TDCell(
                  arrow: false,
                  title: textLocalize('set_lang'),
                  note: textLocalize('lang'),
                  onClick: (cell) {
                    onChangeLanguage();
                  },
                ),
                TDCell(
                  arrow: false,
                  title: textLocalize('set_night'),
                  rightIconWidget: TDSwitch(
                    isOn: AppConfig.isNightMode,
                    onChanged: (cell) {
                      SetConfig.setNightMode(!AppConfig.isNightMode, homeRef);
                      SetConfig.saveMsgToFile();
                      return false;
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        BDPanelCard(
          padding: const EdgeInsets.all(18),
          child: Row(
            children: [
              Expanded(
                child: _InfoTile(
                  label: 'Locale',
                  value: AppConfig.langMap['locale'] ?? 'zh_CN',
                ),
              ),
              Expanded(
                child: _InfoTile(
                  label: 'Theme',
                  value: AppConfig.isNightMode ? 'Dark' : 'Light',
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        if (!SupabaseConfig.isAdminMode)
          TDButton(
            text: textLocalize("set_logout"),
            type: TDButtonType.outline,
            theme: TDButtonTheme.danger,
            isBlock: true,
            shape: TDButtonShape.round,
            onTap: () async {
              await Supabase.instance.client.auth.signOut();
              if (homeRef.context.mounted) {
                Navigator.of(homeRef.context).pushReplacementNamed('/login');
              }
            },
          ),
        if (SupabaseConfig.isAdminMode)
          TDButton(
            text: textLocalize("set_admin_enabled"),
            type: TDButtonType.outline,
            theme: TDButtonTheme.primary,
            isBlock: true,
            shape: TDButtonShape.round,
            onTap: () {},
          ),
      ],
    ),
  );
}

class _InfoTile extends StatelessWidget {
  final String label;
  final String value;

  const _InfoTile({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isDark
                ? Colors.white.withValues(alpha: 0.58)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
          ),
        ),
      ],
    );
  }
}
