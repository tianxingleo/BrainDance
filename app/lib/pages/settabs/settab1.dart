import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/set_config.dart';

Widget setTab1(VoidCallback onUpdate, WidgetRef homeRef) {
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
    padding: const EdgeInsets.all(16.0),
    child: Column(
      children: [
        Container(
          decoration: BoxDecoration(
            color: TDTheme.of(homeRef.context).whiteColor1,
            borderRadius: BorderRadius.circular(TDTheme.of(homeRef.context).radiusLarge),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 10,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(TDTheme.of(homeRef.context).radiusLarge),
            child: TDCellGroup(
              cells: [
                TDCell(
                  //语言切换单元格
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
        const SizedBox(height: 24),
        TDButton(
          text: "退出登录",
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
      ],
    ),
  );
}
