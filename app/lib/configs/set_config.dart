import 'package:braindance/extra_func/locale_provider.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/material.dart';
import '../extra_func/language.dart';
import '../extra_func/theme_provider.dart';
import '../extra_func_v2/file_stream.dart';
import 'app_config.dart';

class SetConfig {
  static const settingsFileName = "settings.txt";
  static const settingsCount = 20;
  static late List<String> settingsMsg; //设置信息暂存

  static void setLanguage(String localeCode, WidgetRef ref) {
    //加载 - 设置，保存需要调用saveMsgToSettings
    AppConfig.langMap = Localize.getLangMap(localeCode);
    settingsMsg[0] = AppConfig.langMap['locale'] ?? 'en_US';
    ref.read(localeProvider.notifier).setLocale(settingsMsg[0]);
  }

  static void setNightMode(bool isNight, WidgetRef ref) {
    AppConfig.isNightMode = isNight;
    settingsMsg[1] = isNight.toString();
    ref
        .read(themeModeProvider.notifier)
        .setThemeMode(isNight ? ThemeMode.dark : ThemeMode.light);
  }

  static void loadSettingsFromMsg(WidgetRef ref) {
    //根据设置信息应用设置，涉及设置信息的意义
    for (int i = 0; i < settingsMsg.length; i++) {
      switch (i) {
        case 0:
          setLanguage(settingsMsg[0], ref);
          break;
        case 1:
          // '' = system (fallback to system default but freeze it), 'true' = dark, 'false' = light
          final val = settingsMsg[1];
          if (val.isEmpty) {
            final isDark =
                WidgetsBinding.instance.platformDispatcher.platformBrightness ==
                Brightness.dark;
            AppConfig.isNightMode = isDark;
            ref
                .read(themeModeProvider.notifier)
                .setThemeMode(isDark ? ThemeMode.dark : ThemeMode.light);
          } else {
            setNightMode(val == 'true', ref);
          }
          break;
      }
    }
  }

  static Future<bool> loadMsgFromFile() async {
    //返回是否成功
    final bool suc;
    settingsMsg = await FileStream.appLoad(AppDir.support, settingsFileName);
    suc = settingsMsg.isNotEmpty;
    settingsMsg = [
      ...settingsMsg,
      ...List.filled(settingsCount - settingsMsg.length, ""),
    ];
    return suc;
  }

  static Future<void> saveMsgToFile() async {
    await FileStream.appSave(AppDir.support, settingsFileName, settingsMsg);
  }
}
