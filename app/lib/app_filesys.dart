import 'package:flutter/material.dart';
import 'package:path/path.dart' as path_joiner;
import 'dart:io';// For Platform.localeName
import 'extra_func/language.dart';
import 'extra_func/dir_and_file.dart';
import 'extra_func/theme_provider.dart';
import 'main.dart';
//App基础设置
class AppConfig {
  static const settingsFileName = "settings.txt";
  static const appName = 'BrainDance';
  static const version = '1.0.0';
  static const publishDate = '2026-01-01';
  static const settingsCount = 20;
  static Color primaryColor = Color.fromRGBO(113, 131, 143, 1);
  static Color accentColor = Color.fromRGBO(232, 234, 220, 1);
  static late Map<String, String> langMap;
  static late List<String> settingsMsg;//设置信息暂存
  static bool isNightMode = false;

  static Future<bool> loadMsgFromSettingsFile() async {//返回是否成功
    //加载文件数据
    final dir = await DirFinder.supportDir();
    final path = path_joiner.join(dir, settingsFileName);
    if (await FileSystem.checkFileExists(path)) {//检测设置文件是否存在
      //从设置文件中读取语言代码
      settingsMsg = await FileSystem.readFile(path);
      settingsMsg = [...settingsMsg, ...List.filled(settingsCount - settingsMsg.length, "")];
      return true;
    }
    settingsMsg = List.filled(settingsCount, "");
    return false;
  }
  static Future<void> saveMsgToSettings() async {
    final dir = await DirFinder.supportDir();
    final path = path_joiner.join(dir, settingsFileName);
    await DirSystem.ensureDir(dir);
    await FileSystem.writeFile(path, settingsMsg);
  }
  static void loadSettingsFromMsg() {//根据设置信息应用设置，涉及设置信息的意义
    for (int i = 0; i < settingsMsg.length; i++) {
      switch (i){
        case 0:
          setLanguage(settingsMsg[0]);
          break;
        case 1:
          setNightMode(settingsMsg[1] == 'true');
          break;
      }
    }
  }
}
String textLocalize(String id) {
  return AppConfig.langMap[id] ?? id;
}
void setLanguage(String localeCode) {//加载 - 设置，保存需要调用saveMsgToSettings
  AppConfig.langMap = Localize.getLangMap(localeCode);
  AppConfig.settingsMsg[0] = AppConfig.langMap['locale'] ?? 'en_US';
  onLanguageChanged?.call();
}
void setNightMode(bool isNight) {
  AppConfig.isNightMode = isNight;
  AppConfig.settingsMsg[1] = isNight.toString();
  ThemeModeProvider.setThemeMode(isNight ? ThemeMode.dark : ThemeMode.light);
}
void initializeAppConfig() {
  AppConfig.langMap = Localize.getLangMap(Platform.localeName);
}