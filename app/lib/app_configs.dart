import 'package:flutter/material.dart';
import 'dart:io';// For Platform.localeName
import 'extra_func/language.dart';
import 'extra_func/dir_and_file.dart';
import 'extra_func/theme_provider.dart';
import 'extra_func_v2/file_stream.dart';
import 'main.dart';
//App基础设置
class AppConfig {
  static const appName = 'BrainDance';
  static const version = '1.0.0';
  static const publishDate = '2026-01-01';
  static late Map<String, String> langMap;
  static bool isNightMode = false;

  static Color primaryColor = Color.fromRGBO(113, 131, 143, 1);
  static Color accentColor = Color.fromRGBO(232, 234, 220, 1);
}
/*
Config规范：
load...FromFile - Future<bool> 返回是否成功
save...ToFile - Future<void>
loadVarFrom... - void
load... - Future<String> / Future<List<String>>
save... - Future<void>
 */
class GenConfig {
  static const imagePathsFileName = "genImagePaths.txt";
  static const videoPathsFileName = "genVideoPaths.txt";
  static const textFileName = "genText.txt";

  static Future<List<String>> loadImagePathsFile() async {
    return await FileStream.appLoad(DirFinder.cacheDir(), imagePathsFileName);
  }
  static Future<String> loadTextFile() async {
    final List<String> result = await FileStream.appLoad(DirFinder.cacheDir(), textFileName);
    return result.join();
  }
  static Future<List<String>> loadVideoPathsFile() async {
    return await FileStream.appLoad(DirFinder.cacheDir(), videoPathsFileName);
  }
  static Future<void> saveImagePathsFile(List<String> paths) async {
    await FileStream.appSave(DirFinder.cacheDir(), imagePathsFileName, paths);
  }
  static Future<void> saveTextFile(String text) async {
    await FileStream.appSave(DirFinder.cacheDir(), textFileName, [text]);
  }
  static Future<void> saveVideoPathsFile(List<String> paths) async {
    await FileStream.appSave(DirFinder.cacheDir(), videoPathsFileName, paths);
  }
  static Future<void> deleteImagePathsFile() async {
    await FileStream.appDel(DirFinder.cacheDir(), imagePathsFileName);
  }
  static Future<void> deleteTextFile() async {
    await FileStream.appDel(DirFinder.cacheDir(), textFileName);
  }
  static Future<void> deleteVideoPathsFile() async {
    await FileStream.appDel(DirFinder.cacheDir(), videoPathsFileName);
  }
}
class SetConfig {
  static const settingsFileName = "settings.txt";
  static const settingsCount = 20;
  static late List<String> settingsMsg;//设置信息暂存
  static Future<bool> loadMsgFromFile() async {//返回是否成功
    final bool suc;
    settingsMsg = await FileStream.appLoad(DirFinder.supportDir(), settingsFileName);
    suc = settingsMsg.isNotEmpty;
    settingsMsg = [...settingsMsg, ...List.filled(settingsCount - settingsMsg.length, "")];
    return suc;
  }
  static Future<void> saveMsgToFile() async {
    await FileStream.appSave(DirFinder.supportDir(), settingsFileName, settingsMsg);
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
  SetConfig.settingsMsg[0] = AppConfig.langMap['locale'] ?? 'en_US';
  onLanguageChanged?.call();
}
void setNightMode(bool isNight) {
  AppConfig.isNightMode = isNight;
  SetConfig.settingsMsg[1] = isNight.toString();
  ThemeModeProvider.setThemeMode(isNight ? ThemeMode.dark : ThemeMode.light);
}
void initializeAppConfig() {
  AppConfig.langMap = Localize.getLangMap(Platform.localeName);
}