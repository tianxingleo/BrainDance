import 'package:flutter/material.dart';
import 'dart:io'; // For Platform.localeName
import '../extra_func/language.dart';

//App基础设置
class AppConfig {
  static const fontFamily = 'HarmonyOS_Sans';
  static const appName = 'BrainDance';
  static const version = '1.0.0';
  static const publishDate = '2026-01-01';
  static late Map<String, String> langMap;
  static bool isNightMode = false;
  static bool hasReadRecordTip = false;

  static final Color primaryColor = Color.fromRGBO(113, 131, 143, 1);
  static final Color accentColor = Color.fromRGBO(232, 234, 220, 1);
  static void initializeAppConfig() {
    try {
      AppConfig.langMap = Localize.getLangMap(Platform.localeName);
    } catch (e) {
      AppConfig.langMap = Localize.getLangMap("en_US");
    }
  }
}

String textLocalize(String id) {
  return AppConfig.langMap[id] ?? id;
}
