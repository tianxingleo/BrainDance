import 'package:flutter/material.dart';
import 'package:flutter_riverpod/legacy.dart';

import '../configs/app_config.dart';

class ThemeModeNotifier extends StateNotifier<ThemeMode> {
  ThemeModeNotifier()
    : super(AppConfig.isNightMode ? ThemeMode.dark : ThemeMode.light);
  ThemeMode get themeMode {
    return state;
  }

  void setThemeMode(ThemeMode mode) {
    state = mode;
  }
}

final themeModeProvider = StateNotifierProvider<ThemeModeNotifier, ThemeMode>(
  (ref) => ThemeModeNotifier(),
);
