import 'package:flutter/material.dart';
import '../main.dart';

class ThemeModeProvider extends ChangeNotifier {
  static ThemeMode _themeMode = ThemeMode.light;

  static ThemeMode get themeMode => _themeMode;

  static void setThemeMode(ThemeMode mode) {
    _themeMode = mode;
    onThemeChanged?.call();
  }
}
