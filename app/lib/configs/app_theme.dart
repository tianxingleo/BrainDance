import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import 'app_config.dart';
import 'vivid_page_transitions.dart';

class AppTheme {
  static const Color darkBackground = Color(0xFF101014);
  static const Color darkSurface = Color(0xFF18181C);
  static const Color darkSurfaceElevated = Color(0xFF23232A);
  static const Color darkBorder = Color(0xFF33333D);

  static ThemeData buildLightTheme(TDThemeData baseTheme) {
    final base = baseTheme.systemThemeDataLight!;
    const scheme = ColorScheme.light(
      primary: AppConfig.primaryColor,
      secondary: AppConfig.accentColor,
      surface: Color(0xFFF7F8FA),
      onPrimary: Colors.white,
      onSecondary: Color(0xFF1F2329),
      onSurface: Color(0xFF1F2329),
    );

    return base.copyWith(
      colorScheme: scheme,
      scaffoldBackgroundColor: scheme.surface,
      canvasColor: Colors.transparent,
      cardColor: Colors.white,
      dividerColor: const Color(0xFFE5E7EB),
      appBarTheme: base.appBarTheme.copyWith(
        backgroundColor: Colors.white.withValues(alpha: 0.92),
        foregroundColor: scheme.onSurface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
      ),
      pageTransitionsTheme: const PageTransitionsTheme(
        builders: <TargetPlatform, PageTransitionsBuilder>{
          TargetPlatform.android: VividPageTransitionsBuilder(),
          TargetPlatform.iOS: VividPageTransitionsBuilder(),
          TargetPlatform.macOS: VividPageTransitionsBuilder(),
          TargetPlatform.windows: VividPageTransitionsBuilder(),
          TargetPlatform.linux: VividPageTransitionsBuilder(),
        },
      ),
      inputDecorationTheme: base.inputDecorationTheme.copyWith(
        filled: true,
        fillColor: Colors.white.withValues(alpha: 0.82),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: scheme.outline.withValues(alpha: 0.25)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: scheme.outline.withValues(alpha: 0.25)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(
            color: AppConfig.primaryColor,
            width: 1.2,
          ),
        ),
      ),
    );
  }

  static ThemeData buildDarkTheme(TDThemeData baseTheme) {
    final base = baseTheme.systemThemeDataDark!;
    const scheme = ColorScheme.dark(
      primary: Color(0xFFAEBAC7),
      secondary: Color(0xFFDDE1D2),
      surface: darkBackground,
      onPrimary: Color(0xFF101014),
      onSecondary: Color(0xFF101014),
      onSurface: Color(0xFFF5F7FA),
      outline: darkBorder,
    );

    return base.copyWith(
      colorScheme: scheme,
      scaffoldBackgroundColor: darkBackground,
      canvasColor: Colors.transparent,
      cardColor: darkSurface,
      dividerColor: darkBorder,
      appBarTheme: base.appBarTheme.copyWith(
        backgroundColor: darkSurface.withValues(alpha: 0.9),
        foregroundColor: scheme.onSurface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
      ),
      pageTransitionsTheme: const PageTransitionsTheme(
        builders: <TargetPlatform, PageTransitionsBuilder>{
          TargetPlatform.android: VividPageTransitionsBuilder(),
          TargetPlatform.iOS: VividPageTransitionsBuilder(),
          TargetPlatform.macOS: VividPageTransitionsBuilder(),
          TargetPlatform.windows: VividPageTransitionsBuilder(),
          TargetPlatform.linux: VividPageTransitionsBuilder(),
        },
      ),
      inputDecorationTheme: base.inputDecorationTheme.copyWith(
        filled: true,
        fillColor: darkSurfaceElevated.withValues(alpha: 0.96),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: darkBorder),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: darkBorder),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Color(0xFFAEBAC7), width: 1.2),
        ),
      ),
    );
  }
}

extension AppThemeContext on BuildContext {
  bool get isDarkMode => Theme.of(this).brightness == Brightness.dark;

  Color get appPageBackground =>
      isDarkMode ? AppTheme.darkBackground : TDTheme.of(this).grayColor1;

  Color get appSurfaceColor =>
      isDarkMode ? AppTheme.darkSurface : TDTheme.of(this).whiteColor1;

  Color get appSurfaceMutedColor => isDarkMode
      ? AppTheme.darkSurfaceElevated
      : TDTheme.of(this).whiteColor1.withValues(alpha: 0.78);

  Color get appBorderColor =>
      isDarkMode ? AppTheme.darkBorder : const Color(0xFFE5E7EB);

  List<BoxShadow> get appCardShadow => [
    BoxShadow(
      color: Colors.black.withValues(alpha: isDarkMode ? 0.22 : 0.06),
      blurRadius: isDarkMode ? 24 : 18,
      spreadRadius: isDarkMode ? 0 : 2,
      offset: const Offset(0, 8),
    ),
  ];
}
