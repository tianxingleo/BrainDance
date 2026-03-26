import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/legacy.dart';

class ThemeAnimationState {
  final ui.Image? screenshot;
  final Offset center;
  final bool isAnimating;

  ThemeAnimationState({
    this.screenshot,
    this.center = Offset.zero,
    this.isAnimating = false,
  });

  ThemeAnimationState copyWith({
    ui.Image? screenshot,
    Offset? center,
    bool? isAnimating,
  }) {
    return ThemeAnimationState(
      screenshot: screenshot ?? this.screenshot,
      center: center ?? this.center,
      isAnimating: isAnimating ?? this.isAnimating,
    );
  }
}

class ThemeAnimationNotifier extends StateNotifier<ThemeAnimationState> {
  ThemeAnimationNotifier() : super(ThemeAnimationState());

  void startBase(ui.Image screenshot, Offset center) {
    state = ThemeAnimationState(
      screenshot: screenshot,
      center: center,
      isAnimating: true,
    );
  }

  void end() {
    state = state.copyWith(isAnimating: false, screenshot: null);
  }
}

final themeAnimationProvider =
    StateNotifierProvider<ThemeAnimationNotifier, ThemeAnimationState>((ref) {
  return ThemeAnimationNotifier();
});

// Key used to capture the RepaintBoundary
final GlobalKey themeAnimationKey = GlobalKey();
