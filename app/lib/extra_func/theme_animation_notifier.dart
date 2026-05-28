import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/legacy.dart';

enum ThemeTransitionMode {
  expandHole,
  shrinkClip,
}

class ThemeAnimationState {
  final ui.Image? screenshot;
  final Offset center;
  final bool isAnimating;
  final bool isReversing;
  final double startFraction;
  final ThemeTransitionMode mode;

  ThemeAnimationState({
    this.screenshot,
    this.center = Offset.zero,
    this.isAnimating = false,
    this.isReversing = false,
    this.startFraction = 0.0,
    this.mode = ThemeTransitionMode.expandHole,
  });

  ThemeAnimationState copyWith({
    ui.Image? screenshot,
    Offset? center,
    bool? isAnimating,
    bool? isReversing,
    double? startFraction,
    ThemeTransitionMode? mode,
  }) {
    return ThemeAnimationState(
      screenshot: screenshot ?? this.screenshot,
      center: center ?? this.center,
      isAnimating: isAnimating ?? this.isAnimating,
      isReversing: isReversing ?? this.isReversing,
      startFraction: startFraction ?? this.startFraction,
      mode: mode ?? this.mode,
    );
  }
}

class ThemeAnimationNotifier extends StateNotifier<ThemeAnimationState> {
  ThemeAnimationNotifier() : super(ThemeAnimationState());

  void start(ui.Image screenshot, Offset center, ThemeTransitionMode mode) {
    state = ThemeAnimationState(
      screenshot: screenshot,
      center: center,
      isAnimating: true,
      isReversing: false,
      mode: mode,
    );
  }

  void toggleDirection(double currentFraction) {
    if (!state.isAnimating) return;
    state = state.copyWith(
      isReversing: !state.isReversing,
      startFraction: currentFraction,
    );
  }

  void end() {
    final oldScreenshot = state.screenshot;
    state = ThemeAnimationState();
    oldScreenshot?.dispose();
  }
}

final themeAnimationProvider =
    StateNotifierProvider<ThemeAnimationNotifier, ThemeAnimationState>((ref) {
  return ThemeAnimationNotifier();
});

final GlobalKey themeAnimationKey = GlobalKey();

final ValueNotifier<double> themeAnimationFraction = ValueNotifier(0.0);
