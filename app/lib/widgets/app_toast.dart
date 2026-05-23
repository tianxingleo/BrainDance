import 'dart:async';

import 'package:flutter/material.dart';

import '../configs/app_config.dart';

OverlayEntry? _currentToast;
Timer? _dismissTimer;

void showAppToast(BuildContext context, String message) {
  _dismissTimer?.cancel();
  _currentToast?.remove();
  _currentToast = null;

  final isDark = AppConfig.isNightMode;
  final bgColor = isDark ? const Color(0xE6282828) : const Color(0xE6F0F2F5);
  final textColor = isDark
      ? Colors.white.withAlpha(220)
      : const Color(0xDD1E1E20);
  final borderColor = isDark
      ? Colors.white.withAlpha(18)
      : const Color(0x22000000);

  final overlay = Overlay.of(context);
  final bottomPadding = MediaQuery.paddingOf(context).bottom;

  final entry = OverlayEntry(
    builder: (_) => IgnorePointer(
      child: Positioned(
        left: 0,
        right: 0,
        bottom: bottomPadding + 100,
        child: Center(
          child: Material(
            color: Colors.transparent,
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 32),
              padding: const EdgeInsets.symmetric(
                horizontal: 20,
                vertical: 13,
              ),
              decoration: BoxDecoration(
                color: bgColor,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: borderColor),
              ),
              child: Text(
                message,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: textColor,
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  height: 1.35,
                ),
              ),
            ),
          ),
        ),
      ),
    ),
  );

  _currentToast = entry;
  overlay.insert(entry);

  _dismissTimer = Timer(const Duration(seconds: 2), () {
    entry.remove();
    if (_currentToast == entry) _currentToast = null;
  });
}
