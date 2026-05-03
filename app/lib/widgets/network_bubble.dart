import 'dart:async';

import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';

import '../services/network_service.dart';

class NetworkBubbleOverlay extends StatefulWidget {
  const NetworkBubbleOverlay({super.key});

  @override
  State<NetworkBubbleOverlay> createState() => _NetworkBubbleOverlayState();
}

class _NetworkBubbleOverlayState extends State<NetworkBubbleOverlay>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _fade;
  late final Animation<double> _scale;
  Timer? _dismissTimer;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      duration: const Duration(milliseconds: 250),
      vsync: this,
    );
    _fade = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
    _scale = Tween<double>(
      begin: 0.88,
      end: 1,
    ).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
  }

  @override
  void dispose() {
    _dismissTimer?.cancel();
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: networkService,
      builder: (context, child) {
        final messageKey = networkService.bubbleMessage;
        if (messageKey == null) {
          _dismissTimer?.cancel();
          if (!_ctrl.isDismissed) _ctrl.reverse();
          return const SizedBox.shrink();
        }

        _dismissTimer?.cancel();
        _ctrl.forward();
        final isReLogin = networkService.navigateToLoginAfterDismiss;
        _dismissTimer = Timer(Duration(seconds: isReLogin ? 3 : 4), () {
          if (mounted) {
            _ctrl.reverse();
            networkService.onBubbleDismissed();
          }
        });

        return Positioned.fill(
          child: AnimatedBuilder(
            animation: _ctrl,
            builder: (context, child) {
              if (_ctrl.isDismissed) return const SizedBox.shrink();

              final icon = isReLogin
                  ? Icons.login_rounded
                  : Icons.wifi_off_rounded;
              final iconColor = isReLogin
                  ? const Color(0xFFFFB74D)
                  : Colors.orange.withAlpha(200);

              return Align(
                alignment: const Alignment(0, 0.30),
                child: FadeTransition(
                  opacity: _fade,
                  child: ScaleTransition(
                    scale: _scale,
                    child: GestureDetector(
                      onTap: () {
                        _dismissTimer?.cancel();
                        _ctrl.reverse();
                        networkService.onBubbleDismissed();
                      },
                      child: Container(
                        margin: const EdgeInsets.symmetric(horizontal: 32),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 13,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xE6282828),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: Colors.white.withAlpha(18)),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(icon, color: iconColor, size: 20),
                            const SizedBox(width: 10),
                            Flexible(
                              child: Text(
                                textLocalize(messageKey),
                                style: TextStyle(
                                  color: Colors.white.withAlpha(220),
                                  fontSize: 13,
                                  fontWeight: FontWeight.w500,
                                  height: 1.35,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              );
            },
          ),
        );
      },
    );
  }
}
