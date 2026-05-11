import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../configs/supabase_config.dart';

class NetworkService extends ChangeNotifier {
  static final NetworkService _instance = NetworkService._internal();
  factory NetworkService() => _instance;
  NetworkService._internal();

  StreamSubscription<List<ConnectivityResult>>? _sub;
  bool _isConnected = true;
  bool get isConnected => _isConnected;
  bool _initialized = false;

  // Bubble state — pushed to the overlay widget
  String? _bubbleMessage;
  String? get bubbleMessage => _bubbleMessage;
  bool _navigateToLoginAfterDismiss = false;
  bool get navigateToLoginAfterDismiss => _navigateToLoginAfterDismiss;

  GlobalKey<NavigatorState>? _navigatorKey;
  void setNavigatorKey(GlobalKey<NavigatorState> key) {
    _navigatorKey = key;
  }

  Future<void> init() async {
    if (_initialized) return;
    _initialized = true;

    final results = await Connectivity().checkConnectivity();
    _isConnected = !results.every((r) => r == ConnectivityResult.none);

    _sub = Connectivity().onConnectivityChanged.listen(_onConnectivityChanged);
  }

  void _onConnectivityChanged(List<ConnectivityResult> results) async {
    final nowConnected = !results.every((r) => r == ConnectivityResult.none);
    if (nowConnected == _isConnected) return;

    final wasConnected = _isConnected;
    _isConnected = nowConnected;

    if (!wasConnected && nowConnected) {
      await _onNetworkRestored();
    } else if (wasConnected && !nowConnected) {
      _showBubble('network_disconnected', navigateToLogin: false);
    }
  }

  Future<void> _onNetworkRestored() async {
    if (SupabaseConfig.isAdminMode) return;

    final hasSession =
        Supabase.instance.client.auth.currentSession != null;
    if (!hasSession) return; // No stored credentials, nothing to do

    try {
      await Supabase.instance.client.auth.refreshSession();
    } catch (_) {
      _showBubble('network_session_expired', navigateToLogin: true);
    }
  }

  void _showBubble(String localizationKey, {required bool navigateToLogin}) {
    _bubbleMessage = localizationKey;
    _navigateToLoginAfterDismiss = navigateToLogin;
    notifyListeners();
  }

  void hideBubble() {
    _bubbleMessage = null;
    _navigateToLoginAfterDismiss = false;
    notifyListeners();
  }

  void onBubbleDismissed() {
    final shouldNavigate = _navigateToLoginAfterDismiss;
    _bubbleMessage = null;
    _navigateToLoginAfterDismiss = false;
    notifyListeners();

    if (shouldNavigate && _navigatorKey?.currentContext != null) {
      Navigator.pushNamed(_navigatorKey!.currentContext!, '/login');
    }
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }
}

final networkService = NetworkService();
