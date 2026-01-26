import 'package:flutter_riverpod/legacy.dart';

class LocaleNotifier extends StateNotifier<String> {
  LocaleNotifier() : super('en_US');
  String get locale {
    return state;
  }

  void setLocale(String code) {
    state = code;
  }
}

final localeProvider = StateNotifierProvider<LocaleNotifier, String>(
  (ref) => LocaleNotifier(),
);
