class TestEnv {
  static const String _supabaseUrl = String.fromEnvironment(
    'IT_SUPABASE_URL',
    defaultValue: '',
  );
  static const String _anonKey = String.fromEnvironment(
    'IT_SUPABASE_ANON_KEY',
    defaultValue: '',
  );
  static const String _serviceRoleKey = String.fromEnvironment(
    'IT_SUPABASE_SERVICE_ROLE_KEY',
    defaultValue: '',
  );
  static const String _profile = String.fromEnvironment(
    'IT_PROFILE',
    defaultValue: 'local',
  );
  static const bool _enableOnlineModelTests = bool.fromEnvironment(
    'IT_ENABLE_ONLINE_MODEL_TESTS',
    defaultValue: false,
  );

  static String get supabaseUrl => _supabaseUrl;
  static String get anonKey => _anonKey;
  static String get serviceRoleKey => _serviceRoleKey;
  static String get profile => _profile;
  static bool get enableOnlineModelTests => _enableOnlineModelTests;

  static bool get isConfigured =>
      _supabaseUrl.isNotEmpty && (_anonKey.isNotEmpty || _serviceRoleKey.isNotEmpty);
}
