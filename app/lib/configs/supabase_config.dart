import 'package:flutter_dotenv/flutter_dotenv.dart';

class SupabaseConfig {
  /// Supabase project URL
  static String get url => dotenv.env['SUPABASE_URL'] ?? '';

  /// Unified Supabase key.
  ///
  /// Priority:
  /// 1. SUPABASE_KEY
  /// 2. SUPABASE_SECRET_KEY / SUPABASE_SERVICE_ROLE_KEY
  /// 3. SUPABASE_ANON_KEY
  static String get apiKey {
    final explicitKey = dotenv.env['SUPABASE_KEY']?.trim();
    if (explicitKey != null && explicitKey.isNotEmpty) {
      return explicitKey;
    }

    final secretKey = dotenv.env['SUPABASE_SECRET_KEY']?.trim();
    if (secretKey != null && secretKey.isNotEmpty) {
      return secretKey;
    }

    final serviceRoleKey = dotenv.env['SUPABASE_SERVICE_ROLE_KEY']?.trim();
    if (serviceRoleKey != null && serviceRoleKey.isNotEmpty) {
      return serviceRoleKey;
    }

    return dotenv.env['SUPABASE_ANON_KEY']?.trim() ?? '';
  }

  /// Backward-compatible alias used by older code paths.
  static String get anonKey => apiKey;

  /// Whether the configured key is a secret/service-role key that bypasses RLS.
  static bool get isAdminMode {
    final key = apiKey;
    if (key.isEmpty) return false;

    return key.startsWith('sb_secret_') ||
        (dotenv.env['SUPABASE_SECRET_KEY']?.trim().isNotEmpty ?? false) ||
        (dotenv.env['SUPABASE_SERVICE_ROLE_KEY']?.trim().isNotEmpty ?? false);
  }

  static String get modeLabel => isAdminMode ? 'admin' : 'rls';
}
