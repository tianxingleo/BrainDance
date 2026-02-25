import 'package:flutter_dotenv/flutter_dotenv.dart';

class SupabaseConfig {
  /// Supabase project URL
  static String get url => dotenv.env['SUPABASE_URL'] ?? 'http://172.28.97.38:54321';

  /// Supabase Anon Key (Publishable)
  static String get anonKey => dotenv.env['SUPABASE_ANON_KEY'] ?? '';
}
