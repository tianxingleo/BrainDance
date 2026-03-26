import 'package:flutter_dotenv/flutter_dotenv.dart';

class SupabaseConfig {
  static const String _defaultLocalModelBucket = 'braindance-models';
  static const List<String> _defaultLocalModelDiscoveryBuckets = <String>[
    'braindance-assets',
    'braindance-models',
  ];
  static const String _defaultLocalModelObjectPath =
      'releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf';

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

  static String get localModelBucket {
    final bucket = dotenv.env['LOCAL_LLM_MODEL_BUCKET']?.trim();
    if (bucket != null && bucket.isNotEmpty) {
      return bucket;
    }
    return _defaultLocalModelBucket;
  }

  static List<String> get localModelDiscoveryBuckets {
    final rawBuckets = dotenv.env['LOCAL_LLM_MODEL_BUCKETS']?.trim() ?? '';
    final buckets = <String>[
      if (rawBuckets.isNotEmpty)
        ...rawBuckets
            .split(',')
            .map((item) => item.trim())
            .where((item) => item.isNotEmpty),
      localModelBucket,
      ..._defaultLocalModelDiscoveryBuckets,
    ];

    final deduplicated = <String>{};
    for (final bucket in buckets) {
      deduplicated.add(bucket);
    }
    return deduplicated.toList(growable: false);
  }

  static String get localModelObjectPath {
    final objectPath = dotenv.env['LOCAL_LLM_MODEL_OBJECT_PATH']?.trim();
    if (objectPath != null && objectPath.isNotEmpty) {
      return objectPath;
    }
    return _defaultLocalModelObjectPath;
  }

  static String get localModelUrl {
    final explicitUrl = dotenv.env['LOCAL_LLM_MODEL_URL']?.trim();
    if (explicitUrl != null && explicitUrl.isNotEmpty) {
      return explicitUrl;
    }

    final baseUrl = url.trim();
    if (baseUrl.isEmpty) {
      return '';
    }

    final normalizedBaseUrl = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return '$normalizedBaseUrl/storage/v1/object/public/$localModelBucket/$localModelObjectPath';
  }
}
