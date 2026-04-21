import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class SupabaseEndpointResolution {
  const SupabaseEndpointResolution({
    required this.url,
    required this.attemptedUrls,
    this.diagnosticMessage,
    this.usedFallback = false,
  });

  final String url;
  final List<String> attemptedUrls;
  final String? diagnosticMessage;
  final bool usedFallback;
}

class SupabaseConfig {
  static const String _defaultLocalModelBucket = 'braindance-models';
  static const List<String> _defaultLocalModelDiscoveryBuckets = <String>[
    'braindance-assets',
    'braindance-models',
  ];
  static const String _defaultLocalModelObjectPath =
      'releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf';
  static String? _runtimeResolvedUrl;
  static String? _runtimeDiagnosticMessage;

  /// Supabase project URL
  static String get url =>
      _runtimeResolvedUrl ?? normalizeUrl(dotenv.env['SUPABASE_URL'] ?? '');

  static String get configuredUrl =>
      normalizeUrl(dotenv.env['SUPABASE_URL'] ?? '');

  static String? get runtimeDiagnosticMessage => _runtimeDiagnosticMessage;

  static void applyRuntimeResolution(SupabaseEndpointResolution resolution) {
    _runtimeResolvedUrl = normalizeUrl(resolution.url);
    _runtimeDiagnosticMessage = resolution.diagnosticMessage;
  }

  static String normalizeUrl(String rawUrl) {
    final trimmed = rawUrl.trim();
    if (trimmed.isEmpty) {
      return '';
    }
    return trimmed.endsWith('/')
        ? trimmed.substring(0, trimmed.length - 1)
        : trimmed;
  }

  static bool isLikelyLocalUrl(String rawUrl) {
    final normalized = normalizeUrl(rawUrl).toLowerCase();
    return normalized.startsWith('http://127.0.0.1') ||
        normalized.startsWith('http://localhost') ||
        normalized.startsWith('http://10.0.2.2') ||
        normalized.startsWith('http://192.168.') ||
        normalized.startsWith('http://10.') ||
        RegExp(r'^http://172\.(1[6-9]|2\d|3[01])\.').hasMatch(normalized);
  }

  static List<String> get urlFallbacks {
    final raw = dotenv.env['SUPABASE_URL_FALLBACKS']?.trim() ?? '';
    if (raw.isEmpty) {
      return const <String>[];
    }
    return raw
        .split(',')
        .map(normalizeUrl)
        .where((item) => item.isNotEmpty)
        .toList(growable: false);
  }

  static List<String> buildUrlCandidates() {
    final configured = configuredUrl;
    final candidates = <String>[
      if (configured.isNotEmpty) configured,
      ...urlFallbacks,
    ];

    final lowerConfigured = configured.toLowerCase();
    if (lowerConfigured.startsWith('http://127.0.0.1') ||
        lowerConfigured.startsWith('http://localhost')) {
      if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
        candidates.add(configured.replaceFirst(
          RegExp(r'://(127\.0\.0\.1|localhost)'),
          '://10.0.2.2',
        ));
      }
    }

    final deduplicated = <String>{};
    for (final item in candidates) {
      final normalized = normalizeUrl(item);
      if (normalized.isNotEmpty) {
        deduplicated.add(normalized);
      }
    }
    return deduplicated.toList(growable: false);
  }

  static String edgeFunctionUrl(String functionName) {
    final baseUrl = url;
    if (baseUrl.isEmpty) {
      return '';
    }
    return '$baseUrl/functions/v1/$functionName';
  }

  static Future<SupabaseEndpointResolution> resolveEndpoint({
    Duration timeout = const Duration(seconds: 2),
  }) async {
    final candidates = buildUrlCandidates();
    if (candidates.isEmpty) {
      return const SupabaseEndpointResolution(
        url: '',
        attemptedUrls: <String>[],
        diagnosticMessage:
            'SUPABASE_URL is missing, so Supabase cannot be initialized.',
      );
    }

    final dio = Dio(
      BaseOptions(
        connectTimeout: timeout,
        receiveTimeout: timeout,
        sendTimeout: timeout,
        validateStatus: (status) => status != null && status < 500,
      ),
    );

    try {
      for (var index = 0; index < candidates.length; index++) {
        final candidate = candidates[index];
        if (await _isEndpointReachable(dio, candidate)) {
          final usedFallback = index > 0 || candidate != configuredUrl;
          final diagnostic = usedFallback
              ? 'Supabase switched to reachable endpoint: $candidate'
              : null;
          return SupabaseEndpointResolution(
            url: candidate,
            attemptedUrls: candidates,
            diagnosticMessage: diagnostic,
            usedFallback: usedFallback,
          );
        }
      }
    } finally {
      dio.close();
    }

    final configured = configuredUrl;
    final isLocal = isLikelyLocalUrl(configured);
    final diagnostic = configured.isEmpty
        ? 'SUPABASE_URL is missing, so Supabase cannot be initialized.'
        : isLocal
        ? 'Current SUPABASE_URL points to a local/LAN address ($configured), but it is unreachable. Start local Supabase first or update app/.env to a reachable endpoint.'
        : 'Current SUPABASE_URL=$configured is unreachable. Check the host, port, and network.';
    return SupabaseEndpointResolution(
      url: configured,
      attemptedUrls: candidates,
      diagnosticMessage: diagnostic,
    );
  }

  static Future<bool> _isEndpointReachable(Dio dio, String baseUrl) async {
    final probes = <String>[
      '$baseUrl/rest/v1/',
      '$baseUrl/auth/v1/health',
    ];
    for (final probe in probes) {
      try {
        final response = await dio.get<Object?>(
          probe,
          options: Options(
            headers: {
              if (apiKey.isNotEmpty) 'apikey': apiKey,
            },
          ),
        );
        if ((response.statusCode ?? 0) > 0) {
          return true;
        }
      } on DioException {
        continue;
      }
    }
    return false;
  }

  static String buildConnectionHelp(String target, {String? endpoint}) {
    final baseUrl = url;
    final configured = configuredUrl;
    final current = baseUrl.isNotEmpty ? baseUrl : configured;
    if (current.isEmpty) {
      return '$target failed: SUPABASE_URL is missing.';
    }
    if (isLikelyLocalUrl(current)) {
      return '$target failed: current Supabase endpoint $current is unreachable. Confirm local Supabase is running and that this device/emulator can reach the host.'
          '${endpoint == null || endpoint.isEmpty ? '' : ' Endpoint: $endpoint'}';
    }
    return '$target failed: current Supabase endpoint $current is unreachable. Check the host, port, or network.'
        '${endpoint == null || endpoint.isEmpty ? '' : ' Endpoint: $endpoint'}';
  }

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

    final normalizedBaseUrl = normalizeUrl(baseUrl);
    return '$normalizedBaseUrl/storage/v1/object/public/$localModelBucket/$localModelObjectPath';
  }
}
