import 'dart:convert';
import 'dart:math' as math;

import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import 'local_text_embedder.dart';

class LocalRagIndexStats {
  const LocalRagIndexStats({
    required this.totalItems,
    required this.rebuiltItems,
    required this.cachedItems,
  });

  final int totalItems;
  final int rebuiltItems;
  final int cachedItems;
}

class LocalRagIndexService {
  LocalRagIndexService({
    LocalTextEmbedder? embedder,
    this.tableName = 'memory_scene_vectors',
  }) : _embedder = embedder ?? HashingTextEmbedder();

  final LocalTextEmbedder _embedder;
  final String tableName;

  Database? _db;

  Future<Database> _database() async {
    if (_db != null) {
      return _db!;
    }

    final dir = await getApplicationDocumentsDirectory();
    final dbPath = path.join(dir.path, 'braindance_memory_rag.db');
    _db = await openDatabase(
      dbPath,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE $tableName (
            model_id TEXT PRIMARY KEY,
            scene_id TEXT NOT NULL,
            user_id TEXT,
            searchable_text TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            updated_at TEXT NOT NULL
          )
        ''');
      },
    );
    return _db!;
  }

  Future<LocalRagIndexStats> syncModels(
    List<Map<String, dynamic>> models,
  ) async {
    final db = await _database();
    final existing = await db.query(
      tableName,
      columns: ['model_id', 'fingerprint'],
    );
    final fingerprintById = <String, String>{
      for (final row in existing)
        row['model_id']?.toString() ?? '': row['fingerprint']?.toString() ?? '',
    }..remove('');

    final keepIds = <String>{};
    var rebuilt = 0;

    final batch = db.batch();
    for (final rawModel in models) {
      final model = Map<String, dynamic>.from(rawModel);
      final modelId = model['id']?.toString();
      if (modelId == null || modelId.isEmpty) {
        continue;
      }

      keepIds.add(modelId);
      final searchableText = _buildSearchableText(model);
      final fingerprint = _fingerprintFor(model, searchableText);
      if (fingerprintById[modelId] == fingerprint) {
        continue;
      }

      final vector = _embedder.embed(searchableText);
      batch.insert(tableName, {
        'model_id': modelId,
        'scene_id': model['scene_id']?.toString() ?? '',
        'user_id': model['user_id']?.toString(),
        'searchable_text': searchableText,
        'vector_json': jsonEncode(vector),
        'payload_json': jsonEncode(model),
        'fingerprint': fingerprint,
        'updated_at': DateTime.now().toUtc().toIso8601String(),
      }, conflictAlgorithm: ConflictAlgorithm.replace);
      rebuilt++;
    }

    for (final row in existing) {
      final modelId = row['model_id']?.toString();
      if (modelId == null || modelId.isEmpty || keepIds.contains(modelId)) {
        continue;
      }
      batch.delete(tableName, where: 'model_id = ?', whereArgs: [modelId]);
    }

    await batch.commit(noResult: true);
    return LocalRagIndexStats(
      totalItems: keepIds.length,
      rebuiltItems: rebuilt,
      cachedItems: math.max(keepIds.length - rebuilt, 0),
    );
  }

  Future<List<Map<String, dynamic>>> search(
    String query, {
    int limit = 24,
    double minScore = 0.12,
  }) async {
    final normalizedQuery = query.trim();
    if (normalizedQuery.isEmpty) {
      return [];
    }

    final queryVector = _embedder.embed(normalizedQuery);
    final db = await _database();
    final rows = await db.query(tableName);
    final loweredQuery = normalizedQuery.toLowerCase();
    final results = <Map<String, dynamic>>[];

    for (final row in rows) {
      final payloadJson = row['payload_json']?.toString();
      final vectorJson = row['vector_json']?.toString();
      final searchableText = row['searchable_text']?.toString() ?? '';
      if (payloadJson == null || vectorJson == null) {
        continue;
      }

      final model = Map<String, dynamic>.from(jsonDecode(payloadJson) as Map);
      final vector = (jsonDecode(vectorJson) as List<dynamic>)
          .map((value) => (value as num).toDouble())
          .toList();

      final cosine = _dot(queryVector, vector);
      final lexical = _lexicalBoost(loweredQuery, searchableText.toLowerCase());
      final score = (cosine * 0.82) + (lexical * 0.18);
      final shouldKeep = score >= minScore || lexical >= 0.9;
      if (!shouldKeep) {
        continue;
      }

      model['similarity'] = score.clamp(0.0, 1.0);
      model['matched_frames'] ??= const [];
      results.add(model);
    }

    results.sort((a, b) {
      final left = (a['similarity'] as num?)?.toDouble() ?? 0;
      final right = (b['similarity'] as num?)?.toDouble() ?? 0;
      return right.compareTo(left);
    });

    if (results.length > limit) {
      return results.sublist(0, limit);
    }
    return results;
  }

  String _buildSearchableText(Map<String, dynamic> model) {
    final parts = <String>[
      model['display_name']?.toString() ?? model['scene_id'] ?? '',
      model['description']?.toString() ?? '',
      ..._stringList(model['tags']),
      ..._stringList(model['objects']),
    ];

    final metaInfo = _toMap(model['meta_info']);
    if (metaInfo.isNotEmpty) {
      parts.addAll(_extractStrings(metaInfo));
    }

    return parts
        .map((value) => value.trim())
        .where((value) => value.isNotEmpty)
        .join(' | ');
  }

  Map<String, dynamic> _toMap(dynamic value) {
    if (value is Map<String, dynamic>) {
      return value;
    }
    if (value is Map) {
      return value.map((key, item) => MapEntry(key.toString(), item));
    }
    return const <String, dynamic>{};
  }

  List<String> _extractStrings(dynamic value) {
    if (value == null) {
      return const [];
    }
    if (value is String) {
      final trimmed = value.trim();
      return trimmed.isEmpty ? const [] : [trimmed];
    }
    if (value is num || value is bool) {
      return [value.toString()];
    }
    if (value is List) {
      return value.expand(_extractStrings).toList();
    }
    if (value is Map) {
      return value.values.expand(_extractStrings).toList();
    }
    return const [];
  }

  List<String> _stringList(dynamic value) {
    if (value is List) {
      return value.map((item) => item.toString()).toList();
    }
    return const [];
  }

  String _fingerprintFor(Map<String, dynamic> model, String searchableText) {
    final createdAt = model['created_at']?.toString() ?? '';
    final preview = model['preview_img_path']?.toString() ?? '';
    final ply = model['ply_path']?.toString() ?? '';
    return '$createdAt|$preview|$ply|$searchableText';
  }

  double _dot(List<double> left, List<double> right) {
    if (left.isEmpty || right.isEmpty || left.length != right.length) {
      return 0;
    }
    var sum = 0.0;
    for (var i = 0; i < left.length; i++) {
      sum += left[i] * right[i];
    }
    return sum;
  }

  double _lexicalBoost(String query, String searchableText) {
    if (query.isEmpty || searchableText.isEmpty) {
      return 0;
    }
    if (searchableText.contains(query)) {
      return 1.0;
    }

    final queryTokens = query
        .split(RegExp(r'\s+'))
        .where((token) => token.isNotEmpty)
        .toList();
    if (queryTokens.isEmpty) {
      return 0;
    }

    var hits = 0;
    for (final token in queryTokens) {
      if (searchableText.contains(token)) {
        hits++;
      }
    }
    return hits / queryTokens.length;
  }
}
