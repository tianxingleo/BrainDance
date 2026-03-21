import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
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

    final prepared = await compute(_prepareSyncPayload, {
      'models': models,
      'fingerprints': fingerprintById,
      'updatedAt': DateTime.now().toUtc().toIso8601String(),
    });
    final keepIds = (prepared['keepIds'] as List<dynamic>)
        .map((item) => item.toString())
        .toSet();
    final upserts = List<Map<String, dynamic>>.from(
      prepared['upserts'] as List<dynamic>,
    );
    final rebuilt = upserts.length;

    final batch = db.batch();
    for (final item in upserts) {
      batch.insert(
        tableName,
        item,
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
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

    final db = await _database();
    final rows = await db.query(
      tableName,
      columns: ['payload_json', 'vector_json', 'searchable_text'],
    );
    final results = await compute(_searchRows, {
      'query': normalizedQuery,
      'limit': limit,
      'minScore': minScore,
      'rows': rows,
      'dimensions': _embedder.dimensions,
    });
    return List<Map<String, dynamic>>.from(results as List<dynamic>);
  }

}

Map<String, dynamic> _prepareSyncPayload(Map<String, dynamic> payload) {
  final models = List<Map<String, dynamic>>.from(payload['models'] as List);
  final fingerprints = Map<String, String>.from(
    payload['fingerprints'] as Map,
  );
  final updatedAt = payload['updatedAt']?.toString() ?? '';
  final embedder = HashingTextEmbedder();
  final keepIds = <String>[];
  final upserts = <Map<String, dynamic>>[];

  for (final rawModel in models) {
    final model = Map<String, dynamic>.from(rawModel);
    final modelId = model['id']?.toString();
    if (modelId == null || modelId.isEmpty) {
      continue;
    }

    keepIds.add(modelId);
    final searchableText = _buildSearchableTextStatic(model);
    final fingerprint = _fingerprintForStatic(model, searchableText);
    if (fingerprints[modelId] == fingerprint) {
      continue;
    }

    upserts.add({
      'model_id': modelId,
      'scene_id': model['scene_id']?.toString() ?? '',
      'user_id': model['user_id']?.toString(),
      'searchable_text': searchableText,
      'vector_json': jsonEncode(embedder.embed(searchableText)),
      'payload_json': jsonEncode(model),
      'fingerprint': fingerprint,
      'updated_at': updatedAt,
    });
  }

  return {
    'keepIds': keepIds,
    'upserts': upserts,
  };
}

List<Map<String, dynamic>> _searchRows(Map<String, dynamic> payload) {
  final query = payload['query']?.toString() ?? '';
  final limit = payload['limit'] as int? ?? 24;
  final minScore = (payload['minScore'] as num?)?.toDouble() ?? 0.12;
  final rows = List<Map<String, dynamic>>.from(payload['rows'] as List);
  final dimensions = payload['dimensions'] as int? ?? 192;
  final embedder = HashingTextEmbedder(dimensions: dimensions);
  final queryVector = embedder.embed(query);
  final loweredQuery = query.toLowerCase();
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
    final cosine = _dotStatic(queryVector, vector);
    final lexical = _lexicalBoostStatic(loweredQuery, searchableText.toLowerCase());
    final score = (cosine * 0.82) + (lexical * 0.18);
    if (score < minScore && lexical < 0.9) {
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

  return results.length > limit ? results.sublist(0, limit) : results;
}

String _buildSearchableTextStatic(Map<String, dynamic> model) {
  final parts = <String>[
    model['display_name']?.toString() ?? '',
    model['scene_id']?.toString() ?? '',
    model['description']?.toString() ?? '',
    ..._stringListStatic(model['tags']),
    ..._stringListStatic(model['objects']),
  ];

  final metaInfo = _toMapStatic(model['meta_info']);
  if (metaInfo.isNotEmpty) {
    parts.addAll(_extractStringsStatic(metaInfo));
  }

  return parts
      .map((value) => value.trim())
      .where((value) => value.isNotEmpty)
      .join(' | ');
}

Map<String, dynamic> _toMapStatic(dynamic value) {
  if (value is Map<String, dynamic>) {
    return value;
  }
  if (value is Map) {
    return value.map((key, item) => MapEntry(key.toString(), item));
  }
  return const <String, dynamic>{};
}

List<String> _extractStringsStatic(dynamic value) {
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
    return value.expand(_extractStringsStatic).toList();
  }
  if (value is Map) {
    return value.values.expand(_extractStringsStatic).toList();
  }
  return const [];
}

List<String> _stringListStatic(dynamic value) {
  if (value is List) {
    return value.map((item) => item.toString()).toList();
  }
  return const [];
}

String _fingerprintForStatic(Map<String, dynamic> model, String searchableText) {
  final createdAt = model['created_at']?.toString() ?? '';
  final preview = model['preview_img_path']?.toString() ?? '';
  final ply = model['ply_path']?.toString() ?? '';
  return '$createdAt|$preview|$ply|$searchableText';
}

double _dotStatic(List<double> left, List<double> right) {
  if (left.isEmpty || right.isEmpty || left.length != right.length) {
    return 0;
  }
  var sum = 0.0;
  for (var i = 0; i < left.length; i++) {
    sum += left[i] * right[i];
  }
  return sum;
}

double _lexicalBoostStatic(String query, String searchableText) {
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
