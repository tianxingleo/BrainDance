import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as path_joiner;
import 'package:path_provider/path_provider.dart';

import '../extra_func/dir_and_file.dart';
import 'thumbnail_cache.dart';

class LocalModelScanner {
  const LocalModelScanner();

  static const _modelExtensions = ['.ply', '.splat', '.ksplat'];

  /// Scan all known directories for fully downloaded 3D model files.
  /// Returns a list of model maps suitable for display in the recall page.
  Future<List<Map<String, dynamic>>> scanDownloadedModels() async {
    final allFiles = <File>[];
    final seenPaths = <String>{};

    Future<void> collectFrom(Directory dir) async {
      if (!await dir.exists()) return;
      try {
        await for (final entity in dir.list(recursive: true)) {
          if (entity is! File) continue;
          final p = entity.path;
          if (seenPaths.contains(p)) continue;
          if (p.endsWith('.tmp')) continue;
          if (p.endsWith('.meta.json')) continue;
          if (!_modelExtensions.any((ext) => p.endsWith(ext))) continue;
          seenPaths.add(p);
          allFiles.add(entity);
        }
      } catch (e) {
        debugPrint('[LocalModelScanner] error scanning ${dir.path}: $e');
      }
    }

    // Scan app documents dir (where WebGL viewer caches downloads)
    try {
      final appDocDir = await getApplicationDocumentsDirectory();
      debugPrint(
        '[LocalModelScanner] scanning appDocDir: ${appDocDir.path}',
      );
      await collectFrom(Directory(appDocDir.path));
    } catch (e) {
      debugPrint('[LocalModelScanner] appDocDir error: $e');
    }

    // Scan downloads dir (where _downloadRecallModel saves files)
    try {
      final dlDir = await DirFinder.downloadsDir();
      if (dlDir.isNotEmpty) {
        debugPrint('[LocalModelScanner] scanning downloadsDir: $dlDir');
        await collectFrom(Directory(dlDir));
      }
    } catch (e) {
      debugPrint('[LocalModelScanner] downloadsDir error: $e');
    }

    // Scan documents dir (fallback for _downloadRecallModel)
    try {
      final docDir = await DirFinder.documentsDir();
      if (docDir.isNotEmpty) {
        debugPrint('[LocalModelScanner] scanning documentsDir: $docDir');
        await collectFrom(Directory(docDir));
      }
    } catch (e) {
      debugPrint('[LocalModelScanner] documentsDir error: $e');
    }

    debugPrint('[LocalModelScanner] found ${allFiles.length} model files');

    final thumbnailCache = ThumbnailCache();
    final models = <Map<String, dynamic>>[];
    for (final file in allFiles) {
      try {
        final stat = await file.stat();
        final meta = await _readMetaSidecar(file.path);
        // Only include models with a .meta.json sidecar (written by recall page)
        if (meta.isEmpty) continue;
        final name = meta['display_name'] as String? ??
            _fileNameWithoutExtension(file.path);
        final sizeMb = (stat.size / (1024 * 1024)).toStringAsFixed(1);

        // Resolve preview image from metadata + thumbnail cache
        String previewPath = '';
        final metaPreviewUrl = meta['preview_img_path']?.toString();
        if (metaPreviewUrl != null && metaPreviewUrl.isNotEmpty) {
          final cachedThumb = await thumbnailCache.getCachedPath(metaPreviewUrl);
          if (cachedThumb != null) {
            previewPath = cachedThumb;
            debugPrint('[LocalModelScanner] resolved thumbnail for $name: $previewPath');
          }
        }

        models.add({
          'id': 'local_${file.path.hashCode}',
          'scene_id': name,
          'display_name': name,
          'description': '$sizeMb MB  ·  ${file.path}',
          'ply_path': file.path,
          'preview_img_path': previewPath,
          'tags': ['local', 'offline'],
          'objects': ['3dgs'],
          'meta_info': <String, dynamic>{},
          'created_at': stat.modified.toUtc().toIso8601String(),
          '_is_local_only': true,
        });
      } catch (e) {
        debugPrint('[LocalModelScanner] error reading ${file.path}: $e');
      }
    }

    // Sort by modification time, newest first
    models.sort((a, b) {
      final ta = DateTime.tryParse(a['created_at']?.toString() ?? '') ??
          DateTime(0);
      final tb = DateTime.tryParse(b['created_at']?.toString() ?? '') ??
          DateTime(0);
      return tb.compareTo(ta);
    });

    return models;
  }

  /// Read a .meta.json sidecar file for [modelPath].
  Future<Map<String, dynamic>> _readMetaSidecar(String modelPath) async {
    final metaFile = File('$modelPath.meta.json');
    if (!await metaFile.exists()) return {};
    try {
      final content = await metaFile.readAsString();
      final decoded = jsonDecode(content) as Map<String, dynamic>;
      return decoded;
    } catch (e) {
      debugPrint('[LocalModelScanner] error parsing meta for $modelPath: $e');
      return {};
    }
  }

  String _fileNameWithoutExtension(String path) =>
      path_joiner.basenameWithoutExtension(path);
}
