import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../configs/supabase_config.dart';

class LocalModelCatalogItem {
  const LocalModelCatalogItem({
    required this.id,
    required this.name,
    required this.downloadUrl,
    required this.fileName,
    required this.bucket,
    this.description,
    this.sizeBytes,
    this.tags = const <String>[],
    this.isRecommended = false,
  });

  final String id;
  final String name;
  final String downloadUrl;
  final String fileName;
  final String bucket;
  final String? description;
  final int? sizeBytes;
  final List<String> tags;
  final bool isRecommended;
}

class LocalModelCatalogService {
  static const String _catalogObjectPath = 'catalog/model_catalog.json';

  const LocalModelCatalogService();

  @visibleForTesting
  List<LocalModelCatalogItem> parseCatalogForTesting(dynamic decoded) =>
      _parseCatalog(decoded);

  Future<List<LocalModelCatalogItem>> fetchCatalog() async {
    final items = <LocalModelCatalogItem>[];

    // 从 catalog/model_catalog.json 获取
    final catalogUrl = _buildPublicUrl(_catalogObjectPath);
    if (catalogUrl.isNotEmpty) {
      try {
        final response = await Dio().get<dynamic>(
          catalogUrl,
          options: Options(responseType: ResponseType.plain),
        );
        final decoded = jsonDecode(response.data.toString());
        items.addAll(_parseCatalog(decoded));
      } catch (_) {
        // 读取 catalog 失败时继续尝试其他方式
      }
    }

    // 历史上端侧模型既放过 braindance-models，也放过 braindance-assets。
    // 这里统一做多 bucket 扫描，避免前端因为迁移历史只能看到一个默认模型。
    try {
      final supabase = Supabase.instance.client;
      for (final bucket in SupabaseConfig.localModelDiscoveryBuckets) {
        final pathsToScan = [''];
        final scannedPaths = <String>{};

        while (pathsToScan.isNotEmpty) {
          final currentPath = pathsToScan.removeLast();
          if (scannedPaths.contains(currentPath)) continue;
          scannedPaths.add(currentPath);

          try {
            final objects = await supabase.storage
                .from(bucket)
                .list(
                  path: currentPath,
                  searchOptions: const SearchOptions(limit: 1000),
                );
            for (final obj in objects) {
              if (obj.name == '.emptyFolderPlaceholder') continue;

              final objPath = currentPath.isEmpty
                  ? obj.name
                  : '$currentPath/${obj.name}';

              if (obj.id == null || obj.metadata == null) {
                pathsToScan.add(objPath);
              } else if (obj.name.toLowerCase().endsWith('.gguf')) {
                final url = _buildPublicUrl(objPath, bucket: bucket);
                items.add(
                  LocalModelCatalogItem(
                    id: '$bucket:$objPath',
                    name: obj.name,
                    downloadUrl: url,
                    fileName: obj.name,
                    bucket: bucket,
                    description: 'Supabase Storage: $bucket/$objPath',
                    sizeBytes: _readInt(obj.metadata?['size']),
                    isRecommended: url == SupabaseConfig.localModelUrl,
                  ),
                );
              }
            }
          } catch (_) {}
        }
      }
    } catch (_) {}

    if (items.isEmpty) {
      items.add(_buildDefaultItem());
    }

    final deduplicated = <String, LocalModelCatalogItem>{};
    for (final item in items) {
      deduplicated[item.downloadUrl] = item;
    }

    final result = deduplicated.values.toList()
      ..sort((left, right) {
        if (left.isRecommended != right.isRecommended) {
          return left.isRecommended ? -1 : 1;
        }
        return left.name.toLowerCase().compareTo(right.name.toLowerCase());
      });
    return result;
  }

  List<LocalModelCatalogItem> _parseCatalog(dynamic decoded) {
    final rawItems = _extractRawItems(decoded);
    final items = rawItems
        .map(_parseItem)
        .whereType<LocalModelCatalogItem>()
        .toList();

    if (items.any((item) => item.downloadUrl == SupabaseConfig.localModelUrl)) {
      return items;
    }
    return <LocalModelCatalogItem>[_buildDefaultItem(), ...items];
  }

  List<dynamic> _extractRawItems(dynamic decoded) {
    if (decoded is List) {
      return decoded;
    }
    if (decoded is! Map) {
      return const <dynamic>[];
    }

    const listKeys = <String>[
      'models',
      'items',
      'releases',
      'candidates',
      'data',
    ];
    for (final key in listKeys) {
      final value = decoded[key];
      if (value is List) {
        return value;
      }
    }

    final nestedCatalog = decoded['catalog'];
    if (nestedCatalog is Map) {
      return _extractRawItems(nestedCatalog);
    }

    return const <dynamic>[];
  }

  LocalModelCatalogItem? _parseItem(dynamic raw) {
    if (raw is! Map) {
      return null;
    }

    final objectPath = _readString(raw, const [
      'object_path',
      'path',
      'key',
      'storage_path',
    ]);
    final prefix = _readString(raw, const ['prefix']);
    final type = _readString(raw, const ['type']);
    final isDirectoryRelease =
        (prefix != null && prefix.isNotEmpty) ||
        (objectPath != null && objectPath.endsWith('/'));
    final isDownloadableModel =
        (objectPath != null && objectPath.toLowerCase().endsWith('.gguf')) ||
        type?.toLowerCase() == 'gguf';

    // 端侧 llamadart 只能直接加载 GGUF 文件。catalog 里可能同时记录
    // HF merged 或 LoRA release 目录，这类目录不应出现在手机下载列表里。
    if (isDirectoryRelease || !isDownloadableModel) {
      return null;
    }

    final name =
        _readString(raw, const [
          'name',
          'title',
          'display_name',
          'model_name',
        ]) ??
        _readString(raw, const ['slug', 'id']) ??
        _buildDisplayNameFromPath(objectPath ?? '');
    final downloadUrl =
        _readString(raw, const ['download_url', 'url', 'public_url', 'href']) ??
        _buildPublicUrl(
          objectPath ?? '',
          bucket:
              _readString(raw, const ['bucket', 'bucket_id']) ??
              SupabaseConfig.localModelBucket,
        );
    final fileName =
        _readString(raw, const ['file_name', 'filename']) ??
        _extractFileName(downloadUrl);

    if (name == null ||
        name.isEmpty ||
        downloadUrl.isEmpty ||
        fileName.isEmpty) {
      return null;
    }

    final id =
        _readString(raw, const ['id', 'slug']) ?? objectPath ?? downloadUrl;

    return LocalModelCatalogItem(
      id: id,
      name: name,
      downloadUrl: downloadUrl,
      fileName: fileName,
      bucket:
          _readString(raw, const ['bucket', 'bucket_id']) ??
          _inferBucketFromUrl(downloadUrl) ??
          SupabaseConfig.localModelBucket,
      description: _readString(raw, const ['description', 'desc', 'summary']),
      sizeBytes: _readInt(raw['size_bytes']) ?? _readInt(raw['size']),
      tags: _readStringList(raw['tags']),
      isRecommended:
          raw['recommended'] == true ||
          raw['default'] == true ||
          downloadUrl == SupabaseConfig.localModelUrl,
    );
  }

  LocalModelCatalogItem _buildDefaultItem() {
    final defaultUrl = SupabaseConfig.localModelUrl;
    return LocalModelCatalogItem(
      id: SupabaseConfig.localModelObjectPath,
      name: 'Qwen3-1.7B BrainDance 移动端默认模型',
      downloadUrl: defaultUrl,
      fileName: _extractFileName(defaultUrl),
      bucket: SupabaseConfig.localModelBucket,
      description: 'Recall 端侧问答默认模型',
      isRecommended: true,
    );
  }

  String _buildPublicUrl(String objectPath, {String? bucket}) {
    final baseUrl = SupabaseConfig.url.trim();
    final normalizedPath = objectPath.trim().replaceFirst(RegExp(r'^/+'), '');
    final normalizedBucket = (bucket ?? SupabaseConfig.localModelBucket).trim();
    if (baseUrl.isEmpty || normalizedPath.isEmpty) {
      return '';
    }
    final normalizedBaseUrl = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return '$normalizedBaseUrl/storage/v1/object/public/$normalizedBucket/$normalizedPath';
  }

  String? _readString(Map raw, List<String> keys) {
    for (final key in keys) {
      final value = raw[key];
      if (value == null) {
        continue;
      }
      final text = value.toString().trim();
      if (text.isNotEmpty) {
        return text;
      }
    }
    return null;
  }

  int? _readInt(dynamic value) {
    if (value is int) {
      return value;
    }
    if (value is num) {
      return value.toInt();
    }
    if (value == null) {
      return null;
    }
    return int.tryParse(value.toString());
  }

  List<String> _readStringList(dynamic raw) {
    if (raw is! List) {
      return const <String>[];
    }
    return raw
        .map((item) => item?.toString().trim() ?? '')
        .where((item) => item.isNotEmpty)
        .toList();
  }

  String _extractFileName(String urlOrPath) {
    if (urlOrPath.isEmpty) {
      return '';
    }
    final uri = Uri.tryParse(urlOrPath);
    final segments = (uri?.pathSegments ?? urlOrPath.split('/'))
        .where((segment) => segment.isNotEmpty)
        .toList();
    if (segments.isEmpty) {
      return '';
    }
    return segments.last;
  }

  String? _buildDisplayNameFromPath(String objectPath) {
    final fileName = _extractFileName(objectPath);
    if (fileName.isEmpty) {
      return null;
    }
    return fileName
        .replaceAll(RegExp(r'\.gguf$', caseSensitive: false), '')
        .replaceAll('-', ' ')
        .replaceAll('_', ' ')
        .trim();
  }

  String? _inferBucketFromUrl(String url) {
    final uri = Uri.tryParse(url);
    final segments = uri?.pathSegments;
    if (segments == null || segments.length < 5) {
      return null;
    }
    final publicIndex = segments.indexOf('public');
    if (publicIndex == -1 || publicIndex + 1 >= segments.length) {
      return null;
    }
    return segments[publicIndex + 1];
  }
}
