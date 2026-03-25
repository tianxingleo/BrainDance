import 'dart:convert';

import 'package:dio/dio.dart';

import '../configs/supabase_config.dart';

class LocalModelCatalogItem {
  const LocalModelCatalogItem({
    required this.id,
    required this.name,
    required this.downloadUrl,
    required this.fileName,
    this.description,
    this.sizeBytes,
    this.tags = const <String>[],
    this.isRecommended = false,
  });

  final String id;
  final String name;
  final String downloadUrl;
  final String fileName;
  final String? description;
  final int? sizeBytes;
  final List<String> tags;
  final bool isRecommended;
}

class LocalModelCatalogService {
  static const String _catalogObjectPath = 'catalog/model_catalog.json';

  const LocalModelCatalogService();

  Future<List<LocalModelCatalogItem>> fetchCatalog() async {
    final items = <LocalModelCatalogItem>[];
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
        // 读取 catalog 失败时回退到默认模型，避免阻断 Recall 页使用。
      }
    }

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

    final name =
        _readString(raw, const [
          'name',
          'title',
          'display_name',
          'model_name',
        ]) ??
        _readString(raw, const ['slug', 'id']);
    final downloadUrl =
        _readString(raw, const ['download_url', 'url', 'public_url', 'href']) ??
        _buildPublicUrl(
          _readString(raw, const [
                'object_path',
                'path',
                'key',
                'storage_path',
              ]) ??
              '',
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

    final id = _readString(raw, const ['id', 'slug']) ?? downloadUrl;

    return LocalModelCatalogItem(
      id: id,
      name: name,
      downloadUrl: downloadUrl,
      fileName: fileName,
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
      name: 'Qwen3-1.7B BrainDance 默认模型',
      downloadUrl: defaultUrl,
      fileName: _extractFileName(defaultUrl),
      description: 'Recall 端侧问答默认模型',
      isRecommended: true,
    );
  }

  String _buildPublicUrl(String objectPath) {
    final baseUrl = SupabaseConfig.url.trim();
    final normalizedPath = objectPath.trim().replaceFirst(RegExp(r'^/+'), '');
    if (baseUrl.isEmpty || normalizedPath.isEmpty) {
      return '';
    }
    final normalizedBaseUrl = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;
    return '$normalizedBaseUrl/storage/v1/object/public/${SupabaseConfig.localModelBucket}/$normalizedPath';
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
}
