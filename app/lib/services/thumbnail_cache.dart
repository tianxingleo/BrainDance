import 'dart:io';

import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path_joiner;

class ThumbnailCache {
  static final ThumbnailCache _instance = ThumbnailCache._();
  factory ThumbnailCache() => _instance;
  ThumbnailCache._();

  String? _cacheDir;

  Future<String> get _dir async {
    if (_cacheDir != null) return _cacheDir!;
    final base = await DirFinder.cacheDir();
    _cacheDir = path_joiner.join(base, 'thumbnails');
    await DirSystem.ensureDir(_cacheDir!);
    return _cacheDir!;
  }

  String _filename(String url) {
    final hash = url.hashCode.toRadixString(36);
    final ext = url.split('.').last.split('?').first;
    final safeExt = ext.length <= 5 ? ext : 'jpg';
    return '$hash.$safeExt';
  }

  /// Returns local file path for [url], downloading and caching on first access.
  Future<String?> getPath(String url) async {
    if (url.isEmpty || !url.startsWith('http')) return null;

    final dir = await _dir;
    final filePath = path_joiner.join(dir, _filename(url));

    if (await FileSystem.checkFileExists(filePath)) return filePath;

    try {
      final response = await http.get(Uri.parse(url)).timeout(
        const Duration(seconds: 8),
      );
      if (response.statusCode == 200) {
        await File(filePath).writeAsBytes(response.bodyBytes);
        return filePath;
      }
    } catch (_) {}

    return null;
  }

  /// Returns path only if already cached, null otherwise.
  Future<String?> getCachedPath(String url) async {
    if (url.isEmpty || !url.startsWith('http')) return null;
    final dir = await _dir;
    final filePath = path_joiner.join(dir, _filename(url));
    if (await FileSystem.checkFileExists(filePath)) return filePath;
    return null;
  }
}
