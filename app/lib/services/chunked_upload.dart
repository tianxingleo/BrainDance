import 'dart:io';
import 'dart:math';

import 'package:dio/dio.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../configs/supabase_config.dart';

/// Resumable upload to Supabase Storage.
///
/// All network calls use [SupabaseConfig.url] via Dio (resolved local IP when
/// on LAN) so that nothing depends on the Supabase SDK's internal URL which
/// may point to an unreachable Cloudflare hostname.
class ChunkedUpload {
  static const _targetChunkSeconds = 60;

  static Future<void> upload({
    required File file,
    required String storagePath,
    required String contentType,
    required void Function(int uploadedBytes, int totalBytes) onProgress,
  }) async {
    final client = Supabase.instance.client;
    final token = client.auth.currentSession?.accessToken ?? '';
    final fileSize = await file.length();
    final url =
        '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$storagePath';

    // --- Step 1: try streaming upload (fast path) ---
    int lastUploaded = 0;
    final stopwatch = Stopwatch()..start();

    try {
      await _streamingPost(
        url: url,
        file: file,
        fileSize: fileSize,
        contentType: contentType,
        token: token,
        onProgress: (count, _) {
          lastUploaded = count;
          onProgress(count, fileSize);
        },
      );
      return; // whole file uploaded in one shot
    } on DioException catch (_) {
      // Any network failure falls through to the adaptive chunked upload.
    }

    // --- Step 2: clean up partial file, then chunked upload ---
    await _storageDelete(token: token, paths: [storagePath]);

    final elapsed = stopwatch.elapsed.inSeconds;
    final speed = elapsed > 0 ? lastUploaded / elapsed : 50000.0;
    final chunkSize = (speed * _targetChunkSeconds)
        .round()
        .clamp(1024 * 1024, 20 * 1024 * 1024);

    await _uploadChunked(
      file: file,
      fileSize: fileSize,
      storagePath: storagePath,
      contentType: contentType,
      token: token,
      chunkSize: chunkSize,
      onProgress: onProgress,
    );
  }

  /// Insert a processing_tasks row via REST (bypasses SDK URL).
  static Future<void> insertTask(Map<String, dynamic> task) async {
    final client = Supabase.instance.client;
    final token = client.auth.currentSession?.accessToken ?? '';
    final dio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 15),
      sendTimeout: const Duration(seconds: 15),
      receiveTimeout: const Duration(seconds: 15),
    ));
    await dio.post(
      '${SupabaseConfig.url}/rest/v1/processing_tasks',
      data: task,
      options: Options(
        headers: {
          'Authorization': 'Bearer $token',
          'apikey': SupabaseConfig.apiKey,
          'Content-Type': 'application/json',
          'Prefer': 'return=minimal',
        },
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Streaming upload (single POST, zero-copy via file.openRead)
  // ---------------------------------------------------------------------------

  static Future<void> _streamingPost({
    required String url,
    required File file,
    required int fileSize,
    required String contentType,
    required String token,
    required void Function(int, int) onProgress,
    int maxAttempts = 2,
  }) async {
    final dio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(minutes: 10),
      receiveTimeout: const Duration(minutes: 5),
    ));

    DioException? lastError;
    for (var attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        await dio.post(
          url,
          data: file.openRead(),
          options: Options(
            headers: {
              'Authorization': 'Bearer $token',
              'apikey': SupabaseConfig.apiKey,
              'Content-Type': contentType,
              'Content-Length': fileSize.toString(),
            },
          ),
          onSendProgress: onProgress,
        );
        return;
      } on DioException catch (e) {
        lastError = e;
        if (attempt < maxAttempts - 1) {
          await Future.delayed(const Duration(seconds: 5));
        }
      }
    }
    throw lastError!;
  }

  // ---------------------------------------------------------------------------
  // Chunked upload with resume detection
  // ---------------------------------------------------------------------------

  static Future<void> _uploadChunked({
    required File file,
    required int fileSize,
    required String storagePath,
    required String contentType,
    required String token,
    required int chunkSize,
    required void Function(int, int) onProgress,
  }) async {
    final totalChunks = (fileSize + chunkSize - 1) ~/ chunkSize;
    final dio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(minutes: 2),
      receiveTimeout: const Duration(minutes: 2),
    ));

    // Resume: detect parts already on the server.
    final done = await _detectExistingChunks(
      token: token,
      storagePath: storagePath,
      totalChunks: totalChunks,
    );

    int uploaded = 0;
    for (var i = 0; i < totalChunks; i++) {
      if (done.contains(i)) {
        // already uploaded in a previous run
        uploaded += _chunkLen(i, chunkSize, fileSize);
        onProgress(uploaded, fileSize);
        continue;
      }

      final len = _chunkLen(i, chunkSize, fileSize);
      final raf = await file.open();
      await raf.setPosition(i * chunkSize);
      final bytes = await raf.read(len);
      await raf.close();

      final partPath = '$storagePath.part.$i';
      final partUrl =
          '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$partPath';

      await _retryPost(
        dio: dio,
        url: partUrl,
        data: bytes,
        token: token,
        contentType: contentType,
        contentLength: len,
      );

      uploaded += len;
      onProgress(uploaded, fileSize);
    }

    // Combine all parts into the final object.
    await _combineChunks(
      token: token,
      dio: dio,
      storagePath: storagePath,
      contentType: contentType,
      totalChunks: totalChunks,
      fileSize: fileSize,
    );
  }

  static int _chunkLen(int index, int chunkSize, int fileSize) {
    final start = index * chunkSize;
    return min(chunkSize, fileSize - start);
  }

  static Future<void> _retryPost({
    required Dio dio,
    required String url,
    required List<int> data,
    required String token,
    required String contentType,
    required int contentLength,
    int maxAttempts = 3,
  }) async {
    DioException? last;
    for (var a = 0; a < maxAttempts; a++) {
      try {
        await dio.post(
          url,
          data: data,
          options: Options(
            headers: {
              'Authorization': 'Bearer $token',
              'apikey': SupabaseConfig.apiKey,
              'Content-Type': contentType,
              'Content-Length': contentLength.toString(),
            },
          ),
        );
        return;
      } on DioException catch (e) {
        last = e;
        if (a < maxAttempts - 1) {
          await Future.delayed(Duration(seconds: (a + 1) * 2));
        }
      }
    }
    throw last!;
  }

  // ---------------------------------------------------------------------------
  // Storage REST helpers (all via SupabaseConfig.url → resolved local IP)
  // ---------------------------------------------------------------------------

  static Future<void> _storageDelete({
    required String token,
    required List<String> paths,
  }) async {
    try {
      final dio = Dio(BaseOptions(
        connectTimeout: const Duration(seconds: 15),
        receiveTimeout: const Duration(seconds: 15),
      ));
      await dio.delete(
        '${SupabaseConfig.url}/storage/v1/object/braindance-assets',
        data: paths,
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'apikey': SupabaseConfig.apiKey,
            'Content-Type': 'application/json',
          },
        ),
      );
    } catch (_) {}
  }

  static Future<Set<int>> _detectExistingChunks({
    required String token,
    required String storagePath,
    required int totalChunks,
  }) async {
    final slash = storagePath.lastIndexOf('/');
    final dir = storagePath.substring(0, slash);
    final filePrefix = '${storagePath.substring(slash + 1)}.part.';
    try {
      final dio = Dio(BaseOptions(
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 10),
      ));
      final res = await dio.post<List<dynamic>>(
        '${SupabaseConfig.url}/storage/v1/object/list/braindance-assets',
        data: {'prefix': '$dir/', 'limit': 1000},
        options: Options(
          responseType: ResponseType.json,
          headers: {
            'Authorization': 'Bearer $token',
            'apikey': SupabaseConfig.apiKey,
          },
        ),
      );
      final found = <int>{};
      for (final item in res.data!) {
        final name = (item as Map<String, dynamic>)['name'] as String;
        if (name.startsWith(filePrefix)) {
          final idx = int.tryParse(name.substring(filePrefix.length));
          if (idx != null && idx >= 0 && idx < totalChunks) found.add(idx);
        }
      }
      return found;
    } catch (_) {
      return {};
    }
  }

  /// Try server-side combine via edge function; fall back to client-side merge.
  static Future<void> _combineChunks({
    required String token,
    required Dio dio,
    required String storagePath,
    required String contentType,
    required int totalChunks,
    required int fileSize,
  }) async {
    // --- Try edge function with generous timeout ---
    final combineDio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 30),
      sendTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(minutes: 5),
    ));
    try {
      await combineDio.post(
        '${SupabaseConfig.url}/functions/v1/combine-chunks',
        data: {
          'bucket': 'braindance-assets',
          'path': storagePath,
          'totalChunks': totalChunks,
        },
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'apikey': SupabaseConfig.apiKey,
            'Content-Type': 'application/json',
          },
        ),
      );
      await _storageDelete(
        token: token,
        paths: List.generate(totalChunks, (i) => '$storagePath.part.$i'),
      );
      return;
    } catch (_) {
      // ignore – fall through to client-side combine
    }

    // --- Client-side fallback: stream chunks to temp file, then upload ---
    final tempFile = File(
      '${Directory.systemTemp.path}/bd_combine_${DateTime.now().millisecondsSinceEpoch}',
    );
    try {
      final sink = tempFile.openWrite();
      try {
        for (var i = 0; i < totalChunks; i++) {
          final partPath = '$storagePath.part.$i';
          final downloadUrl =
              '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$partPath';
          final res = await dio.get<List<int>>(
            downloadUrl,
            options: Options(
              responseType: ResponseType.bytes,
              headers: {
                'Authorization': 'Bearer $token',
                'apikey': SupabaseConfig.apiKey,
              },
            ),
          );
          sink.add(res.data!);
        }
      } finally {
        await sink.close();
      }

      final url =
          '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$storagePath';
      await _streamingPost(
        url: url,
        file: tempFile,
        fileSize: await tempFile.length(),
        contentType: contentType,
        token: token,
        onProgress: (uploadedBytes, totalBytes) {},
      );

      await _storageDelete(
        token: token,
        paths: List.generate(totalChunks, (i) => '$storagePath.part.$i'),
      );
    } finally {
      if (await tempFile.exists()) {
        await tempFile.delete();
      }
    }
  }
}
