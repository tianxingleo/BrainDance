import 'dart:io';
import 'dart:async';
import 'package:flutter/foundation.dart';

import 'package:ffmpeg_kit_extended_flutter/ffmpeg_kit_extended_flutter.dart';
import 'package:path_provider/path_provider.dart';

class VideoPreprocessConfig {
  final int targetFps;
  final int maxHeight;
  final String videoBitrate;
  final String audioBitrate;
  final bool enableFastStart;

  const VideoPreprocessConfig({
    this.targetFps = 5,
    this.maxHeight = 1080,
    this.videoBitrate = '2M',
    this.audioBitrate = '96k',
    this.enableFastStart = true,
  });

  VideoPreprocessConfig copyWith({
    int? targetFps,
    int? maxHeight,
    String? videoBitrate,
    String? audioBitrate,
    bool? enableFastStart,
  }) {
    return VideoPreprocessConfig(
      targetFps: targetFps ?? this.targetFps,
      maxHeight: maxHeight ?? this.maxHeight,
      videoBitrate: videoBitrate ?? this.videoBitrate,
      audioBitrate: audioBitrate ?? this.audioBitrate,
      enableFastStart: enableFastStart ?? this.enableFastStart,
    );
  }
}

class VideoPreprocessResult {
  final File outputFile;
  final int durationMs;
  final int inputSizeBytes;
  final int outputSizeBytes;

  const VideoPreprocessResult({
    required this.outputFile,
    required this.durationMs,
    required this.inputSizeBytes,
    required this.outputSizeBytes,
  });

  double get compressionRatio =>
      inputSizeBytes > 0 ? outputSizeBytes / inputSizeBytes : 1.0;

  String get savedBytesFormatted {
    final saved = inputSizeBytes - outputSizeBytes;
    if (saved <= 0) return '0 B';
    if (saved < 1024) return '$saved B';
    if (saved < 1024 * 1024) {
      return '${(saved / 1024).toStringAsFixed(1)} KB';
    }
    return '${(saved / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
}

class VideoPreprocessCancelledException implements Exception {
  final String? message;
  const VideoPreprocessCancelledException([this.message]);
  @override
  String toString() => message ?? 'Preprocessing cancelled';
}

class VideoPreprocessor {
  static bool _initialized = false;
  static final _durationRe = RegExp(
    r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})',
  );
  static final _timeRe = RegExp(r'time=(\d{2}):(\d{2}):(\d{2}\.\d{2})');
  static final _outTimeMsRe = RegExp(r'out_time_ms=(\d+)');

  static Future<void> ensureInitialized() async {
    if (_initialized) return;
    _initialized = true;
    await FFmpegKitExtended.initialize();
  }

  /// Returns encoders in priority order. First success wins.
  static List<String> _videoEncoders() {
    if (Platform.isIOS)
      return ['hevc_videotoolbox', 'h264_videotoolbox', 'libx265'];
    if (Platform.isAndroid)
      return ['hevc_mediacodec', 'h264_mediacodec', 'libx264'];
    return ['libx265'];
  }

  static bool _isSoftwareEncoder(String encoder) {
    return encoder == 'libx265' || encoder == 'libx264';
  }

  static String _audioEncoder() {
    if (Platform.isIOS) return 'aac_at';
    return 'aac';
  }

  /// Preprocess a video file with the given config.
  ///
  /// If [outputFile] is not provided, a temp file is created automatically.
  /// The caller should delete the output file when no longer needed.
  static Future<VideoPreprocessResult> preprocess(
    File inputFile, {
    VideoPreprocessConfig config = const VideoPreprocessConfig(),
    void Function(double progress)? onProgress,
    File? outputFile,
    Completer<void>? cancelSignal,
  }) async {
    await ensureInitialized();

    if (!await inputFile.exists()) {
      throw Exception('Input video file not found: ${inputFile.path}');
    }

    final inputSize = await inputFile.length();

    final File resolvedOutput;
    if (outputFile != null) {
      resolvedOutput = outputFile;
    } else {
      final tempDir = await getTemporaryDirectory();
      final outputName =
          'preprocessed_${DateTime.now().millisecondsSinceEpoch}.mp4';
      resolvedOutput = File('${tempDir.path}/$outputName');
    }

    final encoders = _videoEncoders();
    String? lastError;

    for (final videoEncoder in encoders) {
      if (cancelSignal?.isCompleted == true) {
        throw const VideoPreprocessCancelledException();
      }
      try {
        return await _tryEncode(
          inputFile: inputFile,
          outputFile: resolvedOutput,
          encoder: videoEncoder,
          config: config,
          onProgress: onProgress,
          inputSize: inputSize,
          cancelSignal: cancelSignal,
        );
      } on VideoPreprocessCancelledException {
        rethrow;
      } catch (e) {
        lastError = e.toString();
        debugPrint(
          '[VideoPreprocessor] encoder $videoEncoder failed: $lastError',
        );
        if (await resolvedOutput.exists()) {
          try {
            await resolvedOutput.delete();
          } catch (_) {}
        }
      }
    }

    throw Exception(
      'FFmpeg preprocessing failed with all encoders:\n${lastError ?? 'unknown error'}',
    );
  }

  static Future<VideoPreprocessResult> _tryEncode({
    required File inputFile,
    required File outputFile,
    required String encoder,
    required VideoPreprocessConfig config,
    required void Function(double)? onProgress,
    required int inputSize,
    Completer<void>? cancelSignal,
  }) async {
    final isSoftware = _isSoftwareEncoder(encoder);

    final extraArgs = <String>[];
    if (encoder == 'hevc_videotoolbox') {
      extraArgs.addAll(['-allow_sw', '1', '-realtime', '1']);
    }

    final commandParts = <String>[
      '-y',
      '-i',
      '"${inputFile.path}"',
      '-vf',
      _buildFilterChain(config),
      '-c:v',
      encoder,
      '-b:v',
      config.videoBitrate,
      if (isSoftware) ...['-preset', 'fast'],
      ...extraArgs,
      '-c:a',
      _audioEncoder(),
      '-b:a',
      config.audioBitrate,
      if (config.enableFastStart) ...['-movflags', '+faststart'],
      '"${outputFile.path}"',
    ];

    final completer = Completer<Session>();
    final logBuf = StringBuffer();
    const maxLogLen = 8192;
    double? totalDurationSec;

    final startMs = DateTime.now().millisecondsSinceEpoch;

    FFmpegKit.executeAsync(
      commandParts.join(' '),
      onComplete: (s) {
        if (!completer.isCompleted) completer.complete(s);
      },
      onLog: (log) {
        final msg = log.message;
        if (logBuf.length < maxLogLen) {
          logBuf.writeln(msg);
        }

        if (onProgress == null) return;

        if (totalDurationSec == null) {
          final durMatch = _durationRe.firstMatch(msg);
          if (durMatch != null) {
            totalDurationSec =
                int.parse(durMatch.group(1)!) * 3600.0 +
                int.parse(durMatch.group(2)!) * 60.0 +
                double.parse(durMatch.group(3)!);
          }
        }

        if (totalDurationSec != null && totalDurationSec! > 0) {
          double? currentSec;
          final timeMatch = _timeRe.firstMatch(msg);
          if (timeMatch != null) {
            currentSec =
                int.parse(timeMatch.group(1)!) * 3600.0 +
                int.parse(timeMatch.group(2)!) * 60.0 +
                double.parse(timeMatch.group(3)!);
          } else {
            final outMatch = _outTimeMsRe.firstMatch(msg);
            if (outMatch != null) {
              currentSec = int.parse(outMatch.group(1)!) / 1000000.0;
            }
          }
          if (currentSec != null) {
            onProgress((currentSec / totalDurationSec!).clamp(0.0, 1.0));
          }
        }
      },
    );

    Future<Session> waitSession() async {
      if (cancelSignal != null) {
        await Future.any([completer.future, cancelSignal.future]);
        if (cancelSignal.isCompleted) {
          FFmpegKitExtended.cancelAllSessions();
          try {
            await completer.future.timeout(const Duration(seconds: 3));
          } catch (_) {}
          throw const VideoPreprocessCancelledException();
        }
      }
      return completer.future.timeout(
        const Duration(minutes: 20),
        onTimeout: () {
          FFmpegKitExtended.cancelAllSessions();
          throw TimeoutException('FFmpeg encoding timed out after 20 minutes');
        },
      );
    }

    final session = await waitSession();
    final returnCode = session.getReturnCode();
    final elapsed = DateTime.now().millisecondsSinceEpoch - startMs;

    if (ReturnCode.isSuccess(returnCode)) {
      final outputSize = await outputFile.exists()
          ? await outputFile.length()
          : 0;
      return VideoPreprocessResult(
        outputFile: outputFile,
        durationMs: elapsed,
        inputSizeBytes: inputSize,
        outputSizeBytes: outputSize,
      );
    }

    try {
      session.cancel();
    } catch (_) {}

    if (cancelSignal?.isCompleted == true) {
      throw const VideoPreprocessCancelledException();
    }

    throw Exception(logBuf.toString());
  }

  static String _buildFilterChain(VideoPreprocessConfig config) {
    final parts = <String>['fps=${config.targetFps}'];
    if (config.maxHeight > 0) {
      parts.add(
        'scale=w=-2:h=${config.maxHeight}:force_original_aspect_ratio=decrease',
      );
    }
    parts.add('format=yuv420p');
    return parts.join(',');
  }

  /// Quick probe: get video metadata without transcoding.
  static Future<Map<String, dynamic>?> probe(File inputFile) async {
    await ensureInitialized();
    if (!await inputFile.exists()) return null;

    final session = await FFprobeKit.getMediaInformationAsync(
      inputFile.path,
      onComplete: (_) {},
    );

    final info = session.getMediaInformation();
    if (info == null) return null;

    return {
      'duration': info.duration,
      'format': info.format,
      'bitrate': info.bitrate,
      'streams': info.streams
          .map(
            (s) => {
              'type': s.type,
              'codec': s.codec,
              'width': s.width,
              'height': s.height,
              'bitrate': s.bitrate,
            },
          )
          .toList(),
    };
  }
}
