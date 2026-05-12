import 'dart:io';

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

class VideoPreprocessor {
  static bool _initialized = false;
  static final _durationRe = RegExp(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})');
  static final _timeRe = RegExp(r'time=(\d{2}):(\d{2}):(\d{2}\.\d{2})');

  static Future<void> ensureInitialized() async {
    if (_initialized) return;
    _initialized = true;
    await FFmpegKitExtended.initialize();
  }

  static String _hevcEncoder() {
    if (Platform.isIOS) return 'hevc_videotoolbox';
    if (Platform.isAndroid) return 'hevc_mediacodec';
    return 'libx265';
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

    final videoEncoder = _hevcEncoder();

    final extraArgs = <String>[];
    if (videoEncoder == 'hevc_videotoolbox') {
      extraArgs.addAll(['-allow_sw', '1', '-realtime', '1']);
    }

    final movflags =
        config.enableFastStart ? '-movflags +faststart' : '';

    final command = [
      '-y',
      '-i', inputFile.path,
      '-vf', _buildFilterChain(config),
      '-c:v', videoEncoder,
      '-b:v', config.videoBitrate,
      '-preset', 'fast',
      ...extraArgs,
      '-c:a', _audioEncoder(),
      '-b:a', config.audioBitrate,
      movflags,
      resolvedOutput.path,
    ].where((s) => s.isNotEmpty).join(' ');

    final startMs = DateTime.now().millisecondsSinceEpoch;
    String? failOutput;
    double? totalDurationSec;

    final session = await FFmpegKit.executeAsync(
      command,
      onComplete: (s) async {
        // completion handled below via session
      },
      onLog: (log) {
        final msg = log.message;
        if (failOutput == null) {
          failOutput = msg;
        } else {
          failOutput = '${failOutput!}\n$msg';
        }

        if (onProgress == null) return;

        // Parse total duration from early log lines: Duration: 00:01:23.45
        if (totalDurationSec == null) {
          final durMatch = _durationRe.firstMatch(msg);
          if (durMatch != null) {
            totalDurationSec = int.parse(durMatch.group(1)!) * 3600.0 +
                int.parse(durMatch.group(2)!) * 60.0 +
                double.parse(durMatch.group(3)!);
          }
        }

        // Parse current encoding time: time=00:00:12.34
        final timeMatch = _timeRe.firstMatch(msg);
        if (timeMatch != null) {
          final currentSec = int.parse(timeMatch.group(1)!) * 3600.0 +
              int.parse(timeMatch.group(2)!) * 60.0 +
              double.parse(timeMatch.group(3)!);
          if (totalDurationSec != null && totalDurationSec! > 0) {
            onProgress((currentSec / totalDurationSec!).clamp(0.0, 1.0));
          }
        }
      },
    );

    final returnCode = session.getReturnCode();
    final elapsed = DateTime.now().millisecondsSinceEpoch - startMs;

    if (ReturnCode.isSuccess(returnCode)) {
      final outputSize =
          await resolvedOutput.exists() ? await resolvedOutput.length() : 0;
      return VideoPreprocessResult(
        outputFile: resolvedOutput,
        durationMs: elapsed,
        inputSizeBytes: inputSize,
        outputSizeBytes: outputSize,
      );
    }

    throw Exception(
      'FFmpeg preprocessing failed:\n${failOutput ?? session.getFailStackTrace() ?? "unknown error"}',
    );
  }

  static String _buildFilterChain(VideoPreprocessConfig config) {
    final parts = <String>['fps=${config.targetFps}'];
    if (config.maxHeight > 0) {
      parts.add('scale=-2:${config.maxHeight}');
    }
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
          ?.map((s) => {
                'type': s.type,
                'codec': s.codec,
                'width': s.width,
                'height': s.height,
                'bitrate': s.bitrate,
              })
          .toList(),
    };
  }
}
