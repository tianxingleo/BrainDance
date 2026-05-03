import 'dart:convert';
import 'dart:io';

import 'package:braindance/configs/app_config.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';

import '../services/download_event_bus.dart';
import '../configs/reco_config.dart';

// ============================================================
// Dev/prod mode notes
// - Debug mode: connect to local Vite dev server
// - Production mode: serve bundled files from assets/webgl/
// ============================================================
class WebGLViewerPage extends StatefulWidget {
  final String initialModelUrl;
  final String? posesUrl;
  final String sceneId;
  final List<double>? initialPose;
  final String? initialPoseId;
  final bool useSparkViewer;
  final bool initialMarkerArMode;
  final List<Map<String, dynamic>> timePeelingModels;

  const WebGLViewerPage({
    super.key,
    this.initialModelUrl = './models/scene_auto_sync_raw.ply',
    this.posesUrl,
    this.sceneId = '3DGS Viewer',
    this.initialPose,
    this.initialPoseId,
    this.useSparkViewer = true,
    this.initialMarkerArMode = false,
    this.timePeelingModels = const [],
  });

  @override
  State<WebGLViewerPage> createState() => _WebGLViewerPageState();
}

class _WebGLViewerPageState extends State<WebGLViewerPage> {
  static const int _downloadProgressThrottleMs = 100;
  static const String _defaultMarkerTargetPath = '/targets/braindance-card.mind';
  static const String _markerArDebugVersion = 'marker-ar-debug-20260503-3';
  WebViewController? _controller;
  bool _isWebReady = false;
  bool _isUnsupportedPlatform = false;
  bool _useExternalBrowserMode = false;
  bool _isOpeningExternalViewer = false;
  bool _didAttemptExternalOpen = false;
  String? _externalViewerUrl;
  String? _lastLoadedViewerUrl;
  HttpServer? _localServer;
  int _localPort = 0;
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  int _downloadedBytes = 0;
  int _totalBytes = -1;
  String? _localModelPath;
  bool _downloadCancelled = false;
  int _lastDownloadUiUpdate = 0;
  late bool _useSparkViewer;
  late bool _isMarkerArMode;

  Future<bool> _ensureRuntimeCameraPermission() async {
    // Android WebView getUserMedia 依赖宿主 App 已拿到运行时 CAMERA 权限。
    // 这里复用现有相机初始化链路主动触发系统授权，再立即释放，避免占用摄像头。
    try {
      final granted = await RecoConfig.cameraInitialize();
      RecoConfig.disposeCamera();
      return granted;
    } catch (error) {
      debugPrint('Request runtime camera permission failed: $error');
      return false;
    }
  }

  void _attachViewerHeaders(HttpResponse response) {
    response.headers.add('Access-Control-Allow-Origin', '*');
    response.headers.add('Cross-Origin-Opener-Policy', 'same-origin');
    response.headers.add('Cross-Origin-Embedder-Policy', 'require-corp');
    response.headers.add('Cross-Origin-Resource-Policy', 'cross-origin');
  }

  String get _viewerAssetRoot =>
      _useSparkViewer ? 'assets/webgl_spark' : 'assets/webgl';

  String get _viewerLabel => _isMarkerArMode
      ? 'Marker AR'
      : (_useSparkViewer ? 'Spark' : '\u539f\u7248');

  @override
  void initState() {
    super.initState();
    _useSparkViewer = widget.useSparkViewer;
    _isMarkerArMode = widget.initialMarkerArMode;
    if (_isMarkerArMode) {
      _useSparkViewer = true;
    }

    // Flutter Web is not supported here. Desktop uses the external browser mode.
    if (kIsWeb) {
      _isUnsupportedPlatform = true;
    } else if (defaultTargetPlatform == TargetPlatform.windows ||
        defaultTargetPlatform == TargetPlatform.linux ||
        defaultTargetPlatform == TargetPlatform.macOS) {
      _useExternalBrowserMode = true;
      _startLocalServer().then((_) {
        if (mounted) _prepareModelAndLoad();
      });
    } else {
      try {
        _startLocalServer().then((_) {
          if (mounted) _prepareModelAndLoad();
        });
      } catch (e) {
        debugPrint('WebView initialization failed: $e');
        _isUnsupportedPlatform = true;
      }
    }
  }

  @override
  void dispose() {
    _downloadCancelled = true;
    _localServer?.close();
    super.dispose();
  }

  void _stopDownload() {
    _downloadCancelled = true;
    if (mounted) {
      setState(() {
        _isDownloading = false;
      });
    }
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '${bytes}B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(0)}KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)}MB';
  }

  Future<void> _startLocalServer() async {
    _localServer = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _localPort = _localServer!.port;
    _localServer!.listen((HttpRequest request) async {
      var path = request.uri.path;

      // /proxy/: proxy HTTPS requests through Dart to avoid WebView SSL issues.
      if (path.startsWith('/proxy/')) {
        final encodedTarget = path.substring('/proxy/'.length);
        final targetUrl = Uri.decodeComponent(encodedTarget);
        try {
          final proxyClient = HttpClient()
            ..badCertificateCallback = (cert, host, port) => true;
          final proxyUri = Uri.parse(targetUrl);
          final proxyReq = await proxyClient.getUrl(proxyUri);
          proxyReq.headers.set('User-Agent', 'BrainDance/1.0 Flutter');
          final proxyResp = await proxyReq.close();
          request.response.statusCode = proxyResp.statusCode;
          _attachViewerHeaders(request.response);
          final ct = proxyResp.headers.contentType;
          if (ct != null) request.response.headers.contentType = ct;
          await request.response.addStream(proxyResp);
        } catch (e) {
          request.response.statusCode = HttpStatus.badGateway;
          request.response.write('Proxy error: $e');
        }
        await request.response.close();
        return;
      }

      // /local_models/: serve already-downloaded local model files.
      if (path.startsWith('/local_models/')) {
        final filePath = Uri.decodeComponent(
          path.substring('/local_models/'.length),
        );
        final file = File(filePath);
        if (await file.exists()) {
          _attachViewerHeaders(request.response);

          if (filePath.endsWith('.ply') ||
              filePath.endsWith('.splat') ||
              filePath.endsWith('.ksplat') ||
              filePath.endsWith('.spz') ||
              filePath.endsWith('.rad') ||
              filePath.endsWith('.sog')) {
            request.response.headers.contentType = ContentType(
              'application',
              'octet-stream',
            );
          }

          // ── HTTP Range support for .rad / .spz streaming ──
          final fileLength = await file.length();
          final rangeHeader = request.headers.value('range');

          if (rangeHeader != null && rangeHeader.startsWith('bytes=')) {
            // Parse "bytes=START-END" (END is optional)
            final rangeSpec = rangeHeader.substring(6);
            final parts = rangeSpec.split('-');

            int start = 0;
            int end = fileLength - 1;

            if (parts.length == 2) {
              if (parts[0].isNotEmpty) {
                start = int.tryParse(parts[0]) ?? 0;
              }
              if (parts[1].isNotEmpty) {
                end = int.tryParse(parts[1]) ?? (fileLength - 1);
              }
            }

            // Clamp range
            if (start < 0) start = 0;
            if (end >= fileLength) end = fileLength - 1;
            if (start > end) start = end;

            final contentLength = end - start + 1;

            request.response.statusCode = HttpStatus.partialContent;
            request.response.headers.set(
              'Content-Range',
              'bytes $start-$end/$fileLength',
            );
            request.response.headers.set('Accept-Ranges', 'bytes');
            request.response.contentLength = contentLength;

            // Read only the requested byte range
            final raf = await file.open(mode: FileMode.read);
            try {
              await raf.setPosition(start);
              var remaining = contentLength;
              const chunkSize = 64 * 1024; // 64KB chunks
              while (remaining > 0) {
                final toRead = remaining > chunkSize ? chunkSize : remaining;
                final buffer = await raf.read(toRead);
                request.response.add(buffer);
                remaining -= buffer.length;
                if (buffer.length < toRead) break;
              }
            } finally {
              await raf.close();
            }
          } else {
            // No Range header — serve full file with Accept-Ranges hint
            request.response.headers.set('Accept-Ranges', 'bytes');
            request.response.contentLength = fileLength;
            await request.response.addStream(file.openRead());
          }

          await request.response.close();
          return;
        }
        request.response.statusCode = HttpStatus.notFound;
        await request.response.close();
        return;
      }

      if (path == '/' || path.isEmpty) {
        path = '/index.html';
      }

      final assetPath = '$_viewerAssetRoot$path';
      if (path == '/index.html' ||
          path.endsWith('.js') ||
          path.endsWith('.mind')) {
        debugPrint('Viewer asset request: $assetPath');
      }

      try {
        final data = await rootBundle.load(assetPath);
        final bytes = data.buffer.asUint8List();

        var contentType = 'text/plain';
        if (path.endsWith('.html')) {
          contentType = 'text/html; charset=utf-8';
        } else if (path.endsWith('.js')) {
          contentType = 'application/javascript; charset=utf-8';
        } else if (path.endsWith('.css')) {
          contentType = 'text/css; charset=utf-8';
        } else if (path.endsWith('.png')) {
          contentType = 'image/png';
        } else if (path.endsWith('.ico')) {
          contentType = 'image/x-icon';
        } else if (path.endsWith('.mind')) {
          contentType = 'application/octet-stream';
        } else if (path.endsWith('.ply') ||
            path.endsWith('.splat') ||
            path.endsWith('.ksplat')) {
          contentType = 'application/octet-stream';
        }

        request.response.headers.contentType = ContentType.parse(contentType);
        request.response.headers.set(
          'Cache-Control',
          'no-store, no-cache, must-revalidate, max-age=0',
        );
        request.response.headers.set('Pragma', 'no-cache');
        request.response.headers.set('Expires', '0');
        _attachViewerHeaders(request.response);
        request.response.add(bytes);
        await request.response.close();
      } catch (_) {
        debugPrint('Viewer asset missing: $assetPath');
        request.response.statusCode = HttpStatus.notFound;
        request.response.write('Not Found');
        await request.response.close();
      }
    });
  }

  Future<void> _prepareModelAndLoad() async {
    final originalUrl = widget.initialModelUrl;

    if (originalUrl.startsWith('http://') ||
        originalUrl.startsWith('https://')) {
      try {
        final encodedUrl = Uri.encodeFull(Uri.decodeFull(originalUrl));
        final uri = Uri.parse(encodedUrl);
        final requestPath = uri.path;
        final sanitizedFileName = requestPath
            .replaceAll('/', '_')
            .replaceAll('\\', '_');

        final dir = await getApplicationDocumentsDirectory();
        final localFile = File('${dir.path}/$sanitizedFileName');

        if (await localFile.exists()) {
          debugPrint('Using cached offline model: ${localFile.path}');
          _localModelPath = localFile.path;
          if (mounted) _launchViewer();
        } else {
          final tmpFile = File('${localFile.path}.tmp');

          int existingBytes = 0;
          if (await tmpFile.exists()) {
            existingBytes = await tmpFile.length();
            debugPrint('Resuming download from byte $existingBytes');
          }

          debugPrint('Starting download from: $originalUrl');
          setState(() {
            _isDownloading = true;
            _downloadProgress = 0.0;
            _downloadedBytes = 0;
            _totalBytes = -1;
          });

          final client = HttpClient()
            ..badCertificateCallback = (cert, host, port) => true;
          final request = await client.getUrl(uri);
          request.headers.set('User-Agent', 'BrainDance/1.0 Flutter');
          if (existingBytes > 0) {
            request.headers.set('Range', 'bytes=$existingBytes-');
          }
          final response = await request.close();

          if (response.statusCode == 200 && existingBytes > 0) {
            debugPrint('Server does not support Range, restarting download');
            if (await tmpFile.exists()) await tmpFile.delete();
            existingBytes = 0;
          } else if (response.statusCode != 200 && response.statusCode != 206) {
            var errorBody = '';
            try {
              errorBody = await response.transform(utf8.decoder).join();
            } catch (_) {}
            throw Exception('HTTP Error ${response.statusCode}: $errorBody');
          }

          final contentLength = response.contentLength;
          final totalBytes = contentLength > 0
              ? contentLength + existingBytes
              : -1;
          var receivedBytes = existingBytes;

          final metaFile = File('${localFile.path}.meta');
          if (totalBytes > 0) {
            await metaFile.writeAsString('$totalBytes');
          }

          final sink = tmpFile.openWrite(
            mode: existingBytes > 0 ? FileMode.append : FileMode.write,
          );

          if (mounted) {
            setState(() {
              _downloadedBytes = receivedBytes;
              _totalBytes = totalBytes;
              _downloadProgress = totalBytes > 0
                  ? receivedBytes / totalBytes
                  : 0.0;
            });
          }

          try {
            await for (final chunk in response) {
              if (_downloadCancelled) {
                await sink.close();
                downloadEventBus.add(
                  ModelDownloadEvent(
                    url: originalUrl,
                    progress: _downloadProgress,
                    downloadedBytes: receivedBytes,
                    totalBytes: totalBytes,
                    isCancelled: true,
                  ),
                );
                return;
              }
              sink.add(chunk);
              receivedBytes += chunk.length;
              if (totalBytes > 0) {
                final progress = receivedBytes / totalBytes;
                _updateDownloadProgress(
                  progress: progress,
                  downloadedBytes: receivedBytes,
                  totalBytes: totalBytes,
                );
                downloadEventBus.add(
                  ModelDownloadEvent(
                    url: originalUrl,
                    progress: progress,
                    downloadedBytes: receivedBytes,
                    totalBytes: totalBytes,
                  ),
                );
              }
            }
            await sink.close();

            await tmpFile.rename(localFile.path);
            if (await metaFile.exists()) await metaFile.delete();
            debugPrint('Download complete: ${localFile.path}');
            _localModelPath = localFile.path;
            downloadEventBus.add(
              ModelDownloadEvent(
                url: originalUrl,
                progress: 1.0,
                isComplete: true,
                downloadedBytes: receivedBytes,
                totalBytes: totalBytes,
              ),
            );
            if (mounted) {
              setState(() {
                _isDownloading = false;
              });
              _initWebView();
            }
          } catch (e) {
            await sink.close();
            debugPrint('Download interrupted, tmp file preserved for resume');
            rethrow;
          }
        }
      } catch (e) {
        debugPrint('Download error: $e');
        if (mounted) {
          setState(() {
            _isDownloading = false;
          });
          TDToast.showText(
            '\u4e0b\u8f7d\u6a21\u578b\u5931\u8d25: $e',
            context: context,
          );
          _launchViewer();
        }
      }
    } else {
      if (mounted) _launchViewer();
    }
  }

  void _updateDownloadProgress({
    required double progress,
    required int downloadedBytes,
    required int totalBytes,
    bool force = false,
  }) {
    final now = DateTime.now().millisecondsSinceEpoch;
    if (!force &&
        now - _lastDownloadUiUpdate < _downloadProgressThrottleMs &&
        downloadedBytes < totalBytes) {
      return;
    }
    _lastDownloadUiUpdate = now;
    if (!mounted) {
      return;
    }
    setState(() {
      _downloadProgress = progress;
      _downloadedBytes = downloadedBytes;
      _totalBytes = totalBytes;
    });
  }

  void _launchViewer() {
    if (_useExternalBrowserMode) {
      _openInExternalBrowser();
      return;
    }
    _initWebView();
  }

  /// Resolve the best model URL for Spark 2.0.
  /// Priority: .rad > .spz > .ksplat > .ply
  /// Checks the local cache for optimized formats before falling back.
  String _resolveBestModelUrl(String originalUrl) {
    if (!_useSparkViewer) return originalUrl;

    // If local model is cached, check for optimized format variants
    if (_localModelPath != null) {
      final baseName = _localModelPath!.replaceAll(
        RegExp(r'\.(ply|splat|ksplat|spz|rad|sog)$'),
        '',
      );

      // Priority order: .rad > .spz > .ksplat > original
      const extensions = ['.rad', '.spz', '.ksplat'];
      for (final ext in extensions) {
        final candidate = File('$baseName$ext');
        if (candidate.existsSync()) {
          debugPrint('Using optimized model format: $ext');
          return 'http://127.0.0.1:$_localPort/local_models/${Uri.encodeComponent(candidate.path)}';
        }
      }
    }

    return originalUrl;
  }

  Map<String, dynamic> _buildViewerPayload() {
    String rawUrl;
    if (_localModelPath != null) {
      rawUrl =
          'http://127.0.0.1:$_localPort/local_models/${Uri.encodeComponent(_localModelPath!)}';
    } else if (widget.initialModelUrl.startsWith('http://') ||
        widget.initialModelUrl.startsWith('https://')) {
      rawUrl =
          'http://127.0.0.1:$_localPort/proxy/${Uri.encodeComponent(widget.initialModelUrl)}';
    } else {
      rawUrl = widget.initialModelUrl;
    }

    // Try to resolve a better format for Spark 2.0
    final targetUrl = _resolveBestModelUrl(rawUrl);

    return {
      'ply': targetUrl,
      if (widget.posesUrl != null && widget.posesUrl!.isNotEmpty)
        'poses': widget.posesUrl,
      if (widget.initialPose != null) 'matrix': widget.initialPose,
      if (widget.initialPoseId != null && widget.initialPoseId!.isNotEmpty)
        'imageId': widget.initialPoseId,
    };
  }

  Future<void> _openInExternalBrowser() async {
    if (_isMarkerArMode) {
      final cacheBust = DateTime.now().millisecondsSinceEpoch;
      final url = _buildViewerUrl(cacheBust);
      _externalViewerUrl = url;

      if (_didAttemptExternalOpen) {
        if (mounted) setState(() {});
        return;
      }

      _didAttemptExternalOpen = true;
      if (mounted) {
        setState(() {
          _isOpeningExternalViewer = true;
        });
      }

      try {
        await _openUrlOnDesktop(url);
      } catch (e) {
        debugPrint('Open external Marker AR viewer failed: $e');
      } finally {
        if (mounted) {
          setState(() {
            _isOpeningExternalViewer = false;
          });
        }
      }
      return;
    }

    final payload = _buildViewerPayload();
    final encodedPayload = Uri.encodeComponent(jsonEncode(payload));
    final url =
        'http://127.0.0.1:$_localPort/index.html?payload=$encodedPayload';
    _externalViewerUrl = url;

    if (_didAttemptExternalOpen) {
      if (mounted) setState(() {});
      return;
    }

    _didAttemptExternalOpen = true;
    if (mounted) {
      setState(() {
        _isOpeningExternalViewer = true;
      });
    }

    try {
      await _openUrlOnDesktop(url);
    } catch (e) {
      debugPrint('Open external viewer failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isOpeningExternalViewer = false;
        });
      }
    }
  }

  Future<void> _openUrlOnDesktop(String url) async {
    if (defaultTargetPlatform == TargetPlatform.windows) {
      await Process.start('cmd', ['/c', 'start', '', url], runInShell: true);
      return;
    }
    if (defaultTargetPlatform == TargetPlatform.macOS) {
      await Process.start('open', [url], runInShell: true);
      return;
    }
    if (defaultTargetPlatform == TargetPlatform.linux) {
      await Process.start('xdg-open', [url], runInShell: true);
      return;
    }
    throw UnsupportedError('Unsupported desktop platform');
  }

  void _initWebView() {
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0x00000000))
      ..addJavaScriptChannel(
        'BrainDanceChannel',
        onMessageReceived: (JavaScriptMessage message) {
          debugPrint('BrainDanceChannel: ${message.message}');
          final dynamic decoded;
          try {
            decoded = jsonDecode(message.message);
          } catch (error) {
            debugPrint('BrainDanceChannel parse error: $error');
            return;
          }
          if (decoded is! Map<String, dynamic>) {
            debugPrint('BrainDanceChannel ignored non-object payload');
            return;
          }
          final data = decoded;
          if (data['status'] == 'ready') {
            setState(() => _isWebReady = true);
            if (!_isMarkerArMode) {
              _sendModelToVue();
            }
          } else if (data['action'] == 'switchModel') {
            _handleSwitchModel(data);
          } else if (data['status'] == 'error') {
            if (mounted) {
              TDToast.showText(
                'Spark \u9519\u8bef: ${data['msg']}',
                context: context,
              );
            }
          } else if (data['status'] == 'info') {
            debugPrint('Spark info: ${data['msg']}');
          }
        },
      )
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (String url) {
            debugPrint('WebView page started: $url marker=$_isMarkerArMode');
          },
          onPageFinished: (String url) {
            debugPrint('WebView page finished: $url marker=$_isMarkerArMode');
            // 注入 overscroll 防护脚本：在 Android/iOS WebView 层面强化禁用弹性边界效果，
            // 防止拖动 WebGL canvas 时出现浏览器特征的画面拉伸（overscroll bounce）。
            // canvas 上的 touchmove 已由 GestureHandler（gestures.ts）调用 e.preventDefault() 处理，
            // 此处只需确保 html/body 的 overscroll 被彻底关闭。
            _controller?.runJavaScript('''
              (function() {
                var html = document.documentElement;
                var body = document.body;
                html.style.overflow = 'hidden';
                body.style.overflow = 'hidden';
                html.style.overscrollBehavior = 'none';
                body.style.overscrollBehavior = 'none';
                html.style.touchAction = 'none';
                body.style.touchAction = 'none';
              })();
            ''');

            Future.delayed(const Duration(seconds: 2), () {
              if (!_isWebReady && mounted) {
                debugPrint(
                  'WebView: no ready signal received, triggering manually ($_markerArDebugVersion)',
                );
                debugPrint(
                  'WebView fallback detail: marker=$_isMarkerArMode url=$_lastLoadedViewerUrl',
                );
                setState(() => _isWebReady = true);
                if (!_isMarkerArMode) {
                  _sendModelToVue();
                }
              }
            });
          },
          onWebResourceError: (WebResourceError error) {
            debugPrint('WebView error: ${error.description}');
          },
        ),
      );

    final platformController = _controller?.platform;
    if (platformController is AndroidWebViewController) {
      platformController.setMediaPlaybackRequiresUserGesture(false);
      platformController.setOnConsoleMessage((message) {
        debugPrint('WebView console [${message.level.name}]: ${message.message}');
      });
      platformController.setOnPlatformPermissionRequest((request) {
        if (!_isMarkerArMode) {
          request.deny();
          return;
        }

        // Marker AR 运行在内置 WebView 中，需要把网页 camera 权限请求
        // 明确转交给 Android WebView，否则 MindAR 无法启动摄像头。
        request.grant();
      });
    }

    if (mounted) setState(() {});
    _loadLocalHtml();
  }

  Future<void> _loadLocalHtml() async {
    try {
      // 本地 WebGL viewer 文件名和内容会频繁更新，进入页面时强制清理 WebView 缓存，
      // 避免 Android WebView 继续执行旧的构建产物导致相机修复看起来没有生效。
      await _controller?.clearCache();
      final cacheBust = DateTime.now().millisecondsSinceEpoch;
      final url = _buildViewerUrl(cacheBust);
      _lastLoadedViewerUrl = url;
      debugPrint('Loading WebGL viewer URL: $url');
      await _controller?.loadRequest(Uri.parse(url));
    } catch (e) {
      debugPrint('Error loading HTML via local server: $e');
    }
  }

  String _buildViewerUrl(int cacheBust) {
    if (_isMarkerArMode) {
      final payload = _buildViewerPayload();
      return Uri(
        scheme: 'http',
        host: '127.0.0.1',
        port: _localPort,
        path: '/index.html',
        queryParameters: {
          'mode': 'marker-ar',
          'model': payload['ply']?.toString() ?? '',
          'target': _defaultMarkerTargetPath,
          'v': '$_markerArDebugVersion-$cacheBust',
        },
      ).toString();
    }

    return 'http://127.0.0.1:$_localPort/index.html?v=rollfix-$cacheBust';
  }

  void _sendModelToVue() {
    if (_isMarkerArMode) return;
    if (!_isWebReady) return;
    final payloadData = _buildViewerPayload();
    final targetUrl = payloadData['ply'];

    debugPrint('Sending model URL to WebView: $targetUrl');
    final payload = jsonEncode(payloadData);
    _controller?.runJavaScript("window.loadModelFromFlutter($payload)");

    // 发送 TimePeeling 模型列表（无条件发送，空列表时 JS 端显示空态）
    _sendTimePeelingList();
  }

  void _sendTimePeelingList() {
    final models = widget.timePeelingModels;

    // 如果没有兄弟模型数据，用当前模型参数构造单元素列表
    if (models.isEmpty) {
      final currentModelUrl = widget.initialModelUrl;
      String plyUrl = '';
      if (currentModelUrl.startsWith('http://') ||
          currentModelUrl.startsWith('https://')) {
        plyUrl =
            'http://127.0.0.1:$_localPort/proxy/${Uri.encodeComponent(currentModelUrl)}';
      }
      final fallbackList = [
        {
          'id': widget.sceneId,
          'name': widget.sceneId,
          'ply': plyUrl,
          'poses': widget.posesUrl ?? '',
          'previewImg': '',
          'createdAt': '',
        },
      ];
      final json = jsonEncode(fallbackList);
      debugPrint('Sending TimePeeling list (1 fallback model) to WebView');
      _controller?.runJavaScript(
        "window.setModelListForTimePeeling($json, '${widget.sceneId}')",
      );
      return;
    }

    final list = models.map((model) {
      final plyPath = model['ply_path'] as String? ?? '';
      String plyUrl = '';
      if (plyPath.isNotEmpty) {
        // 通过本地代理访问远程模型
        try {
          final publicUrl = Supabase.instance.client.storage
              .from('braindance-assets')
              .getPublicUrl(plyPath);
          plyUrl =
              'http://127.0.0.1:$_localPort/proxy/${Uri.encodeComponent(publicUrl)}';
        } catch (_) {}
      }

      String? posesUrl;
      if (plyPath.isNotEmpty) {
        final posesPath = plyPath.replaceAll(
          RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
          'webgl_poses.json',
        );
        if (posesPath != plyPath) {
          try {
            final publicPosesUrl = Supabase.instance.client.storage
                .from('braindance-assets')
                .getPublicUrl(posesPath);
            posesUrl = publicPosesUrl;
          } catch (_) {}
        }
      }

      final previewPath = model['preview_img_path']?.toString() ?? '';
      String previewUrl = '';
      if (previewPath.isNotEmpty) {
        try {
          // preview_img_path 可能已经是完整 URL 或 storage path
          String fullUrl;
          if (previewPath.startsWith('http://') ||
              previewPath.startsWith('https://')) {
            fullUrl = previewPath;
          } else {
            fullUrl = Supabase.instance.client.storage
                .from('braindance-assets')
                .getPublicUrl(previewPath);
          }
          previewUrl =
              'http://127.0.0.1:$_localPort/proxy/${Uri.encodeComponent(fullUrl)}';
        } catch (_) {}
      }

      final displayName =
          model['display_name']?.toString() ??
          model['scene_id']?.toString() ??
          '';

      return {
        'id': model['id']?.toString() ?? model['scene_id']?.toString() ?? '',
        'name': displayName,
        'ply': plyUrl,
        'poses': posesUrl ?? '',
        'previewImg': previewUrl,
        'createdAt': model['created_at']?.toString() ?? '',
      };
    }).toList();

    final json = jsonEncode(list);
    // 找出当前加载的模型 ID
    final currentModelId = _findCurrentModelId();
    debugPrint('Sending TimePeeling list (${list.length} models) to WebView');
    _controller?.runJavaScript(
      "window.setModelListForTimePeeling($json, '$currentModelId')",
    );
  }

  String _findCurrentModelId() {
    final currentPly = widget.initialModelUrl;
    for (final model in widget.timePeelingModels) {
      final plyPath = model['ply_path'] as String? ?? '';
      if (plyPath.isNotEmpty && currentPly.contains(plyPath)) {
        return model['id']?.toString() ?? model['scene_id']?.toString() ?? '';
      }
    }
    // 默认返回第一个
    if (widget.timePeelingModels.isNotEmpty) {
      final first = widget.timePeelingModels.first;
      return first['id']?.toString() ?? first['scene_id']?.toString() ?? '';
    }
    return '';
  }

  Future<void> _handleSwitchModel(Map<String, dynamic> data) async {
    final modelId = data['modelId']?.toString() ?? '';
    debugPrint('TimePeeling: switching to model $modelId');

    // 在 timePeelingModels 中找到目标模型的原始 ply_path
    Map<String, dynamic>? targetModel;
    for (final m in widget.timePeelingModels) {
      final id = m['id']?.toString() ?? m['scene_id']?.toString() ?? '';
      if (id == modelId) {
        targetModel = m;
        break;
      }
    }

    if (targetModel == null) {
      debugPrint('TimePeeling: model $modelId not found');
      return;
    }

    final plyPath = targetModel['ply_path'] as String? ?? '';
    if (plyPath.isEmpty) return;

    // 生成公开 URL
    String publicUrl;
    try {
      publicUrl = Supabase.instance.client.storage
          .from('braindance-assets')
          .getPublicUrl(plyPath);
    } catch (e) {
      debugPrint('TimePeeling: getPublicUrl failed: $e');
      return;
    }

    // 检查是否已缓存到本地
    final encodedUrl = Uri.encodeFull(Uri.decodeFull(publicUrl));
    final uri = Uri.parse(encodedUrl);
    final sanitizedFileName = uri.path
        .replaceAll('/', '_')
        .replaceAll('\\', '_');
    final dir = await getApplicationDocumentsDirectory();
    final localFile = File('${dir.path}/$sanitizedFileName');

    String targetUrl;
    if (await localFile.exists()) {
      debugPrint('TimePeeling: model cached locally');

      // Try optimized format variants (.rad > .spz > .ksplat > original)
      final basePath = localFile.path.replaceAll(
        RegExp(r'\.(ply|splat|ksplat|spz|rad|sog)$'),
        '',
      );
      String resolvedPath = localFile.path;
      for (final ext in ['.rad', '.spz', '.ksplat']) {
        final candidate = File('$basePath$ext');
        if (await candidate.exists()) {
          debugPrint('TimePeeling: using optimized format $ext');
          resolvedPath = candidate.path;
          break;
        }
      }
      targetUrl =
          'http://127.0.0.1:$_localPort/local_models/${Uri.encodeComponent(resolvedPath)}';
    } else {
      // 未缓存，下载模型
      debugPrint('TimePeeling: downloading model...');
      try {
        final client = HttpClient()
          ..badCertificateCallback = (cert, host, port) => true;
        final request = await client.getUrl(uri);
        request.headers.set('User-Agent', 'BrainDance/1.0 Flutter');
        final response = await request.close();
        if (response.statusCode != 200) {
          throw Exception('HTTP ${response.statusCode}');
        }
        final tmpFile = File('${localFile.path}.tmp');
        final sink = tmpFile.openWrite();
        await for (final chunk in response) {
          sink.add(chunk);
        }
        await sink.close();
        await tmpFile.rename(localFile.path);
        debugPrint('TimePeeling: download complete');
        targetUrl =
            'http://127.0.0.1:$_localPort/local_models/${Uri.encodeComponent(localFile.path)}';
      } catch (e) {
        debugPrint('TimePeeling: download failed: $e, using proxy');
        targetUrl =
            'http://127.0.0.1:$_localPort/proxy/${Uri.encodeComponent(publicUrl)}';
      }
    }

    // 构建 poses URL
    String? posesUrl;
    final posesPath = plyPath.replaceAll(
      RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
      'webgl_poses.json',
    );
    if (posesPath != plyPath) {
      try {
        posesUrl = Supabase.instance.client.storage
            .from('braindance-assets')
            .getPublicUrl(posesPath);
      } catch (_) {}
    }

    // 通知 WebView 加载模型
    final payloadData = <String, String>{'ply': targetUrl};
    if (posesUrl != null) {
      payloadData['poses'] = posesUrl;
    }
    final payload = jsonEncode(payloadData);
    _controller?.runJavaScript("window.loadModelFromFlutter($payload)");
  }

  Future<void> _switchViewer(bool useSpark) async {
    if (_useSparkViewer == useSpark) return;
    setState(() {
      _useSparkViewer = useSpark;
      _isMarkerArMode = false;
      _isWebReady = false;
      _didAttemptExternalOpen = false;
      _externalViewerUrl = null;
    });

    if (_useExternalBrowserMode) {
      await _openInExternalBrowser();
      return;
    }

    await _loadLocalHtml();
  }

  Future<void> _switchMarkerArMode() async {
    final nextMarkerMode = !_isMarkerArMode;
    if (nextMarkerMode) {
      final granted = await _ensureRuntimeCameraPermission();
      if (!granted) {
        if (mounted) {
          TDToast.showText('未获得摄像头权限，无法进入 Marker AR', context: context);
        }
        return;
      }
    }

    setState(() {
      _isMarkerArMode = nextMarkerMode;
      if (_isMarkerArMode) {
        _useSparkViewer = true;
      }
      _isWebReady = false;
      _didAttemptExternalOpen = false;
      _externalViewerUrl = null;
    });

    if (_useExternalBrowserMode) {
      await _openInExternalBrowser();
      return;
    }

    await _loadLocalHtml();
  }

  Widget _buildFloatingCircleButton({
    required IconData icon,
    required VoidCallback? onPressed,
    required bool isDark,
    required String tooltip,
  }) {
    final foregroundColor = isDark
        ? const Color(0xFFF7F7FB)
        : const Color(0xFF202226);
    final backgroundColor = isDark
        ? const Color(0xCC1A1D24)
        : const Color(0xEFFFFFFF);

    return Tooltip(
      message: tooltip,
      child: Material(
        color: backgroundColor,
        shape: const CircleBorder(),
        elevation: 10,
        shadowColor: Colors.black.withValues(alpha: 0.22),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onPressed,
          child: SizedBox(
            width: 46,
            height: 46,
            child: Icon(icon, color: foregroundColor, size: 22),
          ),
        ),
      ),
    );
  }

  Widget _buildFloatingViewerToggle(bool isDark) {
    return Material(
      color: isDark ? const Color(0xCC1A1D24) : const Color(0xEFFFFFFF),
      borderRadius: BorderRadius.circular(18),
      elevation: 10,
      shadowColor: Colors.black.withValues(alpha: 0.18),
      child: Padding(
        padding: const EdgeInsets.all(4),
        child: ToggleButtons(
          isSelected: [!_useSparkViewer, _useSparkViewer],
          onPressed: (index) {
            _switchViewer(index == 1);
          },
          borderRadius: BorderRadius.circular(14),
          constraints: const BoxConstraints(minHeight: 34, minWidth: 58),
          fillColor: isDark
              ? Colors.white.withValues(alpha: 0.12)
              : AppConfig.primaryColor.withValues(alpha: 0.10),
          selectedColor: isDark
              ? const Color(0xFFF7F7FB)
              : const Color(0xFF202226),
          color: isDark
              ? Colors.white.withValues(alpha: 0.74)
              : const Color(0xFF5B6470),
          children: const [
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 10),
              child: Text('\u539f\u7248'),
            ),
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 10),
              child: Text('Spark'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMarkerArButton(bool isDark) {
    return _buildFloatingCircleButton(
      icon: _isMarkerArMode
          ? Icons.view_in_ar_rounded
          : Icons.view_in_ar_outlined,
      onPressed: _useSparkViewer || _isMarkerArMode
          ? _switchMarkerArMode
          : null,
      isDark: isDark,
      tooltip: _isMarkerArMode ? '退出 Marker AR' : '进入 Marker AR',
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final iconColor = isDark
        ? const Color(0xFFEEEEEE)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;
    final pageBg = isDark ? const Color(0xFF18181C) : Colors.white;

    return Scaffold(
      backgroundColor: pageBg,
      extendBodyBehindAppBar: true,
      body: _isUnsupportedPlatform
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(32.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.desktop_access_disabled,
                      size: 80,
                      color: iconColor,
                    ),
                    const SizedBox(height: 16),
                    TDText(
                      textLocalize('platform_webview_unsupported'),
                      font: theme.fontTitleLarge,
                      fontWeight: FontWeight.w600,
                      textColor: textColor,
                    ),
                    const SizedBox(height: 12),
                    TDText(
                      '\u5f53\u524d\u5b9e\u73b0\u672a\u8986\u76d6 Flutter Web\u3002\u8bf7\u5728 Android / iOS \u4f7f\u7528\u5185\u5d4c\u67e5\u770b\u5668\uff0c\u6216\u5728\u684c\u9762\u7aef\u4f7f\u7528\u7cfb\u7edf\u6d4f\u89c8\u5668\u6253\u5f00 3D \u67e5\u770b\u5668\u3002',
                      font: theme.fontBodyMedium,
                      textColor: hintTextColor,
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            )
          : _useExternalBrowserMode
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.open_in_browser_rounded,
                      size: 72,
                      color: iconColor,
                    ),
                    const SizedBox(height: 16),
                    TDText(
                      '\u684c\u9762\u7aef\u5df2\u5207\u6362\u4e3a\u6d4f\u89c8\u5668\u9884\u89c8\u6a21\u5f0f',
                      font: theme.fontTitleLarge,
                      fontWeight: FontWeight.w600,
                      textColor: textColor,
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 12),
                    TDText(
                      _isOpeningExternalViewer
                          ? '\u6b63\u5728\u542f\u52a8\u672c\u5730\u67e5\u770b\u670d\u52a1\u5e76\u6253\u5f00\u7cfb\u7edf\u6d4f\u89c8\u5668...'
                          : '\u5f53\u524d\u4f7f\u7528 $_viewerLabel \u67e5\u770b\u5668\u3002\u82e5\u6d4f\u89c8\u5668\u6ca1\u6709\u81ea\u52a8\u5f39\u51fa\uff0c\u53ef\u624b\u52a8\u91cd\u65b0\u6253\u5f00\u3002',
                      font: theme.fontBodyMedium,
                      textColor: hintTextColor,
                      textAlign: TextAlign.center,
                    ),
                    if (_isDownloading) ...[
                      const SizedBox(height: 18),
                      CircularProgressIndicator(color: iconColor),
                      const SizedBox(height: 12),
                      TDText(
                        '\u6b63\u5728\u51c6\u5907\u6a21\u578b...\n${(_downloadProgress * 100).toStringAsFixed(1)}%',
                        textAlign: TextAlign.center,
                        font: theme.fontBodyMedium,
                        textColor: hintTextColor,
                      ),
                    ],
                    const SizedBox(height: 18),
                    ElevatedButton(
                      onPressed: _localPort == 0
                          ? null
                          : _openInExternalBrowser,
                      child: const Text(
                        '\u5728\u6d4f\u89c8\u5668\u4e2d\u6253\u5f00',
                      ),
                    ),
                    if (_externalViewerUrl != null) ...[
                      const SizedBox(height: 12),
                      SelectableText(
                        _externalViewerUrl!,
                        textAlign: TextAlign.center,
                        style: TextStyle(color: hintTextColor, fontSize: 12),
                      ),
                    ],
                  ],
                ),
              ),
            )
          : Stack(
              children: [
                if (_controller != null && !_isDownloading)
                  SizedBox.expand(
                    child: AnimatedOpacity(
                      opacity: _isWebReady ? 1.0 : 0.0,
                      duration: const Duration(milliseconds: 500),
                      child: WebViewWidget(controller: _controller!),
                    ),
                  ),
                if (_isDownloading)
                  Center(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 40.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          TDText(
                            textLocalize('viewer_downloading_title'),
                            textAlign: TextAlign.center,
                            font: theme.fontBodyMedium,
                            textColor: hintTextColor,
                          ),
                          const SizedBox(height: 4),
                          TDText(
                            textLocalize('viewer_downloading_subtitle'),
                            textAlign: TextAlign.center,
                            font: theme.fontBodySmall,
                            textColor: hintTextColor.withAlpha(150),
                          ),
                          const SizedBox(height: 24),
                          TDText(
                            '${(_downloadProgress * 100).toStringAsFixed(1)}%',
                            font: theme.fontTitleLarge,
                            fontWeight: FontWeight.w600,
                            textColor: textColor,
                          ),
                          const SizedBox(height: 8),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: _downloadProgress,
                              minHeight: 6,
                              backgroundColor: isDark
                                  ? Colors.white.withAlpha(20)
                                  : Colors.black.withAlpha(15),
                              valueColor: AlwaysStoppedAnimation<Color>(
                                isDark
                                    ? const Color(0xFF7AA2FF)
                                    : AppConfig.primaryColor,
                              ),
                            ),
                          ),
                          const SizedBox(height: 8),
                          TDText(
                            _totalBytes > 0
                                ? '${_formatBytes(_downloadedBytes)} / ${_formatBytes(_totalBytes)}'
                                : '${_formatBytes(_downloadedBytes)} / --',
                            font: theme.fontBodySmall,
                            textColor: hintTextColor,
                          ),
                          const SizedBox(height: 20),
                          SizedBox(
                            width: 140,
                            child: OutlinedButton.icon(
                              onPressed: () {
                                _stopDownload();
                                Navigator.of(context).pop();
                              },
                              icon: const Icon(Icons.stop_rounded, size: 18),
                              label: Text(textLocalize('viewer_stop_download')),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: isDark
                                    ? const Color(0xFFFF6B6B)
                                    : const Color(0xFFD32F2F),
                                side: BorderSide(
                                  color: isDark
                                      ? const Color(0xFFFF6B6B)
                                      : const Color(0xFFD32F2F),
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                padding: const EdgeInsets.symmetric(
                                  vertical: 10,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  )
                else if (!_isWebReady && _controller != null)
                  Center(child: CircularProgressIndicator(color: iconColor)),
                if (!_isDownloading)
                  SafeArea(
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(14, 12, 14, 0),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildFloatingCircleButton(
                            icon: Icons.arrow_back_rounded,
                            onPressed: () => Navigator.of(context).pop(),
                            isDark: isDark,
                            tooltip: '\u8fd4\u56de',
                          ),
                          const Spacer(),
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              _buildMarkerArButton(isDark),
                              const SizedBox(width: 10),
                              _buildFloatingViewerToggle(isDark),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
    );
  }
}
