import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:braindance/configs/app_config.dart';
import '../services/download_event_bus.dart';

// ============================================================
// 开发/生产模式切换
// - 开发模式（kDebugMode）：连接本地 Vite dev server
//   先在项目目录运行：npm run dev（https://localhost:5173）
// - 生产模式：加载 assets/webgl/ 中的打包静态文件
//   先在项目目录运行：npm run build-only，然后把 dist/ 内容复制到 app/assets/webgl/
// ============================================================
class WebGLViewerPage extends StatefulWidget {
  final String initialModelUrl;
  final String? posesUrl; // 云端 webgl_poses.json 的公开 URL（可选）
  final String sceneId;
  final List<double>? initialPose; // 从 RAG 视角跳转传入的坐标矩阵

  const WebGLViewerPage({
    super.key,
    this.initialModelUrl = './models/scene_auto_sync_raw.ply',
    this.posesUrl,
    this.sceneId = '3DGS Viewer',
    this.initialPose,
  });

  @override
  State<WebGLViewerPage> createState() => _WebGLViewerPageState();
}

class _WebGLViewerPageState extends State<WebGLViewerPage> {
  WebViewController? _controller;
  bool _isWebReady = false;
  bool _isUnsupportedPlatform = false;
  HttpServer? _localServer;
  int _localPort = 0;
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  int _downloadedBytes = 0;
  int _totalBytes = -1;
  String? _localModelPath;
  bool _downloadCancelled = false;

  @override
  void initState() {
    super.initState();
    // Flutter Web 和桌面端均不支持 webview_flutter（需 Android/iOS）
    if (kIsWeb ||
        defaultTargetPlatform == TargetPlatform.windows ||
        defaultTargetPlatform == TargetPlatform.linux ||
        defaultTargetPlatform == TargetPlatform.macOS) {
      _isUnsupportedPlatform = true;
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

  /// 用户主动停止下载
  void _stopDownload() {
    _downloadCancelled = true;
    if (mounted) {
      setState(() {
        _isDownloading = false;
      });
    }
  }

  /// 格式化字节为人类可读的大小字符串
  String _formatBytes(int bytes) {
    if (bytes < 1024) return '${bytes}B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(0)}KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)}MB';
  }

  Future<void> _startLocalServer() async {
    _localServer = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _localPort = _localServer!.port;
    _localServer!.listen((HttpRequest request) async {
      String path = request.uri.path;

      // ---- /proxy/ : Dart 端代理 HTTPS 请求，绕过 WebView 的 SSL 证书限制 ----
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
          request.response.headers.add('Access-Control-Allow-Origin', '*');
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

      // ---- /local_models/ : 本地已下载文件 ----
      if (path.startsWith('/local_models/')) {
        final filePath = Uri.decodeComponent(
          path.substring('/local_models/'.length),
        );
        final file = File(filePath);
        if (await file.exists()) {
          request.response.headers.add('Access-Control-Allow-Origin', '*');
          if (filePath.endsWith('.ply') ||
              filePath.endsWith('.splat') ||
              filePath.endsWith('.ksplat')) {
            request.response.headers.contentType = ContentType(
              'application',
              'octet-stream',
            );
          }
          await request.response.addStream(file.openRead());
          await request.response.close();
          return;
        } else {
          request.response.statusCode = HttpStatus.notFound;
          await request.response.close();
          return;
        }
      }

      if (path == '/' || path.isEmpty) {
        path = '/index.html';
      }

      // 将请求路径映射到 Flutter 的 assets/webgl 目录
      String assetPath = 'assets/webgl$path';

      try {
        final ByteData data = await rootBundle.load(assetPath);
        final List<int> bytes = data.buffer.asUint8List();

        String contentType = 'text/plain';
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
        } else if (path.endsWith('.ply') ||
            path.endsWith('.splat') ||
            path.endsWith('.ksplat')) {
          contentType = 'application/octet-stream';
        }

        request.response.headers.contentType = ContentType.parse(contentType);
        // 允许跨域
        request.response.headers.add('Access-Control-Allow-Origin', '*');
        request.response.add(bytes);
        await request.response.close();
      } catch (e) {
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
        // Fix string containing spaces or unencoded characters causing 400 Bad Request
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
          if (mounted) _initWebView();
        } else {
          // 使用临时文件下载，完成后再重命名，避免部分下载被当作完整文件
          final tmpFile = File('${localFile.path}.tmp');

          // 断点续传：检查已下载的临时文件大小
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

          // 允许自托管 Supabase 的自签名或不被 Android 信任的证书
          final client = HttpClient()
            ..badCertificateCallback = (cert, host, port) => true;
          final request = await client.getUrl(uri);
          request.headers.set('User-Agent', 'BrainDance/1.0 Flutter');
          // 断点续传：设置 Range 头从已下载位置继续
          if (existingBytes > 0) {
            request.headers.set('Range', 'bytes=$existingBytes-');
          }
          final response = await request.close();

          // 206 Partial Content = 服务器支持断点续传
          // 200 OK = 服务器不支持 Range，需从头下载
          if (response.statusCode == 200 && existingBytes > 0) {
            // 服务器不支持续传，删除旧临时文件从头开始
            debugPrint('Server does not support Range, restarting download');
            if (await tmpFile.exists()) await tmpFile.delete();
            existingBytes = 0;
          } else if (response.statusCode != 200 && response.statusCode != 206) {
            String errorBody = '';
            try {
              errorBody = await response.transform(utf8.decoder).join();
            } catch (_) {}
            throw Exception('HTTP Error ${response.statusCode}: $errorBody');
          }

          // 计算总大小：续传时 contentLength 是剩余部分大小
          final contentLength = response.contentLength;
          final totalBytes = contentLength > 0
              ? contentLength + existingBytes
              : -1;
          int receivedBytes = existingBytes;

          // 保存总大小到 .meta 文件，供主页模型卡片读取下载进度
          final metaFile = File('${localFile.path}.meta');
          if (totalBytes > 0) {
            await metaFile.writeAsString('$totalBytes');
          }

          // 续传时追加写入，否则覆盖写入
          final sink = tmpFile.openWrite(
            mode: existingBytes > 0 ? FileMode.append : FileMode.write,
          );

          // 如果有已下载的部分，立即更新进度条
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
                // 断点续传：取消时保留临时文件，下次可继续
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
              if (totalBytes > 0 && mounted) {
                final progress = receivedBytes / totalBytes;
                setState(() {
                  _downloadProgress = progress;
                  _downloadedBytes = receivedBytes;
                });
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

            // 下载完成，将临时文件重命名为正式文件，并清理 meta 文件
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
            // 断点续传：下载失败时保留临时文件，下次可继续
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
          TDToast.showText('下载模型失败: $e', context: context);
          _initWebView();
        }
      }
    } else {
      if (mounted) _initWebView();
    }
  }

  void _initWebView() {
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0x00000000))
      ..addJavaScriptChannel(
        'BrainDanceChannel',
        onMessageReceived: (JavaScriptMessage message) {
          final data = jsonDecode(message.message);
          if (data['status'] == 'ready') {
            setState(() => _isWebReady = true);
            // 优先使用本地路径或代理 URL，避免 WebView JS 直接访问 HTTPS
            _sendModelToVue();
          } else if (data['status'] == 'success') {
            if (mounted) {
              TDToast.showText(data['msg'], context: context);
            }
          }
        },
      )
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageFinished: (String url) {
            // 后备方案：页面加载完 2 秒后如果还没收到 ready 信号，则主动触发
            Future.delayed(const Duration(seconds: 2), () {
              if (!_isWebReady && mounted) {
                debugPrint(
                  'WebView: no ready signal received, triggering manually',
                );
                setState(() => _isWebReady = true);
                _sendModelToVue();
              }
            });
          },
          onWebResourceError: (WebResourceError error) {
            debugPrint('WebView error: ${error.description}');
          },
        ),
      );

    // 通知 Flutter 重建，让 WebViewWidget 真正挂载到树上
    if (mounted) setState(() {});

    // Load the local HTML file matching the server port
    _loadLocalHtml();
  }

  Future<void> _loadLocalHtml() async {
    try {
      final url = 'http://127.0.0.1:$_localPort/index.html';
      await _controller?.loadRequest(Uri.parse(url));
    } catch (e) {
      debugPrint('Error loading HTML via local server: $e');
    }
  }

  /// 统一入口：决定传给 WebView 的模型 URL
  /// - 如果已下载到本地 -> 使用本地 HTTP /local_models/ 路由
  /// - 如果是远程 HTTPS URL  -> 转成本地 HTTP /proxy/ 路由，由 Dart 代理
  /// - 如果是相对路径（本地 demo） -> 直接传递
  void _sendModelToVue() {
    if (!_isWebReady) return;
    String targetUrl;
    if (_localModelPath != null) {
      // 已下载到本地：通过本地 HTTP 服务提供
      targetUrl =
          'http://127.0.0.1:$_localPort/local_models/${Uri.encodeComponent(_localModelPath!)}';
    } else if (widget.initialModelUrl.startsWith('http://') ||
        widget.initialModelUrl.startsWith('https://')) {
      // 远程 URL：通过本地代理，避免 WebView 直接访问 HTTPS
      targetUrl =
          'http://127.0.0.1:$_localPort/proxy/${Uri.encodeComponent(widget.initialModelUrl)}';
    } else {
      // 相对路径（本地 demo 模型）
      targetUrl = widget.initialModelUrl;
    }

    debugPrint('Sending model URL to WebView: $targetUrl');

    if (widget.posesUrl != null && widget.posesUrl!.isNotEmpty) {
      // 新版：同时传模型 URL、webgl_poses.json 公网 URL 以及可能的初始视角矩阵
      final payload = jsonEncode({
        'ply': targetUrl,
        'poses': widget.posesUrl,
        if (widget.initialPose != null) 'matrix': widget.initialPose,
      });
      _controller?.runJavaScript("window.loadModelFromFlutter($payload)");
    } else {
      // 旧版兼容：只传模型 URL，由 WebGL 内部使用默认位姿文件
      final payload = jsonEncode({
        'ply': targetUrl,
        if (widget.initialPose != null) 'matrix': widget.initialPose,
      });
      _controller?.runJavaScript("window.loadModelFromFlutter($payload)");
    }
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
    final appBarBg = isDark ? const Color(0xFF101014) : Colors.white;
    final appBarFg = textColor;
    final pageBg = isDark ? const Color(0xFF18181C) : Colors.white;
    return Scaffold(
      backgroundColor: pageBg,
      appBar: AppBar(
        title: Text(widget.sceneId, style: TextStyle(color: appBarFg)),
        backgroundColor: appBarBg,
        foregroundColor: appBarFg,
        systemOverlayStyle: isDark
            ? SystemUiOverlayStyle.light
            : SystemUiOverlayStyle.dark,
        elevation: 0,
        iconTheme: IconThemeData(color: iconColor),
      ),
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
                      'Flutter 官方的 webview_flutter 插件目前仅支持 Android / iOS / Web 平台。\n如果你正在使用 Windows / macOS 调试，不支持直接原位打开 3D 模型。\n\n请在移动端模拟器（Android Emulator/iOS Simulator）或真实手机设备上调试 3D 查看功能！',
                      font: theme.fontBodyMedium,
                      textColor: hintTextColor,
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            )
          : Stack(
              children: [
                if (_controller != null && !_isDownloading)
                  AnimatedOpacity(
                    opacity: _isWebReady ? 1.0 : 0.0,
                    duration: const Duration(milliseconds: 500),
                    child: WebViewWidget(controller: _controller!),
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
                          // 百分比
                          TDText(
                            '${(_downloadProgress * 100).toStringAsFixed(1)}%',
                            font: theme.fontTitleLarge,
                            fontWeight: FontWeight.w600,
                            textColor: textColor,
                          ),
                          const SizedBox(height: 8),
                          // 进度条
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
                          // 已下载 / 总大小
                          TDText(
                            _totalBytes > 0
                                ? '${_formatBytes(_downloadedBytes)} / ${_formatBytes(_totalBytes)}'
                                : '${_formatBytes(_downloadedBytes)} / --',
                            font: theme.fontBodySmall,
                            textColor: hintTextColor,
                          ),
                          const SizedBox(height: 20),
                          // 停止下载按钮
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
              ],
            ),
    );
  }
}
