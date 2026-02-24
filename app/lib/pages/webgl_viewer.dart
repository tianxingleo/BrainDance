import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'dart:io';
import 'package:flutter/services.dart';
// ============================================================
// 开发/生产模式切换
// - 开发模式（kDebugMode）：连接本地 Vite dev server
//   先在项目目录运行：npm run dev（https://localhost:5173）
// - 生产模式：加载 assets/webgl/ 中的打包静态文件
//   先在项目目录运行：npm run build-only，然后把 dist/ 内容复制到 app/assets/webgl/
// ============================================================
class WebGLViewerPage extends StatefulWidget {
  final String initialModelUrl;
  final String sceneId;

  const WebGLViewerPage({
    super.key,
    this.initialModelUrl = './models/scene_auto_sync_raw.ply',
    this.sceneId = '3DGS Viewer',
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
          if (mounted) _initWebView();
        });
      } catch (e) {
        debugPrint('WebView initialization failed: $e');
        _isUnsupportedPlatform = true;
      }
    }
  }

  @override
  void dispose() {
    _localServer?.close();
    super.dispose();
  }

  Future<void> _startLocalServer() async {
    _localServer = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _localPort = _localServer!.port;
    _localServer!.listen((HttpRequest request) async {
      String path = request.uri.path;
      if (path == '/' || path.isEmpty) {
        path = '/index.html';
      }

      // 将请求路径映射到 Flutter 的 assets/webgl 目录
      String assetPath = 'assets/webgl$path';

      try {
        final ByteData data = await rootBundle.load(assetPath);
        final List<int> bytes = data.buffer.asUint8List();

        String contentType = 'text/plain';
        if (path.endsWith('.html')) contentType = 'text/html; charset=utf-8';
        else if (path.endsWith('.js')) contentType = 'application/javascript; charset=utf-8';
        else if (path.endsWith('.css')) contentType = 'text/css; charset=utf-8';
        else if (path.endsWith('.png')) contentType = 'image/png';
        else if (path.endsWith('.ico')) contentType = 'image/x-icon';
        else if (path.endsWith('.ply')) contentType = 'application/octet-stream';

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
            _sendModelToVue(widget.initialModelUrl);
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
                debugPrint('WebView: no ready signal received, triggering manually');
                setState(() => _isWebReady = true);
                _sendModelToVue(widget.initialModelUrl);
              }
            });
          },
          onWebResourceError: (WebResourceError error) {
            debugPrint('WebView error: ${error.description}');
          },
        ),
      );

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

  void _sendModelToVue(String modelUrl) {
    if (_isWebReady) {
      // 使用 jsonEncode 确保 URL 中的特殊字符被正确转义
      final encodedUrl = jsonEncode(modelUrl);
      _controller?.runJavaScript("window.loadModelFromFlutter($encodedUrl)");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.sceneId),
        backgroundColor: TDTheme.of(context).whiteColor1,
        foregroundColor: TDTheme.of(context).fontGyColor1,
        elevation: 0,
      ),
      body: _isUnsupportedPlatform
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(32.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.desktop_access_disabled, size: 80, color: Colors.grey),
                    const SizedBox(height: 16),
                    TDText(
                      '当前平台不支持内嵌网页',
                      font: TDTheme.of(context).fontTitleLarge,
                      fontWeight: FontWeight.w600,
                    ),
                    const SizedBox(height: 12),
                    TDText(
                      'Flutter 官方的 webview_flutter 插件目前仅支持 Android / iOS / Web 平台。\n如果你正在使用 Windows / macOS 调试，不支持直接原位打开 3D 模型。\n\n请在移动端模拟器（Android Emulator/iOS Simulator）或真实手机设备上调试 3D 查看功能！',
                      font: TDTheme.of(context).fontBodyMedium,
                      textColor: TDTheme.of(context).fontGyColor3,
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            )
          : Stack(
              children: [
                if (_controller != null) WebViewWidget(controller: _controller!),
                if (!_isWebReady)
                  const Center(
                    child: CircularProgressIndicator(),
                  ),
              ],
            ),
    );
  }
}
