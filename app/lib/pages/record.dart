import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:camera/camera.dart';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
import 'package:braindance/pages/video_submit.dart';

class RecordPage extends StatefulWidget {
  const RecordPage({super.key});

  @override
  State<RecordPage> createState() => _RecordPageState();
}

class _RecordPageState extends State<RecordPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _buttonAnimController;
  late Animation<double> _buttonScaleAnimation;

  @override
  void initState() {
    super.initState();
    _buttonAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    _buttonScaleAnimation = Tween<double>(begin: 1.0, end: 0.9).animate(
      CurvedAnimation(parent: _buttonAnimController, curve: Curves.easeInOut),
    );

    //相机初始化
    if (!RecoConfig.cameraEnabled) {
      return;
    }
    RecoConfig.onUpdate = () {
      setState(() {});
    };
    RecoConfig.cameraInitialize();
  }

  @override
  void dispose() {
    _buttonAnimController.dispose();
    RecoConfig.cameraController?.dispose();
    RecoConfig.cameraController = null; // 防止内存泄漏和误用
    RecoConfig.onUpdate = () {};
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    late final Widget cameraView;
    if (!RecoConfig.cameraEnabled) {
      cameraView = Center(
        child: Text(
          textLocalize("reco_camun"),
          style: TextStyle(fontSize: 18, color: Colors.white70),
        ),
      );
    } else if (!RecoConfig.cameraController!.value.isInitialized) {
      cameraView = Center(
        child: Text(
          textLocalize("reco_wait"),
          style: TextStyle(fontSize: 18, color: Colors.white70),
        ),
      );
    } else {
      cameraView = CameraPreview(RecoConfig.cameraController!);
    }
    // 获取所有可用摄像头
    final cameras = RecoConfig.cameras;
    final List<Widget> cameraSwitchButtons = [];
    for (int i = 0; i < cameras.length; i++) {
      final cam = cameras[i];
      String label = '';
      switch (cam.lensDirection) {
        case CameraLensDirection.front:
          label = '前置${i + 1}';
          break;
        case CameraLensDirection.back:
          label = '后置${i + 1}';
          break;
        case CameraLensDirection.external:
          label = '外置${i + 1}';
          break;
      }
      cameraSwitchButtons.add(
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 6),
          child: ElevatedButton.icon(
            icon: Icon(RecoConfig.getCameraLensIcon(cam.lensDirection)),
            label: Text(label),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.black.withAlpha(180),
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
              ),
            ),
            onPressed: () async {
              if (RecoConfig.cameraEnabled) {
                RecoConfig.camNum = i;
                await RecoConfig.cameraInitialize();
                setState(() {});
              }
            },
          ),
        ),
      );
    }
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: cameraView),
          // 相机按钮（位于底边栏上方，居中悬浮）
          Positioned(
            bottom: 100,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTapDown: (_) => _buttonAnimController.forward(),
                onTapUp: (_) async {
                  _buttonAnimController.reverse();
                  // 开始录像
                  if (RecoConfig.cameraController != null &&
                      RecoConfig.cameraController!.value.isInitialized) {
                    final controller = RecoConfig.cameraController!;
                    final videoFile = await controller.startVideoRecording();
                    TDToast.showText('正在录制...', context: context);
                    // 录制 10 秒后自动停止（可改为按钮控制）
                    await Future.delayed(const Duration(seconds: 10));
                    final file = await controller.stopVideoRecording();
                    TDToast.showText('录制完成', context: context);
                    // 生成缩略图
                    String thumbPath = file.path;
                    try {
                      thumbPath = await VThumb.ensureThumb(file.path);
                    } catch (_) {}
                    // 跳转到视频提交页
                    if (context.mounted) {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => VideoSubmitPage(
                            videoPath: file.path,
                            thumbnailPath: thumbPath,
                          ),
                        ),
                      );
                    }
                  }
                },
                onTapCancel: () => _buttonAnimController.reverse(),
                child: ScaleTransition(
                  scale: _buttonScaleAnimation,
                  child: Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: LinearGradient(
                        colors: [
                          TDTheme.of(context).brandColor1,
                          TDTheme.of(context).brandColor4,
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      border: Border.all(
                        color: Colors.white.withAlpha(200),
                        width: 4,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withAlpha(40),
                          blurRadius: 12,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: Center(
                      child: Container(
                        width: 64,
                        height: 64,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withAlpha(20),
                              blurRadius: 6,
                              spreadRadius: 1,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          // 底部摄像头切换栏
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 80,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.black.withAlpha(230),
                    Colors.black.withAlpha(120),
                    Colors.transparent,
                  ],
                  stops: const [0.0, 0.6, 1.0],
                ),
                borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: cameraSwitchButtons,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
