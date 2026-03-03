import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:camera/camera.dart';

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
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: cameraView),
          // 顶部控制栏
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            right: 24,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.black.withAlpha(180),
                    Colors.white.withAlpha(30),
                  ],
                  begin: Alignment.topRight,
                  end: Alignment.bottomLeft,
                ),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white.withAlpha(40), width: 1),
                boxShadow: [
                  BoxShadow(color: Colors.black.withAlpha(30), blurRadius: 8),
                ],
              ),
              child: IconButton(
                icon: Icon(TDIcons.refresh, color: Colors.white, size: 24),
                padding: const EdgeInsets.all(12),
                constraints: const BoxConstraints(),
                onPressed: () {
                  if (RecoConfig.cameraEnabled) {
                    RecoConfig.cameraSwitch();
                  }
                },
              ),
            ),
          ),
          // 底部控制栏
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 140,
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
              child: Center(
                child: GestureDetector(
                  onTapDown: (_) => _buttonAnimController.forward(),
                  onTapUp: (_) {
                    _buttonAnimController.reverse();
                    TDToast.showText('Recording...', context: context);
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
          ),
        ],
      ),
    );
  }
}
