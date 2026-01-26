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

class _RecordPageState extends State<RecordPage> {
  @override
  void initState() {
    super.initState();
    //相机初始化
    if (!RecoConfig.cameraEnabled) {
      return;
    }
    RecoConfig.cameraInitialize();
  }

  @override
  void dispose() {
    RecoConfig.cameraController?.dispose();
    RecoConfig.cameraController = null; // 防止内存泄漏和误用
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    late final Widget cameraView;
    if (!RecoConfig.cameraEnabled) {
      cameraView = Center(child: Text(textLocalize("reco_camun")));
    } else if (!RecoConfig.cameraController!.value.isInitialized) {
      cameraView = Center(child: Text(textLocalize("reco_wait")));
    } else {
      cameraView = CameraPreview(RecoConfig.cameraController!);
    }
    return Scaffold(
      body: cameraView,
      floatingActionButton: Align(
        alignment: Alignment(0, 0.9),
        child: TDButton(
          text: 'Current Camera: ${RecoConfig.camNum}',
          onTap: () {
            if (RecoConfig.cameraEnabled) {
              RecoConfig.cameraSwitch();
            }
          },
        ),
      ),
    );
  }
}
