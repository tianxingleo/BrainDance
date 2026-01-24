import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../app_configs.dart';
import '../main.dart';
import 'package:camera/camera.dart';

class RecordPage extends StatefulWidget {
  const RecordPage({super.key});

  @override
  State<RecordPage> createState() => _RecordPageState();
}
class _RecordPageState extends State<RecordPage> {
  static const ResolutionPreset resolutionPreset = ResolutionPreset.max;
  static int camNum = 0;
  Future<void> cameraSwitch() async {
    camNum++;
    if (camNum == AppConfig.cameras.length) {
      camNum = 0;
    }
    if ((cameraController == null) || !cameraController!.value.isInitialized) {
      await cameraInitialize();
    } else {
      cameraController!.setDescription(AppConfig.cameras[camNum]);
      setState(() {});
    }
  }
  Future<bool> cameraInitialize() async {
    cameraController = CameraController(
      AppConfig.cameras[camNum],
      resolutionPreset,
    );
    bool suc = true;
    await cameraController!.initialize().then((_) {
      if (mounted) {
        setState(() {});
      }
    }).catchError((Object e) {
      suc = false;
    });
    return suc;
  }
  /// Returns a suitable camera icon for [direction].
IconData getCameraLensIcon(CameraLensDirection direction) {
  switch (direction) {
    case CameraLensDirection.back:
      return Icons.camera_rear;
    case CameraLensDirection.front:
      return Icons.camera_front;
    case CameraLensDirection.external:
      return Icons.camera;
  }
  // This enum is from a different package, so a new value could be added at
  // any time. The example should keep working if that happens.
  // ignore: dead_code
  return Icons.camera;
}
 @override
  void initState() {
    super.initState();
    //相机初始化
    if (!cameraEnabled) {
      return;
    }
    onCameraInitialize = cameraInitialize;
    cameraInitialize();
  }
  
@override
void dispose() {
  cameraController?.dispose();
  cameraController = null;  // 防止内存泄漏和误用
  onCameraInitialize = () {};
  super.dispose();
}
  @override
  Widget build(BuildContext context) {
    late final Widget cameraView;
    if (!cameraEnabled) {
      cameraView = Center(child: Text(textLocalize("reco_camun")));
    } else if (!cameraController!.value.isInitialized) {
      cameraView = Center(child: Text(textLocalize("reco_wait")));
    } else {
      cameraView = CameraPreview(cameraController!);
    }
    return Scaffold(
      body: cameraView,
      floatingActionButton: Align(
        alignment: Alignment(0, 0.9),
        child: TDButton(text: 'Current Camera: $camNum', onTap: () {
          if (cameraEnabled) {
            cameraSwitch();
          }
        })
      ),
    );
  }
}
