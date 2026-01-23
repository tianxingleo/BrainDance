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
  static bool firstCheck = true;
  static late final bool cameraEnabled;
  static late CameraController cameraController;
  @override
  void initState() {
    super.initState();
    onCameraUpdate = cameraUpdate;
    bool ce = (AppConfig.cameras.isNotEmpty);
    //相机初始化
    if (!ce) {
      if (firstCheck) {
        firstCheck = false;
        //可直接锁定cameraEnabled状态
        cameraEnabled = false;
      }
      return;
    }
    cameraController = CameraController(AppConfig.cameras[0], ResolutionPreset.max);
    
    ce = cameraUpdate();
        if (firstCheck) {
          firstCheck = false;
          //以下代码只会执行一次
          cameraEnabled = ce;
        }
  }
  bool cameraUpdate() {
    if (!firstCheck && !cameraEnabled) {
      return false;
    }
    bool suc = true;
    
    cameraController.initialize().then((_) {
      if (mounted) {
        setState(() {});
      }
    }).catchError((Object e) {
      suc = false;
    });
    return suc;
  }
  @override
  void dispose() {
    if (cameraEnabled) {
      cameraController.dispose();
    }
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    late final Widget cameraView;
    if(cameraEnabled) {
      cameraView = Expanded(child: CameraPreview(cameraController));
    } else {
      cameraView = Container();
    }
    return Scaffold(
      body: Center(
        child: cameraView,
      ),
      floatingActionButton: Align(
        alignment: Alignment(0, 0.9),
        child: TDButton(text: 'Button1', onTap: () {
          cameraUpdate();
        })
      ),
    );
  }
}
