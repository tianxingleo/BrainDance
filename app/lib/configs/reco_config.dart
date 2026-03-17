import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';

class RecoConfig {
  //更新
  static VoidCallback? onUpdate;
  //程序
  static late final bool cameraEnabled;
  static late final List<CameraDescription> cameras;
  static List<CameraDescription> frontCameras = [];
  static List<CameraDescription> backCameras = [];
  static List<CameraDescription> externalCameras = [];
  static CameraController? cameraController;
  //可变
  static int camNum = 0;
  static const ResolutionPreset resolutionPreset = ResolutionPreset.max;
  //基础函数
  static Future<bool> cameraInitialize() async {
    cameraController = CameraController(cameras[camNum], resolutionPreset);
    bool suc = true;
    await cameraController!.initialize().catchError((Object e) {
      suc = false;
    });
    onUpdate?.call();
    return suc;
  }

  static Future<void> trySwitchCameraDescription(int index) async {
    if (cameraController != null && cameraController!.value.isInitialized) {
      camNum = index;
      await cameraController!.setDescription(cameras[camNum]);
      onUpdate?.call();
    }
  }

  static Future<void> cameraSwitch() async {
    camNum++;
    if (camNum == cameras.length) {
      camNum = 0;
    }
    if ((cameraController == null) || !cameraController!.value.isInitialized) {
      await cameraInitialize();
    } else {
      await cameraController!.setDescription(cameras[camNum]);
    }
    onUpdate?.call();
  }

  //相机自动更新
  static void refreshCamera() {
    if ((cameraController == null) ||
        (!cameraController!.value.isInitialized)) {
      return;
    }
    cameraInitialize();
  }

  static void disposeCamera() {
    if ((cameraController == null) ||
        (!cameraController!.value.isInitialized)) {
      return;
    }
    cameraController?.dispose();
    cameraController = null;
  }

  //获取图标
  static IconData getCameraLensIcon(CameraLensDirection direction) {
    switch (direction) {
      case CameraLensDirection.back:
        return Icons.camera_rear;
      case CameraLensDirection.front:
        return Icons.camera_front;
      case CameraLensDirection.external:
        return Icons.camera;
    }
    // ignore: dead_code
    return Icons.camera;
  }
}
