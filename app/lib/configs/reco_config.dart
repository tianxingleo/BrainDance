import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class RecoConfig {
  //更新
  static VoidCallback? onUpdate;
  //程序
  static bool cameraEnabled = false;
  static List<CameraDescription> cameras = [];
  static List<CameraDescription> frontCameras = [];
  static List<CameraDescription> backCameras = [];
  static List<CameraDescription> externalCameras = [];
  static CameraController? cameraController;
  //可变
  static int camNum = 0;
  static const ResolutionPreset resolutionPreset = ResolutionPreset.max;

  static Future<void> ensureCameraCatalog() async {
    if (cameras.isNotEmpty) {
      cameraEnabled = true;
      return;
    }

    try {
      final discovered = await availableCameras();
      frontCameras = [];
      backCameras = [];
      externalCameras = [];
      for (final cam in discovered) {
        switch (cam.lensDirection) {
          case CameraLensDirection.front:
            frontCameras.add(cam);
            break;
          case CameraLensDirection.back:
            backCameras.add(cam);
            break;
          case CameraLensDirection.external:
            externalCameras.add(cam);
            break;
        }
      }
      cameras = discovered;
      cameraEnabled = discovered.isNotEmpty;
      if (camNum >= cameras.length) {
        camNum = 0;
      }
    } catch (_) {
      cameras = [];
      frontCameras = [];
      backCameras = [];
      externalCameras = [];
      cameraEnabled = false;
      camNum = 0;
    }
  }

  //基础函数
  static Future<bool> cameraInitialize() async {
    await ensureCameraCatalog();
    if (!cameraEnabled || cameras.isEmpty) {
      onUpdate?.call();
      return false;
    }
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
