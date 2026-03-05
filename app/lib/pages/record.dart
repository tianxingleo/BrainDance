import 'dart:async';
import 'dart:math';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:camera/camera.dart';
import 'package:sensors_plus/sensors_plus.dart';

// 相机水平/垂直视场角（度），与实际摄像头接近即可
const double _kFovH = 65.0;
const double _kFovV = 50.0;

/// 记录一帧拍摄时的朝向
class _CapturedFrame {
  final double yaw;
  final double pitch;
  _CapturedFrame(this.yaw, this.pitch);
}

class RecordPage extends StatefulWidget {
  const RecordPage({super.key});

  @override
  State<RecordPage> createState() => _RecordPageState();
}

class _RecordPageState extends State<RecordPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _buttonAnimController;
  late Animation<double> _buttonScaleAnimation;

  // 覆盖扫描模式
  bool _isArMode = false;
  bool _isRecording = false;

  // 传感器订阅
  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<MagnetometerEvent>? _magSub;
  Timer? _sampleTimer;

  // 原始传感器数据
  double _ax = 0, _ay = 0, _az = -9.8;
  double _magX = 30, _magY = 0, _magZ = -40;

  // 融合后的稳定绝对朝向（无漂移）
  double _yaw = 0; // 方位角 0-360°（磁北方向）
  double _pitch = 0; // 俯仰角 -90°~+90°

  // 已录制帧的朝向列表
  final List<_CapturedFrame> _capturedFrames = [];

  @override
  void initState() {
    super.initState();
    _buttonAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    _buttonScaleAnimation = Tween<double>(begin: 1.0, end: 0.9).animate(
      CurvedAnimation(
          parent: _buttonAnimController, curve: Curves.easeInOut),
    );

    if (!RecoConfig.cameraEnabled) return;
    RecoConfig.onUpdate = () {
      if (mounted) setState(() {});
    };
    RecoConfig.cameraInitialize();
  }

  // ─── 传感器融合：加速度计（俯仰/横滚） + 磁力计（方位角） ─────────────
  // 采用绝对值解算，不进行时间积分，从而根治漂移问题
  void _computeOrientation() {
    // 由加速度计推算俯仰角（Gravity Vector 为主）
    final pitch = atan2(-_ax, sqrt(_ay * _ay + _az * _az));
    final roll = atan2(_ay, _az);

    // 倾斜补偿后的磁力计分量（消除手机倾斜对方位角的影响）
    final mx = _magX * cos(pitch) + _magZ * sin(pitch);
    final my = _magX * sin(roll) * sin(pitch) +
        _magY * cos(roll) -
        _magZ * sin(roll) * cos(pitch);

    // 从磁力计计算方位角（磁北，无累积漂移）
    var yaw = atan2(-my, mx) * (180 / pi);
    if (yaw < 0) yaw += 360;

    if (mounted) {
      setState(() {
        _yaw = yaw;
        _pitch = (pitch * (180 / pi)).clamp(-90.0, 90.0);
      });
    }
  }

  void _startSensors() {
    _accelSub = accelerometerEventStream(samplingPeriod: const Duration(milliseconds: 50)).listen((event) {
      _ax = event.x;
      _ay = event.y;
      _az = event.z;
      _computeOrientation();
    });

    _magSub = magnetometerEventStream(samplingPeriod: const Duration(milliseconds: 50)).listen((event) {
      _magX = event.x;
      _magY = event.y;
      _magZ = event.z;
      _computeOrientation();
    });

    // 扫描模式实时记录采样
    _sampleTimer = Timer.periodic(const Duration(milliseconds: 300), (timer) {
      if (_isRecording && _isArMode) {
        setState(() {
          // 只在模式开启且正在“录制”时记录当前朝向点
          _capturedFrames.add(_CapturedFrame(_yaw, _pitch));
        });
      }
    });
  }

  void _stopSensors() {
    _accelSub?.cancel();
    _magSub?.cancel();
    _sampleTimer?.cancel();
  }

  void _toggleArMode() {
    setState(() {
      _isArMode = !_isArMode;
      if (_isArMode) {
        _capturedFrames.clear();
        _startSensors();
      } else {
        _stopSensors();
        _isRecording = false;
      }
    });
  }

  @override
  void dispose() {
    _buttonAnimController.dispose();
    RecoConfig.cameraController?.dispose();
    RecoConfig.cameraController = null;
    RecoConfig.onUpdate = () {};
    _stopSensors();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Widget cameraWidget;
    if (!RecoConfig.cameraEnabled) {
      cameraWidget = Center(child: Text(textLocalize("reco_camun")));
    } else if (!RecoConfig.cameraController!.value.isInitialized) {
      cameraWidget = Center(child: Text(textLocalize("reco_wait")));
    } else {
      // ─── 1. 解决相机拉伸：根据屏幕比例动态计算缩放 ───
      final controller = RecoConfig.cameraController!;
      final size = MediaQuery.of(context).size;
      final deviceRatio = size.width / size.height;
      final cameraRatio = controller.value.aspectRatio; // 通常是 9/16 ≈ 0.56
      
      // 如果相机比屏幕还窄，则按高度撑满并裁切宽度
      var scale = 1 / (cameraRatio * deviceRatio);
      if (scale < 1) scale = 1 / scale;

      cameraWidget = Transform.scale(
        scale: scale,
        child: Center(child: CameraPreview(controller)),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // ─── 底层相机画面 ───
          Positioned.fill(child: cameraWidget),

          // ─── 2. 覆盖可见性层 (Fog of War) ───
          // 实现扫描覆盖效果：未拍摄区域全黑，已拍摄区域完全透明
          if (_isArMode)
            Positioned.fill(
              child: CustomPaint(
                painter: _FogOfWarPainter(
                  capturedFrames: _capturedFrames,
                  currentYaw: _yaw,
                  currentPitch: _pitch,
                ),
              ),
            ),

          // ─── UI 控制栏 ───
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            right: 24,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(TDTheme.of(context).radiusRound),
                border: Border.all(color: Colors.white.withValues(alpha: 0.2), width: 1),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: Icon(
                      _isArMode ? TDIcons.scan : TDIcons.scan,
                      color: _isArMode ? const Color(0xFF00E5FF) : Colors.white,
                      size: 24,
                    ),
                    onPressed: _toggleArMode,
                  ),
                  if (!_isArMode)
                    IconButton(
                      icon: const Icon(TDIcons.refresh, color: Colors.white, size: 24),
                      onPressed: () => RecoConfig.cameraSwitch(),
                    ),
                ],
              ),
            ),
          ),

          // ─── 录制按钮 ───
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: () {
                  setState(() {
                    _isRecording = !_isRecording;
                  });
                  if (!_isArMode && _isRecording) {
                     TDToast.showText('录制中...', context: context);
                  }
                },
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 4),
                  ),
                  child: Center(
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      width: _isRecording ? 32 : 64,
                      height: _isRecording ? 32 : 64,
                      decoration: BoxDecoration(
                        color: Colors.red,
                        borderRadius: BorderRadius.circular(_isRecording ? 8 : 32),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          
          if (_isArMode)
            Positioned(
              top: 100,
              left: 20,
              child: Container(
                padding: const EdgeInsets.all(8),
                color: Colors.black45,
                child: Text(
                  '方位(Yaw): ${_yaw.toStringAsFixed(1)}°\n俯仰(Pitch): ${_pitch.toStringAsFixed(1)}°\n已记录覆盖帧: ${_capturedFrames.length}',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// ─── 3. 战争迷雾 Painter：实现全局覆盖检查 ───
class _FogOfWarPainter extends CustomPainter {
  final List<_CapturedFrame> capturedFrames;
  final double currentYaw;
  final double currentPitch;

  _FogOfWarPainter({
    required this.capturedFrames,
    required this.currentYaw,
    required this.currentPitch,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // A. 绘制一层半透明黑色遮罩，作为“未拍摄区域”
    final layerPaint = Paint()..color = Colors.black.withValues(alpha: 0.85);
    canvas.saveLayer(Rect.fromLTWH(0, 0, size.width, size.height), Paint());
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), layerPaint);

    // B. 使用 BlendMode.clear 在遮罩上挖出已拍摄的画面（洞口）
    final clearPaint = Paint()
      ..blendMode = BlendMode.clear
      ..style = PaintingStyle.fill;

    // 逻辑：遍历所有 capture 过的帧，将它们在当前视角下的相对位置投影到屏幕
    for (var frame in capturedFrames) {
      double dx = (frame.yaw - currentYaw);
      // 处理 0/360 度循环越界
      if (dx > 180) dx -= 360;
      if (dx < -180) dx += 360;
      double dy = (frame.pitch - currentPitch);

      // 投影：角度偏移 -> 屏幕像素偏移
      double screenX = size.width / 2 + (dx / _kFovH) * size.width;
      double screenY = size.height / 2 - (dy / _kFovV) * size.height;

      // 既然这一帧当时拍到了整个屏幕内容，那我们就把这块区域挖掉
      canvas.drawRect(
        Rect.fromCenter(
          center: Offset(screenX, screenY), 
          width: size.width, 
          height: size.height
        ),
        clearPaint,
      );
    }

    canvas.restore();
    
    // 绘制中心引导框（提示当前正在录制的范围）
    final guidePaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawRect(
      Rect.fromCenter(center: size.center(Offset.zero), width: size.width, height: size.height),
      guidePaint,
    );
  }

  @override
  bool shouldRepaint(_FogOfWarPainter oldDelegate) => true;
}