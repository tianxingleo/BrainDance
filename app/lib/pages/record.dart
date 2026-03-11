import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
import 'package:braindance/main.dart' show isRecordingProvider;
import 'package:braindance/pages/video_submit.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

const double _kFovH = 65.0;
const double _kFovV = 50.0;

class _CapturedFrame {
  final double yaw;
  final double pitch;

  const _CapturedFrame(this.yaw, this.pitch);
}

class RecordPage extends ConsumerStatefulWidget {
  const RecordPage({super.key});

  @override
  ConsumerState<RecordPage> createState() => _RecordPageState();
}

class _RecordPageState extends ConsumerState<RecordPage>
    with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  late AnimationController _buttonAnimController;
  late Animation<double> _buttonScaleAnimation;

  bool _showTips = false;
  bool _isArMode = false;
  bool _isArSampling = false;

  Timer? _recordTimer;
  int _recordSeconds = 0;

  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<MagnetometerEvent>? _magSub;
  Timer? _sampleTimer;

  double _ax = 0;
  double _ay = 0;
  double _az = -9.8;
  double _magX = 30;
  double _magY = 0;
  double _magZ = -40;

  double _yaw = 0;
  double _pitch = 0;

  final List<_CapturedFrame> _capturedFrames = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _showTips = !AppConfig.hasReadRecordTip;

    _buttonAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    _buttonScaleAnimation = Tween<double>(begin: 1.0, end: 0.9).animate(
      CurvedAnimation(parent: _buttonAnimController, curve: Curves.easeInOut),
    );

    if (!RecoConfig.cameraEnabled) {
      return;
    }

    RecoConfig.onUpdate = () {
      if (mounted) {
        setState(() {});
      }
    };
    RecoConfig.cameraInitialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _recordTimer?.cancel();
    _recordTimer = null;
    _stopSensors();
    _setGlobalRecording(false);
    _buttonAnimController.dispose();
    RecoConfig.onUpdate = () {};
    RecoConfig.disposeCamera();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    final controller = RecoConfig.cameraController;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      if (_isArMode) {
        _stopSensors();
        if (_isArSampling) {
          setState(() {
            _isArSampling = false;
          });
          _setGlobalRecording(false);
        }
      }

      if (controller != null && controller.value.isInitialized) {
        if (controller.value.isRecordingVideo) {
          _stopVideoRecording(
            controller,
            showToast: false,
            navigateToSubmit: true,
          ).whenComplete(() {
            RecoConfig.disposeCamera();
            if (mounted) {
              setState(() {});
              TDToast.showText(context: context, '应用切换导致录像中断，已保存录制内容');
            }
          });
        } else {
          RecoConfig.disposeCamera();
          if (mounted) {
            setState(() {});
          }
        }
      }
    } else if (state == AppLifecycleState.resumed) {
      if (RecoConfig.cameraController == null) {
        RecoConfig.cameraInitialize();
      }
      if (_isArMode) {
        _startSensors();
      }
    }
  }

  void _setGlobalRecording(bool value) {
    if (value) {
      WakelockPlus.enable();
    } else {
      WakelockPlus.disable();
    }
    ref.read(isRecordingProvider.notifier).state = value;
  }

  Future<void> _stopVideoRecording(
    CameraController controller, {
    bool showToast = true,
    bool navigateToSubmit = true,
  }) async {
    _recordTimer?.cancel();
    _recordTimer = null;

    if (!controller.value.isRecordingVideo) {
      return;
    }

    _setGlobalRecording(false);
    var file = await controller.stopVideoRecording();

    if (showToast && mounted) {
      TDToast.showText('录制完成', context: context);
    }

    final permissionState = await PhotoManager.requestPermissionExtend();
    if (!permissionState.isAuth) {
      if (showToast && mounted) {
        TDToast.showText('无法保存视频到相册。视频文件暂存于缓存中，注意缓存清理。', context: context);
      }
    } else {
      try {
        final newAsset = await PhotoManager.editor.saveVideo(
          File(file.path),
          title: file.name,
        );
        await FileSystem.deleteFile(file.path);
        final savedFile = await newAsset.originFile;
        if (savedFile != null) {
          file = XFile(savedFile.path);
        } else if (mounted) {
          TDToast.showText('保存视频到相册时发生错误', context: context);
        }
      } catch (_) {
        if (mounted) {
          TDToast.showText('保存视频到相册时发生错误', context: context);
        }
      }
    }

    var thumbPath = file.path;
    try {
      thumbPath = await VThumb.ensureThumb(file.path);
    } catch (_) {}

    if (navigateToSubmit && mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) =>
              VideoSubmitPage(videoPath: file.path, thumbnailPath: thumbPath),
        ),
      );
    }
  }

  Future<void> _toggleVideoRecording() async {
    final controller = RecoConfig.cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    final isVideoRecording = ref.read(isRecordingProvider);
    if (isVideoRecording) {
      await _stopVideoRecording(controller);
      return;
    }

    _setGlobalRecording(true);
    await controller.startVideoRecording();
    try {
      await RecoConfig.trySwitchCameraDescription(RecoConfig.camNum);
    } catch (_) {}
    _recordSeconds = 0;

    _recordTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      _recordSeconds++;
      if (_recordSeconds >= 180) {
        _stopVideoRecording(controller);
      } else if (mounted) {
        setState(() {});
      }
    });
  }

  void _computeOrientation() {
    final pitch = atan2(-_ax, sqrt(_ay * _ay + _az * _az));
    final roll = atan2(_ay, _az);

    final mx = _magX * cos(pitch) + _magZ * sin(pitch);
    final my =
        _magX * sin(roll) * sin(pitch) +
        _magY * cos(roll) -
        _magZ * sin(roll) * cos(pitch);

    var yaw = atan2(-my, mx) * (180 / pi);
    if (yaw < 0) {
      yaw += 360;
    }

    if (mounted) {
      setState(() {
        _yaw = yaw;
        _pitch = (pitch * (180 / pi)).clamp(-90.0, 90.0);
      });
    }
  }

  void _startSensors() {
    if (_accelSub != null || _magSub != null) {
      return;
    }

    _accelSub =
        accelerometerEventStream(
          samplingPeriod: const Duration(milliseconds: 50),
        ).listen((event) {
          _ax = event.x;
          _ay = event.y;
          _az = event.z;
          _computeOrientation();
        });

    _magSub =
        magnetometerEventStream(
          samplingPeriod: const Duration(milliseconds: 50),
        ).listen((event) {
          _magX = event.x;
          _magY = event.y;
          _magZ = event.z;
          _computeOrientation();
        });

    _sampleTimer ??= Timer.periodic(const Duration(milliseconds: 300), (_) {
      if (_isArMode && _isArSampling && mounted) {
        setState(() {
          _capturedFrames.add(_CapturedFrame(_yaw, _pitch));
        });
      }
    });
  }

  void _stopSensors() {
    _accelSub?.cancel();
    _accelSub = null;
    _magSub?.cancel();
    _magSub = null;
    _sampleTimer?.cancel();
    _sampleTimer = null;
  }

  void _toggleArMode() {
    final isVideoRecording = ref.read(isRecordingProvider) && !_isArSampling;
    if (isVideoRecording) {
      return;
    }

    setState(() {
      _isArMode = !_isArMode;
      if (_isArMode) {
        _capturedFrames.clear();
        _startSensors();
      } else {
        _stopSensors();
        _isArSampling = false;
        _setGlobalRecording(false);
      }
    });
  }

  void _toggleArSampling() {
    if (!_isArMode) {
      return;
    }

    setState(() {
      _isArSampling = !_isArSampling;
    });
    _setGlobalRecording(_isArSampling);

    if (mounted) {
      TDToast.showText(_isArSampling ? '开始扫描覆盖' : '已暂停扫描', context: context);
    }
  }

  Future<void> _onRecordTap() async {
    if (_isArMode) {
      _toggleArSampling();
      return;
    }

    await _toggleVideoRecording();
  }

  List<Widget> _buildCameraSwitchButtons(
    BuildContext context,
    bool isAnyRecording,
  ) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final cameras = RecoConfig.cameras;
    final cameraSwitchButtons = <Widget>[];

    int getLensPriority(CameraLensDirection dir) {
      if (dir == CameraLensDirection.back) {
        return 1;
      }
      if (dir == CameraLensDirection.front) {
        return 2;
      }
      return 3;
    }

    final sortedIndices = List<int>.generate(cameras.length, (i) => i)
      ..sort(
        (a, b) => getLensPriority(
          cameras[a].lensDirection,
        ).compareTo(getLensPriority(cameras[b].lensDirection)),
      );

    var backCount = 1;
    var frontCount = 1;
    var externalCount = 1;

    for (final i in sortedIndices) {
      final cam = cameras[i];
      late final String label;
      switch (cam.lensDirection) {
        case CameraLensDirection.back:
          label = '后置$backCount';
          backCount++;
          break;
        case CameraLensDirection.front:
          label = '前置$frontCount';
          frontCount++;
          break;
        case CameraLensDirection.external:
          label = '外置$externalCount';
          externalCount++;
          break;
      }

      final isSelected = RecoConfig.camNum == i;

      cameraSwitchButtons.add(
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(24),
              onTap: () async {
                if (!RecoConfig.cameraEnabled) {
                  return;
                }

                if (isAnyRecording) {
                  try {
                    await RecoConfig.trySwitchCameraDescription(i);
                  } catch (_) {
                    if (mounted) {
                      TDToast.showText('录像中无法直接切换传感器', context: context);
                    }
                  }
                  return;
                }

                RecoConfig.camNum = i;
                await RecoConfig.cameraInitialize();
                if (mounted) {
                  setState(() {});
                }
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: isSelected
                      ? (isDark
                            ? const Color(0xFF4582FF).withAlpha(200)
                            : theme.brandColor7.withAlpha(220))
                      : (isDark
                            ? const Color(0xFF23232A).withAlpha(150)
                            : Colors.white.withAlpha(150)),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: isSelected
                        ? (isDark ? const Color(0xFF4582FF) : theme.brandColor7)
                        : (isDark
                              ? Colors.white.withAlpha(30)
                              : Colors.black.withAlpha(20)),
                    width: isSelected ? 1.5 : 1,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      RecoConfig.getCameraLensIcon(cam.lensDirection),
                      size: 20,
                      color: isSelected
                          ? Colors.white
                          : (isDark ? Colors.white70 : const Color(0xFF555555)),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      label,
                      style: TextStyle(
                        color: isSelected
                            ? Colors.white
                            : (isDark
                                  ? Colors.white70
                                  : const Color(0xFF555555)),
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
    }

    return cameraSwitchButtons;
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final isVideoRecording = ref.watch(isRecordingProvider) && !_isArSampling;
    final isAnyRecording = isVideoRecording || _isArSampling;

    Widget cameraView;
    if (!RecoConfig.cameraEnabled) {
      cameraView = Center(
        child: Text(
          textLocalize('reco_camun'),
          style: const TextStyle(fontSize: 18, color: Colors.white70),
        ),
      );
    } else if (RecoConfig.cameraController == null ||
        !RecoConfig.cameraController!.value.isInitialized) {
      cameraView = Center(
        child: Text(
          textLocalize('reco_wait'),
          style: const TextStyle(fontSize: 18, color: Colors.white70),
        ),
      );
    } else {
      final controller = RecoConfig.cameraController!;
      final size = MediaQuery.of(context).size;
      final deviceRatio = size.width / size.height;
      final cameraRatio = controller.value.aspectRatio;

      var scale = 1 / (cameraRatio * deviceRatio);
      if (scale < 1) {
        scale = 1 / scale;
      }

      cameraView = Transform.scale(
        scale: scale,
        child: Center(child: CameraPreview(controller)),
      );
    }

    final cameraSwitchButtons = _buildCameraSwitchButtons(
      context,
      isAnyRecording,
    );
    final mediaQuery = MediaQuery.of(context);
    final bottomOffset = mediaQuery.padding.bottom + 32;

    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF101014) : Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: cameraView),
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
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(
                top: mediaQuery.padding.top + 10,
                bottom: 20,
              ),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withAlpha(200),
                    Colors.black.withAlpha(0),
                  ],
                ),
              ),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                physics: const BouncingScrollPhysics(),
                child: Padding(
                  padding: const EdgeInsets.only(right: 140),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: cameraSwitchButtons,
                  ),
                ),
              ),
            ),
          ),
          if (!isVideoRecording)
            Positioned(
              top: mediaQuery.padding.top + 60,
              right: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(110),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: Colors.white.withAlpha(30)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon: Icon(
                        TDIcons.scan,
                        color: _isArMode
                            ? const Color(0xFF00E5FF)
                            : Colors.white70,
                        size: 24,
                      ),
                      onPressed: _toggleArMode,
                    ),
                    if (!_isArMode)
                      IconButton(
                        icon: const Icon(
                          TDIcons.refresh,
                          color: Colors.white70,
                          size: 24,
                        ),
                        onPressed: isAnyRecording
                            ? null
                            : () => RecoConfig.cameraSwitch(),
                      ),
                    IconButton(
                      icon: const Icon(
                        Icons.info_outline,
                        color: Colors.white70,
                        size: 24,
                      ),
                      onPressed: () {
                        setState(() {
                          _showTips = true;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),
          Positioned(
            top: mediaQuery.padding.top + 72,
            left: 20,
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 200),
              opacity: _isArMode ? 1 : 0,
              child: IgnorePointer(
                ignoring: !_isArMode,
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.black45,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '方位(Yaw): ${_yaw.toStringAsFixed(1)}°\n'
                    '俯仰(Pitch): ${_pitch.toStringAsFixed(1)}°\n'
                    '已记录覆盖帧: ${_capturedFrames.length}',
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            left: 0,
            right: 0,
            bottom: bottomOffset,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (isVideoRecording)
                  Container(
                    margin: const EdgeInsets.only(bottom: 24),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.redAccent.withAlpha(200),
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.redAccent.withAlpha(100),
                          blurRadius: 8,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          '${(_recordSeconds ~/ 60).toString().padLeft(2, '0')}:${(_recordSeconds % 60).toString().padLeft(2, '0')}',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            fontFeatures: [FontFeature.tabularFigures()],
                          ),
                        ),
                      ],
                    ),
                  ),
                if (_isArSampling)
                  Container(
                    margin: const EdgeInsets.only(bottom: 24),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF00BCD4).withAlpha(210),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(
                      '扫描中 ${_capturedFrames.length} 帧',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                Center(
                  child: GestureDetector(
                    onTapDown: (_) => _buttonAnimController.forward(),
                    onTapUp: (_) async {
                      _buttonAnimController.reverse();
                      await _onRecordTap();
                    },
                    onTapCancel: () => _buttonAnimController.reverse(),
                    child: ScaleTransition(
                      scale: _buttonScaleAnimation,
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        width: 84,
                        height: 84,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: LinearGradient(
                            colors: isAnyRecording
                                ? [Colors.redAccent, Colors.red]
                                : isDark
                                ? [
                                    const Color(0xFF4582FF),
                                    const Color(0xFF2156CC),
                                  ]
                                : [theme.brandColor4, theme.brandColor7],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          border: Border.all(
                            color: isDark
                                ? const Color(0xFF18181C)
                                : Colors.white,
                            width: 4,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: isAnyRecording
                                  ? Colors.redAccent.withAlpha(100)
                                  : isDark
                                  ? const Color(0xFF4582FF).withAlpha(80)
                                  : Colors.black.withAlpha(40),
                              blurRadius: 16,
                              spreadRadius: 4,
                            ),
                          ],
                        ),
                        child: Center(
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 300),
                            width: 66,
                            height: 66,
                            decoration: BoxDecoration(
                              color: isDark
                                  ? const Color(0xFF18181C)
                                  : Colors.white,
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withAlpha(20),
                                  blurRadius: 6,
                                  spreadRadius: 1,
                                ),
                              ],
                            ),
                            child: Center(
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 300),
                                width: isAnyRecording ? 28 : 24,
                                height: isAnyRecording ? 28 : 24,
                                decoration: BoxDecoration(
                                  color: isAnyRecording
                                      ? Colors.redAccent
                                      : isDark
                                      ? const Color(0xFF4582FF)
                                      : theme.brandColor6,
                                  borderRadius: BorderRadius.circular(
                                    isAnyRecording ? 6 : 12,
                                  ),
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
          ),
          if (_showTips)
            Positioned.fill(
              child: Container(
                color: Colors.black87,
                padding: const EdgeInsets.only(bottom: 80),
                child: Center(
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 32),
                    padding: const EdgeInsets.all(24),
                    constraints: BoxConstraints(
                      maxHeight: mediaQuery.size.height * 0.70,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E1E1E),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Tips',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Flexible(
                          child: SingleChildScrollView(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  textLocalize('reco_tip_title1'),
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  textLocalize('reco_tip1'),
                                  style: const TextStyle(color: Colors.white70),
                                ),
                                const SizedBox(height: 10),
                                Text(
                                  textLocalize('reco_tip_title2'),
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  textLocalize('reco_tip2'),
                                  style: const TextStyle(color: Colors.white70),
                                ),
                                const SizedBox(height: 10),
                                Text(
                                  textLocalize('reco_tip_title3'),
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  textLocalize('reco_tip3'),
                                  style: const TextStyle(color: Colors.white70),
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        Align(
                          alignment: Alignment.centerRight,
                          child: ElevatedButton(
                            onPressed: () {
                              SetConfig.setHasReadRecordTip(true);
                              setState(() {
                                _showTips = false;
                              });
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.blueAccent,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                            child: const Padding(
                              padding: EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 8,
                              ),
                              child: Text(
                                'OK',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
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

class _FogOfWarPainter extends CustomPainter {
  final List<_CapturedFrame> capturedFrames;
  final double currentYaw;
  final double currentPitch;

  const _FogOfWarPainter({
    required this.capturedFrames,
    required this.currentYaw,
    required this.currentPitch,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final layerPaint = Paint()..color = Colors.black.withAlpha(217);
    canvas.saveLayer(Rect.fromLTWH(0, 0, size.width, size.height), Paint());
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), layerPaint);

    final clearPaint = Paint()
      ..blendMode = BlendMode.clear
      ..style = PaintingStyle.fill;

    for (final frame in capturedFrames) {
      var dx = frame.yaw - currentYaw;
      if (dx > 180) {
        dx -= 360;
      }
      if (dx < -180) {
        dx += 360;
      }
      final dy = frame.pitch - currentPitch;

      final screenX = size.width / 2 + (dx / _kFovH) * size.width;
      final screenY = size.height / 2 - (dy / _kFovV) * size.height;

      canvas.drawRect(
        Rect.fromCenter(
          center: Offset(screenX, screenY),
          width: size.width,
          height: size.height,
        ),
        clearPaint,
      );
    }

    canvas.restore();

    final guidePaint = Paint()
      ..color = Colors.white.withAlpha(77)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawRect(
      Rect.fromCenter(
        center: size.center(Offset.zero),
        width: size.width,
        height: size.height,
      ),
      guidePaint,
    );
  }

  @override
  bool shouldRepaint(_FogOfWarPainter oldDelegate) {
    return oldDelegate.currentYaw != currentYaw ||
        oldDelegate.currentPitch != currentPitch ||
        oldDelegate.capturedFrames.length != capturedFrames.length;
  }
}
