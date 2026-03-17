import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
import 'package:braindance/main.dart' show isRecordingProvider;
import 'package:braindance/pages/video_submit.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../configs/motion_tokens.dart';
import 'record/record_hud_painter.dart';

const double _kFovH = 65.0;
const double _kFovV = 50.0;
const double _kIdealAccelMin = 0.08;
const double _kIdealAccelMax = 0.65;
const double _kCautionAccelMax = 1.35;
const double _kDangerAccelMax = 2.10;
const double _kInstantSpikeAccel = 2.60;
const double _kJerkThreshold = 0.95;
const int _kAccelHistoryLength = 40;
const double _kFloatingNavReservedHeight = 112.0;

class _CapturedFrame {
  final double yaw;
  final double pitch;

  const _CapturedFrame(this.yaw, this.pitch);
}

enum _MotionState { steady, ideal, caution, danger }

class RecordPage extends ConsumerStatefulWidget {
  const RecordPage({super.key});

  @override
  ConsumerState<RecordPage> createState() => _RecordPageState();
}

class _RecordPageState extends ConsumerState<RecordPage>
    with TickerProviderStateMixin, WidgetsBindingObserver {
  late AnimationController _buttonAnimController;
  late Animation<double> _buttonScaleAnimation;
  late AnimationController _hudAnimController; // 用于HUD角点呼吸

  bool _showTips = false;
  bool _isArMode = false;
  bool _isArSampling = false;
  bool _isMovingTooFast = false; // 用于异常运动警告
  int _warningEndTime = 0; // 用于抖动警告显示状态的防抖停留时长

  Timer? _recordTimer;
  int _recordSeconds = 0;

  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<UserAccelerometerEvent>? _userAccelSub;
  StreamSubscription<MagnetometerEvent>? _magSub;
  Timer? _sampleTimer;

  double _ax = 0;
  double _ay = 0;
  double _az = -9.8;
  double _magX = 30;
  double _magY = 0;
  double _magZ = -40;
  double _linearAccel = 0;
  double _smoothedLinearAccel = 0;
  double _peakLinearAccel = 0;
  double _accelDelta = 0;
  double _motionMeter = 0;
  String _motionHint = textLocalize('reco_motion_steady');
  String _motionDetail = textLocalize('reco_motion_detail');
  int _lastFastToastTime = 0;
  Timer? _hapticLoopTimer;
  _MotionState? _hapticLoopState;
  _MotionState _motionState = _MotionState.steady;

  double _yaw = 0;
  double _pitch = 0;

  final List<_CapturedFrame> _capturedFrames = [];
  final List<double> _accelHistory = [];

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

    _hudAnimController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat(reverse: true);

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
    _hudAnimController.dispose();
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
              TDToast.showText(context: context, textLocalize('reco_app_switch'));
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
      TDToast.showText(textLocalize('reco_done'), context: context);
    }

    final permissionState = await PhotoManager.requestPermissionExtend();
    if (!permissionState.isAuth) {
      if (showToast && mounted) {
        TDToast.showText(textLocalize('reco_save_fail'), context: context);
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
          TDToast.showText(textLocalize('reco_save_error'), context: context);
        }
      } catch (_) {
        if (mounted) {
          TDToast.showText(textLocalize('reco_save_error'), context: context);
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

  void _updateMotionFeedback(
    UserAccelerometerEvent event, {
    bool forceRefresh = false,
  }) {
    final linearAccel = sqrt(
      event.x * event.x + event.y * event.y + event.z * event.z,
    );
    final accelDelta = (linearAccel - _linearAccel).abs();
    final smoothedAccel = (_smoothedLinearAccel * 0.72) + (linearAccel * 0.28);
    final motionMeter = max(
      smoothedAccel,
      linearAccel * 0.78 + accelDelta * 0.42,
    );

    final nextState = switch (motionMeter) {
      <= _kIdealAccelMin => _MotionState.steady,
      <= _kIdealAccelMax => _MotionState.ideal,
      <= _kCautionAccelMax => _MotionState.caution,
      _ => _MotionState.danger,
    };

    final instantDanger =
        linearAccel >= _kInstantSpikeAccel ||
        motionMeter >= _kDangerAccelMax ||
        (accelDelta >= _kJerkThreshold && smoothedAccel >= _kIdealAccelMax);

    if (instantDanger) {
      _warningEndTime = DateTime.now().millisecondsSinceEpoch + 1200;
    }

    final now = DateTime.now().millisecondsSinceEpoch;
    final currentWarning = now < _warningEndTime;
    final effectiveState = currentWarning ? _MotionState.danger : nextState;
    final nextHint = switch (effectiveState) {
      _MotionState.steady => textLocalize('reco_hint_steady'),
      _MotionState.ideal => textLocalize('reco_hint_ideal'),
      _MotionState.caution => textLocalize('reco_hint_caution'),
      _MotionState.danger => textLocalize('reco_hint_danger'),
    };
    final nextDetail = switch (effectiveState) {
      _MotionState.steady => textLocalize('reco_detail_steady'),
      _MotionState.ideal => textLocalize('reco_detail_ideal'),
      _MotionState.caution => textLocalize('reco_detail_caution'),
      _MotionState.danger => textLocalize('reco_detail_danger'),
    };

    if (currentWarning != _isMovingTooFast) {
      _isMovingTooFast = currentWarning;
      _hudAnimController.duration = _isMovingTooFast
          ? const Duration(milliseconds: 150)
          : const Duration(seconds: 1);
      _hudAnimController.repeat(reverse: true);
    }

    if (_isMovingTooFast && mounted && now - _lastFastToastTime > 1800) {
      _lastFastToastTime = now;
      TDToast.showText(context: context, textLocalize('reco_accel_warning'));
    }

    _syncMotionHaptics(effectiveState);

    final nextPeak = max(
      _peakLinearAccel * 0.965,
      max(linearAccel, motionMeter),
    );
    if (_accelHistory.length >= _kAccelHistoryLength) {
      _accelHistory.removeAt(0);
    }
    _accelHistory.add(motionMeter);

    if (mounted) {
      setState(() {
        _linearAccel = linearAccel;
        _smoothedLinearAccel = smoothedAccel;
        _peakLinearAccel = nextPeak;
        _accelDelta = accelDelta;
        _motionMeter = motionMeter;
        _motionHint = nextHint;
        _motionDetail = nextDetail;
        _motionState = effectiveState;
      });
    } else {
      _linearAccel = linearAccel;
      _smoothedLinearAccel = smoothedAccel;
      _peakLinearAccel = nextPeak;
      _accelDelta = accelDelta;
      _motionMeter = motionMeter;
      _motionHint = nextHint;
      _motionDetail = nextDetail;
      _motionState = effectiveState;
    }

    if (forceRefresh && mounted) {
      setState(() {});
    }
  }

  void _syncMotionHaptics(_MotionState state) {
    if (state == _MotionState.ideal) {
      _stopHapticLoop();
      return;
    }

    final nextInterval = state == _MotionState.danger
        ? const Duration(milliseconds: 420)
        : const Duration(milliseconds: 700);

    if (_hapticLoopTimer != null && _hapticLoopTimer!.isActive) {
      if (_hapticLoopState == state) {
        return;
      }
      _stopHapticLoop();
    }

    _hapticLoopState = state;
    _fireHapticForState(state);
    _hapticLoopTimer = Timer.periodic(nextInterval, (_) {
      _fireHapticForState(_motionState);
    });
  }

  void _fireHapticForState(_MotionState state) {
    if (state == _MotionState.ideal) {
      _stopHapticLoop();
      return;
    }

    if (state == _MotionState.danger) {
      unawaited(HapticFeedback.heavyImpact());
      return;
    }

    unawaited(HapticFeedback.mediumImpact());
  }

  void _stopHapticLoop() {
    _hapticLoopTimer?.cancel();
    _hapticLoopTimer = null;
    _hapticLoopState = null;
  }

  void _startSensors() {
    if (_accelSub != null || _userAccelSub != null || _magSub != null) {
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

    _userAccelSub =
        userAccelerometerEventStream(
          samplingPeriod: const Duration(milliseconds: 50),
        ).listen((event) {
          _updateMotionFeedback(event);
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
    _stopHapticLoop();
    _accelSub?.cancel();
    _accelSub = null;
    _userAccelSub?.cancel();
    _userAccelSub = null;
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
        _linearAccel = 0;
        _smoothedLinearAccel = 0;
        _peakLinearAccel = 0;
        _accelDelta = 0;
        _motionMeter = 0;
        _motionHint = textLocalize('reco_motion_steady');
        _motionDetail = textLocalize('reco_motion_detail');
        _motionState = _MotionState.steady;
        _isMovingTooFast = false;
        _warningEndTime = 0;
        _accelHistory.clear();
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
      TDToast.showText(_isArSampling ? textLocalize('reco_scan_start') : textLocalize('reco_scan_pause'), context: context);
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
                    if (context.mounted) {
                      TDToast.showText(textLocalize('reco_no_switch'), context: context);
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
    final isDark = AppConfig.isNightMode;
    final isVideoRecording = ref.watch(isRecordingProvider) && !_isArSampling;
    final isAnyRecording = isVideoRecording || _isArSampling;
    final darkInput = const Color(0xFF23232A);

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
    final bottomOffset =
        mediaQuery.padding.bottom +
        (isAnyRecording ? 36 : _kFloatingNavReservedHeight + 18);

    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF101014) : Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: cameraView),
          Positioned.fill(
            child: CustomPaint(
              painter: RecordHUDPainter(
                isWarning: _isMovingTooFast,
                isCaution: _motionState == _MotionState.caution,
                motionValue: _motionMeter,
                animation: _hudAnimController,
              ),
            ),
          ),
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
            top: mediaQuery.padding.top + 12,
            left: 16,
            right: 16,
            child: _RecordOverlayPanel(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                physics: const BouncingScrollPhysics(),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: cameraSwitchButtons,
                ),
              ),
            ),
          ),
          if (!isVideoRecording)
            Positioned(
              top: mediaQuery.padding.top + 82,
              right: 16,
              child: _RecordOverlayPanel(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon: Icon(
                        TDIcons.scan,
                        color: _isArMode
                            ? BDDesign.colorPaperWhite
                            : BDDesign.colorAshGray,
                        size: 24,
                      ),
                      onPressed: _toggleArMode,
                    ),
                    if (!_isArMode)
                      IconButton(
                        icon: const Icon(
                          TDIcons.refresh,
                          color: BDDesign.colorAshGray,
                          size: 24,
                        ),
                        onPressed: isAnyRecording
                            ? null
                            : () => RecoConfig.cameraSwitch(),
                      ),
                    IconButton(
                      icon: const Icon(
                        Icons.info_outline,
                        color: BDDesign.colorAshGray,
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
            child: AnimatedSwitcher(
              duration: BDMotion.durationFast,
              switchInCurve: BDMotion.curveEnter,
              switchOutCurve: BDMotion.curveExit,
              child: !_isArMode
                  ? const SizedBox.shrink()
                  : _MotionGuidanceCard(
                      yaw: _yaw,
                      pitch: _pitch,
                      frameCount: _capturedFrames.length,
                      linearAccel: _linearAccel,
                      smoothedLinearAccel: _smoothedLinearAccel,
                      peakLinearAccel: _peakLinearAccel,
                      accelDelta: _accelDelta,
                      motionMeter: _motionMeter,
                      motionState: _motionState,
                      motionDetail: _motionDetail,
                      accelHistory: _accelHistory,
                      isMovingTooFast: _isMovingTooFast,
                      motionHint: _motionHint,
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
                if (isVideoRecording || _isArSampling) ...[
                  AnimatedSwitcher(
                    duration: BDMotion.durationFast,
                    switchInCurve: BDMotion.curveEnter,
                    switchOutCurve: BDMotion.curveExit,
                    child: isVideoRecording
                        ? _StatusPill(
                            key: ValueKey<String>(
                              'rec_${_recordSeconds}_${_capturedFrames.length}',
                            ),
                            label:
                                'REC ${_recordSeconds ~/ 60}:${(_recordSeconds % 60).toString().padLeft(2, '0')}',
                            color: Colors.redAccent,
                            backgroundColor: darkInput,
                            isSquareDot: true,
                            compact: true,
                          )
                        : _StatusPill(
                            key: ValueKey<String>(
                              'scan_${_capturedFrames.length}',
                            ),
                            label: '${_capturedFrames.length} F',
                            color: BDDesign.colorPaperWhite,
                            backgroundColor: darkInput,
                            compact: true,
                          ),
                  ),
                  const SizedBox(height: 10),
                ],
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
                        duration: BDMotion.durationFast,
                        curve: BDMotion.curveEnter,
                        width: 78,
                        height: 78,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: isAnyRecording
                              ? Colors.redAccent
                              : BDDesign.colorPaperWhite,
                          border: Border.all(
                            color: BDDesign.colorInkBlack,
                            width: 2,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color:
                                  (isAnyRecording
                                          ? Colors.redAccent
                                          : BDDesign.colorMutedBlue)
                                      .withAlpha(isAnyRecording ? 92 : 46),
                              blurRadius: 14,
                              spreadRadius: isAnyRecording ? 2 : 0,
                            ),
                          ],
                        ),
                        child: Center(
                          child: AnimatedContainer(
                            duration: BDMotion.durationFast,
                            curve: BDMotion.curveEnter,
                            width: 60,
                            height: 60,
                            decoration: const BoxDecoration(
                              color: BDDesign.colorInkBlack,
                              shape: BoxShape.circle,
                            ),
                            child: Center(
                              child: AnimatedContainer(
                                duration: BDMotion.durationFast,
                                curve: BDMotion.curveEnter,
                                width: isAnyRecording ? 26 : 22,
                                height: isAnyRecording ? 26 : 22,
                                decoration: BoxDecoration(
                                  color: isAnyRecording
                                      ? Colors.redAccent
                                      : BDDesign.colorPaperWhite,
                                  borderRadius: BorderRadius.circular(
                                    isAnyRecording ? 4 : 12,
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
                color: Colors.black.withAlpha(176),
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
                      borderRadius: BDDesign.radiusNormal,
                      border: Border.all(color: Colors.white.withAlpha(20)),
                      boxShadow: [BDDesign.shadowElevated],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                textLocalize('reco_scan_tip'),
                                style: TextStyle(
                                  color: BDDesign.colorPaperWhite,
                                  fontSize: 20,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                            _StatusPill(
                              label: _isArMode ? 'HUD' : 'VIDEO',
                              color: _isArMode
                                  ? BDDesign.colorFadedOlive
                                  : BDDesign.colorMutedBlueLight,
                              backgroundColor: const Color(0xFF23232A),
                              compact: true,
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Flexible(
                          child: SingleChildScrollView(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                _TipBlock(
                                  title: textLocalize('reco_tip_title1'),
                                  body: textLocalize('reco_tip1'),
                                ),
                                const SizedBox(height: 12),
                                _TipBlock(
                                  title: textLocalize('reco_tip_title2'),
                                  body: textLocalize('reco_tip2'),
                                ),
                                const SizedBox(height: 12),
                                _TipBlock(
                                  title: textLocalize('reco_tip_title3'),
                                  body: textLocalize('reco_tip3'),
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 18),
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                textLocalize('reco_scan_before'),
                                style: TextStyle(
                                  color: Colors.white.withAlpha(168),
                                  fontSize: 12.5,
                                  height: 1.35,
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            TDButton(
                              onTap: () {
                                SetConfig.setHasReadRecordTip(true);
                                setState(() {
                                  _showTips = false;
                                });
                              },
                              text: textLocalize('reco_scan_ok'),
                              style: TDButtonStyle(
                                backgroundColor: BDDesign.colorMutedBlue,
                                textColor: Colors.white,
                                radius: BorderRadius.circular(18),
                              ),
                              type: TDButtonType.fill,
                              shape: TDButtonShape.rectangle,
                              theme: TDButtonTheme.primary,
                              size: TDButtonSize.small,
                            ),
                          ],
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

class _TipBlock extends StatelessWidget {
  final String title;
  final String body;

  const _TipBlock({required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white.withAlpha(12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withAlpha(18)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: BDDesign.colorPaperWhite,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            body,
            style: TextStyle(color: Colors.white.withAlpha(176), height: 1.45),
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

class _StatusPill extends StatelessWidget {
  final String label;
  final Color color;
  final Color backgroundColor;
  final bool isSquareDot;
  final bool compact;

  const _StatusPill({
    super.key,
    required this.label,
    required this.color,
    required this.backgroundColor,
    this.isSquareDot = false,
    this.compact = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 10 : 12,
        vertical: compact ? 5 : 7,
      ),
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withAlpha(120)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: compact ? 7 : 8,
            height: compact ? 7 : 8,
            decoration: BoxDecoration(
              color: color,
              shape: isSquareDot ? BoxShape.rectangle : BoxShape.circle,
            ),
          ),
          SizedBox(width: compact ? 6 : 8),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: compact ? 10 : 11,
              fontWeight: FontWeight.w700,
              letterSpacing: compact ? 0.3 : 0.6,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

class _RecordOverlayPanel extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const _RecordOverlayPanel({required this.child, required this.padding});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: BDDesign.colorInkBlack.withAlpha(216),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withAlpha(28)),
        boxShadow: [BDDesign.shadowElevated],
      ),
      child: child,
    );
  }
}

class _MotionGuidanceCard extends StatelessWidget {
  final double yaw;
  final double pitch;
  final int frameCount;
  final double linearAccel;
  final double smoothedLinearAccel;
  final double peakLinearAccel;
  final double accelDelta;
  final double motionMeter;
  final _MotionState motionState;
  final String motionDetail;
  final List<double> accelHistory;
  final bool isMovingTooFast;
  final String motionHint;

  const _MotionGuidanceCard({
    required this.yaw,
    required this.pitch,
    required this.frameCount,
    required this.linearAccel,
    required this.smoothedLinearAccel,
    required this.peakLinearAccel,
    required this.accelDelta,
    required this.motionMeter,
    required this.motionState,
    required this.motionDetail,
    required this.accelHistory,
    required this.isMovingTooFast,
    required this.motionHint,
  });

  @override
  Widget build(BuildContext context) {
    final safePeak = max(peakLinearAccel, _kInstantSpikeAccel);
    final normalizedAccel = (motionMeter / safePeak).clamp(0.0, 1.0);
    final guideColor = switch (motionState) {
      _MotionState.steady => BDDesign.colorMutedBlue,
      _MotionState.ideal => BDDesign.colorFadedOlive,
      _MotionState.caution => const Color(0xFFB88746),
      _MotionState.danger => BDDesign.colorDarkRed,
    };
    final stateLabel = switch (motionState) {
      _MotionState.steady => textLocalize('reco_state_steady'),
      _MotionState.ideal => textLocalize('reco_state_ideal'),
      _MotionState.caution => textLocalize('reco_state_caution'),
      _MotionState.danger => textLocalize('reco_state_danger'),
    };

    return Container(
      width: 260,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: BDDesign.colorInkBlack.withAlpha(216),
        borderRadius: BDDesign.radiusSmall,
        border: Border.all(color: guideColor.withAlpha(160)),
        boxShadow: [BDDesign.shadowElevated],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  color: guideColor,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  motionHint,
                  style: TextStyle(
                    color: BDDesign.colorPaperWhite,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: guideColor.withAlpha(36),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  stateLabel,
                  style: TextStyle(
                    color: guideColor,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              minHeight: 8,
              value: normalizedAccel,
              backgroundColor: Colors.white.withAlpha(28),
              valueColor: AlwaysStoppedAnimation<Color>(guideColor),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '实时 ${motionMeter.toStringAsFixed(2)}  平滑 ${smoothedLinearAccel.toStringAsFixed(2)}  峰值 ${peakLinearAccel.toStringAsFixed(2)}',
            style: const TextStyle(
              color: BDDesign.colorPaperWhite,
              fontFamily: 'Courier',
              fontWeight: FontWeight.bold,
              fontSize: 11,
              letterSpacing: 0.8,
            ),
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 68,
            child: CustomPaint(
              painter: _AccelHistoryPainter(
                samples: accelHistory,
                color: guideColor,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '推荐区间 ${_kIdealAccelMin.toStringAsFixed(2)} - ${_kIdealAccelMax.toStringAsFixed(2)} m/s^2',
            style: TextStyle(
              color: Colors.white.withAlpha(170),
              fontSize: 10,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            motionDetail,
            style: TextStyle(
              color: Colors.white.withAlpha(210),
              fontSize: 11,
              height: 1.35,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'YAW ${yaw.toStringAsFixed(1)}°   PTH ${pitch.toStringAsFixed(1)}°   Δ ${accelDelta.toStringAsFixed(2)}',
            style: const TextStyle(
              color: BDDesign.colorPaperWhite,
              fontFamily: 'Courier',
              fontWeight: FontWeight.bold,
              fontSize: 11,
              letterSpacing: 0.8,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'FRM $frameCount   RAW ${linearAccel.toStringAsFixed(2)}  建议让曲线尽量停在绿色带',
            style: TextStyle(
              color: BDDesign.colorAshGray.withAlpha(220),
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }
}

class _AccelHistoryPainter extends CustomPainter {
  final List<double> samples;
  final Color color;

  const _AccelHistoryPainter({required this.samples, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final bounds = Offset.zero & size;
    final background = Paint()..color = Colors.white.withAlpha(10);
    canvas.drawRRect(
      RRect.fromRectAndRadius(bounds, const Radius.circular(10)),
      background,
    );

    double yFor(double value) {
      final normalized = (value / _kInstantSpikeAccel).clamp(0.0, 1.0);
      return size.height - (size.height * normalized);
    }

    final safeBand = Paint()..color = BDDesign.colorFadedOlive.withAlpha(26);
    final cautionBand = Paint()..color = const Color(0xFFB88746).withAlpha(22);
    final dangerBand = Paint()..color = BDDesign.colorDarkRed.withAlpha(20);

    canvas.drawRect(
      Rect.fromLTRB(
        0,
        yFor(_kIdealAccelMax),
        size.width,
        yFor(_kIdealAccelMin),
      ),
      safeBand,
    );
    canvas.drawRect(
      Rect.fromLTRB(
        0,
        yFor(_kCautionAccelMax),
        size.width,
        yFor(_kIdealAccelMax),
      ),
      cautionBand,
    );
    canvas.drawRect(
      Rect.fromLTRB(
        0,
        yFor(_kInstantSpikeAccel),
        size.width,
        yFor(_kCautionAccelMax),
      ),
      dangerBand,
    );

    final gridPaint = Paint()
      ..color = Colors.white.withAlpha(20)
      ..strokeWidth = 1;
    for (final marker in <double>[
      _kIdealAccelMax,
      _kCautionAccelMax,
      _kDangerAccelMax,
    ]) {
      final y = yFor(marker);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    if (samples.isEmpty) {
      return;
    }

    final path = Path();
    for (var i = 0; i < samples.length; i++) {
      final x = samples.length == 1
          ? size.width
          : (size.width * i) / (samples.length - 1);
      final y = yFor(samples[i]);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final linePaint = Paint()
      ..shader = LinearGradient(
        colors: [color.withAlpha(120), color],
      ).createShader(bounds)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.4
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(path, linePaint);
    canvas.drawCircle(
      Offset(size.width, yFor(samples.last)),
      3.5,
      Paint()..color = color,
    );
  }

  @override
  bool shouldRepaint(covariant _AccelHistoryPainter oldDelegate) {
    return oldDelegate.samples != samples || oldDelegate.color != color;
  }
}
