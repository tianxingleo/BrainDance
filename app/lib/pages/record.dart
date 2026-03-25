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

const double _kIdealAccelMin = 0.08;
const double _kIdealAccelMax = 0.65;
const double _kCautionAccelMax = 1.35;
const double _kDangerAccelMax = 2.10;
const double _kInstantSpikeAccel = 2.60;
const double _kJerkThreshold = 0.95;
const int _kAccelHistoryLength = 40;
const double _kFloatingNavReservedHeight = 112.0;

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
  late AnimationController _hudAnimController; // HUD animation

  bool _showTips = false;
  bool _isMotionHudEnabled = false;
  bool _isMovingTooFast = false; // fast movement warning state
  int _warningEndTime = 0; // warning hold duration

  Timer? _recordTimer;
  int _recordSeconds = 0;

  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<UserAccelerometerEvent>? _userAccelSub;
  StreamSubscription<MagnetometerEvent>? _magSub;

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
      if (_isMotionHudEnabled) {
        _stopSensors();
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
              TDToast.showText(
                context: context,
                textLocalize('reco_app_switch'),
              );
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
      if (_isMotionHudEnabled) {
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
    final displayMotionMeter =
        (_motionMeter * 0.82) + (motionMeter * 0.18);

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
        _motionMeter = displayMotionMeter;
        _motionHint = nextHint;
        _motionDetail = nextDetail;
        _motionState = effectiveState;
      });
    } else {
      _linearAccel = linearAccel;
      _smoothedLinearAccel = smoothedAccel;
      _peakLinearAccel = nextPeak;
      _accelDelta = accelDelta;
      _motionMeter = displayMotionMeter;
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

  void _resetMotionState() {
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
  }

  void _stopSensors() {
    _stopHapticLoop();
    _accelSub?.cancel();
    _accelSub = null;
    _userAccelSub?.cancel();
    _userAccelSub = null;
    _magSub?.cancel();
    _magSub = null;
  }

  void _toggleMotionHud() {
    setState(() {
      _isMotionHudEnabled = !_isMotionHudEnabled;
      if (_isMotionHudEnabled) {
        _resetMotionState();
        _startSensors();
      } else {
        _stopSensors();
        _resetMotionState();
      }
    });
  }

  Future<void> _onRecordTap() async {
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
          label = 'Rear$backCount';
          backCount++;
          break;
        case CameraLensDirection.front:
          label = 'Front$frontCount';
          frontCount++;
          break;
        case CameraLensDirection.external:
          label = 'External$externalCount';
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
                      TDToast.showText(
                        textLocalize('reco_no_switch'),
                        context: context,
                      );
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

  int? _findPrimaryCameraIndex(CameraLensDirection direction) {
    final cameras = RecoConfig.cameras;
    for (var i = 0; i < cameras.length; i++) {
      if (cameras[i].lensDirection == direction) {
        return i;
      }
    }
    return null;
  }

  Future<void> _switchPrimaryCamera() async {
    if (!RecoConfig.cameraEnabled || RecoConfig.cameras.isEmpty) {
      return;
    }

    final currentDirection =
        RecoConfig.cameras[RecoConfig.camNum].lensDirection;
    final targetDirection = currentDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;
    final targetIndex = _findPrimaryCameraIndex(targetDirection);

    if (targetIndex == null || targetIndex == RecoConfig.camNum) {
      if (mounted) {
        TDToast.showText(textLocalize('reco_no_switch'), context: context);
      }
      return;
    }

    try {
      if (ref.read(isRecordingProvider)) {
        await RecoConfig.trySwitchCameraDescription(targetIndex);
      } else {
        RecoConfig.camNum = targetIndex;
        await RecoConfig.cameraInitialize();
      }
      if (mounted) {
        setState(() {});
      }
    } catch (_) {
      if (mounted) {
        TDToast.showText(textLocalize('reco_no_switch'), context: context);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AppConfig.isNightMode;
    final isVideoRecording = ref.watch(isRecordingProvider);
    final isAnyRecording = isVideoRecording;
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

    final mediaQuery = MediaQuery.of(context);
    final currentLensDirection = RecoConfig.cameras.isNotEmpty
        ? RecoConfig.cameras[RecoConfig.camNum].lensDirection
        : CameraLensDirection.back;
    final canSwitchPrimaryCamera =
        _findPrimaryCameraIndex(CameraLensDirection.front) != null &&
        _findPrimaryCameraIndex(CameraLensDirection.back) != null;
    final cornerControlBottom = mediaQuery.padding.bottom + 28;
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
                isWarning: _isMotionHudEnabled && _isMovingTooFast,
                isCaution:
                    _isMotionHudEnabled && _motionState == _MotionState.caution,
                motionValue: _isMotionHudEnabled ? _motionMeter : 0,
                animation: _hudAnimController,
              ),
            ),
          ),
          Positioned(
            top: mediaQuery.padding.top + 16,
            left: 16,
            child: AnimatedSwitcher(
              duration: BDMotion.durationFast,
              switchInCurve: BDMotion.curveEnter,
              switchOutCurve: BDMotion.curveExit,
              child: !_isMotionHudEnabled
                  ? const SizedBox.shrink()
                  : _SimpleMotionGuidanceCard(
                      motionMeter: _motionMeter,
                      motionState: _motionState,
                      motionHint: _motionHint,
                      motionDetail: _motionDetail,
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
                if (isVideoRecording) ...[
                  AnimatedSwitcher(
                    duration: BDMotion.durationFast,
                    switchInCurve: BDMotion.curveEnter,
                    switchOutCurve: BDMotion.curveExit,
                    child: _StatusPill(
                      key: ValueKey<String>('rec_$_recordSeconds'),
                      label:
                          'REC ${_recordSeconds ~/ 60}:${(_recordSeconds % 60).toString().padLeft(2, '0')}',
                      color: Colors.redAccent,
                      backgroundColor: darkInput,
                      isSquareDot: true,
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
          Positioned(
            left: 16,
            bottom: cornerControlBottom,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _RecordOverlayPanel(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 4,
                    vertical: 4,
                  ),
                  child: IconButton(
                    icon: Icon(
                      currentLensDirection == CameraLensDirection.front
                          ? Icons.camera_front
                          : Icons.camera_rear,
                      color: canSwitchPrimaryCamera
                          ? BDDesign.colorPaperWhite
                          : BDDesign.colorAshGray,
                      size: 24,
                    ),
                    onPressed: canSwitchPrimaryCamera
                        ? _switchPrimaryCamera
                        : null,
                  ),
                ),
                if (!isVideoRecording) ...[
                  const SizedBox(width: 12),
                  _RecordOverlayPanel(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 4,
                    ),
                    child: IconButton(
                      icon: const Icon(
                        Icons.close,
                        color: BDDesign.colorAshGray,
                        size: 24,
                      ),
                      onPressed: () => Navigator.maybePop(context),
                    ),
                  ),
                ],
              ],
            ),
          ),
          Positioned(
            right: 16,
            bottom: cornerControlBottom,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (!isVideoRecording) ...[
                  _RecordOverlayPanel(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 4,
                    ),
                    child: IconButton(
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
                  ),
                  const SizedBox(width: 12),
                ],
                _RecordOverlayPanel(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 4,
                    vertical: 4,
                  ),
                  child: IconButton(
                    icon: Icon(
                      _isMotionHudEnabled
                          ? Icons.speed_rounded
                          : Icons.speed_outlined,
                      color: _isMotionHudEnabled
                          ? BDDesign.colorPaperWhite
                          : BDDesign.colorAshGray,
                      size: 24,
                    ),
                    onPressed: _toggleMotionHud,
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
                              label: _isMotionHudEnabled
                                  ? textLocalize('sensor_on')
                                  : textLocalize('sensor_off'),
                              color: _isMotionHudEnabled
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

class _SimpleMotionGuidanceCard extends StatelessWidget {
  final double motionMeter;
  final _MotionState motionState;
  final String motionHint;
  final String motionDetail;

  const _SimpleMotionGuidanceCard({
    required this.motionMeter,
    required this.motionState,
    required this.motionHint,
    required this.motionDetail,
  });

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final cardWidth = (size.width * 0.42).clamp(170.0, 220.0);
    final progressValue = switch (motionState) {
      _MotionState.steady => 0.18,
      _MotionState.ideal => 0.42 + (motionMeter / _kIdealAccelMax) * 0.18,
      _MotionState.caution =>
        0.68 +
            ((motionMeter - _kIdealAccelMax) /
                    (_kCautionAccelMax - _kIdealAccelMax)) *
                0.18,
      _MotionState.danger =>
        0.9 +
            ((motionMeter - _kCautionAccelMax) /
                    (_kInstantSpikeAccel - _kCautionAccelMax)) *
                0.1,
    };
    final normalizedAccel = progressValue.clamp(0.0, 1.0);
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
      width: cardWidth,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: BDDesign.colorInkBlack.withAlpha(216),
        borderRadius: BorderRadius.circular(18),
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
                width: 9,
                height: 9,
                decoration: BoxDecoration(
                  color: guideColor,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  textLocalize('sensor_on'),
                  style: const TextStyle(
                    color: BDDesign.colorPaperWhite,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                decoration: BoxDecoration(
                  color: guideColor.withAlpha(36),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  stateLabel,
                  style: TextStyle(
                    color: guideColor,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          TweenAnimationBuilder<double>(
            tween: Tween<double>(end: normalizedAccel),
            duration: const Duration(milliseconds: 260),
            curve: Curves.easeOutCubic,
            builder: (context, value, _) {
              return ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: LinearProgressIndicator(
                  minHeight: 9,
                  value: value,
                  backgroundColor: Colors.white.withAlpha(28),
                  valueColor: AlwaysStoppedAnimation<Color>(guideColor),
                ),
              );
            },
          ),
          const SizedBox(height: 8),
          Text(
            motionHint,
            style: TextStyle(
              color: Colors.white.withAlpha(210),
              fontSize: 11,
              height: 1.35,
            ),
          ),
          const SizedBox(height: 5),
          Text(
            motionDetail,
            style: TextStyle(
              color: BDDesign.colorAshGray.withAlpha(220),
              fontSize: 10,
              height: 1.3,
            ),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
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

