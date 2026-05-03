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
import 'package:flutter_riverpod/legacy.dart';
part 'record/record_ui_widgets.dart';
part 'record/record_motion_guidance_card.dart';

const double _kIdealAccelMin = 0.08;
const double _kIdealAccelMax = 0.65;
const double _kCautionAccelMax = 1.35;
const double _kDangerAccelMax = 2.10;
const double _kInstantSpikeAccel = 2.60;
const double _kJerkThreshold = 0.95;
const int _kAccelHistoryLength = 40;
const double _kFloatingNavReservedHeight = 112.0;

/// 加速度过快 → 顶部横幅
final showAccelBannerProvider = StateProvider<bool>((ref) => false);

/// 保存到相册失败 → 中下方悬浮气泡（null = 隐藏）
final saveFailBubbleProvider = StateProvider<String?>((ref) => null);

/// 录制时间过短 → 中央气泡
final showTooShortBubbleProvider = StateProvider<bool>((ref) => false);

/// 录制完成 → 中央气泡
final showRecoDoneBubbleProvider = StateProvider<bool>((ref) => false);

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
  bool _isToggling = false;
  DateTime? _recordingStartTime;
  static const _minRecordingMs = 500;

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
  double _motionMeter = 0;
  String _motionHint = textLocalize('reco_motion_steady');
  String _motionDetail = textLocalize('reco_motion_detail');
  int _lastFastToastTime = 0;
  Timer? _hapticLoopTimer;
  _MotionState? _hapticLoopState;
  _MotionState _motionState = _MotionState.steady;

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

  void _resetRecordingState({bool updateUi = false}) {
    _recordTimer?.cancel();
    _recordTimer = null;
    _recordSeconds = 0;
    _recordingStartTime = null;
    _setGlobalRecording(false);
    if (updateUi && mounted) {
      setState(() {});
    }
  }

  Future<void> _stopVideoRecording(
    CameraController controller, {
    bool showToast = true,
    bool navigateToSubmit = true,
  }) async {
    final startTime = _recordingStartTime;
    _resetRecordingState(updateUi: true);

    if (!controller.value.isRecordingVideo) {
      return;
    }

    if (startTime != null) {
      final elapsed = DateTime.now().difference(startTime).inMilliseconds;
      if (elapsed < _minRecordingMs) {
        _isToggling = true;
        await Future.delayed(
          Duration(milliseconds: _minRecordingMs - elapsed),
        );
        _isToggling = false;
      }
    }

    XFile file;
    try {
      file = await controller.stopVideoRecording();
    } catch (_) {
      if (showToast && mounted) {
        ref.read(showTooShortBubbleProvider.notifier).state = true;
      }
      return;
    }

    final permissionState = await PhotoManager.requestPermissionExtend();
    if (!permissionState.isAuth) {
      if (showToast && mounted) {
        ref.read(saveFailBubbleProvider.notifier).state = textLocalize(
          'reco_save_fail',
        );
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
          ref.read(saveFailBubbleProvider.notifier).state = textLocalize(
            'reco_save_error',
          );
        }
      } catch (_) {
        if (mounted) {
          ref.read(saveFailBubbleProvider.notifier).state = textLocalize(
            'reco_save_error',
          );
        }
      }
    }

    var thumbPath = file.path;
    try {
      thumbPath = await VThumb.ensureThumb(file.path);
    } catch (_) {}

    if (thumbPath.startsWith('assets/')) {
      if (showToast && mounted) {
        ref.read(showTooShortBubbleProvider.notifier).state = true;
      }
      return;
    }

    if (showToast && mounted) {
      ref.read(showRecoDoneBubbleProvider.notifier).state = true;
    }

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
    if (_isToggling) return;
    _isToggling = true;

    final controller = RecoConfig.cameraController;
    if (controller == null || !controller.value.isInitialized) {
      _isToggling = false;
      return;
    }

    final isVideoRecording = ref.read(isRecordingProvider);
    if (isVideoRecording) {
      try {
        await _stopVideoRecording(controller);
      } finally {
        _isToggling = false;
      }
      return;
    }

    _resetRecordingState(updateUi: false);
    _setGlobalRecording(true);
    try {
      await controller.startVideoRecording();
    } catch (_) {
      _resetRecordingState(updateUi: true);
      _isToggling = false;
      rethrow;
    }
    try {
      await RecoConfig.trySwitchCameraDescription(RecoConfig.camNum);
    } catch (_) {}
    _recordSeconds = 0;
    _recordingStartTime = DateTime.now();

    _recordTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      _recordSeconds++;
      if (_recordSeconds >= 180) {
        _stopVideoRecording(controller);
      } else if (mounted) {
        setState(() {});
      }
    });

    _isToggling = false;
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
    final displayMotionMeter = (_motionMeter * 0.82) + (motionMeter * 0.18);

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

    if (_isMovingTooFast &&
        mounted &&
        !ref.read(showAccelBannerProvider) &&
        now - _lastFastToastTime > 1800) {
      _lastFastToastTime = now;
      ref.read(showAccelBannerProvider.notifier).state = true;
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
        _motionMeter = displayMotionMeter;
        _motionHint = nextHint;
        _motionDetail = nextDetail;
        _motionState = effectiveState;
      });
    } else {
      _linearAccel = linearAccel;
      _smoothedLinearAccel = smoothedAccel;
      _peakLinearAccel = nextPeak;
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

    return PopScope(
      onPopInvokedWithResult: (didPop, _) {
        if (didPop) FocusManager.instance.primaryFocus?.unfocus();
      },
      child: Scaffold(
        backgroundColor: isDark ? const Color(0xFF101014) : Colors.black,
        body: Stack(
          children: [
            Positioned.fill(child: cameraView),
            Positioned.fill(
              child: CustomPaint(
                painter: RecordHUDPainter(
                  isWarning: _isMotionHudEnabled && _isMovingTooFast,
                  isCaution:
                      _isMotionHudEnabled &&
                      _motionState == _MotionState.caution,
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
            const _AccelWarningBanner(),
            const _SaveFailBubble(),
            _CenterBubble(
              provider: showTooShortBubbleProvider,
              message: textLocalize('reco_record_too_short'),
              icon: Icons.info_outline_rounded,
              iconColor: Colors.white.withAlpha(180),
            ),
            _CenterBubble(
              provider: showRecoDoneBubbleProvider,
              message: textLocalize('reco_done'),
              icon: Icons.check_circle_outline_rounded,
              iconColor: BDDesign.colorFadedOlive,
              durationSeconds: 2,
            ),
          ],
        ),
      ),
    );
  }
}
