import 'dart:async';
import 'dart:io';
import 'dart:isolate';
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
import 'package:braindance/widgets/app_toast.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:image/image.dart' as img;
import 'package:supabase_flutter/supabase_flutter.dart';
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

/// 流式拍摄完成气泡（null = 隐藏）
final streamingDoneBubbleProvider = StateProvider<String?>((ref) => null);

/// 流式传输模式开关
final streamingModeProvider = StateProvider<bool>((ref) => false);

/// 拍摄键上方模式切换气泡（null = 隐藏）
final recordModeBubbleProvider = StateProvider<String?>((ref) => null);

/// 流式拍摄实时帧计数
final streamingFrameCountProvider = StateProvider<int>((ref) => 0);

/// 流式上传成功帧数（进度条分子）
final streamingUploadSuccessProvider = StateProvider<int>((ref) => 0);

/// 流式上传失败帧数
final streamingFailCountProvider = StateProvider<int>((ref) => 0);

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

  // ---- streaming mode state ----
  Timer? _streamingTimer;
  String? _streamingSceneId;
  int _streamingFrameIndex = 0;
  final List<(String, String)> _uploadQueue = []; // (filePath, frameLabel)
  bool _queueLocked = false;
  int _streamingSuccessCount = 0;
  int _streamingFailCount = 0;
  static const _streamingIntervalSec = 1;
  static const _streamingMaxDurationSec = 600; // 10 min
  bool _streamingActive = false;
  bool _streamingAborted = false;

  bool get _hasPendingUpload => _uploadQueue.isNotEmpty || _queueLocked;

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
  bool _blockHaptics = false;
  int _currentBackCamPosition = 0;

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
        _syncBackCamPosition();
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
    if (_streamingActive) {
      _streamingTimer?.cancel();
      _streamingTimer = null;
      _handleStreamingAbort();
    } else {
      _streamingTimer?.cancel();
      _streamingTimer = null;
    }
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
        if (ref.read(streamingModeProvider) && _streamingTimer != null) {
          _stopStreaming(finalize: false);
        } else if (controller.value.isRecordingVideo) {
          _stopVideoRecording(
            controller,
            showToast: false,
            navigateToSubmit: true,
          ).whenComplete(() {
            RecoConfig.disposeCamera();
            if (mounted) {
              setState(() {});
              showAppToast(context, textLocalize('reco_app_switch'));
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
    if (mounted) {
      ref.read(isRecordingProvider.notifier).state = value;
    }
  }

  void _resetRecordingState({bool updateUi = false}) {
    _recordTimer?.cancel();
    _recordTimer = null;
    _streamingTimer?.cancel();
    _streamingTimer = null;
    _resetStreamingState();
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
        await Future.delayed(Duration(milliseconds: _minRecordingMs - elapsed));
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

    if (navigateToSubmit && mounted) {
      _blockHaptics = true;
      _stopHapticLoop();
      await Navigator.push(
        context,
        PageRouteBuilder(
          transitionDuration: BDMotion.durationNormal,
          reverseTransitionDuration: BDMotion.durationNormal,
          opaque: true,
          pageBuilder: (_, __, ___) =>
              VideoSubmitPage(videoPath: file.path, thumbnailPath: thumbPath),
          transitionsBuilder: (ctx, animation, __, child) {
            final curved = animation.drive(
              CurveTween(curve: Curves.easeInOutCubic),
            );
            final screenHeight = MediaQuery.sizeOf(ctx).height;
            return AnimatedBuilder(
              animation: curved,
              builder: (_, child) {
                return Transform.translate(
                  offset: Offset(0, -(1.0 - curved.value) * screenHeight),
                  child: child,
                );
              },
              child: child,
            );
          },
        ),
      );
      _blockHaptics = false;
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

    final isRecording = ref.read(isRecordingProvider);
    if (isRecording) {
      try {
        if (ref.read(streamingModeProvider)) {
          await _stopStreaming();
        } else {
          await _stopVideoRecording(controller);
        }
      } finally {
        _isToggling = false;
        if (mounted) setState(() {});
      }
      return;
    }

    // 流式拍摄进行中，阻止新一轮
    if (_streamingActive) {
      _isToggling = false;
      return;
    }

    if (ref.read(streamingModeProvider)) {
      await _startStreaming(controller);
    } else {
      _resetRecordingState(updateUi: false);
      _setGlobalRecording(true);
      try {
        await controller.startVideoRecording();
      } catch (_) {
        _resetRecordingState(updateUi: true);
        _isToggling = false;
        return;
      }
      try {
        await RecoConfig.trySwitchCameraDescription(RecoConfig.camNum);
      } catch (_) {}
      _recordSeconds = 0;
      _recordingStartTime = DateTime.now();

      _recordTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        _recordSeconds++;
        if (_recordSeconds >= 600) {
          _stopVideoRecording(controller);
        } else if (mounted) {
          setState(() {});
        }
      });
    }

    _isToggling = false;
  }

  String _generateStreamingSceneId() {
    final time = DateTime.now();
    final r = Random();
    return 'scene_'
        '${time.year}'
        '${time.month.toString().padLeft(2, '0')}'
        '${time.day.toString().padLeft(2, '0')}'
        '_'
        '${r.nextInt(1000000).toString().padLeft(6, '0')}';
  }

  Future<bool> _showStopStreamingDialog() async {
    final hasPending = _hasPendingUpload;
    final captured = _streamingFrameIndex;
    final uploaded = _streamingSuccessCount;
    final pending = _uploadQueue.length + (_queueLocked ? 1 : 0);

    bool deleteUploaded = true;

    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        final isDark = AppConfig.isNightMode;
        return StatefulBuilder(
          builder: (builderContext, setDialogState) {
            return AlertDialog(
              title: Text(
                textLocalize('stream_stop_title'),
                style: TextStyle(
                  color: isDark ? Colors.white : Colors.black87,
                  fontWeight: FontWeight.w600,
                ),
              ),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    hasPending
                        ? textLocalize('stream_stop_uploading')
                            .replaceFirst('%d', pending.toString())
                            .replaceFirst('%d', uploaded.toString())
                        : textLocalize('stream_stop_capturing')
                            .replaceFirst('%d', captured.toString())
                            .replaceFirst('%d', uploaded.toString()),
                    style: TextStyle(
                      color: isDark ? Colors.white70 : Colors.black54,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      SizedBox(
                        width: 24,
                        height: 24,
                        child: Checkbox(
                          value: deleteUploaded,
                          onChanged: (v) {
                            deleteUploaded = v ?? true;
                            setDialogState(() {});
                          },
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        textLocalize('stream_stop_delete'),
                        style: TextStyle(
                          color: isDark ? Colors.white70 : Colors.black87,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(false),
                  child: Text(textLocalize('stream_stop_btn_cancel')),
                ),
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(true),
                  style: TextButton.styleFrom(
                    foregroundColor: Colors.redAccent,
                  ),
                  child: Text(textLocalize('stream_stop_btn_confirm')),
                ),
              ],
            );
          },
        );
      },
    );

    if (result == true) {
      await _performStopFromDialog(
        hasPendingUploads: hasPending,
        deleteUploaded: deleteUploaded,
      );
      return true;
    }
    return false;
  }

  Future<void> _performStopFromDialog({
    required bool hasPendingUploads,
    required bool deleteUploaded,
  }) async {
    if (deleteUploaded) {
      _handleStreamingAbort();
      return;
    }

    _streamingActive = false;
    _streamingTimer?.cancel();
    _streamingTimer = null;
    _setGlobalRecording(false);
    if (mounted) {
      ref.read(streamingDoneBubbleProvider.notifier).state =
          textLocalize('stream_stop_title');
    }

    if (hasPendingUploads) {
      final drainStart = DateTime.now();
      while (_uploadQueue.isNotEmpty || _queueLocked) {
        if (DateTime.now().difference(drainStart).inSeconds > 30) break;
        await Future.delayed(const Duration(milliseconds: 100));
      }
      while (_queueLocked) {
        if (DateTime.now().difference(drainStart).inSeconds > 40) break;
        await Future.delayed(const Duration(milliseconds: 100));
      }
    } else {
      final drainStart = DateTime.now();
      while (_queueLocked) {
        if (DateTime.now().difference(drainStart).inSeconds > 5) break;
        await Future.delayed(const Duration(milliseconds: 100));
      }
    }

    final finalCount = _streamingSuccessCount;

    if (finalCount < 3) {
      // 用户选择保留已上传帧 → 尊重用户选择，不删除
      _resetStreamingState();
      if (mounted) {
        ref.read(streamingDoneBubbleProvider.notifier).state =
            '${textLocalize('stream_upload_fail')}（至少需要 3 帧，当前 $finalCount 帧）';
      }
      return;
    }

    final user = Supabase.instance.client.auth.currentUser;
    final sceneId = _streamingSceneId;
    if (user == null || sceneId == null) {
      _resetStreamingState();
      return;
    }

    try {
      await Supabase.instance.client.from('processing_tasks').insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'task_type': 'video_dual_chain',
        'task_params': {'image_count': finalCount},
        'status': 'pending',
      });
      debugPrint(
        '[streaming] task created via dialog: $sceneId ($finalCount frames)',
      );
      _resetStreamingState();
      if (mounted) {
        ref.read(streamingDoneBubbleProvider.notifier).state =
            '${textLocalize('stream_success')}（$finalCount 帧）';
      }
    } catch (e) {
      debugPrint('[streaming] dialog task creation error: $e');
      await _deleteStreamingAssets(user.id, sceneId);
      _resetStreamingState();
      if (mounted) {
        ref.read(streamingDoneBubbleProvider.notifier).state =
            textLocalize('stream_fail_task');
      }
    }
  }

  Future<void> _startStreaming(CameraController controller) async {
    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) {
      _isToggling = false;
      return;
    }

    _streamingSceneId = _generateStreamingSceneId();
    _streamingFrameIndex = 0;
    _streamingSuccessCount = 0;
    _streamingFailCount = 0;
    _uploadQueue.clear();
    _queueLocked = false;
    _streamingAborted = false;
    _streamingActive = true;
    _recordSeconds = 0;
    _setGlobalRecording(true);

    if (mounted) {
      ref.read(streamingFrameCountProvider.notifier).state = 0;
      ref.read(streamingUploadSuccessProvider.notifier).state = 0;
      ref.read(streamingFailCountProvider.notifier).state = 0;
      ref.read(streamingDoneBubbleProvider.notifier).state = null;
    }

    // Capture first frame immediately, then every 1s
    _captureAndQueueFrame(controller);
    _streamingTimer = Timer.periodic(
      const Duration(seconds: _streamingIntervalSec),
      (_) {
        _recordSeconds++;
        if (_recordSeconds >= _streamingMaxDurationSec) {
          _stopStreaming();
          return;
        }
        if (mounted) {
          setState(() {});
          _captureAndQueueFrame(controller);
        }
      },
    );

    if (mounted) setState(() {});
  }

  Future<void> _captureAndQueueFrame(CameraController controller) async {
    if (!_streamingActive) return;

    XFile? photo;
    try {
      photo = await controller.takePicture();
    } catch (e) {
      debugPrint('[_captureAndQueue] takePicture failed: $e');
      if (mounted) setState(() {});
      return;
    }

    final frameIndex = ++_streamingFrameIndex;
    final frameLabel = 'picture_${frameIndex.toString().padLeft(5, '0')}';
    _uploadQueue.add((photo.path, frameLabel));
    debugPrint('[_capture] $frameLabel → queue=${_uploadQueue.length}');

    if (mounted) {
      ref.read(streamingFrameCountProvider.notifier).state =
          _streamingFrameIndex;
    }

    _processUploadQueue();
  }

  Future<void> _processUploadQueue() async {
    if (_queueLocked) {
      debugPrint('[_uploadQueue] skipped (locked)');
      return;
    }
    debugPrint('[_uploadQueue] start, queue=${_uploadQueue.length}');
    _queueLocked = true;

    final userId = Supabase.instance.client.auth.currentUser?.id;
    final sceneId = _streamingSceneId;
    if (userId == null || sceneId == null) {
      debugPrint(
        '[_uploadQueue] abort: userId=$userId sceneId=$sceneId',
      );
      _queueLocked = false;
      return;
    }

    try {
      while (_uploadQueue.isNotEmpty) {
        final (filePath, frameLabel) = _uploadQueue.removeAt(0);

        String? compressedPath;
        try {
          compressedPath = await _compressImage(filePath);
          final storagePath = '$userId/$sceneId/raw/$frameLabel.jpg';

          await Supabase.instance.client.storage
              .from('braindance-assets')
              .upload(storagePath, File(compressedPath));

          _streamingSuccessCount++;
          debugPrint(
            '[_uploadQueue] ok $frameLabel ($_streamingSuccessCount total)',
          );
        } catch (e) {
          _streamingFailCount++;
          debugPrint('[_uploadQueue] fail $frameLabel: $e');
          if (mounted) {
            ref.read(streamingFailCountProvider.notifier).state =
                _streamingFailCount;
            if (_streamingFailCount <= 1) {
              showAppToast(context, textLocalize('stream_frame_fail'));
            }
          }
        } finally {
          unawaited(File(filePath).delete().catchError((_) {}));
          if (compressedPath != null) {
            unawaited(File(compressedPath).delete().catchError((_) {}));
          }
        }

        if (mounted) {
          ref.read(streamingUploadSuccessProvider.notifier).state =
              _streamingSuccessCount;
        }
      }
    } catch (e) {
      debugPrint('[_uploadQueue] processor crashed: $e');
      _handleStreamingAbort();
      return;
    }

    _queueLocked = false;
    debugPrint('[_uploadQueue] done, success=$_streamingSuccessCount fail=$_streamingFailCount');
    if (_uploadQueue.isNotEmpty) {
      _processUploadQueue();
    }
  }

  void _handleStreamingAbort() {
    print(
      '[abort] called, aborted=$_streamingAborted active=$_streamingActive',
    );
    if (_streamingAborted) return;
    _streamingAborted = true;
    _streamingActive = false;
    _streamingTimer?.cancel();
    _streamingTimer = null;
    _setGlobalRecording(false);
    _queueLocked = false;

    final userId = Supabase.instance.client.auth.currentUser?.id;
    final sceneId = _streamingSceneId;
    print('[abort] userId=$userId sceneId=$sceneId');
    if (userId != null && sceneId != null) {
      print('[abort] firing delete for $userId/$sceneId');
      unawaited(_deleteStreamingAssets(userId, sceneId));
    }
    _resetStreamingState();

    if (mounted) {
      ref.read(streamingDoneBubbleProvider.notifier).state = textLocalize(
        'stream_upload_fail',
      );
    }
  }

  Future<String> _compressImage(String filePath) async {
    return Isolate.run(() async {
      final bytes = await File(filePath).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) {
        throw Exception('Failed to decode image');
      }
      final compressed = img.encodeJpg(decoded, quality: 85);
      final compressedPath = '${filePath}_c.jpg';
      await File(compressedPath).writeAsBytes(compressed);
      return compressedPath;
    });
  }

  Future<void> _stopStreaming({bool finalize = true}) async {
    _streamingActive = false;
    _streamingTimer?.cancel();
    _streamingTimer = null;
    _setGlobalRecording(false);
    if (mounted && finalize) {
      ref.read(streamingDoneBubbleProvider.notifier).state =
          textLocalize('stream_stop_title');
    }

    if (!mounted) {
      _resetStreamingState();
      return;
    }
    final _drainStart = DateTime.now();
    while (_uploadQueue.isNotEmpty || _queueLocked) {
      if (DateTime.now().difference(_drainStart).inSeconds > 30) {
        debugPrint('[_stopStreaming] queue drain timeout');
        break;
      }
      await Future.delayed(const Duration(milliseconds: 100));
    }
    // 等飞行中的最后一帧上传完成，确保 _streamingSuccessCount 准确
    while (_queueLocked) {
      if (DateTime.now().difference(_drainStart).inSeconds > 40) break;
      await Future.delayed(const Duration(milliseconds: 100));
    }

    if (!mounted) {
      _resetStreamingState();
      return;
    }

    final finalCount = _streamingSuccessCount;

    if (finalCount < 3) {
      if (finalize) {
        final u = Supabase.instance.client.auth.currentUser;
        final sid = _streamingSceneId;
        if (u != null && sid != null) {
          await _deleteStreamingAssets(u.id, sid);
        }
      }
      debugPrint('[streaming] too few frames: $finalCount, finalize=$finalize');
      _resetStreamingState();
      ref.read(streamingDoneBubbleProvider.notifier).state = finalize
          ? textLocalize('stream_upload_fail')
          : textLocalize('stream_stop_title');
      if (mounted) setState(() {});
      return;
    }

    final user = Supabase.instance.client.auth.currentUser;
    final sceneId = _streamingSceneId;
    if (user == null || sceneId == null) {
      _resetStreamingState();
      if (mounted) {
        ref.read(streamingDoneBubbleProvider.notifier).state = textLocalize(
          'stream_fail_task',
        );
      }
      return;
    }

    try {
      await Supabase.instance.client.from('processing_tasks').insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'task_type': 'video_dual_chain',
        'task_params': {'image_count': finalCount},
        'status': 'pending',
      });

      debugPrint('[streaming] task created: $sceneId ($finalCount frames)');
      _resetStreamingState();
      if (mounted) {
        ref.read(streamingDoneBubbleProvider.notifier).state =
            '${textLocalize('stream_success')}（$finalCount 帧）';
      }
    } catch (e) {
      print('[_stopStreaming] task creation error: $e');
      await _deleteStreamingAssets(user.id, sceneId);
      _resetStreamingState();
      if (mounted) {
        ref.read(streamingDoneBubbleProvider.notifier).state =
            textLocalize('stream_fail_task');
      }
    }
    if (mounted) setState(() {});
  }

  Future<void> _deleteStreamingAssets(String userId, String sceneId) async {
    final prefix = '$userId/$sceneId/raw';
    print('[del] attempting cleanup prefix=$prefix');
    try {
      final result = await Supabase.instance.client.storage
          .from('braindance-assets')
          .list(path: prefix);
      final paths = result.map((f) => '$prefix/${f.name}').toList();
      print(
        '[del] found ${paths.length} files: ${paths.map((p) => p.split('/').last).toList()}',
      );
      if (paths.isNotEmpty) {
        await Supabase.instance.client.storage
            .from('braindance-assets')
            .remove(paths);
        print('[del] deleted ${paths.length} orphan files');
      } else {
        print('[del] no files to delete');
      }
    } catch (e) {
      print('[del] cleanup error: $e');
    }
  }

  void _resetStreamingState() {
    for (final (filePath, _) in _uploadQueue) {
      unawaited(File(filePath).delete().catchError((_) {}));
    }
    _uploadQueue.clear();
    _queueLocked = false;
    _streamingAborted = false;
    _streamingSceneId = null;
    _streamingFrameIndex = 0;
    _streamingSuccessCount = 0;
    _streamingFailCount = 0;
    _streamingActive = false;
    if (mounted) {
      ref.read(streamingFrameCountProvider.notifier).state = 0;
      ref.read(streamingUploadSuccessProvider.notifier).state = 0;
      ref.read(streamingFailCountProvider.notifier).state = 0;
    }
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
    if (_blockHaptics) {
      _stopHapticLoop();
      return;
    }
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

  void _syncBackCamPosition() {
    if (RecoConfig.cameras.isEmpty) return;
    if (RecoConfig.cameras[RecoConfig.camNum].lensDirection !=
        CameraLensDirection.back) {
      return;
    }
    final backIndices = RecoConfig.backCameraIndices;
    final pos = backIndices.indexOf(RecoConfig.camNum);
    if (pos >= 0) {
      _currentBackCamPosition = pos;
    }
  }

  Future<void> _cycleRearCamera() async {
    if (!RecoConfig.cameraEnabled) return;

    final backIndices = RecoConfig.backCameraIndices;
    if (backIndices.length <= 1) return;

    _currentBackCamPosition = (_currentBackCamPosition + 1) % backIndices.length;
    final targetIndex = backIndices[_currentBackCamPosition];

    try {
      if (ref.read(isRecordingProvider)) {
        await RecoConfig.trySwitchCameraDescription(targetIndex);
      } else {
        RecoConfig.camNum = targetIndex;
        await RecoConfig.cameraInitialize();
      }
      if (mounted) setState(() {});
    } catch (_) {
      if (mounted) {
        showAppToast(context, textLocalize('reco_no_switch'));
      }
    }
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

    int targetIndex;

    if (currentDirection == CameraLensDirection.front) {
      final backIndices = RecoConfig.backCameraIndices;
      if (backIndices.isEmpty) {
        if (mounted) showAppToast(context, textLocalize('reco_no_switch'));
        return;
      }
      if (_currentBackCamPosition >= backIndices.length) {
        _currentBackCamPosition = 0;
      }
      targetIndex = backIndices[_currentBackCamPosition];
    } else {
      final frontIndex = _findPrimaryCameraIndex(CameraLensDirection.front);
      if (frontIndex == null) {
        if (mounted) showAppToast(context, textLocalize('reco_no_switch'));
        return;
      }
      targetIndex = frontIndex;
    }

    if (targetIndex == RecoConfig.camNum) return;

    try {
      if (ref.read(isRecordingProvider)) {
        await RecoConfig.trySwitchCameraDescription(targetIndex);
      } else {
        RecoConfig.camNum = targetIndex;
        await RecoConfig.cameraInitialize();
      }
      if (mounted) setState(() {});
    } catch (_) {
      if (mounted) {
        showAppToast(context, textLocalize('reco_no_switch'));
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
      final size = MediaQuery.sizeOf(context);
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

    final bottomPadding = MediaQuery.paddingOf(context).bottom;
    final currentLensDirection = RecoConfig.cameras.isNotEmpty
        ? RecoConfig.cameras[RecoConfig.camNum].lensDirection
        : CameraLensDirection.back;
    final canSwitchPrimaryCamera =
        _findPrimaryCameraIndex(CameraLensDirection.front) != null &&
        _findPrimaryCameraIndex(CameraLensDirection.back) != null;
    final backCameraCount = RecoConfig.backCameraIndices.length;
    final showRearCamBadge =
        currentLensDirection == CameraLensDirection.back &&
        backCameraCount > 1;
    final cornerControlBottom = bottomPadding + 28;
    final bottomOffset =
        bottomPadding +
        (isAnyRecording ? 36 : _kFloatingNavReservedHeight + 18);

    return PopScope(
      canPop: !_streamingActive && !_hasPendingUpload,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) {
          FocusManager.instance.primaryFocus?.unfocus();
        } else {
          final shouldStop = await _showStopStreamingDialog();
          if (shouldStop && mounted) setState(() {});
        }
      },
      child: Scaffold(
        resizeToAvoidBottomInset: false,
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
              top: MediaQuery.paddingOf(context).top + 16,
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
              top: MediaQuery.paddingOf(context).top + 90,
              left: 0,
              right: 0,
              child: const Center(child: _StreamingStatusCard()),
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
                  if (showRearCamBadge) ...[
                    const SizedBox(width: 8),
                    _RecordOverlayPanel(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 6,
                      ),
                      child: GestureDetector(
                        onTap: _cycleRearCamera,
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(
                              Icons.lens,
                              color: BDDesign.colorPaperWhite,
                              size: 13,
                            ),
                            const SizedBox(width: 5),
                            Text(
                              '${_currentBackCamPosition + 1}/$backCameraCount',
                              style: const TextStyle(
                                color: BDDesign.colorPaperWhite,
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                fontFeatures: [FontFeature.tabularFigures()],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                  if (!isVideoRecording) ...[
                    const SizedBox(width: 12),
                    _RecordOverlayPanel(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 4,
                        vertical: 4,
                      ),
                      child: IconButton(
                        icon: Icon(
                          ref.watch(streamingModeProvider)
                              ? Icons.photo_camera
                              : Icons.videocam_outlined,
                          color: ref.watch(streamingModeProvider)
                              ? BDDesign.colorPaperWhite
                              : BDDesign.colorAshGray,
                          size: 24,
                        ),
                        onPressed: () async {
                          final next = !ref.read(streamingModeProvider);
                          ref.read(streamingModeProvider.notifier).state = next;
                          ref.read(recordModeBubbleProvider.notifier).state = next
                              ? '已切换为流式传输模式'
                              : '已切换为普通录制模式';
                        },
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
                        maxHeight: MediaQuery.sizeOf(context).height * 0.70,
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
            _TipBubble(
              provider: saveFailBubbleProvider,
              alignment: const Alignment(0, 0.30),
              iconBuilder: () => Icons.error_outline_rounded,
              iconColorBuilder: () => Colors.orange.withAlpha(200),
            ),
            _TipBubble(
              provider: streamingDoneBubbleProvider,
              alignment: const Alignment(0, 0.72),
              iconBuilder: () {
                final s = ref.read(streamingUploadSuccessProvider);
                final f = ref.read(streamingFailCountProvider);
                return (s > 0 && f == 0)
                    ? Icons.check_circle_outline
                    : Icons.error_outline;
              },
              iconColorBuilder: () {
                final s = ref.read(streamingUploadSuccessProvider);
                final f = ref.read(streamingFailCountProvider);
                return (s > 0 && f == 0)
                    ? Colors.white
                    : Colors.redAccent.withAlpha(200);
              },
              bgColorBuilder: () {
                final s = ref.read(streamingUploadSuccessProvider);
                final f = ref.read(streamingFailCountProvider);
                return (s > 0 && f == 0)
                    ? const Color(0xE62E7D32)
                    : null;
              },
            ),
            _TipBubble(
              provider: recordModeBubbleProvider,
              alignment: const Alignment(0, 0.72),
              iconBuilder: () => Icons.info_outline_rounded,
              iconColorBuilder: () => Colors.white.withAlpha(180),
            ),
            _CenterBubble(
              provider: showTooShortBubbleProvider,
              message: textLocalize('reco_record_too_short'),
              icon: Icons.info_outline_rounded,
              iconColor: Colors.white.withAlpha(180),
            ),
          ],
        ),
      ),
    );
  }
}

class _StreamingStatusCard extends ConsumerStatefulWidget {
  const _StreamingStatusCard();

  @override
  ConsumerState<_StreamingStatusCard> createState() =>
      _StreamingStatusCardState();
}

class _StreamingStatusCardState extends ConsumerState<_StreamingStatusCard> {
  double _prevProportion = 0.0;

  @override
  Widget build(BuildContext context) {
    final captured = ref.watch(streamingFrameCountProvider);
    final uploaded = ref.watch(streamingUploadSuccessProvider);
    final failed = ref.watch(streamingFailCountProvider);

    if (captured == 0) return const SizedBox.shrink();

    final cardWidth = (MediaQuery.sizeOf(context).width * 0.44).clamp(
      170.0,
      240.0,
    );

    // Live mode: streaming is active
    final rawProportion = captured > 0 ? uploaded / captured : 0.0;
    final proportion = rawProportion > _prevProportion
        ? rawProportion
        : _prevProportion;
    final hasFailures = failed > 0;

    final result = Container(
      width: cardWidth,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: BDDesign.colorInkBlack.withAlpha(216),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: (hasFailures ? Colors.orangeAccent : BDDesign.colorFadedOlive)
              .withAlpha(160),
        ),
        boxShadow: [BDDesign.shadowElevated],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Icon(
                Icons.cloud_upload_outlined,
                color: BDDesign.colorFadedOlive,
                size: 14,
              ),
              const SizedBox(width: 8),
              Text(
                textLocalize('stream_uploading').replaceFirst(
                  '%d',
                  uploaded.toString(),
                ),
                style: const TextStyle(
                  color: BDDesign.colorPaperWhite,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              if (hasFailures) ...[
                const SizedBox(width: 8),
                Icon(Icons.close, color: Colors.redAccent, size: 12),
                const SizedBox(width: 2),
                Text(
                  failed.toString(),
                  style: const TextStyle(
                    color: Colors.redAccent,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              const Icon(
                Icons.photo_camera_outlined,
                color: BDDesign.colorAshGray,
                size: 12,
              ),
              const SizedBox(width: 8),
              Text(
                textLocalize('stream_progress').replaceFirst(
                  '%d',
                  captured.toString(),
                ),
                style: const TextStyle(
                  color: BDDesign.colorAshGray,
                  fontSize: 11,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              minHeight: 6,
              value: proportion,
              backgroundColor: Colors.white.withAlpha(28),
              valueColor: AlwaysStoppedAnimation<Color>(
                hasFailures
                    ? Colors.orangeAccent
                    : BDDesign.colorFadedOlive,
              ),
            ),
          ),
        ],
      ),
    );

    _prevProportion = proportion;
    return result;
  }
}
