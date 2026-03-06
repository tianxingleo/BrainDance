import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'package:camera/camera.dart';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
import 'package:braindance/pages/video_submit.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:braindance/main.dart' show isRecordingProvider;
import 'dart:async';

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

    //相机初始化
    if (!RecoConfig.cameraEnabled) {
      return;
    }
    RecoConfig.onUpdate = () {
      setState(() {});
    };
    RecoConfig.cameraInitialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _recordTimer?.cancel();
    _buttonAnimController.dispose();
    RecoConfig.disposeCamera();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    final controller = RecoConfig.cameraController;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      if (controller != null && controller.value.isInitialized) {
        if (controller.value.isRecordingVideo) {
          _stopRecording(controller).whenComplete(() {
            RecoConfig.disposeCamera();
            if (mounted) setState(() {});
          });
        } else {
          RecoConfig.disposeCamera();
          if (mounted) setState(() {});
        }
      }
    } else if (state == AppLifecycleState.resumed) {
      if (RecoConfig.cameraController == null) {
        RecoConfig.cameraInitialize();
      }
    }
  }

  Timer? _recordTimer;
  int _recordSeconds = 0;

  Future<void> _stopRecording(CameraController controller) async {
    _recordTimer?.cancel();
    _recordTimer = null;
    if (controller.value.isRecordingVideo) {
      if (mounted) {
        ref.read(isRecordingProvider.notifier).state = false;
      }
      final file = await controller.stopVideoRecording();
      if (mounted) TDToast.showText('录制完成', context: context);

      String thumbPath = file.path;
      try {
        thumbPath = await VThumb.ensureThumb(file.path);
      } catch (_) {}

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) =>
                VideoSubmitPage(videoPath: file.path, thumbnailPath: thumbPath),
          ),
        );
      }
    }
  }

  void _onRecordTap() async {
    final controller = RecoConfig.cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    final isRecording = ref.read(isRecordingProvider);

    if (isRecording) {
      // Manual stop
      _stopRecording(controller);
    } else {
      // Start recording
      await controller.startVideoRecording();
      ref.read(isRecordingProvider.notifier).state = true;
      _recordSeconds = 0;

      _recordTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        _recordSeconds++;
        if (_recordSeconds >= 180) {
          // 3 min
          _stopRecording(controller);
        } else if (mounted) {
          // Trigger a minor rebuild to pulse or update something if needed
          setState(() {});
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final isRecording = ref.watch(isRecordingProvider);

    late final Widget cameraView;
    if (!RecoConfig.cameraEnabled) {
      cameraView = Center(
        child: Text(
          textLocalize("reco_camun"),
          style: TextStyle(fontSize: 18, color: Colors.white70),
        ),
      );
    } else if (RecoConfig.cameraController == null ||
        !RecoConfig.cameraController!.value.isInitialized) {
      cameraView = Center(
        child: Text(
          textLocalize("reco_wait"),
          style: TextStyle(fontSize: 18, color: Colors.white70),
        ),
      );
    } else {
      cameraView = CameraPreview(RecoConfig.cameraController!);
    }
    // 获取所有可用摄像头
    final cameras = RecoConfig.cameras;
    final List<Widget> cameraSwitchButtons = [];

    int getLensPriority(CameraLensDirection dir) {
      if (dir == CameraLensDirection.back) return 1;
      if (dir == CameraLensDirection.front) return 2;
      return 3;
    }

    final sortedIndices = List<int>.generate(cameras.length, (i) => i)
      ..sort(
        (a, b) => getLensPriority(
          cameras[a].lensDirection,
        ).compareTo(getLensPriority(cameras[b].lensDirection)),
      );

    int backCount = 1;
    int frontCount = 1;
    int externalCount = 1;

    for (int i in sortedIndices) {
      final cam = cameras[i];
      String label = '';
      switch (cam.lensDirection) {
        case CameraLensDirection.back:
          label = backCount == 1 ? '主摄' : '广角';
          backCount++;
          break;
        case CameraLensDirection.front:
          label = frontCount == 1 ? '自拍' : '前置$frontCount';
          frontCount++;
          break;
        case CameraLensDirection.external:
          label = '外置$externalCount';
          externalCount++;
          break;
      }

      final bool isSelected = RecoConfig.camNum == i;

      cameraSwitchButtons.add(
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(24),
              onTap: isRecording
                  ? null
                  : () async {
                      if (RecoConfig.cameraEnabled) {
                        RecoConfig.camNum = i;
                        await RecoConfig.cameraInitialize();
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
    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF101014) : Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: cameraView),
          // 相机按钮
          Positioned(
            bottom: isRecording ? 80 : 130, // 略微上移，录制时如果底部菜单隐藏，可以稍下沉或保持
            left: 0,
            right: 0,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (isRecording)
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
                Center(
                  child: GestureDetector(
                    onTapDown: (_) => _buttonAnimController.forward(),
                    onTapUp: (_) async {
                      _buttonAnimController.reverse();
                      _onRecordTap();
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
                            colors: isRecording
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
                              color: isRecording
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
                                width: isRecording ? 28 : 24,
                                height: isRecording ? 28 : 24,
                                decoration: BoxDecoration(
                                  color: isRecording
                                      ? Colors.redAccent
                                      : isDark
                                      ? const Color(0xFF4582FF)
                                      : theme.brandColor6,
                                  borderRadius: BorderRadius.circular(
                                    isRecording ? 6.0 : 12.0,
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
          // 顶部摄像头切换栏
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(
                top: MediaQuery.of(context).padding.top + 10,
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
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: cameraSwitchButtons,
                ),
              ),
            ),
          ),

          // Info Button
          Positioned(
            top: MediaQuery.of(context).padding.top + 10,
            right: 16,
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: () {
                setState(() {
                  _showTips = true;
                });
              },
              child: const Icon(
                Icons.info_outline,
                color: Colors.white70,
                size: 42,
              ),
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
                      maxHeight: MediaQuery.of(context).size.height * 0.70,
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
                          "Tips",
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
