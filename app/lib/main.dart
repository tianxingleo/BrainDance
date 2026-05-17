import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/extra_func/theme_provider.dart';
import 'package:braindance/extra_func/locale_provider.dart';
import 'pages/recall.dart';
import 'pages/agent_chat.dart';
import 'pages/settings.dart';
import 'pages/create_guide.dart';
import 'pages/community.dart';
import 'pages/login.dart';
import 'pages/task_list.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/configs/gen_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'services/task_notification_service.dart';
import 'services/network_service.dart';
import 'floating_nav_bar.dart';
import 'widgets/bd_surfaces.dart';
import 'widgets/network_bubble.dart';
import 'widgets/theme_animation_overlay.dart';

//App Data
final themeData = TDTheme.defaultData();
//MainScreen
final pageIndexProvider = StateProvider((ref) => 0);
final loadingProvider = StateProvider((ref) => true);
final isRecordingProvider = StateProvider((ref) => false);
final pendingSubmitTitleProvider = StateProvider<String?>((ref) => null);
final recallScrollToTopSignal = StateProvider<int>((ref) => 0);
final pageAnimatingProvider = StateProvider<bool>((ref) => false);

// OverviewCard 统计数据，recall 写入，manage 读取
final overviewStatsProvider = StateProvider<Map<String, int>>(
  (ref) => {
    'allModelCount': 0,
    'processingTaskCount': 0,
    'ragCount': 0,
    'recentCount': 0,
  },
);
final overviewLocalIndexingProvider = StateProvider<bool>((ref) => false);

// 全局 NavigatorKey
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

void main() async {
  WidgetsBinding widgetsBinding = WidgetsFlutterBinding.ensureInitialized();
  FlutterNativeSplash.preserve(widgetsBinding: widgetsBinding);
  TDTheme.needMultiTheme(true);

  /// 开启多套主题功能
  AppConfig.initializeAppConfig(); //加载默认数据
  await dotenv.load(fileName: ".env");
  final supabaseResolution = await SupabaseConfig.resolveEndpoint();
  SupabaseConfig.applyRuntimeResolution(supabaseResolution);
  if (supabaseResolution.diagnosticMessage?.isNotEmpty ?? false) {
    debugPrint('Supabase bootstrap: ${supabaseResolution.diagnosticMessage}');
  }
  await Supabase.initialize(
    url: SupabaseConfig.url,
    anonKey: SupabaseConfig.apiKey,
  ); //Supabase

  // 初始化全局任务通知服务
  taskNotificationService.setNavigatorKey(navigatorKey);
  await taskNotificationService.init();

  // 初始化全局网络状态监测服务
  networkService.setNavigatorKey(navigatorKey);
  await networkService.init();
  //Camera
  try {
    final List<CameraDescription> camsTemp = await availableCameras();
    //摄像机分类。
    for (var cam in camsTemp) {
      switch (cam.lensDirection) {
        case CameraLensDirection.front:
          RecoConfig.frontCameras.add(cam);
          break;
        case CameraLensDirection.back:
          RecoConfig.backCameras.add(cam);
          break;
        case CameraLensDirection.external:
          RecoConfig.externalCameras.add(cam);
          break;
      }
    }
    RecoConfig.cameras = camsTemp;
    RecoConfig.cameraEnabled = camsTemp.isNotEmpty;
  } catch (e) {
    //print(e.toString()); *未来考虑添加根据不同异常信息，改变相机页面错误信息
    RecoConfig.cameras = [];
    RecoConfig.frontCameras = [];
    RecoConfig.backCameras = [];
    RecoConfig.externalCameras = [];
    RecoConfig.cameraEnabled = false;
  }
  //
  runApp(const ProviderScope(child: MyApp()));
}

//App定义
class MyApp extends StatelessWidget with WidgetsBindingObserver {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    WidgetsBinding.instance.addObserver(this); // 注册观察者
    return const Home();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    switch (state) {
      case AppLifecycleState.inactive:
        // Handled by RecordPage locally to prevent recording interruption
        break;
      case AppLifecycleState.resumed:
        // Handled by RecordPage locally
        break;
      case AppLifecycleState.paused: // 应用进入后台（例如用户按了Home键、切换到其他应用）
        GenConfig.saveUploadedAssets();
        break;
      default:
    }
  }
}

class Home extends ConsumerWidget {
  const Home({super.key});
  void loadSettings(WidgetRef ref) async {
    if (!ref.read(loadingProvider)) {
      return;
    }
    final bool suc = await SetConfig.loadMsgFromFile();
    if (suc) {
      SetConfig.loadSettingsFromMsg(ref);
    }
    ref.read(loadingProvider.notifier).state = false;
    FlutterNativeSplash.remove();
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final localeCode = ref.watch(localeProvider);
    final themeModeAsync = ref.watch(themeModeProvider);

    // 启动时检查是否有会话
    final hasSession = Supabase.instance.client.auth.currentSession != null;
    final canEnterApp = SupabaseConfig.isAdminMode || hasSession;

    return MaterialApp(
      navigatorKey: navigatorKey,
      debugShowCheckedModeBanner: false,
      title: "Brain Dance",
      locale: _parseLocale(localeCode),
      theme: AppTheme.buildLightTheme(themeData).copyWith(
        textTheme: themeData.systemThemeDataLight?.textTheme.apply(
          fontFamily: AppConfig.fontFamily,
        ),
      ),
      darkTheme: AppTheme.buildDarkTheme(themeData).copyWith(
        textTheme: themeData.systemThemeDataDark?.textTheme.apply(
          fontFamily: AppConfig.fontFamily,
        ),
      ),
      themeMode: themeModeAsync,
      initialRoute: canEnterApp
          ? '/'
          : '/login', // secret key 走管理员模式，anon key 仍按登录态进入
      onGenerateRoute: (settings) {
        WidgetBuilder builder;
        bool useSlide = false;
        switch (settings.name) {
          case '/':
            builder = (context) {
              loadSettings(ref);
              return MainScreen();
            };
          case '/login':
            builder = (_) => const LoginPage();
          case '/example':
            builder = (_) => RecallPage();
          case '/tasks':
            builder = (_) => const TaskListPage();
            useSlide = true;
          default:
            return null;
        }
        if (useSlide) {
          return PageRouteBuilder(
            settings: settings,
            transitionDuration: BDMotion.durationNormal,
            reverseTransitionDuration: BDMotion.durationNormal,
            opaque: true,
            pageBuilder: (ctx, _, _) => builder(ctx),
            transitionsBuilder: (_, animation, _a, child) {
              final curved = animation.drive(
                CurveTween(curve: Curves.easeInOutCubic),
              );
              return AnimatedBuilder(
                animation: curved,
                builder: (ctx, child) {
                  final screenHeight = MediaQuery.of(ctx).size.height;
                  return Transform.translate(
                    offset: Offset(0, -(1.0 - curved.value) * screenHeight),
                    child: child,
                  );
                },
                child: child,
              );
            },
          );
        }
        return MaterialPageRoute(settings: settings, builder: builder);
      },
      // 使用 builder 创建全局 Overlay，确保通知弹窗能在任意界面显示
      builder: (context, child) {
        return KeyedSubtree(
          key: ValueKey<String>(localeCode),
          child: ThemeAnimationOverlay(
            child: Stack(
              children: [
                child!,
                // 语言切换时一并重建全局通知和页面树，避免旧文案残留
                const GlobalNotificationOverlay(),
                const NetworkBubbleOverlay(),
              ],
            ),
          ),
        );
      },
    );
  }

  Locale _parseLocale(String localeCode) {
    final parts = localeCode.split('_');
    if (parts.length == 2) {
      return Locale(parts[0], parts[1]);
    }
    return Locale(parts.first);
  }
}

/// 全局通知 Overlay 层
/// 使用 ListenableBuilder 监听通知状态变化
class GlobalNotificationOverlay extends StatefulWidget {
  const GlobalNotificationOverlay({super.key});

  @override
  State<GlobalNotificationOverlay> createState() =>
      _GlobalNotificationOverlayState();
}

class _GlobalNotificationOverlayState extends State<GlobalNotificationOverlay>
    with TickerProviderStateMixin {
  late AnimationController _showController;
  late AnimationController _hideController;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _fadeAnimation;
  Timer? _autoHideTimer;

  @override
  void initState() {
    super.initState();

    // 显示动画控制器 (300ms)
    _showController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    // 隐藏动画控制器 (1秒逐渐消失)
    _hideController = AnimationController(
      duration: const Duration(seconds: 1),
      vsync: this,
    );

    _slideAnimation =
        Tween<Offset>(begin: const Offset(0, -1), end: Offset.zero).animate(
          CurvedAnimation(parent: _showController, curve: Curves.easeOutCubic),
        );

    // 淡出动画：从1.0到0.0
    _fadeAnimation = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(parent: _hideController, curve: Curves.easeInOut),
    );

    // 监听隐藏动画完成
    _hideController.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        taskNotificationService.hideNotification();
      }
    });
  }

  @override
  void dispose() {
    _autoHideTimer?.cancel();
    _showController.dispose();
    _hideController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: taskNotificationService,
      builder: (context, child) {
        final notification = taskNotificationService.currentNotification;
        if (notification == null) {
          // 重置动画控制器
          _showController.reset();
          _hideController.reset();
          _autoHideTimer?.cancel();
          return const SizedBox.shrink();
        }

        // 检查当前路由是否允许显示通知
        final currentRoute = taskNotificationService.currentRoute;
        if (!taskNotificationService.isNotificationEnabledForRoute(
          currentRoute,
        )) {
          return const SizedBox.shrink();
        }

        // 显示动画：滑入
        _showController.forward();

        // 启动1秒后开始淡出动画（总显示时间约2秒）
        _autoHideTimer?.cancel();
        _autoHideTimer = Timer(const Duration(seconds: 1), () {
          // 等待显示动画完成后，开始1秒淡出动画
          if (mounted && taskNotificationService.currentNotification != null) {
            _hideController.forward(from: 0);
          }
        });

        return _buildNotificationWidget(notification);
      },
    );
  }

  Widget _buildNotificationWidget(TaskNotificationData notification) {
    final isDark = AppConfig.isNightMode;
    final hasCompleted = notification.completedCount > 0;
    final hasFailed = notification.failedCount > 0;

    // 构建通知内容
    String message = '';
    IconData icon = Icons.check_circle;
    Color iconColor = Colors.green;

    if (hasCompleted && hasFailed) {
      message =
          '${notification.completedCount} ${textLocalize('task_completed')}，${notification.failedCount} ${textLocalize('task_failed')}';
      icon = Icons.info;
      iconColor = Colors.orange;
    } else if (hasCompleted) {
      message =
          '${notification.completedCount} ${textLocalize('task_notification_completed')}';
      icon = Icons.check_circle;
      iconColor = Colors.green;
    } else if (hasFailed) {
      message =
          '${notification.failedCount} ${textLocalize('task_notification_failed')}';
      icon = Icons.error;
      iconColor = Colors.red;
    }

    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: SafeArea(
        child: SlideTransition(
          position: _slideAnimation,
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: Material(
              color: Colors.transparent,
              child: Center(
                child: Container(
                  margin: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                  decoration: BoxDecoration(
                    color: isDark ? const Color(0xFF2A2A30) : Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withAlpha(30),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                    border: Border.all(
                      color: isDark
                          ? const Color(0xFF3A3A40)
                          : const Color(0xFFE0E0E0),
                      width: 1,
                    ),
                  ),
                  child: InkWell(
                    onTap: () => taskNotificationService.navigateToTaskList(),
                    borderRadius: BorderRadius.circular(12),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: iconColor.withAlpha(20),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(icon, color: iconColor, size: 20),
                        ),
                        const SizedBox(width: 12),
                        Flexible(
                          child: Text(
                            message,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: isDark
                                  ? Colors.white
                                  : const Color(0xFF333333),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Icon(
                          Icons.keyboard_arrow_right,
                          color: isDark
                              ? const Color(0xFF888888)
                              : const Color(0xFF999999),
                          size: 20,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

//主屏幕
class MainScreen extends ConsumerStatefulWidget {
  const MainScreen({super.key});

  @override
  ConsumerState<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends ConsumerState<MainScreen>
    with TickerProviderStateMixin {
  static const int _pageCount = 5;

  int _previousIndex = 0;
  int _lastTabIndex = 0; // 上次不同的 tab，用于同 tab 重复点击时回退
  int _slideDirection = 1; // 1 = slide from right, -1 = slide from left

  late final AnimationController _animController;
  late final Animation<double> _curvedAnimation;
  bool _isAnimating = false;

  /// 懒缓存：首次访问时创建，之后一直保留在 widget tree 中
  final List<Widget?> _cachedPages = List.filled(_pageCount, null);
  final Set<int> _builtPages = {0}; // 首屏默认构建

  @override
  void initState() {
    super.initState();
    _animController =
        AnimationController(duration: BDMotion.durationNormal, vsync: this)
          ..addStatusListener((status) {
            if (status == AnimationStatus.completed) {
              ref.read(pageAnimatingProvider.notifier).state = false;
              if (mounted) {
                setState(() {
                  _isAnimating = false;
                  _previousIndex = ref.read(pageIndexProvider);
                });
              }
            }
          });
    _curvedAnimation = _animController.drive(
      CurveTween(curve: Curves.easeInOutCubic),
    );

    // 首帧只渲染当前页，首帧结束后立即预热其余 Tab 页面（Offstage），消除首次切换掉帧
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      setState(() => _builtPages.addAll({1, 2, 3, 4}));
    });
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  Widget _ensurePage(int index) {
    if (_cachedPages[index] != null) return _cachedPages[index]!;
    late final Widget page;
    switch (index) {
      case 0:
        page = const RecallPage();
      case 1:
        page = const AgentChatPage();
      case 2:
        page = const CreateGuidePage();
      case 3:
        page = const CommunityPage();
      case 4:
        page = const SettingsPage();
      default:
        page = const RecallPage();
    }
    _cachedPages[index] = page;
    return page;
  }

  void _switchToPage(int newIndex) {
    final oldIndex = ref.read(pageIndexProvider);
    if (newIndex == oldIndex) {
      final createIdx = 2;
      if ((newIndex == 1 || newIndex == createIdx) && _lastTabIndex != oldIndex) {
        _switchToPage(_lastTabIndex);
      } else if (newIndex == 0) {
        ref.read(recallScrollToTopSignal.notifier).update((s) => s + 1);
      }
      return;
    }

    _lastTabIndex = oldIndex;
    ref.read(pageAnimatingProvider.notifier).state = true;
    setState(() {
      _slideDirection = newIndex > oldIndex ? 1 : -1;
      _previousIndex = oldIndex;
      _builtPages.add(newIndex);
      _isAnimating = true;
    });
    _animController.forward(from: 0);
    ref.read(pageIndexProvider.notifier).state = newIndex;
  }

  @override
  Widget build(BuildContext context) {
    final bool isLoading = ref.watch(loadingProvider);
    final int pageIndex = ref.watch(pageIndexProvider);
    final bool isRecording = ref.watch(isRecordingProvider);

    // 外部改 pageIndex 时（如 provider 直接修改），同步方向
    if (pageIndex != _previousIndex && !_isAnimating) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        ref.read(pageAnimatingProvider.notifier).state = true;
        setState(() {
          _slideDirection = pageIndex > _previousIndex ? 1 : -1;
          _builtPages.add(pageIndex);
          _isAnimating = true;
        });
        _animController.forward(from: 0);
      });
    }

    return Scaffold(
      extendBody: true,
      body: BDPageBackdrop(
        child: isLoading
            ? const SizedBox.shrink()
            : Stack(
                children: [
                  // ── 页面槽位：保证每个页面的 State 不被销毁 ──
                  ...List.generate(_pageCount, (i) {
                    if (!_builtPages.contains(i)) {
                      return const SizedBox.shrink();
                    }

                    return AnimatedBuilder(
                      animation: _curvedAnimation,
                      builder: (context, child) {
                        final bool isActive = i == pageIndex;
                        // 外部直接修改 pageIndexProvider 时，_isAnimating 在当前帧还未置 true，
                        // 需要按 t=0 的状态渲染，避免新页面在最终位置闪现一帧。
                        final bool pendingAnim =
                            !_isAnimating && pageIndex != _previousIndex;
                        final bool effectiveAnim = _isAnimating || pendingAnim;
                        final int effectiveDir = pendingAnim
                            ? (pageIndex > _previousIndex ? 1 : -1)
                            : _slideDirection;
                        final bool isLeaving =
                            effectiveAnim &&
                            i == _previousIndex &&
                            i != pageIndex;
                        final double t = pendingAnim
                            ? 0.0
                            : _curvedAnimation.value;
                        double dx = 0;

                        if (effectiveAnim && isActive) {
                          dx =
                              effectiveDir *
                              (1.0 - t) *
                              MediaQuery.of(context).size.width;
                        } else if (isLeaving) {
                          dx =
                              -effectiveDir *
                              t *
                              MediaQuery.of(context).size.width;
                        }

                        final bool isVisible = isActive || isLeaving;
                        return Offstage(
                          offstage: !isVisible,
                          child: IgnorePointer(
                            ignoring: !isActive,
                            child: Transform.translate(
                              offset: Offset(dx, 0),
                              child: child,
                            ),
                          ),
                        );
                      },
                      child: RepaintBoundary(child: _ensurePage(i)),
                    );
                  }),
                  if (!isRecording)
                    FloatingNavBar(
                      skipBlur: _isAnimating || (pageIndex != _previousIndex),
                      currentIndex: pageIndex,
                      onTap: _switchToPage,
                      items: [
                        NavIslandItem(
                          icon: Icons.history_edu_rounded,
                          label: textLocalize("recall"),
                        ),
                        NavIslandItem(
                          icon: Icons.travel_explore_rounded,
                          label: textLocalize("agent"),
                        ),
                        NavIslandItem(
                          icon: Icons.add_rounded,
                          label: textLocalize("create"),
                          isLarge: true,
                        ),
                        NavIslandItem(
                          icon: Icons.groups_rounded,
                          label: textLocalize("community"),
                        ),
                        NavIslandItem(
                          icon: Icons.settings_rounded,
                          label: textLocalize("manage"),
                        ),
                      ],
                    ),
                ],
              ),
      ),
    );
  }
}
