import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_riverpod/legacy.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:braindance/configs/reco_config.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/extra_func/theme_provider.dart';
import 'package:braindance/extra_func/locale_provider.dart';
import 'pages/recall.dart';
import 'pages/record.dart';
import 'pages/generate.dart';
import 'pages/community.dart';
import 'pages/settings.dart';
import 'pages/login.dart';
import 'pages/task_list.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/gen_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'services/task_notification_service.dart';
import 'floating_nav_bar.dart';
import 'widgets/bd_surfaces.dart';

//App Data
final themeData = TDTheme.defaultData();
//MainScreen
final pageIndexProvider = StateProvider((ref) => 0);
final loadingProvider = StateProvider((ref) => true);
final isRecordingProvider = StateProvider((ref) => false);

// 全局 NavigatorKey
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  TDTheme.needMultiTheme(true);

  /// 开启多套主题功能
  AppConfig.initializeAppConfig(); //加载默认数据
  await dotenv.load(fileName: ".env");
  await Supabase.initialize(
    url: SupabaseConfig.url,
    anonKey: SupabaseConfig.apiKey,
  ); //Supabase

  // 初始化全局任务通知服务
  taskNotificationService.setNavigatorKey(navigatorKey);
  await taskNotificationService.init();
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
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    ref.watch(localeProvider);
    final themeModeAsync = ref.watch(themeModeProvider);

    // 启动时检查是否有会话
    final hasSession = Supabase.instance.client.auth.currentSession != null;
    final canEnterApp = SupabaseConfig.isAdminMode || hasSession;

    return MaterialApp(
      navigatorKey: navigatorKey,
      debugShowCheckedModeBanner: false,
      title: "Brain Dance",
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
      routes: {
        // 路由表：路径 -> 页面构建器
        '/': (context) {
          loadSettings(ref);
          return MainScreen();
        }, // 根路径对应主屏幕
        '/login': (context) => const LoginPage(), // 登录页
        '/example': (context) => RecallPage(), // "/example"路径对应....
        '/tasks': (context) => const TaskListPage(), // 任务列表页
      },
      // 使用 builder 创建全局 Overlay，确保通知弹窗能在任意界面显示
      builder: (context, child) {
        return Stack(
          children: [
            child!,
            // 全局通知 Overlay 层
            const GlobalNotificationOverlay(),
          ],
        );
      },
    );
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
  static const double unselectedSize = 30;
  static const double selectedSize = 34;

  @override
  ConsumerState<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends ConsumerState<MainScreen> {
  int _previousIndex = 0;
  int _slideDirection = 1; // 1 = slide from right, -1 = slide from left

  Widget getPage(int pageIndex) {
    switch (pageIndex) {
      case 0:
        return RecallPage(); // 页面0: 主页：过往回忆
      case 1:
        return RecordPage(); // 页面1: 相机记录
      case 2:
        return GeneratePage(); // 页面2: 图文生成
      case 3:
        return const CommunityPage(); // 页面3: 社区
      case 4:
        return SettingsPage(homeRef: ref); // 页面4: 设置
    }
    return RecallPage();
  }

  @override
  Widget build(BuildContext context) {
    final bool isLoading = ref.watch(loadingProvider);
    final int pageIndex = ref.watch(pageIndexProvider);
    final bool isRecording = ref.watch(isRecordingProvider);

    // Update slide direction when page changes
    if (pageIndex != _previousIndex) {
      _slideDirection = pageIndex > _previousIndex ? 1 : -1;
      _previousIndex = pageIndex;
    }

    return Scaffold(
      extendBody: true,
      body: BDPageBackdrop(
        darken: pageIndex == 1,
        child: isLoading
            ? const Center(child: CircularProgressIndicator())
            : Stack(
                children: [
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 360),
                    switchInCurve: Curves.easeOutCubic,
                    switchOutCurve: Curves.easeInCubic,
                    transitionBuilder: (child, animation) {
                      // Determine if this is the incoming or outgoing child
                      final isIncoming =
                          child.key == ValueKey<int>(pageIndex);
                      final beginOffset = isIncoming
                          ? Offset(0.15 * _slideDirection, 0)
                          : Offset(-0.15 * _slideDirection, 0);
                      return FadeTransition(
                        opacity: animation,
                        child: SlideTransition(
                          position: Tween<Offset>(
                            begin: beginOffset,
                            end: Offset.zero,
                          ).animate(animation),
                          child: child,
                        ),
                      );
                    },
                    child: KeyedSubtree(
                      key: ValueKey<int>(pageIndex),
                      child: getPage(pageIndex),
                    ),
                  ),
                  if (!isRecording)
                    FloatingNavBar(
                      currentIndex: pageIndex,
                      onTap: (index) {
                        ref.read(pageIndexProvider.notifier).state = index;
                      },
                      items: [
                        NavIslandItem(
                          icon: Icons.history_edu_rounded,
                          label: textLocalize("recall"),
                        ),
                        NavIslandItem(
                          icon: Icons.camera_rounded,
                          label: textLocalize("record"),
                        ),
                        NavIslandItem(
                          icon: Icons.auto_awesome_rounded,
                          label: textLocalize("generate"),
                        ),
                        NavIslandItem(
                          icon: Icons.public_rounded,
                          label: textLocalize("community"),
                        ),
                        NavIslandItem(
                          icon: Icons.settings_rounded,
                          label: textLocalize("settings"),
                        ),
                      ],
                    ),
                ],
              ),
      ),
    );
  }
}
