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
import 'pages/settings.dart';
import 'pages/login.dart';
import 'pages/task_list.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/gen_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/configs/set_config.dart';
import 'services/task_notification_service.dart';

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
    anonKey: SupabaseConfig.anonKey,
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

    return MaterialApp(
      navigatorKey: navigatorKey,
      debugShowCheckedModeBanner: false,
      title: "Brain Dance",
      theme: themeData.systemThemeDataLight?.copyWith(
        textTheme: themeData.systemThemeDataLight?.textTheme.apply(
          fontFamily: AppConfig.fontFamily,
        ),
      ),
      darkTheme: themeData.systemThemeDataDark?.copyWith(
        textTheme: themeData.systemThemeDataDark?.textTheme.apply(
          fontFamily: AppConfig.fontFamily,
        ),
      ),
      themeMode: themeModeAsync,
      initialRoute: hasSession ? '/' : '/login', // 初始路由路径，根据是否有Session判断
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
  State<GlobalNotificationOverlay> createState() => _GlobalNotificationOverlayState();
}

class _GlobalNotificationOverlayState extends State<GlobalNotificationOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, -1),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: taskNotificationService,
      builder: (context, child) {
        final notification = taskNotificationService.currentNotification;
        if (notification == null) {
          _controller.reverse();
          return const SizedBox.shrink();
        }

        // 显示动画
        _controller.forward();

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
                  margin:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
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
class MainScreen extends ConsumerWidget {
  const MainScreen({super.key});
  static const double unselectedSize = 30;
  static const double selectedSize = 34;
  Widget getPage(int pageIndex, WidgetRef ref) {
    switch (pageIndex) {
      case 0:
        return RecallPage(); // 页面0: 主页：过往回忆
      case 1:
        return RecordPage(); // 页面1: 相机记录
      case 2:
        return GeneratePage(); // 页面2: 图文生成
      case 3:
        return SettingsPage(homeRef: ref); // 页面3: 设置
    }
    return RecallPage();
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool isLoading = ref.watch(loadingProvider);
    final int pageIndex = ref.watch(pageIndexProvider);
    final bool isRecording = ref.watch(isRecordingProvider);
    final isDark = AppConfig.isNightMode;
    // 黑夜模式下强制使用更明亮的蓝色，以确保底层文字和图标的高可见性
    final brandColor = isDark
        ? Colors.white
        : Colors.blueAccent; // 统一为蓝色基调，白天使用系统蓝色，夜晚使用更亮的蓝色以增强对比度
    final lightBrandColor = brandColor.withAlpha(160); // 统一为蓝色基调，半透明使得未选中状态易于区分
    final selectedTextStyle = TextStyle(
      fontSize: 12,
      fontWeight: FontWeight.bold,
      color: brandColor,
      height: 1.5,
    );
    final unselectedTextStyle = TextStyle(
      fontSize: 11,
      fontWeight: FontWeight.w500,
      color: lightBrandColor,
      height: 1.5,
    );
    return Scaffold(
      extendBody: true,
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : AnimatedSwitcher(
              duration: const Duration(milliseconds: 400),
              switchInCurve: Curves.easeOutCubic,
              switchOutCurve: Curves.easeInCubic,
              child: Container(
                key: ValueKey<int>(pageIndex),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(32),
                  gradient: LinearGradient(
                    colors: [
                      TDTheme.of(context).brandColor1,
                      TDTheme.of(context).brandColor4,
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: getPage(pageIndex, ref),
              ),
            ),
      bottomNavigationBar: isRecording
          ? null
          : Padding(
              padding: const EdgeInsets.only(bottom: 18, left: 18, right: 18),
              child: PhysicalModel(
                color: Colors.transparent,
                elevation: 16,
                borderRadius: BorderRadius.circular(32),
                shadowColor: Colors.black.withOpacity(0.70),
                clipBehavior: Clip.antiAlias,
                child: Container(
                  decoration: BoxDecoration(
                    color: AppConfig.isNightMode
                        ? const Color(0xFF18181C)
                        : const Color(0xFF23232A),
                    borderRadius: BorderRadius.circular(32),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 24,
                        offset: Offset(16, 16),
                      ),
                    ],
                  ),
                  child: TDBottomTabBar(
                    TDBottomTabBarBasicType.iconText,
                    componentType: TDBottomTabBarComponentType.normal,
                    useVerticalDivider: false,
                    centerDistance: 0,
                    barHeight: 90,
                    navigationTabs: [
                      TDBottomTabBarTabConfig(
                        tabText: textLocalize("recall"),
                        selectTabTextStyle: selectedTextStyle,
                        unselectTabTextStyle: unselectedTextStyle,
                        selectedIcon: Icon(
                          Icons.home_rounded,
                          size: selectedSize,
                          color: brandColor,
                        ),
                        unselectedIcon: Icon(
                          Icons.home_outlined,
                          size: unselectedSize,
                          color: lightBrandColor,
                        ),
                        onTap: () =>
                            ref.read(pageIndexProvider.notifier).state = 0,
                      ),
                      TDBottomTabBarTabConfig(
                        tabText: textLocalize("record"),
                        selectTabTextStyle: selectedTextStyle,
                        unselectTabTextStyle: unselectedTextStyle,
                        selectedIcon: Icon(
                          Icons.videocam_rounded,
                          size: selectedSize,
                          color: brandColor,
                        ),
                        unselectedIcon: Icon(
                          Icons.videocam_outlined,
                          size: unselectedSize,
                          color: lightBrandColor,
                        ),
                        onTap: () =>
                            ref.read(pageIndexProvider.notifier).state = 1,
                      ),
                      TDBottomTabBarTabConfig(
                        tabText: textLocalize("generate"),
                        selectTabTextStyle: selectedTextStyle,
                        unselectTabTextStyle: unselectedTextStyle,
                        selectedIcon: Icon(
                          Icons.image_rounded,
                          size: selectedSize,
                          color: brandColor,
                        ),
                        unselectedIcon: Icon(
                          Icons.image_outlined,
                          size: unselectedSize,
                          color: lightBrandColor,
                        ),
                        onTap: () =>
                            ref.read(pageIndexProvider.notifier).state = 2,
                      ),
                      TDBottomTabBarTabConfig(
                        tabText: textLocalize("settings"),
                        selectTabTextStyle: selectedTextStyle,
                        unselectTabTextStyle: unselectedTextStyle,
                        selectedIcon: Icon(
                          Icons.settings_rounded,
                          size: selectedSize,
                          color: brandColor,
                        ),
                        unselectedIcon: Icon(
                          Icons.settings_outlined,
                          size: unselectedSize,
                          color: lightBrandColor,
                        ),
                        onTap: () =>
                            ref.read(pageIndexProvider.notifier).state = 3,
                      ),
                    ],
                  ),
                ),
              ),
            ),
    );
  }
}
