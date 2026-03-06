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
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/gen_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:braindance/configs/set_config.dart';

//App Data
final themeData = TDTheme.defaultData();
//MainScreen
final pageIndexProvider = StateProvider((ref) => 0);
final loadingProvider = StateProvider((ref) => true);
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
        RecoConfig.disposeCamera();
        break;
      case AppLifecycleState.resumed:
        RecoConfig.refreshCamera();
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
      },
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
    final isDark = AppConfig.isNightMode;
    // 黑夜模式下强制使用更明亮的蓝色，以确保底层文字和图标的高可见性
    final brandColor = isDark
        ? const Color(0xFF4582FF)
        : TDTheme.of(context).brandColor7;
    final lightBrandColor = brandColor.withAlpha(
      isDark ? 160 : 128,
    ); // 统一为蓝色基调，半透明使得未选中状态易于区分
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
                child: getPage(pageIndex, ref),
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
              ),
            ),
      bottomNavigationBar: Padding(
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
                  onTap: () => ref.read(pageIndexProvider.notifier).state = 0,
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
                  onTap: () => ref.read(pageIndexProvider.notifier).state = 1,
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
                  onTap: () => ref.read(pageIndexProvider.notifier).state = 2,
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
                  onTap: () => ref.read(pageIndexProvider.notifier).state = 3,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
