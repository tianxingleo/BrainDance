import 'package:camera/camera.dart';
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
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/gen_config.dart';
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
  await Supabase.initialize(url: 'any', anonKey: 'any');//Supabase
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
class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}
class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this); // 注册观察者
  }
  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this); // 移除观察者
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
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
      initialRoute: '/', // 初始路由路径
      routes: {
        // 路由表：路径 -> 页面构建器
        '/': (context) {
          loadSettings(ref);
          return MainScreen();
        }, // 根路径对应主屏幕
        '/example': (context) => RecallPage(), // "/example"路径对应....
      },
    );
  }
}

//主屏幕
class MainScreen extends ConsumerWidget {
  const MainScreen({super.key});
  static const TextStyle selectedTextStyle = TextStyle(fontSize: 10);
  static const double unselectedSize = 32;
  static const double selectedSize = 36;
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
    final bool isLoading = ref.watch(loadingProvider); //加载状态
    final int pageIndex = ref.watch(pageIndexProvider);
    return Scaffold(
      body: isLoading
          ? Scaffold(body: Center(child: Text('Now Loading...')))
          : getPage(pageIndex, ref),
      bottomNavigationBar: TDBottomTabBar(
        // 底部导航栏
        TDBottomTabBarBasicType.iconText,
        componentType: TDBottomTabBarComponentType.normal,
        useVerticalDivider: false,
        centerDistance: 4,
        navigationTabs: [
          TDBottomTabBarTabConfig(
            tabText: textLocalize("recall"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.home_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.home, size: unselectedSize),
            onTap: () =>
                ref.read(pageIndexProvider.notifier).state = 0, // 点击切换索引并更新状态
          ),
          TDBottomTabBarTabConfig(
            tabText: textLocalize("record"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.camera_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.camera, size: unselectedSize),
            onTap: () =>
                ref.read(pageIndexProvider.notifier).state = 1, // 点击切换索引并更新状态
          ),
          TDBottomTabBarTabConfig(
            tabText: textLocalize("generate"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.file_word_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.file_word, size: unselectedSize),
            onTap: () =>
                ref.read(pageIndexProvider.notifier).state = 2, // 点击切换索引并更新状态
          ),
          TDBottomTabBarTabConfig(
            tabText: textLocalize("settings"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.setting_1_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.setting_1, size: unselectedSize),
            onTap: () =>
                ref.read(pageIndexProvider.notifier).state = 3, // 点击切换索引并更新状态
          ),
        ],
        //backgroundColor: AppConfig.primaryColor,
        currentIndex: pageIndex, // 当前选中索引
        barHeight: 74,
      ),
    );
  }
}
