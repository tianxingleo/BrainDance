import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'extra_func/theme_provider.dart';
import 'pages/recall.dart';
import 'pages/record.dart';
import 'pages/generate.dart';
import 'pages/settings.dart';
import 'app_filesys.dart';
//App Run
late final TDThemeData themeData;
late VoidCallback? onThemeChanged;
late VoidCallback? onLanguageChanged;
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  var themeJsonString = await rootBundle.loadString('assets/theme.json');
  /// 开启多套主题功能
  TDTheme.needMultiTheme(true);
  /// 默认浅色主题,xxxDark为深色主题
  themeData = TDThemeData.fromJson('red', themeJsonString, darkName: 'redDark') ??
      TDTheme.defaultData();

  initializeAppConfig(); //加载默认数据
  runApp(const MyApp());
}
//App定义
class MyApp extends StatefulWidget {
  const MyApp({super.key});
  
  @override
  State<MyApp> createState() => _MyAppState();
}
class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    onThemeChanged = _updateTheme;
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Brain Dance",
      theme: themeData.systemThemeDataLight?.copyWith(
        textTheme: themeData.systemThemeDataLight?.textTheme.apply(
          fontFamily: 'MSYH',
        ),
      ),
      darkTheme: themeData.systemThemeDataDark?.copyWith(
        textTheme: themeData.systemThemeDataDark?.textTheme.apply(
          fontFamily: 'MSYH',
        ),
      ),
      themeMode: ThemeModeProvider.themeMode,
      initialRoute: '/',  // 初始路由路径
      routes: {  // 路由表：路径 -> 页面构建器
        '/': (context) => MainScreen(),  // 根路径对应主屏幕
        '/example': (context) => RecallPage(),  // "/example"路径对应....
      },
    );
  }
  void _updateTheme() {
    setState(() {}); // 触发状态更新以应用新的主题
  }
}
//主屏幕
class MainScreen extends StatefulWidget {  // 主屏幕StatefulWidget
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();  // 创建状态
}
//占位页面
class LoadingPage extends StatelessWidget {
  const LoadingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Text('Now Loading...'),
      ),
    );
  }
}
//主屏幕导航
class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;  // 当前选中的底部导航索引
  bool isLoading = true; //加载状态
  
  late final List<Widget> _pages = [  // 页面列表
    RecallPage(),      // 页面0: 主页：过往回忆
    RecordPage(),    // 页面1: 相机记录
    GeneratePage(),   // 页面2: 图文生成
    SettingsPage(), // 页面3: 设置
  ];
  //final TextStyle unselectedTextStyle = TextStyle(fontSize: 10);
  final TextStyle selectedTextStyle = TextStyle(fontSize: 10);
  static const double unselectedSize = 32;
  static const double selectedSize = 36;
  
  @override
  Widget build(BuildContext context) {
    onLanguageChanged = _updateState;
    return Scaffold(
      body: _getPage(_currentIndex),  // 根据索引显示对应页面
      bottomNavigationBar: TDBottomTabBar(  // 底部导航栏
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
            onTap: () => setState(() => _currentIndex = 0),  // 点击切换索引并更新状态
          ),
          TDBottomTabBarTabConfig(
            tabText: textLocalize("record"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.camera_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.camera, size: unselectedSize),
            onTap: () => setState(() => _currentIndex = 1),  // 点击切换索引并更新状态
          ),
          TDBottomTabBarTabConfig(
            tabText: textLocalize("generate"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.file_word_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.file_word, size: unselectedSize),
            onTap: () => setState(() => _currentIndex = 2),  // 点击切换索引并更新状态
          ),
          TDBottomTabBarTabConfig(
            tabText: textLocalize("settings"),
            selectTabTextStyle: selectedTextStyle,
            unselectTabTextStyle: selectedTextStyle,
            selectedIcon: Icon(TDIcons.setting_1_filled, size: selectedSize),
            unselectedIcon: Icon(TDIcons.setting_1, size: unselectedSize),
            onTap: () => setState(() => _currentIndex = 3),  // 点击切换索引并更新状态
          ),
        ],
        //backgroundColor: AppConfig.primaryColor,
        currentIndex: _currentIndex,  // 当前选中索引
        barHeight: 74,
      )
    );
  }
  @override
  void initState() {
    super.initState();  // 必须调用父类方法
    _loading();//加载AppConfig
  }
  void _loading() async {
    bool suc = await AppConfig.loadMsgFromSettingsFile();
    if (suc) {
      AppConfig.loadSettingsFromMsg();
    }
    isLoading = false;
    _updateState(); // 更新状态以显示主界面
  }
  void _updateState() {
    setState(() {}); // 触发状态更新
  }
  Widget _getPage(int index) {
    if (isLoading) {
      return LoadingPage(); // 显示加载页面
    }
    return _pages[index];
  }
}