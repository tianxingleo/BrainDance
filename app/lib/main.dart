import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as path_joiner;
import 'dart:io';// For Platform.localeName
import 'language.dart';
import 'dir_and_file.dart';
//App基础设置
class AppConfig {
  static const settingsFileName = "settings.txt";
  static const appName = 'BrainDance';
  static const version = '1.0.0';
  static Color primaryColor = Color.fromRGBO(113, 131, 143, 1);
  static Color accentColor = Color.fromRGBO(232, 234, 220, 1);
  static late Map<String, String> langMap;

  static Future<void> loadFromSettings() async {
    //加载文件数据
    final dir = await DirFinder.supportDir();
    final path = path_joiner.join(dir, settingsFileName);
    if (await FileSystem.checkFileExists(path)) {//检测设置文件是否存在
      //从设置文件中读取语言代码
      List<String> contents = await FileSystem.readFile(path);
      for (int i = 0; i < contents.length; i++) {
        switch (i) {
        case 0:
          //语言代码
          AppConfig.langMap = Localize.getLangMap(contents[i]);
          break;
        }
      }
    }
  }
  static Future<void> saveToSettings() async {
    final dir = await DirFinder.supportDir();
    final path = path_joiner.join(dir, settingsFileName);
    await DirSystem.ensureDir(dir);
    List<String> contents = [];
    contents.add(AppConfig.langMap['locale'] ?? 'en_US');
    await FileSystem.writeFile(path, contents);
  }
}
String textLocalize(String id) {
  return AppConfig.langMap[id] ?? id;
}
//App定义
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Brain Dance",
      theme: ThemeData(
        colorScheme: .fromSeed(seedColor: AppConfig.primaryColor),
      ),
      initialRoute: '/',  // 初始路由路径
      routes: {  // 路由表：路径 -> 页面构建器
        '/': (context) => MainScreen(),  // 根路径对应主屏幕
        '/example': (context) => RecallPage(),  // "/example"路径对应....
      },
    );
  }
}


void main() async {
  //加载默认数据
  AppConfig.langMap = Localize.getLangMap(Platform.localeName);
  runApp(const MyApp());
}


//主屏幕
class MainScreen extends StatefulWidget {  // 主屏幕StatefulWidget
  @override
  State<MainScreen> createState() => _MainScreenState();  // 创建状态
}
//4大页面
class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}
class RecordPage extends StatelessWidget {
  const RecordPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Record Page'),
      ),
      body: Center(
        child: Text('This is the Record Page'),
      ),
    );
  }
}
class GeneratePage extends StatelessWidget {
  const GeneratePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Generate Page'),
      ),
      body: Center(
        child: Text('This is the Generate Page'),
      ),
    );
  }
}
class SettingsPage extends StatefulWidget {
  final VoidCallback? onLanguageChanged;
  const SettingsPage({super.key, this.onLanguageChanged});

  @override
  State<SettingsPage> createState() => _SettingsPageState();  // 创建状态
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
    SettingsPage(onLanguageChanged: _updateState), // 页面3: 设置
  ];
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _getPage(_currentIndex),  // 根据索引显示对应页面
      bottomNavigationBar: BottomNavigationBar(  // 底部导航栏
        backgroundColor: AppConfig.primaryColor,
        selectedItemColor: Colors.white,
        type: BottomNavigationBarType.fixed,
        currentIndex: _currentIndex,  // 当前选中索引
        onTap: (index) => setState(() => _currentIndex = index),  // 点击切换索引并更新状态
        items: [  // 导航项列表
          BottomNavigationBarItem(icon: Icon(Icons.home), label : textLocalize("recall")),
          BottomNavigationBarItem(icon: Icon(Icons.camera), label : textLocalize("record")),
          BottomNavigationBarItem(icon: Icon(Icons.add_photo_alternate), label : textLocalize("generate")),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label : textLocalize("settings")),
        ],
      )
    );
  }
  @override
  void initState() {
    super.initState();  // 必须调用父类方法
    _loading();//加载AppConfig
  }
  void _loading() async {
    await AppConfig.loadFromSettings();
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
//过往回忆界面状态
class _RecallPageState extends State<RecallPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppConfig.accentColor,
      appBar: AppBar(
        backgroundColor: AppConfig.primaryColor,//Theme.of(context).colorScheme.inversePrimary
        title: Container(
          alignment: Alignment.centerLeft, // 关键：顶部对齐
          child: Text(
            textLocalize("home_page"),
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              fontFamily: "宋体",
              color: Colors.white,
            )
          ),
        ),
        toolbarHeight: 60,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: .start,
          children: [
            Text(''),
            Text(
              ''
            ),
          ],
        ),
      ),
      /*floatingActionButton: Stack(
        children: [
          
        ],
      ),*/
    );
  }
}
//设置界面状态
class _SettingsPageState extends State<SettingsPage> {
  void _changeLanguage() {
    if (AppConfig.langMap['locale'] == 'en_US') {
      AppConfig.langMap = Localize.getLangMap('zh_CN');
    } else {
      AppConfig.langMap = Localize.getLangMap('en_US');
    }
    AppConfig.saveToSettings();
    widget.onLanguageChanged?.call();
    setState(() {});
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(textLocalize("settings")),
      ),
      body: Center(
        child: TDButton(
                onTap: _changeLanguage,
                text: textLocalize('set_lang'),
                size: TDButtonSize.large,
                shape: TDButtonShape.rectangle,
                theme: TDButtonTheme.primary,
              ),
      ),
    );
  }
}