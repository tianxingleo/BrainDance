import 'package:flutter/material.dart';
import 'package:path/path.dart' as path_joiner;
import 'dart:io';// For Platform.localeName
import 'language.dart';
import 'dir_and_file.dart';

class AppConfig {
  static const settingsFileName = "settings.txt";
  static const appName = 'BrainDance';
  static const version = '1.0.0';
  static Color primaryColor = Colors.blue;
  static Color accentColor = Colors.orange;
  static late Map<String, String> langMap;

  static Future<void> loadFromSettings() async {
    //加载默认数据
    AppConfig.langMap = Localize.getLangMap(Platform.localeName);
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
    List<String> contents = [];
    contents.add(AppConfig.langMap['locale'] ?? 'en_US');
    await FileSystem.writeFile(path, contents);
  }
}
String textLocalize(String id) {
  return AppConfig.langMap[id] ?? id;
}
void main() async {
  await AppConfig.loadFromSettings();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "title",
      theme: ThemeData(
        colorScheme: .fromSeed(seedColor: AppConfig.accentColor),
      ),
      home: MyHomePage(title : "home_page"),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late String _currentTitle;
  @override
  void initState() {
    super.initState();  // 必须调用父类方法
    _currentTitle = widget.title;  // 2. 用传入的标题初始化
  }
  int _counter = 0;

  void _incrementCounter() async {
    String c = await DirFinder.documentsDir();
    setState(() {
      _currentTitle = "Directory:\n$c";
      _counter += 1;
    });
  }
  void _decrementCounter() {
    if (AppConfig.langMap['locale'] == 'en_US') {
      AppConfig.langMap = Localize.getLangMap('zh_CN');
    } else {
      AppConfig.langMap = Localize.getLangMap('en_US');
    }
    AppConfig.saveToSettings();
    setState(() {
      _counter -= 1;
    });
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Container(
          alignment: Alignment.topLeft, // 关键：顶部对齐
          height: 200, // AppBar 的标准高度
          child: Text(
            textLocalize("home_page"),
          ),
        ),
        toolbarHeight: 200,
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          //
          // TRY THIS: Invoke "debug painting" (choose the "Toggle Debug Paint"
          // action in the IDE, or press "p" in the console), to see the
          // wireframe for each widget.
          mainAxisAlignment: .start,
          children: [
            Text(textLocalize("main")),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
      ),
      floatingActionButton: Stack(
        children: [
          Positioned(
            bottom: 16,
            right: 160,
            child:
              FloatingActionButton(
                onPressed: _incrementCounter,
                tooltip: 'Increment',
                child: const Icon(Icons.add),
                
              ),
          ),
          Positioned(
            bottom: 16,
            left: 160,
            child:
              FloatingActionButton(
                onPressed: _decrementCounter,
                tooltip: textLocalize("main_2"),
                child: const Icon(Icons.remove),
              ),
          ),
        ],
      ),
    );
  }
}
