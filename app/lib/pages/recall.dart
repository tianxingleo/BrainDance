import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../configs/app_config.dart';
import '../debug_tool.dart';

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
  @override
  void initState() {
    super.initState();
    DebugVar.onUpdate = () {setState(() {
      
    });};
  }
  @override
  void dispose() {
    DebugVar.onUpdate = null;
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppConfig.accentColor,
      appBar: AppBar(
        backgroundColor: AppConfig
            .primaryColor, //Theme.of(context).colorScheme.inversePrimary
        title: Container(
          alignment: Alignment.centerLeft, // 关键：顶部对齐
          child: Text(
            textLocalize("home_page"),
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        ),
        toolbarHeight: 60,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: .start,
          children: [Text(DebugVar.debugText)],
        ),
      ),
      floatingActionButton: TDButton(
        text: "Test Button",
        onTap: SupabaseApi.test,
      ),
    );
  }
}
