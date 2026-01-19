import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../app_filesys.dart';
class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();  // 创建状态
}
class _SettingsPageState extends State<SettingsPage> with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  static const TextStyle tabTextStyle = TextStyle(fontSize: 16, fontFamily : 'MSYH');
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this, animationDuration: Duration(milliseconds: 200));  // 正确初始化
  }

  @override
  void dispose() {
    _tabController.dispose();  // 在 dispose 中释放资源
    super.dispose();
  }
  void _changeLanguage() {
    if (AppConfig.langMap['locale'] == 'en_US') {
      setLanguage('zh_CN');
    } else {
      setLanguage('en_US');
    }
    AppConfig.saveMsgToSettings();
    if (mounted) {
      setState(() {});
    }
  }
  
  @override
  Widget build(BuildContext context) {
    List<Widget> tabContents = [
      TDTabBar(
        tabs: [
          TDTab(text: textLocalize('set_tab1')),
          TDTab(text: textLocalize('set_tab2')),
          TDTab(text: textLocalize('set_tab3')),
          TDTab(text: textLocalize('set_tab4')),
        ],
        controller: _tabController,
        showIndicator: true,
        indicatorPadding: EdgeInsets.all(4.0),
        indicatorWidth: 60,
        onTap: (index) {
          setState(() {});
        },
        labelStyle: tabTextStyle,
        unselectedLabelStyle: tabTextStyle,
      )
    ];
    switch (_tabController.index) {
      case 0:
        TDCellGroup cells = TDCellGroup(
          cells: [
            TDCell(//语言切换单元格
              arrow: false,
              title: textLocalize('set_lang'),
              note: textLocalize('lang'),
              onClick: (cell) {
                _changeLanguage();
              },
            ),
            TDCell(
              arrow: false,
              title: textLocalize('set_night'),
              rightIconWidget: TDSwitch(
                isOn: AppConfig.isNightMode,
                onChanged: (cell) {
                  setState(() {
                    setNightMode(!AppConfig.isNightMode);
                    AppConfig.saveMsgToSettings();
                  });
                  return true;
                }
              ),
            ),
          ],
        );
        tabContents.add(cells);
        break;
      case 1:
        break;
      case 2:
        break;
      case 3:
        break;
    }
    return Scaffold(
      appBar: AppBar(
        title: Text(textLocalize("settings")),
      ),
      body: Column(
        children: tabContents
      )
    );
  }
}