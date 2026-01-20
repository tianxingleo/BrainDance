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
  late final ScrollController _scrollController;
  static const TextStyle tabTextStyle = TextStyle(fontSize: 16, fontFamily : 'MSYH');
  late List<int> _pickerSelectedIndex; // 选择器选中项索引;
  @override
  void initState() {//若 tab 数量有变，需同步修改 length
    super.initState();
    _tabController = TabController(length: 4, vsync: this, animationDuration: Duration(milliseconds: 200));  // 正确初始化
    _pickerSelectedIndex = List.filled(2, -1); // 初始化选择器索引
    _scrollController = ScrollController();
  }

  @override
  void dispose() {
    _tabController.dispose();  // 在 dispose 中释放资源
    _scrollController.dispose();
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
        TDCellGroup cells = TDCellGroup(
          cells: [
            _buildPicker(context, pickerTitle: 'PickerTest1', pickerIndex: 0),
            _buildPicker(context, pickerTitle: 'PickerTest2', pickerIndex: 1),
          ],
        );
        tabContents.add(cells);
        break;
      case 2:
        TDCellGroup cells = TDCellGroup(
          cells: [
            TDCell(//版本信息
              arrow: false,
              title: textLocalize('set_ver'),
              note: AppConfig.version,
            ),
            TDCell(
              arrow: false,
              title: textLocalize('set_pub'),
              note: AppConfig.publishDate,
            ),
          ],
        );
        tabContents.add(cells);
        break;
      case 3:
        Widget sb = Scrollbar(
          controller: _scrollController,
          child: ListView.builder(
            controller: _scrollController,
            itemCount: 50,
            itemBuilder: (context, index) => ListTile(title: Text('Item $index')),
          ),
        );
        tabContents.add(Expanded(child: sb));
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

  TDCell _buildPicker(BuildContext context, {
    String pickerTitle = '',
    int pickerIndex = 0,
    List<String> pickerData = const ['Option 1', 'Option 2', 'Option 3'],
  }) {
    int selectedIndex = _pickerSelectedIndex[pickerIndex];
    return TDCell(//版本信息
      arrow: true,
      title: pickerTitle,
      note: (selectedIndex == -1) ? textLocalize("pick_null") : pickerData[selectedIndex],
      onClick: (click) {
        TDPicker.showMultiPicker(
          context,
          title: pickerTitle,
          onConfirm: (selected) {
            setState(() {
              _pickerSelectedIndex[pickerIndex] = selected[0];
            });
            Navigator.of(context).pop();
          },
          data: [pickerData],
          initialIndexes: (selectedIndex == -1) ? [0] : [selectedIndex],
        );
      },
    );
  }
}