import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../app_configs.dart';
class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();  // 创建状态
}
class _SettingsPageState extends State<SettingsPage> with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  late final ScrollController _scrollController;
  static const TextStyle tabTextStyle = TextStyle(fontSize: 16, fontFamily : 'MSYH');
  static final List<int> _pickerSelectedIndex = List.filled(2, -1); // 选择器选中项索引;
  @override
  void initState() {//若 tab 数量有变，需同步修改 length
    super.initState();
    _tabController = TabController(length: 4, vsync: this, animationDuration: Duration(milliseconds: 200));  // 正确初始化
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
    SetConfig.saveMsgToFile();
    if (mounted) {
      setState(() {});
    }
  }
  @override
  Widget build(BuildContext context) {
    final TDTabBar myTabBar = 
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
        labelStyle: tabTextStyle,
        unselectedLabelStyle: tabTextStyle,
      );
    final TDCellGroup cells0 = TDCellGroup(
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
                    SetConfig.saveMsgToFile();
                  });
                  return true;
                }
              ),
            ),
          ],
        );
    final TDCellGroup cells1 = TDCellGroup(
          cells: [
            _buildPicker(context, pickerTitle: 'PickerTest1', pickerIndex: 0),
            _buildPicker(context, pickerTitle: 'PickerTest2', pickerIndex: 1),
          ],
        );
    final TDCellGroup cells2 = TDCellGroup(
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
            TDCell(
              arrow: false,
              title: textLocalize('set_cache'),
              onClick:(cell) async {
                TDToast.showText(textLocalize("tip_cache"), context: context);
                await DirSystem.deleteDir(await DirFinder.cacheDir());
              },
            ),
          ],
        );
    final Widget sb = Scrollbar(
          controller: _scrollController,
          child: ListView.builder(
            controller: _scrollController,
            itemCount: 50,
            itemBuilder: (context, index) => ListTile(title: Text('Item $index')),
          ),
        );
    return Scaffold(
      appBar: AppBar(
        title: Text(textLocalize("settings")),
      ),
      body: Column(
        children: [
          myTabBar,
          Expanded(
            child: TDTabBarView(
              controller: _tabController,
              children: [
                cells0, cells1, cells2, sb
              ]
            ) 
          ),
        ]
      )
    );
  }

  TDCell _buildPicker(BuildContext context, {
    String pickerTitle = '',
    int pickerIndex = 0,
    List<String> pickerData = const ['Option 1', 'Option 2', 'Option 3'],
  }) {
    final int selectedIndex = _pickerSelectedIndex[pickerIndex];
    return TDCell(//版本信息
      arrow: true,
      title: pickerTitle,
      note: (selectedIndex == -1) ? textLocalize("pick_null") : pickerData[selectedIndex],
      onClick: (click) {
        TDPicker.showMultiPicker(
          context,
          titleHeight: 40,
          pickerHeight: 200,
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