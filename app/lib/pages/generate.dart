import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../app_filesys.dart';
class GeneratePage extends StatefulWidget {
  const GeneratePage({super.key});

  @override
  State<GeneratePage> createState() => _GeneratePageState();
}

class _GeneratePageState extends State<GeneratePage> with SingleTickerProviderStateMixin{
  late final TabController _tabController;
  late final ScrollController _scrollController;
  late List<TDUploadFile> _uploadedFiles;
  late Key _uploadKey;
  static const TextStyle tabTextStyle = TextStyle(
    fontSize: 16,
    fontFamily : 'MSYH',
  );
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this, animationDuration: Duration(milliseconds: 200));
    _scrollController = ScrollController();
    _uploadedFiles = [];
    _uploadKey = UniqueKey();
  }
  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    List<Widget> tabContents = [
      TDTabBar(
        outlineType: TDTabBarOutlineType.card,
        tabs: [
          TDTab(text: textLocalize('gen_pic')),
          TDTab(text: textLocalize('gen_text')),
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
        TDUpload myTDUpload = TDUpload(
          key: _uploadKey,
          files: _uploadedFiles,
          multiple: true,
          max: 3,
          onValidate: (err) {
            switch(err) {
              case TDUploadValidatorError.overQuantity:
                TDToast.showText(textLocalize("tip_overquan"), context: context);
                break;
              case TDUploadValidatorError.overSize:
                TDToast.showText(textLocalize("tip_oversize"), context: context);
                break;
            }
          },
          onChange: (files, type) {
            switch (type) {
            case TDUploadType.add:
              _uploadedFiles = [..._uploadedFiles, ...files];
              break;
            case TDUploadType.remove:
              for (var f in files) {
                _uploadedFiles.remove(f);
              }
              break;
            case TDUploadType.replace:
              break;
            }
            setState(() {
              _uploadKey = UniqueKey();
            });
          },
          mediaType: [TDUploadMediaType.image],
          width: 150,
          height: 150,
          sizeLimit: 4096,//文件大小限制
        );
        List<Widget> stackChildren = [
          Positioned(
            top: 20,
            left: 20,
            child: Text(textLocalize('gen_tip_pic')),
          ),
          Positioned(
            top: 60,
            left: 20,
            width: MediaQuery.of(context).size.width - 20,
            child: myTDUpload,
          ),
        ];
        int capacity = (MediaQuery.of(context).size.width - 9)~/161;
        capacity = capacity > 0 ? capacity : 1;
        Widget sb = Scrollbar(
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            child: SizedBox(
              height: 80 + ((_uploadedFiles.length + 1)/capacity).ceil() * 171, // 明确高度
              child: Stack(
                children: stackChildren, // Positioned 放在 Stack 中
              ),
            )),
        );
        tabContents.add(
          Expanded(
            child: sb
          ),
        );
        break;
      case 1:
        tabContents.add(
          Expanded(
            child: Center(
              child: Text(textLocalize('gen_tip_text')),
            ),
          ),
        );
        break;
    }
    return Scaffold(
      appBar: AppBar(
        title: Text(textLocalize('gen_top')),
      ),
      body: Stack (
        children: [
          Column(
            children: tabContents,
          ),
          Align(
            alignment: Alignment(0, 0.9),
            child: TDButton(
              onTap: () {
                TDToast.showText(textLocalize('tip_unava'), context: context);
              },
              activeStyle: TDButtonStyle(
                backgroundColor: Theme.of(context).primaryColorLight,
                textColor: Theme.of(context).shadowColor,
              ),
              type: TDButtonType.fill,
              shape: TDButtonShape.round,
              theme: TDButtonTheme.primary,
              width: 300,
              height: 40,
              text: textLocalize('gen_button'),
            ),
          ),
        ]
      ),
    );
  }
}