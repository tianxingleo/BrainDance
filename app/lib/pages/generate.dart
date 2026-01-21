import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/app_configs.dart';
import 'package:braindance/main.dart';
import 'dart:io';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
class GeneratePage extends StatefulWidget {
  const GeneratePage({super.key});

  @override
  State<GeneratePage> createState() => _GeneratePageState();
}

class _GeneratePageState extends State<GeneratePage> with SingleTickerProviderStateMixin{
  late final TabController _tabController;
  late final ScrollController _scrollController;
  late final TextEditingController _textEditingController;
  static Key _uploadKey = UniqueKey();
  static Key _uploadKey2 = UniqueKey();
  static const TextStyle tabTextStyle = TextStyle(
    fontSize: 16,
    fontFamily : 'MSYH',
  );
  static bool firstCheck = true;//检测用户是否是第一次打开该界面
  void loadCache() async {
      //Image
      final List<String> paths = await GenConfig.loadImagePathsFile();
      for (String path in paths) {
        uploadedImages.add(TDUploadFile(
          key : 1,
          assetPath: path,
          file: File(path),
        ));
      }
      //Text
      final String text = await GenConfig.loadTextFile();
      if (text.isNotEmpty) {
        uploadedText = text;
      }
      //Video
      final List<String> paths2 = await GenConfig.loadVideoPathsFile();
      for (String path in paths2) {
        uploadedVideos.add(TDUploadFile(
          key : 1,
          assetPath: path,
          file: File(await VThumb.ensureThumb(path)),
        ));
      }
      setState(() {
        
      });
  }
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this, animationDuration: Duration(milliseconds: 200));
    _scrollController = ScrollController();
    _textEditingController = TextEditingController();
    if (firstCheck) {
      //在此处执行加载缓存数据操作
      loadCache();
      //Cancel firstCheck
      firstCheck = false;
    }
  }
  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    final List<Widget> tabContents = [
      TDTabBar(
        outlineType: TDTabBarOutlineType.card,
        tabs: [
          TDTab(text: textLocalize('gen_pic')),
          TDTab(text: textLocalize('gen_text')),
          TDTab(text: textLocalize('gen_video')),
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
        final TDUpload myTDUpload = TDUpload(
          key: _uploadKey,
          files: uploadedImages,
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
              uploadedImages = [...uploadedImages, ...files];
              break;
            case TDUploadType.remove:
              for (var f in files) {
                uploadedImages.remove(f);
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
        final List<Widget> stackChildren = [
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
        final Widget sb = Scrollbar(
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            child: SizedBox(
              height: 80 + ((uploadedImages.length + 1)/capacity).ceil() * 171, // 明确高度
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
        var lineCount = (MediaQuery.of(context).size.height - 350)~/25;
        lineCount = (lineCount > 0) ? lineCount : 1;
        _textEditingController.text = uploadedText;
        tabContents.add(
          Expanded(
          child : Stack(
            children: [
              Positioned(
                top : 20,
                left : 20,
                child : Text(textLocalize('gen_tip_text'))
              ),
              Positioned(
                width: MediaQuery.of(context).size.width - 20,
                top : 60,
                left : 10,
                child : TDTextarea(
                  controller: _textEditingController,
                  hintText: textLocalize("gen_tip_textbox"),
                  maxLines: lineCount,
                  minLines: lineCount,
                  onChanged: (value) {
                    uploadedText = value;
                  },
                  decoration: BoxDecoration(
                    color: TDTheme.of(context).bgColorContainer,
                    borderRadius:
                        BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
                  ),
                  margin: EdgeInsets.only(
                      right: TDTheme.of(context).spacer16,
                      left: TDTheme.of(context).spacer16),
                )
              )
            ]
          )),
        );
        break;
      case 2:
        final TDUpload myTDUpload = TDUpload(
          type: TDUploadBoxType.circle,
          key: _uploadKey2,
          files: uploadedVideos,
          multiple: false,
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
          onChange: (files, type) async {
            switch (type) {
            case TDUploadType.add:
              for (var file in files) {
                TDUploadFile newFile = TDUploadFile(
                  key: 1,
                  assetPath : file.assetPath,
                  file : File(await VThumb.ensureThumb(file.assetPath.toString())),
                );
                uploadedVideos.add(newFile);
              }
              break;
            case TDUploadType.remove:
              for (var f in files) {
                uploadedVideos.remove(f);
              }
              break;
            case TDUploadType.replace:
              break;
            }
            setState(() {
              _uploadKey2 = UniqueKey();
            });
          },
          mediaType: [TDUploadMediaType.video],
          width: 150,
          height: 150,
        );
        final List<Widget> stackChildren = [
          Positioned(
            top: 20,
            left: 20,
            child: Text(textLocalize('gen_tip_video')),
          ),
          Positioned(
            top: 100,
            left: 20,
            width: MediaQuery.of(context).size.width - 20,
            child: myTDUpload,
          ),
        ];
        int capacity = (MediaQuery.of(context).size.width - 9)~/161;
        capacity = capacity > 0 ? capacity : 1;
        final Widget sb = Scrollbar(
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            child: SizedBox(
              height: 120 + ((uploadedVideos.length + 1)/capacity).ceil() * 171, // 明确高度
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