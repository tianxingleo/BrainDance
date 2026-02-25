import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import 'dart:io';
import 'package:braindance/extra_func_v2/video_thumbnail.dart';
import 'package:image_picker/image_picker.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:braindance/configs/gen_config.dart';
import '../extra_func/dynamic_background.dart';

class GeneratePage extends StatefulWidget {
  const GeneratePage({super.key});

  @override
  State<GeneratePage> createState() => _GeneratePageState();
}

class _GeneratePageState extends State<GeneratePage>
    with TickerProviderStateMixin {
  late final TabController _tabController;
  late final ScrollController _scrollController;
  late final TextEditingController _textEditingController;
  final ImagePicker _picker = ImagePicker();
  static Key _uploadKey = UniqueKey();
  static Key _uploadKey2 = UniqueKey();
  static const TextStyle tabTextStyle = TextStyle(
    fontSize: 16,
    fontFamily: AppConfig.fontFamily,
  );
  static const int maxImageCount = 3;
  static const int sizeLimit = 4096; //鏂囦欢澶у皬闄愬埗(kb)
  static bool firstCheck = false; //妫€娴嬬敤鎴锋槸鍚︿笉鏄涓€娆℃墦寮€璇ョ晫闈?
  void loadCache() async {
    //Image
    final List<String> paths = await GenConfig.loadImagePathsFile();
    for (String path in paths) {
      GenConfig.uploadedImages.add(
        TDUploadFile(key: 1, assetPath: path, file: File(path)),
      );
    }
    //Text
    final String text = await GenConfig.loadTextFile();
    if (text.isNotEmpty) {
      GenConfig.uploadedText = text;
    }
    //Video
    final List<String> paths2 = await GenConfig.loadVideoPathsFile();
    for (String path in paths2) {
      GenConfig.uploadedVideos.add(
        TDUploadFile(
          key: 1,
          assetPath: path,
          file: File(await VThumb.ensureThumb(path)),
        ),
      );
    }
    setState(() {});
  }

  Future<String> _uploadImages(List<XFile> images) async {
    String msg = "";
    final capacity = maxImageCount - GenConfig.uploadedImages.length;
    if (images.length > capacity) {
      msg = textLocalize("tip_overquan");
    }
    final minLength = (capacity > images.length) ? images.length : capacity;
    for (int i = 0; i < minLength; i++) {
      var image = images[i];
      if (await image.length() ~/ 1024 > sizeLimit) {
        msg = textLocalize("tip_oversize");
        continue;
      }
      GenConfig.uploadedImages.add(
        TDUploadFile(key: 1, assetPath: image.path, file: File(image.path)),
      );
    }
    return msg;
  }

  Future<String> _uploadVideo(XFile? video) async {
    if (video == null) {
      return textLocalize("tip_fail");
    }

    GenConfig.uploadedVideos.add(
      TDUploadFile(
        key: 1,
        assetPath: video.path,
        file: File(await VThumb.ensureThumb(video.path)),
      ),
    );
    return "";
  }

  void _showActionSheet(BuildContext context, bool isImage) {
    TDActionSheet(
      context,
      items: [
        TDActionSheetItem(label: textLocalize("gen_shot")),
        TDActionSheetItem(label: textLocalize("gen_gallery")),
      ],
      cancelText: textLocalize("gen_cancel"),
      onSelected: (item, index) async {
        if (index == 0) {
          final XFile? file;
          late final AssetEntity newAsset;
          final PermissionState ps =
              await PhotoManager.requestPermissionExtend();
          if (!ps.isAuth) {
            if (context.mounted) {
              TDToast.showText(
                textLocalize("tip_no_permission"),
                context: context,
              );
            }
            return;
          }
          if (isImage) {
            file = await _picker.pickImage(source: ImageSource.camera);
            if (file == null) {
              return;
            }
            newAsset = await PhotoManager.editor.saveImageWithPath(
              file.path,
              title: file.name,
            );
          } else {
            file = await _picker.pickVideo(
              source: ImageSource.camera,
              maxDuration: const Duration(minutes: 3),
            );
            if (file == null) {
              return;
            }
            newAsset = await PhotoManager.editor.saveVideo(
              File(file.path),
              title: file.name,
            );
          }
          try {
            await FileSystem.deleteFile(file.path);
            File? f = await newAsset.originFile;
            if (f == null) {
              throw ();
            }
            XFile fileSaved = XFile(f.path);
            late final String msg;
            if (isImage) {
              msg = await _uploadImages(List.filled(1, fileSaved));
            } else {
              msg = await _uploadVideo(fileSaved);
            }
            if (msg.isNotEmpty && context.mounted) {
              TDToast.showText(msg, context: context);
            }
          } catch (e) {
            if (context.mounted) {
              TDToast.showText(textLocalize("tip_fail"), context: context);
            }
          }
        } else {
          late final String msg;
          if (isImage) {
            msg = await _uploadImages(await _picker.pickMultiImage());
          } else {
            msg = await _uploadVideo(
              await _picker.pickVideo(source: ImageSource.gallery),
            );
          }
          if (msg.isNotEmpty && context.mounted) {
            TDToast.showText(msg, context: context);
          }
        }
        setState(() {});
      },
    ).show();
  }

  @override
  void initState() {
    super.initState();
    _tabController = TabController(
      length: 3,
      vsync: this,
      animationDuration: Duration(milliseconds: 200),
    );
    _scrollController = ScrollController();
    _textEditingController = TextEditingController();

    if (firstCheck) {
      return;
    }
    firstCheck = true;
    //浠ヤ笅浠ｇ爜鍙細鎵ц涓€娆?
    //鍦ㄦ澶勬墽琛屽姞杞界紦瀛樻暟鎹搷浣?
    loadCache();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _scrollController.dispose();
    _textEditingController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> tabContents = [
      Container(
        color: TDTheme.of(context).whiteColor1,
        child: TDTabBar(
          tabs: [
            TDTab(text: textLocalize('gen_pic')),
            TDTab(text: textLocalize('gen_text')),
            TDTab(text: textLocalize('gen_video')),
          ],
          controller: _tabController,
          showIndicator: true,
          indicatorPadding: const EdgeInsets.all(4.0),
          indicatorWidth: 24, // 鏇寸煭鐨勬寚绀哄櫒鏄惧緱鏇寸簿鑷?
          indicatorHeight: 3,
          indicatorColor: AppConfig.primaryColor,
          onTap: (index) {
            setState(() {});
          },
          labelStyle: tabTextStyle.copyWith(
            fontWeight: FontWeight.w600,
            color: AppConfig.primaryColor,
          ),
          unselectedLabelStyle: tabTextStyle.copyWith(
            fontWeight: FontWeight.w400,
            color: TDTheme.of(context).fontGyColor3,
          ),
        ),
      ),
    ];
    switch (_tabController.index) {
      case 0:
        final TDUpload myTDUpload = TDUpload(
          key: _uploadKey,
          files: GenConfig.uploadedImages,
          multiple: true,
          max: maxImageCount,
          onUploadTap: () {
            _showActionSheet(context, true);
          },
          onChange: (files, type) {
            switch (type) {
              case TDUploadType.add:
                GenConfig.uploadedImages = [
                  ...GenConfig.uploadedImages,
                  ...files,
                ];
                break;
              case TDUploadType.remove:
                for (var f in files) {
                  GenConfig.uploadedImages.remove(f);
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
        );
        final Widget sb = Scrollbar(
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TDText(
                  textLocalize('gen_tip_pic'),
                  font: TDTheme.of(context).fontTitleMedium,
                  fontWeight: FontWeight.w600,
                  textColor: TDTheme.of(context).fontGyColor1,
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
                    border: Border.all(
                      color: TDTheme.of(context).whiteColor1,
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 20,
                        spreadRadius: 5,
                      )
                    ],
                  ),
                  child: myTDUpload,
                ),
                const SizedBox(height: 100), // 涓哄簳閮ㄦ寜閽暀鍑虹┖闂?
              ],
            ),
          ),
        );
        tabContents.add(Expanded(child: sb));
        break;
      case 1:
        _textEditingController.text = GenConfig.uploadedText;
        final Widget sb = Scrollbar(
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TDText(
                  textLocalize('gen_tip_text'),
                  font: TDTheme.of(context).fontTitleMedium,
                  fontWeight: FontWeight.w600,
                  textColor: TDTheme.of(context).fontGyColor1,
                ),
                const SizedBox(height: 24),
                Container(
                  decoration: BoxDecoration(
                    color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
                    border: Border.all(
                      color: TDTheme.of(context).whiteColor1,
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 20,
                        spreadRadius: 5,
                      )
                    ],
                  ),
                  child: TDTextarea(
                    controller: _textEditingController,
                    hintText: textLocalize("gen_tip_textbox"),
                    minLines: 8,
                    maxLines: 20,
                    onChanged: (value) {
                      GenConfig.uploadedText = value;
                    },
                    decoration: BoxDecoration(
                      color: Colors.transparent,
                      borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
                      border: Border.all(color: Colors.transparent),
                    ),
                    textStyle: TextStyle(
                      color: TDTheme.of(context).fontGyColor1,
                      fontSize: 16,
                    ),
                  ),
                ),
                const SizedBox(height: 100), // 涓哄簳閮ㄦ寜閽暀鍑虹┖闂?
              ],
            ),
          ),
        );
        tabContents.add(Expanded(child: sb));
        break;
      case 2:
        final TDUpload myTDUpload = TDUpload(
          type: TDUploadBoxType.circle,
          key: _uploadKey2,
          files: GenConfig.uploadedVideos,
          multiple: false,
          onUploadTap: () {
            _showActionSheet(context, false);
          },
          onChange: (files, type) {
            switch (type) {
              case TDUploadType.add:
                GenConfig.uploadedVideos = [
                  ...GenConfig.uploadedVideos,
                  ...files,
                ];
                break;
              case TDUploadType.remove:
                for (var f in files) {
                  GenConfig.uploadedVideos.remove(f);
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
        final Widget sb = Scrollbar(
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TDText(
                  textLocalize('gen_tip_video'),
                  font: TDTheme.of(context).fontTitleMedium,
                  fontWeight: FontWeight.w600,
                  textColor: TDTheme.of(context).fontGyColor1,
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
                    border: Border.all(
                      color: TDTheme.of(context).whiteColor1,
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 20,
                        spreadRadius: 5,
                      )
                    ],
                  ),
                  child: myTDUpload,
                ),
                const SizedBox(height: 100), // 涓哄簳閮ㄦ寜閽暀鍑虹┖闂?
              ],
            ),
          ),
        );
        tabContents.add(Expanded(child: sb));
        break;
    }
    return Scaffold(
      backgroundColor: TDTheme.of(context).grayColor1,
      appBar: AppBar(
        title: TDText(
          textLocalize('gen_top'),
          font: TDTheme.of(context).fontTitleLarge,
          fontWeight: FontWeight.w600,
          textColor: TDTheme.of(context).fontGyColor1,
        ),
        backgroundColor: TDTheme.of(context).whiteColor1.withValues(alpha: 0.95),
        elevation: 0,
        centerTitle: true,
      ),
      extendBodyBehindAppBar: true,
      body: DynamicGradientBackground(
        child: Stack(
          children: [
            SafeArea(
              child: Column(children: tabContents),
            ),
            Align(
              alignment: Alignment.bottomCenter,
              child: Container(
                padding: const EdgeInsets.only(bottom: 32, top: 24),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      TDTheme.of(context).grayColor1,
                      TDTheme.of(context).grayColor1.withValues(alpha: 0.9),
                      TDTheme.of(context).grayColor1.withValues(alpha: 0.0),
                    ],
                    stops: const [0.0, 0.6, 1.0],
                  ),
                ),
                child: TDButton(
                  onTap: () {
                    TDToast.showText(textLocalize('tip_unava'), context: context);
                  },
                  style: TDButtonStyle(
                    backgroundColor: AppConfig.primaryColor,
                    textColor: Colors.white,
                    radius: BorderRadius.circular(TDTheme.of(context).radiusRound),
                  ),
                  type: TDButtonType.fill,
                  shape: TDButtonShape.round,
                  theme: TDButtonTheme.primary,
                  size: TDButtonSize.large,
                  width: MediaQuery.of(context).size.width * 0.85,
                  text: textLocalize('gen_button'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
