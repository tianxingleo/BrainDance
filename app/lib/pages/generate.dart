import 'dart:async';
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

import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:braindance/main.dart' show pageIndexProvider;
import 'package:dio/dio.dart';
import 'dart:math';
import '../configs/supabase_config.dart';

class GeneratePage extends ConsumerStatefulWidget {
  const GeneratePage({super.key});

  @override
  ConsumerState<GeneratePage> createState() => _GeneratePageState();
}

class _GeneratePageState extends ConsumerState<GeneratePage>
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
  static const int maxImageCount = 1;
  static const int sizeLimit = 40960; //限制大小(kb)
  static bool firstCheck = false;

  bool _isUploading = false;
  double _uploadProgress = 0.0;
  String? _generatedImageUrl;
  bool _isGenerating = false;

  static final Random _rdg = Random();

  static String _generateSceneId() {
    DateTime time = DateTime.now();
    return 'scene_'
        '${time.year.toString().padLeft(4, '0')}'
        '${time.month.toString().padLeft(2, '0')}'
        '${time.day.toString().padLeft(2, '0')}'
        '_'
        '${_rdg.nextInt(1000000).toString().padLeft(6, '0')}';
  }

  /// 弹出 ActionSheet 让用户选择图片任务类型，返回 task_type 字符串，取消返回 null。
  Future<String?> _showImageTaskTypeSheet() {
    final completer = Completer<String?>();
    TDActionSheet(
      context,
      description: '拍摄的内容为……（用于优化生成效果）',
      items: [
        TDActionSheetItem(label: '物体'),
        TDActionSheetItem(label: '场景'),
      ],
      cancelText: textLocalize("gen_cancel"),
      onSelected: (item, index) {
        if (index == 0) {
          completer.complete('single_image_sam3d');
        } else {
          completer.complete('single_image_sharp');
        }
      },
      onCancel: () {
        if (!completer.isCompleted) completer.complete(null);
      },
      onClose: () {
        if (!completer.isCompleted) completer.complete(null);
      },
    ).show();
    return completer.future;
  }

  /// 显示文生图预览底部弹窗，支持重新生成和确认使用
  void _showTextImagePreview(String prompt) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        return StatefulBuilder(
          builder: (builderContext, setSheetState) {
            return Container(
              height: MediaQuery.of(context).size.height * 0.75,
              decoration: BoxDecoration(
                color: TDTheme.of(context).whiteColor1,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(20),
                ),
              ),
              child: Column(
                children: [
                  // 顶部标题栏
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        TDText(
                          textLocalize('gen_text_preview_title'),
                          font: TDTheme.of(context).fontTitleMedium,
                          fontWeight: FontWeight.w600,
                        ),
                        GestureDetector(
                          onTap: () => Navigator.pop(sheetContext),
                          child: Icon(
                            Icons.close,
                            color: TDTheme.of(context).fontGyColor3,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const Divider(height: 1),
                  // 图片预览
                  Expanded(
                    child: _isGenerating
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const CircularProgressIndicator(),
                                const SizedBox(height: 16),
                                TDText(
                                  textLocalize('gen_text_generating'),
                                  textColor:
                                      TDTheme.of(context).fontGyColor2,
                                ),
                              ],
                            ),
                          )
                        : _generatedImageUrl != null
                            ? Padding(
                                padding: const EdgeInsets.all(16),
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(12),
                                  child: Image.network(
                                    _generatedImageUrl!,
                                    fit: BoxFit.contain,
                                    loadingBuilder: (context, child,
                                        loadingProgress) {
                                      if (loadingProgress == null) {
                                        return child;
                                      }
                                      return const Center(
                                        child:
                                            CircularProgressIndicator(),
                                      );
                                    },
                                    errorBuilder:
                                        (context, error, stackTrace) {
                                      return Center(
                                        child: TDText(
                                          '图片加载失败',
                                          textColor: TDTheme.of(context)
                                              .fontGyColor3,
                                        ),
                                      );
                                    },
                                  ),
                                ),
                              )
                            : const SizedBox.shrink(),
                  ),
                  // 底部按钮
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
                    child: Row(
                      children: [
                        // 重新生成按钮
                        Expanded(
                          child: TDButton(
                            onTap: _isGenerating
                                ? () {}
                                : () async {
                                    setSheetState(() {});
                                    setState(() {
                                      _isGenerating = true;
                                    });
                                    try {
                                      final response = await Supabase
                                          .instance.client.functions
                                          .invoke(
                                        'text-to-image',
                                        body: {'prompt': prompt},
                                      );
                                      final data = response.data;
                                      if (data is Map &&
                                          data['success'] == true &&
                                          data['image_url'] != null) {
                                        setState(() {
                                          _generatedImageUrl =
                                              data['image_url'] as String;
                                        });
                                      } else {
                                        if (mounted) {
                                          TDToast.showText(
                                            '重新生成失败',
                                            context: context,
                                          );
                                        }
                                      }
                                    } catch (e) {
                                      if (mounted) {
                                        TDToast.showText(
                                          '重新生成失败: $e',
                                          context: context,
                                        );
                                      }
                                    } finally {
                                      setState(() {
                                        _isGenerating = false;
                                      });
                                      setSheetState(() {});
                                    }
                                  },
                            text: textLocalize('gen_text_regenerate'),
                            theme: TDButtonTheme.light,
                            size: TDButtonSize.large,
                            shape: TDButtonShape.round,
                          ),
                        ),
                        const SizedBox(width: 12),
                        // 确认使用按钮
                        Expanded(
                          child: TDButton(
                            onTap: (_isGenerating ||
                                    _generatedImageUrl == null)
                                ? () {}
                                : () async {
                                    Navigator.pop(sheetContext);
                                    await _confirmTextImage(prompt);
                                  },
                            text: textLocalize('gen_text_confirm'),
                            style: TDButtonStyle(
                              backgroundColor: AppConfig.primaryColor,
                              textColor: Colors.white,
                              radius: BorderRadius.circular(
                                TDTheme.of(context).radiusRound,
                              ),
                            ),
                            type: TDButtonType.fill,
                            theme: TDButtonTheme.primary,
                            size: TDButtonSize.large,
                            shape: TDButtonShape.round,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  /// 确认使用文生图片：分类 + 上传 + 创建任务
  Future<void> _confirmTextImage(String prompt) async {
    final client = Supabase.instance.client;
    var user = client.auth.currentUser;
    if (user == null) {
      if (mounted) {
        TDToast.showText('未登录，即将跳转登录页面...', context: context);
        await Navigator.pushNamed(context, '/login');
      }
      user = client.auth.currentUser;
      if (user == null) {
        if (mounted) TDToast.showText('登录已取消或未完成', context: context);
        return;
      }
    }

    setState(() {
      _isUploading = true;
    });

    try {
      final response = await client.functions.invoke(
        'confirm-text-image',
        body: {
          'image_url': _generatedImageUrl,
          'prompt': prompt,
        },
      );

      final data = response.data;
      if (data is Map && data['success'] == true) {
        if (mounted) {
          TDToast.showText('提交成功，任务已创建', context: context);
          ref.read(pageIndexProvider.notifier).state = 0;
          final nav = Navigator.of(context);
          _generatedImageUrl = null;
          _textEditingController.clear();
          GenConfig.uploadedText = '';
          nav.pushNamed('/tasks');
        }
      } else {
        final errMsg = (data is Map) ? (data['error'] ?? '提交失败') : '服务器返回异常';
        throw Exception(errMsg);
      }
    } on FunctionException catch (e) {
      if (mounted) {
        TDToast.showText('提交失败: ${e.details}', context: context);
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('提交失败: $e', context: context);
      }
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  Future<void> _submit() async {
    if (_tabController.index == 0) {
      if (GenConfig.uploadedImages.isEmpty) {
        if (mounted) TDToast.showText('请先选择图片', context: context);
        return;
      }

      // 弹窗让用户选择拍摄内容类型
      final taskType = await _showImageTaskTypeSheet();
      if (taskType == null) return; // 用户取消

      final client = Supabase.instance.client;
      var user = client.auth.currentUser;
      if (user == null) {
        if (mounted) {
          TDToast.showText('未登录，即将跳转登录页面...', context: context);
          await Navigator.pushNamed(context, '/login');
        }
        user = client.auth.currentUser;
        if (user == null) {
          if (mounted) TDToast.showText('登录已取消或未完成', context: context);
          return;
        } else {
          if (mounted) TDToast.showText('登录成功，开始上传', context: context);
        }
      }

      setState(() {
        _isUploading = true;
      });

      try {
        final sceneId = _generateSceneId();

        // 上传图片
        final imageStoragePath = '${user.id}/$sceneId/raw/image.png';
        final imagePath = GenConfig.uploadedImages[0].assetPath!;
        final file = File(imagePath);
        final fileSize = await file.length();
        final url =
            '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$imageStoragePath';
        final dio = Dio();

        await dio.post(
          url,
          data: file.openRead(),
          options: Options(
            headers: {
              'Authorization':
                  'Bearer ${client.auth.currentSession?.accessToken}',
              'apikey': SupabaseConfig.anonKey,
              'Content-Type': 'image/png',
              'Content-Length': fileSize.toString(),
            },
          ),
          onSendProgress: (count, total) {
            if (mounted) {
              setState(() {
                _uploadProgress = count / fileSize;
              });
            }
          },
        );

        // 创建任务
        await client.from("processing_tasks").insert({
          'scene_id': sceneId,
          'user_id': user.id,
          'status': 'pending',
          'task_type': taskType,
        });

        if (mounted) {
          TDToast.showText('提交成功，任务已创建', context: context);
          ref.read(pageIndexProvider.notifier).state = 0;
          final nav = Navigator.of(context);
          GenConfig.uploadedImages.clear();
          nav.pushNamed('/tasks');
        }
      } catch (e) {
        if (mounted) {
          print(e);
          TDToast.showText('提交失败: $e', context: context);
        }
      } finally {
        if (mounted) {
          setState(() {
            _isUploading = false;
          });
        }
      }
    } else if (_tabController.index == 1) {
      // 文本生成流程
      final prompt = _textEditingController.text.trim();
      if (prompt.isEmpty) {
        if (mounted) TDToast.showText('请先输入描述文本', context: context);
        return;
      }

      setState(() {
        _isGenerating = true;
      });

      try {
        // 调用 text-to-image edge function
        final response = await Supabase.instance.client.functions.invoke(
          'text-to-image',
          body: {'prompt': prompt},
        );

        final data = response.data;
        if (data is Map && data['success'] == true && data['image_url'] != null) {
          final imageUrl = data['image_url'] as String;
          setState(() {
            _generatedImageUrl = imageUrl;
            _isGenerating = false;
          });
          if (mounted) {
            _showTextImagePreview(prompt);
          }
        } else {
          final errMsg = (data is Map) ? (data['error'] ?? '生成失败') : '服务器返回异常';
          throw Exception(errMsg);
        }
      } on FunctionException catch (e) {
        if (mounted) {
          TDToast.showText('生成失败: ${e.details}', context: context);
        }
      } catch (e) {
        if (mounted) {
          TDToast.showText('生成失败: $e', context: context);
        }
      } finally {
        if (mounted) {
          setState(() {
            _isGenerating = false;
          });
        }
      }
    } else if (_tabController.index == 2) {
      // 视频生成，按 record 页（video_submit）逻辑实现：
      if (GenConfig.uploadedVideos.isEmpty) {
        if (mounted) TDToast.showText('请先选择视频', context: context);
        return;
      }

      final client = Supabase.instance.client;
      var user = client.auth.currentUser;
      if (user == null) {
        if (mounted) {
          TDToast.showText('未登录，即将跳转登录页面...', context: context);
          await Navigator.pushNamed(context, '/login');
        }
        user = client.auth.currentUser;
        if (user == null) {
          if (mounted) TDToast.showText('登录已取消或未完成', context: context);
          return;
        } else {
          if (mounted) TDToast.showText('登录成功，开始上传', context: context);
        }
      }

      setState(() {
        _isUploading = true;
      });

      try {
        final sceneId = _generateSceneId();

        // 上传视频
        final videoStoragePath = '${user.id}/$sceneId/raw/video.mp4';
        final videoPath = GenConfig.uploadedVideos[0].assetPath!;
        final file = File(videoPath);
        final fileSize = await file.length();
        final url =
            '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$videoStoragePath';
        final dio = Dio();

        await dio.post(
          url,
          data: file.openRead(),
          options: Options(
            headers: {
              'Authorization':
                  'Bearer ${client.auth.currentSession?.accessToken}',
              'apikey': SupabaseConfig.anonKey,
              'Content-Type': 'video/mp4',
              'Content-Length': fileSize.toString(),
            },
          ),
          onSendProgress: (count, total) {
            if (mounted) {
              setState(() {
                _uploadProgress = count / fileSize;
              });
            }
          },
        );

        // 创建任务 (跟 video_submit 相同)
        await client.from("processing_tasks").insert({
          'scene_id': sceneId,
          'user_id': user.id,
          'status': 'pending',
          'task_params': {'mapper_type': 'da3'},
        });

        if (mounted) {
          TDToast.showText('提交成功，任务已创建', context: context);
          ref.read(pageIndexProvider.notifier).state = 0;
          final nav = Navigator.of(context);
          // 跳转到第一页的任务列表，清空当前页面缓存
          GenConfig.uploadedVideos.clear();
          nav.pushNamed('/tasks');
        }
      } catch (e) {
        if (mounted) {
          print(e);
          TDToast.showText('提交失败: $e', context: context);
        }
      } finally {
        if (mounted) {
          setState(() {
            _isUploading = false;
          });
        }
      }
    }
  }

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
    Widget? currentTabContent;
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
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(0),
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
                    color: TDTheme.of(
                      context,
                    ).whiteColor1.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(
                      TDTheme.of(context).radiusExtraLarge,
                    ),
                    border: Border.all(
                      color: TDTheme.of(context).whiteColor1,
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: myTDUpload,
                ),
                const SizedBox(height: 100), // 涓哄簳閮ㄦ寜閽暀鍑虹┖闂?
              ],
            ),
          ),
        );
        break;
      case 1:
        _textEditingController.text = GenConfig.uploadedText;
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(1),
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
                    color: TDTheme.of(
                      context,
                    ).whiteColor1.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(
                      TDTheme.of(context).radiusExtraLarge,
                    ),
                    border: Border.all(
                      color: TDTheme.of(context).whiteColor1,
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
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
                      borderRadius: BorderRadius.circular(
                        TDTheme.of(context).radiusExtraLarge,
                      ),
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
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(2),
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
                    color: TDTheme.of(
                      context,
                    ).whiteColor1.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(
                      TDTheme.of(context).radiusExtraLarge,
                    ),
                    border: Border.all(
                      color: TDTheme.of(context).whiteColor1,
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.05),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: myTDUpload,
                ),
                const SizedBox(height: 100), // 涓哄簳閮ㄦ寜閽暀鍑虹┖闂?
              ],
            ),
          ),
        );
        break;
    }

    if (currentTabContent != null) {
      tabContents.add(
        Expanded(
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            switchInCurve: Curves.easeOutCubic,
            switchOutCurve: Curves.easeInCubic,
            transitionBuilder: (Widget child, Animation<double> animation) {
              return FadeTransition(
                opacity: animation,
                child: SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0.05, 0.0),
                    end: Offset.zero,
                  ).animate(animation),
                  child: child,
                ),
              );
            },
            child: currentTabContent,
          ),
        ),
      );
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
        backgroundColor: TDTheme.of(
          context,
        ).whiteColor1.withValues(alpha: 0.95),
        elevation: 0,
        centerTitle: true,
      ),
      extendBodyBehindAppBar: true,
      body: DynamicGradientBackground(
        child: Stack(
          children: [
            SafeArea(child: Column(children: tabContents)),
            Align(
              alignment: Alignment(0, 0.7),
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
                  onTap: (_isUploading || _isGenerating) ? () {} : _submit,
                  style: TDButtonStyle(
                    backgroundColor: AppConfig.primaryColor,
                    textColor: Colors.white,
                    radius: BorderRadius.circular(
                      TDTheme.of(context).radiusRound,
                    ),
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
            if (_isUploading || _isGenerating)
              Container(
                color: Colors.black45,
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(
                        _isGenerating
                            ? textLocalize('gen_text_generating')
                            : '正在上传... ${(_uploadProgress * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
