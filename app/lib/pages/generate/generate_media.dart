part of '../generate.dart';

extension _GenerateMediaX on _GeneratePageState {
  void loadCache() async {
    final List<String> imagePaths = await GenConfig.loadImagePathsFile();
    for (final path in imagePaths) {
      GenConfig.uploadedImages.add(
        TDUploadFile(key: 1, assetPath: path, file: File(path)),
      );
    }

    final String text = await GenConfig.loadTextFile();
    if (text.isNotEmpty) {
      GenConfig.uploadedText = text;
    }

    final List<String> videoPaths = await GenConfig.loadVideoPathsFile();
    for (final path in videoPaths) {
      GenConfig.uploadedVideos.add(
        TDUploadFile(
          key: 1,
          assetPath: path,
          file: File(await VThumb.ensureThumb(path)),
        ),
      );
    }

    _refresh(() {});
  }

  Future<String> _uploadImages(List<XFile> images) async {
    String msg = "";
    final capacity =
        _GeneratePageState.maxImageCount - GenConfig.uploadedImages.length;
    if (images.length > capacity) {
      msg = textLocalize("tip_overquan");
    }

    final minLength = capacity > images.length ? images.length : capacity;
    for (int i = 0; i < minLength; i++) {
      final image = images[i];
      if (await image.length() > _GeneratePageState.imageSizeLimitBytes) {
        msg = textLocalize("gen_image_too_large");
        continue;
      }
      GenConfig.uploadedImages.add(
        TDUploadFile(key: 1, assetPath: image.path, file: File(image.path)),
      );
    }
    return msg;
  }

  static Future<double?> _probeVideoDuration(File file) async {
    final info = await VideoPreprocessor.probe(file);
    if (info == null) return null;
    final durRaw = info['duration'];
    if (durRaw == null) return null;

    final durStr = durRaw.toString();
    final secs = double.tryParse(durStr);
    if (secs != null) return secs;

    final parts = durStr.split(':');
    if (parts.length == 3) {
      final h = double.tryParse(parts[0]) ?? 0;
      final m = double.tryParse(parts[1]) ?? 0;
      final s = double.tryParse(parts[2]) ?? 0;
      return h * 3600 + m * 60 + s;
    }
    return null;
  }

  Future<String> _uploadVideo(XFile? video) async {
    if (video == null) {
      return textLocalize("tip_fail");
    }

    final file = File(video.path);

    // Check file size (1.5 GB limit)
    final fileSize = await file.length();
    if (fileSize > _GeneratePageState.videoSizeLimitBytes) {
      debugPrint(
        '[GenerateMedia] video too large: ${_GeneratePageState._formatBytes(fileSize)}',
      );
      return textLocalize("gen_video_too_large");
    }

    // Check duration (10 min limit)
    final durationSeconds = await _probeVideoDuration(file);
    if (durationSeconds != null &&
        durationSeconds > _GeneratePageState.videoDurationLimitSeconds) {
      debugPrint(
        '[GenerateMedia] video too long: ${durationSeconds.toStringAsFixed(1)}s',
      );
      return textLocalize("gen_video_too_long");
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

  Future<XFile> _cacheCapturedMedia(XFile file, {required bool isImage}) async {
    final cacheRoot = await DirFinder.cacheDir();
    if (cacheRoot.isEmpty) {
      return file;
    }

    final captureDir = path.join(
      cacheRoot,
      'generate_capture',
      isImage ? 'images' : 'videos',
    );
    final ensured = await DirSystem.ensureDir(captureDir);
    if (!ensured) {
      return file;
    }

    final originalExt = path.extension(file.path);
    final fallbackExt = isImage ? '.jpg' : '.mp4';
    final fileName =
        '${DateTime.now().millisecondsSinceEpoch}_${_GeneratePageState._rdg.nextInt(1000000).toString().padLeft(6, '0')}'
        '${originalExt.isNotEmpty ? originalExt : fallbackExt}';
    final cachedPath = path.join(captureDir, fileName);
    final copiedFile = await File(file.path).copy(cachedPath);
    await FileSystem.deleteFile(file.path);
    return XFile(copiedFile.path, name: path.basename(copiedFile.path));
  }

  Future<XFile> _resolveCapturedMedia(
    XFile file, {
    required bool isImage,
  }) async {
    final PermissionState ps = await PhotoManager.requestPermissionExtend();
    if (!ps.isAuth) {
      // 未授予相册权限时，把拍摄结果转存到缓存目录，后续直接从缓存路径加载。
      return _cacheCapturedMedia(file, isImage: isImage);
    }

    try {
      final AssetEntity newAsset = isImage
          ? await PhotoManager.editor.saveImageWithPath(
              file.path,
              title: file.name,
            )
          : await PhotoManager.editor.saveVideo(
              File(file.path),
              title: file.name,
            );
      final File? savedFile = await newAsset.originFile;
      if (savedFile != null) {
        await FileSystem.deleteFile(file.path);
        return XFile(savedFile.path, name: path.basename(savedFile.path));
      }
    } catch (_) {
      // 保存到相册失败时回退到缓存目录，避免拍照上传流程被权限阻断。
    }

    return _cacheCapturedMedia(file, isImage: isImage);
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
        var shouldShowVideoTaskTypeSheet = false;
        if (index == 0) {
          final XFile? file;
          if (isImage) {
            file = await _picker.pickImage(source: ImageSource.camera);
            if (file == null) {
              return;
            }
          } else {
            file = await _picker.pickVideo(
              source: ImageSource.camera,
              maxDuration: const Duration(minutes: 10),
            );
            if (file == null) {
              return;
            }
          }
          try {
            final savedXFile = await _resolveCapturedMedia(
              file,
              isImage: isImage,
            );
            final String msg = isImage
                ? await _uploadImages([savedXFile])
                : await _uploadVideo(savedXFile);
            if (msg.isNotEmpty && context.mounted) {
              showAppToast(context, msg);
            }
            shouldShowVideoTaskTypeSheet = !isImage && msg.isEmpty;
          } catch (_) {
            if (context.mounted) {
              showAppToast(context, textLocalize("tip_fail"));
            }
          }
        } else {
          final String msg = isImage
              ? await _uploadImages(await _picker.pickMultiImage())
              : await _uploadVideo(
                  await _picker.pickVideo(source: ImageSource.gallery),
                );
          if (msg.isNotEmpty && context.mounted) {
            showAppToast(context, msg);
          }
          shouldShowVideoTaskTypeSheet = !isImage && msg.isEmpty;
        }
        _refresh(() {});
        if (shouldShowVideoTaskTypeSheet && context.mounted) {
          await _showVideoTaskTypeSheet();
        }
      },
    ).show();
  }
}
