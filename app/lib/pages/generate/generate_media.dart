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
      if (await image.length() ~/ 1024 > _GeneratePageState.sizeLimit) {
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
        var shouldShowVideoTaskTypeSheet = false;
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
            final File? savedFile = await newAsset.originFile;
            if (savedFile == null) {
              throw Exception('Failed to save captured asset.');
            }
            final savedXFile = XFile(savedFile.path);
            final String msg = isImage
                ? await _uploadImages([savedXFile])
                : await _uploadVideo(savedXFile);
            if (msg.isNotEmpty && context.mounted) {
              TDToast.showText(msg, context: context);
            }
            shouldShowVideoTaskTypeSheet = !isImage && msg.isEmpty;
          } catch (_) {
            if (context.mounted) {
              TDToast.showText(textLocalize("tip_fail"), context: context);
            }
          }
        } else {
          final String msg = isImage
              ? await _uploadImages(await _picker.pickMultiImage())
              : await _uploadVideo(
                  await _picker.pickVideo(source: ImageSource.gallery),
                );
          if (msg.isNotEmpty && context.mounted) {
            TDToast.showText(msg, context: context);
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
