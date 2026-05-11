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

  /// Reads video duration in seconds from an MP4/MOV file header.
  /// Returns null if the file cannot be parsed.
  static double? _readVideoDurationSeconds(File file) {
    try {
      final raf = file.openSync();
      try {
        final fileSize = raf.lengthSync();
        int offset = 0;
        while (offset + 8 <= fileSize) {
          raf.setPositionSync(offset);
          final atomSize =
              (raf.readByteSync() << 24) |
              (raf.readByteSync() << 16) |
              (raf.readByteSync() << 8) |
              raf.readByteSync();
          final type = String.fromCharCodes(raf.readSync(4));
          if (atomSize < 8 || offset + atomSize > fileSize) break;

          if (type == 'moov') {
            return _parseMoovMvhd(raf, offset + 8, atomSize - 8);
          }
          offset += atomSize;
        }

        // If moov wasn't found in the forward scan, try reading
        // backward from the end of the file (streaming-optimized MP4s
        // often place moov at the tail).
        final scanBuf = Uint8List(32);
        int tailOffset = fileSize - scanBuf.length;
        while (tailOffset >= 0) {
          raf.setPositionSync(tailOffset);
          raf.readSync(scanBuf.length);
          for (int i = scanBuf.length - 8; i >= 0; i--) {
            final t = String.fromCharCodes(scanBuf.sublist(i + 4, i + 8));
            if (t == 'moov') {
              final moovSize =
                  (scanBuf[i] << 24) |
                  (scanBuf[i + 1] << 16) |
                  (scanBuf[i + 2] << 8) |
                  scanBuf[i + 3];
              final moovStart = tailOffset + i;
              if (moovSize >= 8 && moovStart + moovSize <= fileSize) {
                return _parseMoovMvhd(raf, moovStart + 8, moovSize - 8);
              }
            }
          }
          tailOffset -= scanBuf.length ~/ 2;
        }
      } finally {
        raf.closeSync();
      }
    } catch (e) {
      debugPrint('[GenerateMedia] mp4 parse error: $e');
    }
    return null;
  }

  static double? _parseMoovMvhd(RandomAccessFile raf, int start, int length) {
    final end = start + length;
    int offset = start;
    while (offset + 16 <= end) {
      raf.setPositionSync(offset);
      final size =
          (raf.readByteSync() << 24) |
          (raf.readByteSync() << 16) |
          (raf.readByteSync() << 8) |
          raf.readByteSync();
      final type = String.fromCharCodes(raf.readSync(4));
      if (size < 8) break;

      if (type == 'mvhd') {
        raf.setPositionSync(offset + 8);
        final version = raf.readByteSync();
        raf.readSync(3);

        int timescale;
        int duration;
        if (version == 1) {
          raf.readSync(16);
          final tsB = raf.readSync(4);
          timescale = (tsB[0] << 24) | (tsB[1] << 16) | (tsB[2] << 8) | tsB[3];
          final dHi =
              (raf.readByteSync() << 24) |
              (raf.readByteSync() << 16) |
              (raf.readByteSync() << 8) |
              raf.readByteSync();
          final dLo =
              (raf.readByteSync() << 24) |
              (raf.readByteSync() << 16) |
              (raf.readByteSync() << 8) |
              raf.readByteSync();
          duration = (dHi << 32) | dLo;
        } else {
          raf.readSync(8);
          final tsB = raf.readSync(4);
          timescale = (tsB[0] << 24) | (tsB[1] << 16) | (tsB[2] << 8) | tsB[3];
          final dB = raf.readSync(4);
          duration = (dB[0] << 24) | (dB[1] << 16) | (dB[2] << 8) | dB[3];
        }
        if (timescale > 0) {
          return duration / timescale;
        }
        break;
      }
      offset += size;
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
    final durationSeconds = _readVideoDurationSeconds(file);
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
