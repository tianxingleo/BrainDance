import 'dart:io';
import 'dart:math';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:braindance/widgets/app_toast.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:ffmpeg_kit_extended_flutter/ffmpeg_kit_extended_flutter.dart';

import '../configs/supabase_config.dart';
import '../services/video_preprocessor.dart';

class DateFormat {
  static String format(int number, int length) {
    return number.toString().padLeft(length, '0');
  }
}

class VideoSubmitPage extends ConsumerStatefulWidget {
  final String videoPath;
  final String thumbnailPath;

  const VideoSubmitPage({
    super.key,
    required this.videoPath,
    required this.thumbnailPath,
  });

  @override
  ConsumerState<VideoSubmitPage> createState() => _VideoSubmitPageState();
}

class _VideoSubmitPageState extends ConsumerState<VideoSubmitPage> {
  final TextEditingController nameController = TextEditingController();
  final FocusNode _nameFocusNode = FocusNode();
  bool _nameFocused = false;
  bool _isUploading = false;
  bool _isPreprocessing = false;
  double _uploadProgress = 0.0;
  double _preprocessProgress = 0.0;
  int _uploadedBytes = 0;
  int _totalFileSize = 0;
  CancelToken? _cancelToken;
  bool _dialogShowing = false;
  bool _shouldClosePage = false;
  bool _deleteVideo = true;
  bool _preprocessCancelled = false;
  VideoPreprocessResult? _preprocessResult;
  String? _uploadedStoragePath;

  static final Random _rdg = Random();

  static String _generateSceneId() {
    final time = DateTime.now();
    return 'scene_'
        '${DateFormat.format(time.year, 4)}'
        '${DateFormat.format(time.month, 2)}'
        '${DateFormat.format(time.day, 2)}'
        '_'
        '${DateFormat.format(_rdg.nextInt(1000000), 6)}';
  }

  static const _preprocessConfig = VideoPreprocessConfig(
    targetFps: 5,
    maxHeight: 1080,
    videoBitrate: '2M',
    audioBitrate: '96k',
    enableFastStart: true,
  );

  String _progressHint() {
    final c = _preprocessConfig;
    return textLocalize('video_preprocess_progress')
        .replaceFirst('%s', c.videoBitrate)
        .replaceFirst('%d', c.targetFps.toString())
        .replaceFirst('%d', c.maxHeight.toString());
  }

  Future<bool> _showCancelUploadDialog() async {
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        final isDark = Theme.of(ctx).brightness == Brightness.dark;
        return AlertDialog(
          title: Text(
            textLocalize('video_upload_cancel_title'),
            style: TextStyle(
              color: isDark ? Colors.white : Colors.black87,
              fontWeight: FontWeight.w600,
            ),
          ),
          content: Text(
            textLocalize('video_upload_cancel_message'),
            style: TextStyle(
              color: isDark ? Colors.white70 : Colors.black54,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(false),
              child: Text(textLocalize('video_upload_cancel_continue')),
            ),
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(true),
              style: TextButton.styleFrom(
                foregroundColor: Colors.redAccent,
              ),
              child: Text(textLocalize('video_upload_cancel_confirm')),
            ),
          ],
        );
      },
    );
    return result ?? false;
  }

  Future<bool> _showCancelPreprocessDialog() async {
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        final isDark = Theme.of(ctx).brightness == Brightness.dark;
        return StatefulBuilder(
          builder: (builderContext, setDialogState) {
            return AlertDialog(
              title: Text(
                textLocalize('video_preprocess_cancel_title'),
                style: TextStyle(
                  color: isDark ? Colors.white : Colors.black87,
                  fontWeight: FontWeight.w600,
                ),
              ),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    textLocalize('video_preprocess_cancel_message'),
                    style: TextStyle(
                      color: isDark ? Colors.white70 : Colors.black54,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      SizedBox(
                        width: 24,
                        height: 24,
                        child: Checkbox(
                          value: _deleteVideo,
                          onChanged: (v) {
                            _deleteVideo = v ?? true;
                            setDialogState(() {});
                          },
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        textLocalize('video_exit_delete_checkbox'),
                        style: TextStyle(
                          color: isDark ? Colors.white70 : Colors.black87,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(false),
                  child: Text(textLocalize('video_preprocess_cancel_continue')),
                ),
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(true),
                  style: TextButton.styleFrom(
                    foregroundColor: Colors.redAccent,
                  ),
                  child: Text(textLocalize('video_preprocess_cancel_confirm')),
                ),
              ],
            );
          },
        );
      },
    );
    return result ?? false;
  }

  Future<bool> _showExitConfirmDialog() async {
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        final isDark = Theme.of(ctx).brightness == Brightness.dark;
        return StatefulBuilder(
          builder: (builderContext, setDialogState) {
            return AlertDialog(
              title: Text(
                textLocalize('video_exit_title'),
                style: TextStyle(
                  color: isDark ? Colors.white : Colors.black87,
                  fontWeight: FontWeight.w600,
                ),
              ),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    textLocalize('video_exit_message'),
                    style: TextStyle(
                      color: isDark ? Colors.white70 : Colors.black54,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      SizedBox(
                        width: 24,
                        height: 24,
                        child: Checkbox(
                          value: _deleteVideo,
                          onChanged: (v) {
                            _deleteVideo = v ?? true;
                            setDialogState(() {});
                          },
                        ),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        textLocalize('video_exit_delete_checkbox'),
                        style: TextStyle(
                          color: isDark ? Colors.white70 : Colors.black87,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(false),
                  child: Text(textLocalize('video_exit_cancel')),
                ),
                TextButton(
                  onPressed: () {
                    debugPrint('[VideoSubmit] exit confirmed, deleteVideo=$_deleteVideo');
                    Navigator.of(ctx).pop(true);
                  },
                  style: TextButton.styleFrom(
                    foregroundColor: Colors.redAccent,
                  ),
                  child: Text(textLocalize('video_exit_confirm')),
                ),
              ],
            );
          },
        );
      },
    );
    return result ?? false;
  }

  void _deleteRecordedVideo() {
    if (!_deleteVideo) {
      debugPrint('[VideoSubmit] keeping video files (deleteVideo=false)');
      return;
    }
    debugPrint('[VideoSubmit] deleting video: ${widget.videoPath}');
    try {
      final videoFile = File(widget.videoPath);
      if (videoFile.existsSync()) {
        videoFile.deleteSync();
        debugPrint('[VideoSubmit] deleted video file');
      } else {
        debugPrint('[VideoSubmit] video file not found at path');
      }
      final thumbFile = File(widget.thumbnailPath);
      if (thumbFile.existsSync()) {
        thumbFile.deleteSync();
        debugPrint('[VideoSubmit] deleted thumbnail file');
      }
    } catch (e) {
      debugPrint('[VideoSubmit] file deletion error: $e');
    }
  }

  void _cleanupPreprocess() {
    if (_preprocessResult == null) return;
    try {
      final f = _preprocessResult!.outputFile;
      if (f.existsSync()) {
        f.deleteSync();
        debugPrint('[VideoSubmit] deleted preprocessed temp file');
      }
    } catch (e) {
      debugPrint('[VideoSubmit] error deleting preprocessed file: $e');
    }
    _preprocessResult = null;
  }

  Future<void> _deleteUploadedContent() async {
    if (_uploadedStoragePath == null) return;
    try {
      final client = Supabase.instance.client;
      await client.storage
          .from('braindance-assets')
          .remove([_uploadedStoragePath!]);
      debugPrint('[VideoSubmit] deleted uploaded content: $_uploadedStoragePath');
    } catch (e) {
      debugPrint('[VideoSubmit] error deleting uploaded content: $e');
    }
  }

  Future<void> _submit() async {
    final client = Supabase.instance.client;
    var user = client.auth.currentUser;
    if (user == null) {
      if (SupabaseConfig.isAdminMode) {
        if (mounted) {
          showAppToast(context, textLocalize('admin_mode_msg'));
        }
        return;
      }
      if (mounted) {
        showAppToast(context, textLocalize('not_logged_in'));
        await Navigator.pushNamed(context, '/login');
      }
      user = client.auth.currentUser;
      if (user == null) {
        if (mounted) {
          showAppToast(context, textLocalize('login_cancelled'));
        }
        return;
      } else {
        if (mounted) {
          showAppToast(context, textLocalize('login_success_upload'));
        }
      }
    }

    _cancelToken = CancelToken();
    _preprocessCancelled = false;

    setState(() {
      _isUploading = true;
      _isPreprocessing = true;
      _preprocessProgress = 0.0;
    });

    // --- Phase 1: Preprocessing ---
    VideoPreprocessResult? preprocessResult;
    try {
      preprocessResult = await VideoPreprocessor.preprocess(
        File(widget.videoPath),
        config: _preprocessConfig,
        onProgress: (progress) {
          if (mounted) {
            setState(() {
              _preprocessProgress = progress;
            });
          }
        },
      );
    } catch (e) {
      if (_preprocessCancelled) {
        debugPrint('[VideoSubmit] preprocessing cancelled by user');
        return;
      }
      debugPrint('[VideoSubmit] preprocessing error: $e');
      if (mounted) {
        showAppToast(context, textLocalize('video_preprocess_fail'));
        setState(() {
          _isUploading = false;
          _isPreprocessing = false;
        });
      }
      return;
    }

    if (!mounted) return;

    setState(() {
      _isPreprocessing = false;
      _preprocessResult = preprocessResult;
    });

    // --- Phase 2: Upload ---
    try {
      final sceneId = _generateSceneId();

      final videoStoragePath = '${user.id}/$sceneId/raw/video.mp4';
      _uploadedStoragePath = videoStoragePath;
      final file = preprocessResult.outputFile;
      final fileSize = await file.length();
      final url =
          '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$videoStoragePath';
      final dio = Dio();

      setState(() {
        _totalFileSize = fileSize;
        _uploadedBytes = 0;
      });

      await dio.post(
        url,
        data: file.openRead(),
        options: Options(
          headers: {
            'Authorization':
                'Bearer ${client.auth.currentSession?.accessToken}',
            'apikey': SupabaseConfig.apiKey,
            'Content-Type': 'video/mp4',
            'Content-Length': fileSize.toString(),
          },
        ),
        cancelToken: _cancelToken,
        onSendProgress: (count, total) {
          if (mounted) {
            setState(() {
              _uploadedBytes = count;
              _uploadProgress = count / fileSize;
            });
          }
        },
      );

      await client.from('processing_tasks').insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'display_name': nameController.text.trim().isEmpty
            ? null
            : nameController.text.trim(),
        'task_type': 'video_dual_chain',
        'task_params': {
          'slow_pipeline': 'video_3dgs',
          'sam3d_vram_threshold_gb': 25,
          'best_frame_sample_count': 8,
          'mapper_type': 'da3',
        },
        'status': 'pending',
      });

      _cleanupPreprocess();

      if (mounted) {
        showAppToast(context, textLocalize('gen_submit_success'));
        if (_dialogShowing) {
          _shouldClosePage = true;
          Navigator.of(context).pop();
        } else {
          Navigator.of(context).pop();
        }
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.cancel) {
        debugPrint('[VideoSubmit] upload cancelled by user');
        _deleteUploadedContent();
        _cleanupPreprocess();
        return;
      }
      debugPrint('[VideoSubmit] error: $e');
      if (mounted) {
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
      _cleanupPreprocess();
    } catch (e) {
      debugPrint('[VideoSubmit] error: $e');
      if (mounted) {
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
      _cleanupPreprocess();
    } finally {
      _cancelToken = null;
      _uploadedStoragePath = null;
      if (mounted) {
        setState(() {
          _isUploading = false;
          _isPreprocessing = false;
        });
      }
    }
  }

  static String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    if (bytes < 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
  }

  @override
  void initState() {
    super.initState();
    _nameFocusNode.addListener(() {
      if (!mounted) return;
      setState(() {
        _nameFocused = _nameFocusNode.hasFocus;
      });
    });
  }

  @override
  void dispose() {
    _nameFocusNode.dispose();
    nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;
    final inputBg = isDark ? const Color(0xFF23232A) : const Color(0xFFF6F8FC);
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        final navigator = Navigator.of(context);
        if (_isPreprocessing) {
          debugPrint('[VideoSubmit] back intercepted while preprocessing, showing cancel dialog');
          _dialogShowing = true;
          final shouldCancel = await _showCancelPreprocessDialog();
          if (!mounted) return;
          _dialogShowing = false;
          if (shouldCancel) {
            debugPrint('[VideoSubmit] preprocessing cancelled by user via back button');
            _preprocessCancelled = true;
            FFmpegKitExtended.cancelAllSessions();
            setState(() {
              _isUploading = false;
              _isPreprocessing = false;
            });
            _deleteRecordedVideo();
            navigator.pop();
          } else if (_shouldClosePage) {
            _shouldClosePage = false;
            navigator.pop();
          }
        } else if (_isUploading) {
          debugPrint('[VideoSubmit] back intercepted while uploading, showing upload cancel dialog');
          _dialogShowing = true;
          final shouldCancel = await _showCancelUploadDialog();
          if (!mounted) return;
          _dialogShowing = false;
          if (shouldCancel) {
            debugPrint('[VideoSubmit] upload cancelled by user');
            _cancelToken?.cancel();
            _deleteUploadedContent();
            _cleanupPreprocess();
            setState(() {
              _isUploading = false;
            });
            navigator.pop();
          } else if (_shouldClosePage) {
            _shouldClosePage = false;
            navigator.pop();
          }
        } else {
          debugPrint('[VideoSubmit] back intercepted before upload, showing exit confirm dialog');
          final shouldExit = await _showExitConfirmDialog();
          if (!mounted) return;
          if (shouldExit) {
            _deleteRecordedVideo();
            navigator.pop();
          }
        }
      },
      child: Scaffold(
        backgroundColor: Colors.transparent,
        extendBody: true,
        resizeToAvoidBottomInset: false,
        body: BDPageBackdrop(
          child: SafeArea(
            child: Column(
              children: [
                Expanded(
                  child: Stack(
                    children: [
                      SingleChildScrollView(
                        padding: const EdgeInsets.fromLTRB(24, 16, 24, 16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const SizedBox(height: 52),
                            Text(
                              textLocalize('video_submit_title'),
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.w700,
                                color: textColor,
                              ),
                            ),
                            if (_isUploading) ...[
                              const SizedBox(height: 14),
                              ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: _isPreprocessing
                                      ? _preprocessProgress
                                      : _uploadProgress,
                                  minHeight: 5,
                                  backgroundColor: isDark
                                      ? Colors.white.withAlpha(20)
                                      : Colors.black.withAlpha(15),
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    isDark
                                        ? const Color(0xFFFFB74D)
                                        : const Color(0xFFF57C00),
                                  ),
                                ),
                              ),
                              const SizedBox(height: 6),
                              Row(
                                children: [
                                  Icon(
                                    _isPreprocessing
                                        ? Icons.video_settings_outlined
                                        : Icons.cloud_upload_outlined,
                                    size: 12,
                                    color: hintColor,
                                  ),
                                  const SizedBox(width: 4),
                                  Text(
                                    _isPreprocessing
                                        ? '${(_preprocessProgress * 100).toStringAsFixed(0)}%'
                                        : '${(_uploadProgress * 100).toStringAsFixed(0)}%',
                                    style: TextStyle(fontSize: 12, color: hintColor),
                                  ),
                                  const Spacer(),
                                  Text(
                                    _isPreprocessing
                                        ? _progressHint()
                                        : '${_formatBytes(_uploadedBytes)} / ${_formatBytes(_totalFileSize)}',
                                    style: TextStyle(fontSize: 11, color: hintColor),
                                  ),
                                ],
                              ),
                            ],
                            const SizedBox(height: 20),
                            Text(
                              textLocalize('video_submit_name'),
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                                color: textColor,
                              ),
                            ),
                            const SizedBox(height: 10),
                            _NameTextField(
                              controller: nameController,
                              focusNode: _nameFocusNode,
                              focused: _nameFocused,
                              textColor: textColor,
                              hintColor: hintColor,
                              inputBg: inputBg,
                              isDark: isDark,
                            ),
                            const SizedBox(height: 24),
                            Text(
                              textLocalize('video_submit_thumbnail'),
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                                color: textColor,
                              ),
                            ),
                            const SizedBox(height: 10),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(16),
                              child: Image.file(
                                File(widget.thumbnailPath),
                                width: 200,
                                height: 128,
                                fit: BoxFit.cover,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Positioned(
                        left: 16,
                        top: 4,
                        child: GestureDetector(
                          onTap: () {
                            if (_isPreprocessing) {
                              _dialogShowing = true;
                              _showCancelPreprocessDialog().then((shouldCancel) {
                                if (!mounted) return;
                                _dialogShowing = false;
                                if (shouldCancel) {
                                  _preprocessCancelled = true;
                                  FFmpegKitExtended.cancelAllSessions();
                                  setState(() {
                                    _isUploading = false;
                                    _isPreprocessing = false;
                                  });
                                  _deleteRecordedVideo();
                                  Navigator.of(context).pop();
                                }
                              });
                            } else if (_isUploading) {
                              _dialogShowing = true;
                              _showCancelUploadDialog().then((shouldCancel) {
                                if (!mounted) return;
                                _dialogShowing = false;
                                if (shouldCancel) {
                                  _cancelToken?.cancel();
                                  _deleteUploadedContent();
                                  _cleanupPreprocess();
                                  setState(() {
                                    _isUploading = false;
                                  });
                                  Navigator.of(context).pop();
                                }
                              });
                            } else {
                              _showExitConfirmDialog().then((shouldExit) {
                                if (!mounted) return;
                                if (shouldExit) {
                                  _deleteRecordedVideo();
                                  Navigator.of(context).pop();
                                }
                              });
                            }
                          },
                          child: Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              color: isDark
                                  ? BDDesign.colorInkBlack.withAlpha(216)
                                  : Colors.white.withAlpha(230),
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: isDark
                                    ? Colors.white.withAlpha(28)
                                    : Colors.black.withAlpha(12),
                              ),
                              boxShadow: [BDDesign.shadowElevated],
                            ),
                            child: Icon(
                              Icons.close_rounded,
                              color: isDark
                                  ? BDDesign.colorAshGray
                                  : BDDesign.colorInkBlack,
                              size: 22,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(24, 8, 24, 0),
                  child: TDButton(
                    onTap: _isUploading ? () {} : _submit,
                    style: TDButtonStyle(
                      backgroundColor: _isUploading
                          ? (isDark ? const Color(0xFF1A1A1E) : const Color(0xFFD8D8DF))
                          : (isDark ? const Color(0xFF2A2A2E) : BDDesign.colorMutedBlue),
                      textColor: _isUploading
                          ? (isDark ? Colors.white.withValues(alpha: 0.28) : Colors.white.withValues(alpha: 0.45))
                          : Colors.white,
                      radius: BorderRadius.circular(18),
                    ),
                    type: TDButtonType.fill,
                    shape: TDButtonShape.rectangle,
                    theme: TDButtonTheme.primary,
                    size: TDButtonSize.large,
                    width: double.infinity,
                    text: _isPreprocessing
                        ? '${textLocalize('video_preprocess_cancel_title')}...'
                        : _isUploading
                            ? '${textLocalize('gen_uploading')}...'
                            : textLocalize('video_submit_btn'),
                  ),
                ),
                const SizedBox(height: 32.0),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _NameTextField extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool focused;
  final Color textColor;
  final Color hintColor;
  final Color inputBg;
  final bool isDark;

  const _NameTextField({
    required this.controller,
    required this.focusNode,
    required this.focused,
    required this.textColor,
    required this.hintColor,
    required this.inputBg,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeOutCubic,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: focused
              ? BDDesign.colorMutedBlue
              : (isDark
                  ? Colors.white.withValues(alpha: 0.08)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.10)),
          width: focused ? 1.5 : 1,
        ),
      ),
      child: TextField(
        controller: controller,
        focusNode: focusNode,
        style: TextStyle(color: textColor, fontSize: 15),
        decoration: InputDecoration(
          hintText: textLocalize('video_submit_name_hint'),
          hintStyle: TextStyle(color: hintColor, fontSize: 15),
          filled: true,
          fillColor: inputBg,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 18,
            vertical: 16,
          ),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide.none,
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide.none,
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide.none,
          ),
        ),
      ),
    );
  }
}
