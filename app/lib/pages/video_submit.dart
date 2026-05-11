import 'dart:io';
import 'dart:math';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/main.dart' show pageIndexProvider, pendingSubmitTitleProvider;
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../configs/supabase_config.dart';

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
  bool _isUploading = false;
  bool _didPrefillTitle = false;
  double _uploadProgress = 0.0;
  int _uploadedBytes = 0;
  int _totalFileSize = 0;

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

  Future<void> _submit() async {
    final client = Supabase.instance.client;
    var user = client.auth.currentUser;
    if (user == null) {
      if (SupabaseConfig.isAdminMode) {
        if (mounted) {
          TDToast.showText(textLocalize('admin_mode_msg'), context: context);
        }
        return;
      }
      if (mounted) {
        TDToast.showText(textLocalize('not_logged_in'), context: context);
        await Navigator.pushNamed(context, '/login');
      }
      user = client.auth.currentUser;
      if (user == null) {
        if (mounted) {
          TDToast.showText(textLocalize('login_cancelled'), context: context);
        }
        return;
      } else {
        if (mounted) {
          TDToast.showText(
            textLocalize('login_success_upload'),
            context: context,
          );
        }
      }
    }

    setState(() {
      _isUploading = true;
    });

    try {
      final sceneId = _generateSceneId();

      final videoStoragePath = '${user.id}/$sceneId/raw/video.mp4';
      final file = File(widget.videoPath);
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
        if (nameController.text.isNotEmpty) 'display_name': nameController.text,
      });

      if (mounted) {
        TDToast.showText(textLocalize('gen_submit_success'), context: context);
        ref.read(pageIndexProvider.notifier).state = 0;
        // 统一收敛为一次导航，避免先 pop 再 push 导致返回手势期间路由栈抖动。
        Navigator.of(
          context,
        ).pushNamedAndRemoveUntil('/tasks', (route) => route.isFirst);
      }
    } catch (e) {
      if (mounted) {
        debugPrint(e.toString());
        TDToast.showText(
          '${textLocalize('gen_submit_fail')}: $e',
          context: context,
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
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
  void dispose() {
    nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_didPrefillTitle) {
      _didPrefillTitle = true;
      final pending = ref.read(pendingSubmitTitleProvider);
      if (pending != null && pending.isNotEmpty) {
        nameController.text = pending;
        ref.read(pendingSubmitTitleProvider.notifier).state = null;
      }
    }
    return Scaffold(
      appBar: AppBar(title: Text(textLocalize('video_submit_title'))),
      body: Stack(
        children: [
          SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  textLocalize('video_submit_name'),
                  style: const TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: nameController,
                  decoration: InputDecoration(
                    border: const OutlineInputBorder(),
                    hintText: textLocalize('video_submit_name_hint'),
                  ),
                ),
                const SizedBox(height: 24),
                Text(
                  textLocalize('video_submit_thumbnail'),
                  style: const TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 8),
                Center(
                  child: Image.file(
                    File(widget.thumbnailPath),
                    width: 180,
                    height: 120,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(height: 48),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _isUploading ? null : _submit,
                    child: Text(textLocalize('video_submit_btn')),
                  ),
                ),
              ],
            ),
          ),
          if (_isUploading)
            Container(
              color: Colors.black45,
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 40),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '${textLocalize('gen_uploading')} ${(_uploadProgress * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: _uploadProgress,
                          minHeight: 6,
                          backgroundColor: Colors.white.withAlpha(40),
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            Color(0xFF7AA2FF),
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        '${_formatBytes(_uploadedBytes)} / ${_formatBytes(_totalFileSize)}',
                        style: TextStyle(
                          color: Colors.white.withAlpha(200),
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
