import 'package:flutter/material.dart';
import 'package:dio/dio.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../configs/supabase_config.dart';
import 'dart:io';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'dart:math';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:braindance/main.dart' show pageIndexProvider;

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
  double _uploadProgress = 0.0;

  static final Random _rdg = Random();

  static String _generateSceneId() {
    DateTime time = DateTime.now();
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
          TDToast.showText('当前为管理员浏览模式，未绑定用户，暂不支持上传任务。', context: context);
        }
        return;
      }
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
      final file = File(widget.videoPath);
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
            'apikey': SupabaseConfig.apiKey,
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

      // 创建任务
      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'display_name': nameController.text.trim().isEmpty ? null : nameController.text.trim(),
        'task_type': 'video_dual_chain',
        'task_params': {
          'slow_pipeline': 'video_3dgs',
          'sam3d_vram_threshold_gb': 25,
          'best_frame_sample_count': 8,
        },
        'status': 'pending',
        if (nameController.text.isNotEmpty) 'display_name': nameController.text,
        'task_params': {'mapper_type': 'da3'},
      });

      if (mounted) {
        TDToast.showText('提交成功，任务已创建', context: context);
        // 回到 Recall (也就是主页列表) 页面查看生成的模型状态
        ref.read(pageIndexProvider.notifier).state = 0;
        final nav = Navigator.of(context);
        nav.popUntil((route) => route.isFirst); // 退出到主页
        nav.pushNamed('/tasks'); // 自动打开任务列表
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

  @override
  void dispose() {
    nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('视频信息提交')),
      body: Stack(
        children: [
          SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('视频名称 (可选)', style: TextStyle(fontSize: 16)),
                const SizedBox(height: 8),
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    hintText: '请输入视频名称',
                  ),
                ),
                const SizedBox(height: 24),
                const Text('视频缩略图', style: TextStyle(fontSize: 16)),
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
                    child: const Text('提交并上传'),
                  ),
                ),
              ],
            ),
          ),
          if (_isUploading)
            Container(
              color: Colors.black45,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 16),
                    Text(
                      '正在上传... ${(_uploadProgress * 100).toStringAsFixed(1)}%',
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
    );
  }
}
