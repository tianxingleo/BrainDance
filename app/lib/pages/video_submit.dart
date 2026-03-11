import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
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
      if (mounted) {
        TDToast.showText('未登录，即将跳转登录页面...', context: context);
        await Navigator.pushNamed(context, '/login');
      }
      user = client.auth.currentUser;
      if (user == null) {
        if (mounted) TDToast.showText('登录已取消或未完成', context: context);
        return;
      } else {
        if (mounted) TDToast.showText('登录成功，请再次点击提交以开始上传', context: context);
        return;
      }
    }

    setState(() {
      _isUploading = true;
    });

    try {
      final sceneId = _generateSceneId();

      // 上传视频
      final videoStoragePath = '${user.id}/$sceneId/raw/video.mp4';
      await client.storage
          .from('braindance-assets')
          .upload(
            videoStoragePath,
            File(widget.videoPath),
            fileOptions: const FileOptions(
              contentType: 'video/mp4',
              upsert: true,
            ),
          );

      // 创建任务
      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'status': 'pending',
        if (nameController.text.isNotEmpty) 'display_name': nameController.text,
        'task_params': {'mapper_type': 'da3'},
      });

      if (mounted) {
        TDToast.showText('提交成功，任务已创建', context: context);
        // 回到 Recall (也就是主页列表) 页面查看生成的模型状态
        ref.read(pageIndexProvider.notifier).state = 0;
        Navigator.pop(context); // 退出 submitting 页面
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
              child: const Center(child: CircularProgressIndicator()),
            ),
        ],
      ),
    );
  }
}
