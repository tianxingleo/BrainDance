import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'dart:io';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'dart:math';

class DateFormat {
  static String format(int number, int length) {
    return number.toString().padLeft(length, '0');
  }
}

class VideoSubmitPage extends StatefulWidget {
  final String videoPath;
  final String thumbnailPath;

  const VideoSubmitPage({
    super.key,
    required this.videoPath,
    required this.thumbnailPath,
  });

  @override
  State<VideoSubmitPage> createState() => _VideoSubmitPageState();
}

class _VideoSubmitPageState extends State<VideoSubmitPage> {
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
    final user = client.auth.currentUser;
    if (user == null) {
      TDToast.showText('未登录，请先登录', context: context);
      return;
    }

    setState(() {
      _isUploading = true;
    });

    try {
      final sceneId = _generateSceneId();
      
      // 上传视频
      final videoStoragePath = '${user.id}/$sceneId/raw/video.mp4';
      await client.storage.from('braindance-assets').upload(
        videoStoragePath,
        File(widget.videoPath),
      );

      // 上传封面
      final thumbnailStoragePath = '${user.id}/$sceneId/raw/thumbnail.jpg';
      await client.storage.from('braindance-assets').upload(
        thumbnailStoragePath,
        File(widget.thumbnailPath),
      );

      // 创建任务
      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'status': 'pending',
        'description': nameController.text,
      });

      if (mounted) {
        TDToast.showText('提交成功，任务已创建', context: context);
        Navigator.pop(context); // 退回拍摄页面，或可根据需求退回主页 Navigator.of(context).popUntil((route) => route.isFirst);
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
          Padding(
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
                const Spacer(),
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
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            ),
        ],
      ),
    );
  }
}
