import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'dart:io';

class VideoSubmitPage extends StatelessWidget {
  final String videoPath;
  final String thumbnailPath;

  const VideoSubmitPage({
    super.key,
    required this.videoPath,
    required this.thumbnailPath,
  });

  @override
  Widget build(BuildContext context) {
    final TextEditingController nameController = TextEditingController();
    return Scaffold(
      appBar: AppBar(title: const Text('视频信息提交')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('视频名称', style: TextStyle(fontSize: 16)),
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
                // 这里假设 thumbnailPath 是本地文件路径
                File(thumbnailPath),
                width: 180,
                height: 120,
                fit: BoxFit.cover,
              ),
            ),
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  // 提交逻辑
                  TDToast.showText('已提交', context: context);
                  Navigator.pop(context);
                },
                child: const Text('提交'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
