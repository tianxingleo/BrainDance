import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../../configs/app_config.dart';
import '../../services/download_event_bus.dart';

/// 模型下载状态
enum ModelDownloadStatus {
  /// 未下载
  notDownloaded,

  /// 部分下载（有 .tmp 文件）
  partial,

  /// 已完成下载
  downloaded,
}

/// 独立的下载状态徽章组件，可嵌入任意卡片
/// [modelUrl] 为模型的完整 HTTP URL
class ModelDownloadBadge extends StatefulWidget {
  final String modelUrl;
  final bool isDark;
  final TDThemeData theme;

  const ModelDownloadBadge({
    super.key,
    required this.modelUrl,
    required this.isDark,
    required this.theme,
  });

  @override
  State<ModelDownloadBadge> createState() => _ModelDownloadBadgeState();
}

class _ModelDownloadBadgeState extends State<ModelDownloadBadge> {
  ModelDownloadStatus _status = ModelDownloadStatus.notDownloaded;
  double _progress = 0.0;
  StreamSubscription<ModelDownloadEvent>? _sub;

  @override
  void initState() {
    super.initState();
    _checkDownloadStatus();
    _sub = downloadEventBus.stream.listen((event) {
      if (event.url == widget.modelUrl) {
        if (mounted) {
          setState(() {
            if (event.isDeleted) {
              _status = ModelDownloadStatus.notDownloaded;
              _progress = 0.0;
            } else if (event.isComplete) {
              _status = ModelDownloadStatus.downloaded;
            } else if (event.isCancelled) {
              _status = ModelDownloadStatus.partial;
              _progress = event.progress;
            } else {
              _status = ModelDownloadStatus.partial;
              _progress = event.progress;
            }
          });
        }
      }
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  Future<void> _checkDownloadStatus() async {
    final modelUrl = widget.modelUrl;
    if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
      if (mounted) setState(() => _status = ModelDownloadStatus.downloaded);
      return;
    }
    try {
      final encodedUrl = Uri.encodeFull(Uri.decodeFull(modelUrl));
      final uri = Uri.parse(encodedUrl);
      final sanitizedFileName = uri.path
          .replaceAll('/', '_')
          .replaceAll('\\', '_');
      final dir = await getApplicationDocumentsDirectory();
      final localFile = File('${dir.path}/$sanitizedFileName');
      final tmpFile = File('${dir.path}/$sanitizedFileName.tmp');
      final metaFile = File('${dir.path}/$sanitizedFileName.meta');

      if (await localFile.exists()) {
        if (mounted) setState(() => _status = ModelDownloadStatus.downloaded);
      } else if (await tmpFile.exists()) {
        final downloadedBytes = await tmpFile.length();
        int totalBytes = 0;
        if (await metaFile.exists()) {
          totalBytes = int.tryParse(await metaFile.readAsString()) ?? 0;
        }
        if (mounted) {
          setState(() {
            _status = ModelDownloadStatus.partial;
            _progress = totalBytes > 0 ? downloadedBytes / totalBytes : 0.0;
          });
        }
      } else {
        if (mounted) {
          setState(() => _status = ModelDownloadStatus.notDownloaded);
        }
      }
    } catch (_) {}
  }

  /// 供外部调用刷新状态
  void refresh() => _checkDownloadStatus();

  @override
  Widget build(BuildContext context) {
    final isDark = widget.isDark;
    final theme = widget.theme;
    final lightColor = const Color.fromARGB(255, 89, 176, 182);
    final darkColor = const Color.fromARGB(255, 101, 242, 252);

    switch (_status) {
      case ModelDownloadStatus.notDownloaded:
        return Row(
          children: [
            Icon(
              Icons.cloud_download_outlined,
              size: 14,
              color: isDark ? const Color(0xFF888888) : const Color(0xFF999999),
            ),
            const SizedBox(width: 4),
            TDText(
              textLocalize('recall_model_not_downloaded'),
              font: theme.fontBodyExtraSmall,
              textColor: isDark
                  ? const Color(0xFF888888)
                  : const Color(0xFF999999),
            ),
          ],
        );

      case ModelDownloadStatus.partial:
        final percent = (_progress * 100).toStringAsFixed(0);
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.downloading_rounded,
                  size: 14,
                  color: isDark ? darkColor : lightColor,
                ),
                const SizedBox(width: 4),
                TDText(
                  '${textLocalize('recall_model_downloading')} $percent%',
                  font: theme.fontBodyExtraSmall,
                  textColor: isDark ? darkColor : lightColor,
                ),
              ],
            ),
            const SizedBox(height: 4),
            ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(
                value: _progress,
                minHeight: 4,
                backgroundColor: isDark
                    ? Colors.white.withAlpha(20)
                    : Colors.black.withAlpha(15),
                valueColor: AlwaysStoppedAnimation<Color>(
                  isDark ? darkColor : lightColor,
                ),
              ),
            ),
          ],
        );

      case ModelDownloadStatus.downloaded:
        return Row(
          children: [
            Icon(
              Icons.check_circle_outline,
              size: 14,
              color: isDark ? const Color(0xFF666666) : const Color(0xFFBBBBBB),
            ),
            const SizedBox(width: 4),
            TDText(
              textLocalize('recall_model_available'),
              font: theme.fontBodyExtraSmall,
              textColor: isDark
                  ? const Color(0xFF666666)
                  : const Color(0xFFBBBBBB),
            ),
          ],
        );
    }
  }
}
