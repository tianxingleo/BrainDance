import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../../configs/app_config.dart';
import '../webgl_viewer.dart';

/// 模型下载状态
enum ModelDownloadStatus {
  /// 未下载
  notDownloaded,

  /// 部分下载（有 .tmp 文件）
  partial,

  /// 已完成下载
  downloaded,
}

/// 模型卡片组件
class ModelCard extends StatefulWidget {
  final Map<String, dynamic> model;
  final bool isDark;
  final Color textColor;
  final Color hintTextColor;
  final TDThemeData theme;
  final Color darkCard;
  final Color darkInput;
  final String Function(String) toPublicUrl;
  final String? Function(String?) toPosesUrl;

  const ModelCard({
    super.key,
    required this.model,
    required this.isDark,
    required this.textColor,
    required this.hintTextColor,
    required this.theme,
    required this.darkCard,
    required this.darkInput,
    required this.toPublicUrl,
    required this.toPosesUrl,
  });

  @override
  State<ModelCard> createState() => _ModelCardState();
}

class _ModelCardState extends State<ModelCard> {
  ModelDownloadStatus _status = ModelDownloadStatus.notDownloaded;
  double _progress = 0.0;

  @override
  void initState() {
    super.initState();
    _checkDownloadStatus();
  }

  /// 根据本地文件判断下载状态
  Future<void> _checkDownloadStatus() async {
    final modelUrl = _getModelUrl();
    if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
      // 本地模型，直接视为已下载
      if (mounted) setState(() => _status = ModelDownloadStatus.downloaded);
      return;
    }

    try {
      final encodedUrl = Uri.encodeFull(Uri.decodeFull(modelUrl));
      final uri = Uri.parse(encodedUrl);
      final sanitizedFileName =
          uri.path.replaceAll('/', '_').replaceAll('\\', '_');
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
    } catch (_) {
      // 检查失败时默认未下载
    }
  }

  String _getModelUrl() {
    final plyPath = widget.model['ply_path'] as String? ?? '';
    return plyPath.isNotEmpty
        ? widget.toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
  }

  @override
  Widget build(BuildContext context) {
    final model = widget.model;
    final isDark = widget.isDark;
    final theme = widget.theme;
    final sceneId = model['scene_id'] ?? 'Unknown Scene';
    final desc = model['description'] ?? textLocalize("recall_no_desc");
    final similarity = model['similarity'] as double?;

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return Transform.translate(
          offset: Offset(0, 50 * (1 - value)),
          child: Opacity(opacity: value, child: child),
        );
      },
      child: GestureDetector(
        onTap: () => _navigateToViewer(context),
        child: Container(
          decoration: BoxDecoration(
            color: isDark
                ? widget.darkCard
                : theme.whiteColor1.withAlpha(220),
            borderRadius: BorderRadius.circular(28.0),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withAlpha(20),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        color: isDark ? widget.darkInput : theme.grayColor3,
                        borderRadius: const BorderRadius.vertical(
                          top: Radius.circular(28.0),
                        ),
                      ),
                      clipBehavior: Clip.hardEdge,
                      child: model['preview_img_path'] != null &&
                              model['preview_img_path'].toString().isNotEmpty
                          ? Image.network(
                              model['preview_img_path'],
                              fit: BoxFit.cover,
                              errorBuilder: (context, error, stackTrace) =>
                                  _ModelMockCover(
                                    isDark: isDark,
                                    accentColor: isDark
                                        ? const Color(0xFF7AA2FF)
                                        : AppConfig.primaryColor,
                                  ),
                              loadingBuilder:
                                  (context, child, loadingProgress) {
                                    if (loadingProgress == null) return child;
                                    return const Center(
                                      child: CircularProgressIndicator(),
                                    );
                                  },
                            )
                          : _ModelMockCover(
                              isDark: isDark,
                              accentColor: isDark
                                  ? const Color(0xFF7AA2FF)
                                  : AppConfig.primaryColor,
                            ),
                    ),
                    if (similarity != null)
                      Positioned(
                        top: 8,
                        right: 8,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: theme.brandColor4.withAlpha(220),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: TDText(
                            '${(similarity * 100).toStringAsFixed(1)}%',
                            font: theme.fontBodyExtraSmall,
                            textColor: isDark
                                ? const Color(0xFFFFFFFF)
                                : Colors.white,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TDText(
                      sceneId,
                      font: theme.fontTitleMedium,
                      fontWeight: FontWeight.w600,
                      maxLines: 1,
                      textColor: widget.textColor,
                    ),
                    const SizedBox(height: 4),
                    TDText(
                      desc,
                      font: theme.fontBodySmall,
                      textColor: widget.hintTextColor,
                      maxLines: 2,
                    ),
                    const SizedBox(height: 6),
                    _buildDownloadStatus(isDark, theme),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// 构建下载状态指示器
  Widget _buildDownloadStatus(bool isDark, TDThemeData theme) {
    switch (_status) {
      case ModelDownloadStatus.notDownloaded:
        return Row(
          children: [
            Icon(
              Icons.cloud_download_outlined,
              size: 14,
              color: isDark
                  ? const Color(0xFF888888)
                  : const Color(0xFF999999),
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
                  color: isDark
                      ? const Color(0xFFFFB74D)
                      : const Color(0xFFF57C00),
                ),
                const SizedBox(width: 4),
                TDText(
                  '${textLocalize('recall_model_downloading')} $percent%',
                  font: theme.fontBodyExtraSmall,
                  textColor: isDark
                      ? const Color(0xFFFFB74D)
                      : const Color(0xFFF57C00),
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
                  isDark
                      ? const Color(0xFFFFB74D)
                      : const Color(0xFFF57C00),
                ),
              ),
            ),
          ],
        );

      case ModelDownloadStatus.downloaded:
        // 已下载完成：浅色文字显示"模型可用"
        return Row(
          children: [
            Icon(
              Icons.check_circle_outline,
              size: 14,
              color: isDark
                  ? const Color(0xFF666666)
                  : const Color(0xFFBBBBBB),
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

  void _navigateToViewer(BuildContext context) {
    final model = widget.model;
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty
        ? widget.toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? widget.toPosesUrl(plyPath) : null;
    final sceneId = model['scene_id'] ?? 'Unknown Scene';

    dynamic transformMatrix;
    // 如果传入的 matrix 为空，尝试从模型元数据中获取智能初始视角
    if (model['meta_info'] != null) {
      if (model['meta_info'] is Map &&
          model['meta_info']['initial_camera_pose'] != null) {
        transformMatrix = model['meta_info']['initial_camera_pose'];
      }
    }

    // Convert transformMatrix if not null to List<double>
    List<double>? initialPose;
    if (transformMatrix != null && transformMatrix is List) {
      initialPose =
          transformMatrix.map((e) => (e as num).toDouble()).toList();
    }

    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) =>
            WebGLViewerPage(
              initialModelUrl: modelUrl,
              posesUrl: posesUrl,
              sceneId: sceneId,
              initialPose: initialPose,
            ),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(
            opacity: animation,
            child: ScaleTransition(
              scale: Tween<double>(begin: 0.95, end: 1.0).animate(
                CurvedAnimation(
                  parent: animation,
                  curve: Curves.easeOutCubic,
                ),
              ),
              child: child,
            ),
          );
        },
      ),
    ).then((_) {
      // 从 viewer 返回后刷新下载状态
      _checkDownloadStatus();
    });
  }
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

  @override
  void initState() {
    super.initState();
    _checkDownloadStatus();
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
      final sanitizedFileName =
          uri.path.replaceAll('/', '_').replaceAll('\\', '_');
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

    switch (_status) {
      case ModelDownloadStatus.notDownloaded:
        return Row(
          children: [
            Icon(
              Icons.cloud_download_outlined,
              size: 14,
              color: isDark
                  ? const Color(0xFF888888)
                  : const Color(0xFF999999),
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
                  color: isDark
                      ? const Color(0xFFFFB74D)
                      : const Color(0xFFF57C00),
                ),
                const SizedBox(width: 4),
                TDText(
                  '${textLocalize('recall_model_downloading')} $percent%',
                  font: theme.fontBodyExtraSmall,
                  textColor: isDark
                      ? const Color(0xFFFFB74D)
                      : const Color(0xFFF57C00),
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
                  isDark
                      ? const Color(0xFFFFB74D)
                      : const Color(0xFFF57C00),
                ),
              ),
            ),
          ],
        );

      case ModelDownloadStatus.downloaded:
        // 已下载完成：浅色文字显示"模型可用"
        return Row(
          children: [
            Icon(
              Icons.check_circle_outline,
              size: 14,
              color: isDark
                  ? const Color(0xFF666666)
                  : const Color(0xFFBBBBBB),
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

class _ModelMockCover extends StatelessWidget {
  final bool isDark;
  final Color accentColor;

  const _ModelMockCover({required this.isDark, required this.accentColor});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1A1E27) : const Color(0xFFF6F8FC),
        border: Border.all(
          color: isDark
              ? Colors.white.withAlpha(18)
              : accentColor.withAlpha(35),
        ),
      ),
      child: Center(
        child: Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: isDark
                ? Colors.white.withAlpha(6)
                : Colors.white.withAlpha(190),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: isDark
                  ? Colors.white.withAlpha(18)
                  : accentColor.withAlpha(28),
            ),
          ),
          child: Icon(
            Icons.auto_awesome_mosaic_rounded,
            size: 28,
            color: accentColor.withAlpha(210),
          ),
        ),
      ),
    );
  }
}
