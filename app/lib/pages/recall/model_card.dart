import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../../configs/app_config.dart';
import '../webgl_viewer.dart';

/// 模型卡片组件
class ModelCard extends StatelessWidget {
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
  Widget build(BuildContext context) {
    final sceneId = model['display_name'] ?? model['scene_id'] ?? 'Unknown Scene';
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
            color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
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
                        color: isDark ? darkInput : theme.grayColor3,
                        borderRadius: const BorderRadius.vertical(
                          top: Radius.circular(28.0),
                        ),
                      ),
                      clipBehavior: Clip.hardEdge,
                      child:
                          model['preview_img_path'] != null &&
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
                      textColor: textColor,
                    ),
                    const SizedBox(height: 4),
                    TDText(
                      desc,
                      font: theme.fontBodySmall,
                      textColor: hintTextColor,
                      maxLines: 2,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _navigateToViewer(BuildContext context) {
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty
        ? toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? toPosesUrl(plyPath) : null;
    final sceneId = model['display_name'] ?? model['scene_id'] ?? 'Unknown Scene';

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
      initialPose = transformMatrix.map((e) => (e as num).toDouble()).toList();
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
                CurvedAnimation(parent: animation, curve: Curves.easeOutCubic),
              ),
              child: child,
            ),
          );
        },
      ),
    );
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
