import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:braindance/configs/app_config.dart';
import '../../services/viewer_navigation.dart';

/// 搜索结果卡片组件（带匹配帧列表）
class SearchResultCard extends StatelessWidget {
  final Map<String, dynamic> model;
  final bool isDark;
  final Color textColor;
  final Color hintTextColor;
  final TDThemeData theme;
  final Color darkCard;
  final Color darkInput;
  final String Function(String) toPublicUrl;
  final String? Function(String?) toPosesUrl;

  const SearchResultCard({
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
    final sceneId =
        model['display_name']?.toString() ??
        model['scene_id'] ??
        'Unknown Scene';
    final desc = model['description'] ?? textLocalize('recall_no_desc');
    final similarity = model['similarity'] as double?;
    final userId = model['user_id'] ?? '';
    final matchedFrames = model['matched_frames'] as List<dynamic>? ?? [];

    return Container(
      margin: const EdgeInsets.only(bottom: 16.0),
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: BorderRadius.circular(theme.radiusLarge),
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
          // Top header: Model Info
          GestureDetector(
            onTap: () => _navigateToViewer(context, null),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
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
                  if (similarity != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: theme.brandColor4.withAlpha(220),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: TDText(
                        '${(similarity * 100).toStringAsFixed(1)}%',
                        font: theme.fontBodySmall,
                        textColor: isDark
                            ? const Color(0xFFFFFFFF)
                            : Colors.white,
                      ),
                    ),
                ],
              ),
            ),
          ),
          // Horizontal list of frames
          if (matchedFrames.isNotEmpty)
            SizedBox(
              height: 120,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(
                  horizontal: 16.0,
                ).copyWith(bottom: 16.0),
                itemCount: matchedFrames.length,
                itemBuilder: (context, frameIndex) {
                  final frame = matchedFrames[frameIndex];
                  final imageName = frame['image_name'];
                  final frameSim = frame['similarity'] as double?;

                  final imageUrl =
                      "https://kntcynswgrmgbbgntkiv.supabase.co/storage/v1/object/public/braindance-assets/$userId/$sceneId/output/images/$imageName";

                  return GestureDetector(
                    onTap: () => _navigateToViewer(context, frame),
                    child: Container(
                      width: 140,
                      margin: const EdgeInsets.only(right: 12.0),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(8.0),
                        color: isDark ? darkInput : theme.grayColor3,
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8.0),
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            ColoredBox(
                              color: isDark ? darkInput : theme.grayColor3,
                              child: Padding(
                                padding: const EdgeInsets.all(6.0),
                                child: Image.network(
                                  imageUrl,
                                  fit: BoxFit.contain,
                                  alignment: Alignment.center,
                                  errorBuilder: (context, error, stackTrace) =>
                                      const Center(
                                        child: Icon(
                                          Icons.broken_image,
                                          color: Colors.grey,
                                        ),
                                      ),
                                ),
                              ),
                            ),
                            if (frameSim != null)
                              Positioned(
                                bottom: 4,
                                left: 4,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 4,
                                    vertical: 2,
                                  ),
                                  decoration: BoxDecoration(
                                    color: Colors.black.withAlpha(100),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Text(
                                    '${(frameSim * 100).toStringAsFixed(1)}%',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 10,
                                    ),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }

  void _navigateToViewer(BuildContext context, dynamic transformMatrix) {
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty
        ? toPublicUrl(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final posesUrlResolved = plyPath.isNotEmpty ? toPosesUrl(plyPath) : null;
    final sceneId =
        model['display_name']?.toString() ??
        model['scene_id'] ??
        'Unknown Scene';
    String? initialPoseId;

    if (transformMatrix is Map) {
      initialPoseId = transformMatrix['image_name']?.toString();
      transformMatrix = transformMatrix['transform_matrix'];
    }

    if (transformMatrix == null && model['meta_info'] != null) {
      if (model['meta_info'] is Map &&
          model['meta_info']['initial_camera_pose'] != null) {
        transformMatrix = model['meta_info']['initial_camera_pose'];
      }
    }

    List<double>? initialPose;
    if (transformMatrix != null && transformMatrix is List) {
      initialPose = transformMatrix.map((e) => (e as num).toDouble()).toList();
    }

    openViewer(
      context,
      initialModelUrl: modelUrl,
      posesUrl: posesUrlResolved,
      sceneId: sceneId,
      initialPose: initialPose,
      initialPoseId: initialPoseId,
    );
  }
}
