import 'dart:math' as math;

class CommunityComposerResult {
  final String title;
  final String caption;
  final String placeName;
  final double latitude;
  final double longitude;
  final CommunityModelOption model;

  const CommunityComposerResult({
    required this.title,
    required this.caption,
    required this.placeName,
    required this.latitude,
    required this.longitude,
    required this.model,
  });
}

class CommunityModelOption {
  final String id;
  final String sceneId;
  final String description;
  final String modelUrl;
  final String? posesUrl;
  final String? coverUrl;

  const CommunityModelOption({
    required this.id,
    required this.sceneId,
    required this.description,
    required this.modelUrl,
    required this.posesUrl,
    required this.coverUrl,
  });
}

class CommunityPost {
  final String id;
  final String title;
  final String caption;
  final String placeName;
  final double latitude;
  final double longitude;
  final String authorName;
  final String modelName;
  final String modelUrl;
  final String? posesUrl;
  final String? coverUrl;
  final DateTime createdAt;
  final List<String> tags;

  const CommunityPost({
    required this.id,
    required this.title,
    required this.caption,
    required this.placeName,
    required this.latitude,
    required this.longitude,
    required this.authorName,
    required this.modelName,
    required this.modelUrl,
    required this.posesUrl,
    required this.coverUrl,
    required this.createdAt,
    required this.tags,
  });

  String get relativeTimeLabel {
    final difference = DateTime.now().difference(createdAt);
    if (difference.inMinutes < 60) {
      return '${math.max(difference.inMinutes, 1)} 分钟前';
    }
    if (difference.inHours < 24) {
      return '${difference.inHours} 小时前';
    }
    return '${difference.inDays} 天前';
  }
}
