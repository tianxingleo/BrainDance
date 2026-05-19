import 'dart:math' as math;

String formatRelativeTime(DateTime createdAt) {
  final difference = DateTime.now().difference(createdAt);
  if (difference.inMinutes < 60) {
    return '${math.max(difference.inMinutes, 1)} 分钟前';
  }
  if (difference.inHours < 24) {
    return '${difference.inHours} 小时前';
  }
  return '${difference.inDays} 天前';
}

class CommunityComposerResult {
  final String title;
  final String caption;
  final String placeName;
  final double latitude;
  final double longitude;
  final List<CommunityModelOption> models;
  final List<String> tags;
  final bool isPublic;

  const CommunityComposerResult({
    required this.title,
    required this.caption,
    required this.placeName,
    required this.latitude,
    required this.longitude,
    required this.models,
    required this.tags,
    this.isPublic = true,
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
  final bool isPublic;
  final int likeCount;
  final int favoriteCount;
  final int commentCount;
  final int viewCount;
  final bool isLikedByCurrentUser;
  final bool isFavoritedByCurrentUser;

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
    this.isPublic = true,
    this.likeCount = 0,
    this.favoriteCount = 0,
    this.commentCount = 0,
    this.viewCount = 0,
    this.isLikedByCurrentUser = false,
    this.isFavoritedByCurrentUser = false,
    this.extraImages = const [],
  });

  /// Additional model images per post (from metadata.images).
  final List<Map<String, dynamic>> extraImages;

  CommunityPost copyWith({
    int? likeCount,
    int? favoriteCount,
    int? commentCount,
    int? viewCount,
    bool? isLikedByCurrentUser,
    bool? isFavoritedByCurrentUser,
  }) {
    return CommunityPost(
      id: id,
      title: title,
      caption: caption,
      placeName: placeName,
      latitude: latitude,
      longitude: longitude,
      authorName: authorName,
      modelName: modelName,
      modelUrl: modelUrl,
      posesUrl: posesUrl,
      coverUrl: coverUrl,
      createdAt: createdAt,
      tags: tags,
      isPublic: isPublic,
      likeCount: likeCount ?? this.likeCount,
      favoriteCount: favoriteCount ?? this.favoriteCount,
      commentCount: commentCount ?? this.commentCount,
      viewCount: viewCount ?? this.viewCount,
      isLikedByCurrentUser:
          isLikedByCurrentUser ?? this.isLikedByCurrentUser,
      isFavoritedByCurrentUser:
          isFavoritedByCurrentUser ?? this.isFavoritedByCurrentUser,
      extraImages: extraImages,
    );
  }

  String get relativeTimeLabel => formatRelativeTime(createdAt);
}

class CommunityStats {
  final int postCount;
  final int viewCount;
  final int likeCount;
  final int favoriteCount;
  final int commentCount;
  final int draftCount;
  final int shareableModelCount;

  const CommunityStats({
    this.postCount = 0,
    this.viewCount = 0,
    this.likeCount = 0,
    this.favoriteCount = 0,
    this.commentCount = 0,
    this.draftCount = 0,
    this.shareableModelCount = 0,
  });
}

class CommunityComment {
  final String id;
  final String postId;
  final String userId;
  final String userName;
  final String text;
  final DateTime createdAt;

  const CommunityComment({
    required this.id,
    required this.postId,
    required this.userId,
    required this.userName,
    required this.text,
    required this.createdAt,
  });

  String get relativeTimeLabel => formatRelativeTime(createdAt);
}

class CommunityDraft {
  final List<String> modelIds;
  final String title;
  final String caption;
  final String placeName;
  final double latitude;
  final double longitude;
  final List<String> tags;
  final bool isPublic;

  const CommunityDraft({
    this.modelIds = const [],
    this.title = '',
    this.caption = '',
    this.placeName = '',
    this.latitude = 0,
    this.longitude = 0,
    this.tags = const [],
    this.isPublic = true,
  });

  bool get isEmpty =>
      modelIds.isEmpty &&
      title.isEmpty &&
      caption.isEmpty &&
      placeName.isEmpty;

  Map<String, dynamic> toJson() => {
        'modelIds': modelIds,
        'title': title,
        'caption': caption,
        'placeName': placeName,
        'latitude': latitude,
        'longitude': longitude,
        'tags': tags,
        'isPublic': isPublic,
      };

  factory CommunityDraft.fromJson(Map<String, dynamic> json) {
    return CommunityDraft(
      modelIds: List<String>.from(json['modelIds'] ?? []),
      title: json['title']?.toString() ?? '',
      caption: json['caption']?.toString() ?? '',
      placeName: json['placeName']?.toString() ?? '',
      latitude: (json['latitude'] as num?)?.toDouble() ?? 0,
      longitude: (json['longitude'] as num?)?.toDouble() ?? 0,
      tags: List<String>.from(json['tags'] ?? []),
      isPublic: json['isPublic'] != false,
    );
  }
}
