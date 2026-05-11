import 'package:braindance/configs/app_config.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'models.dart';

class CommunityRepository {
  static final List<CommunityPost> _localDrafts = [];

  SupabaseClient get _client => Supabase.instance.client;

  Future<List<CommunityPost>> fetchPosts() async {
    try {
      final response = await _client
          .from('community_posts')
          .select('''
            id,
            title,
            caption,
            place_name,
            latitude,
            longitude,
            user_id,
            created_at,
            model_name,
            cover_image_url,
            model_assets (
              scene_id,
              description,
              ply_path,
              preview_img_path
            )
          ''')
          .order('created_at', ascending: false)
          .limit(24);

      final posts = response.map<CommunityPost>((raw) {
        final map = Map<String, dynamic>.from(raw);
        final model = map['model_assets'] is Map
            ? Map<String, dynamic>.from(map['model_assets'] as Map)
            : <String, dynamic>{};
        final modelUrl = _normalizeStorageUrl(model['ply_path']?.toString() ?? '');
        final previewUrl = _normalizeStorageUrl(
          map['cover_image_url']?.toString().isNotEmpty == true
              ? map['cover_image_url']!.toString()
              : (model['preview_img_path']?.toString() ?? ''),
        );

        return CommunityPost(
          id: map['id'].toString(),
          title:
              map['title']?.toString() ??
              textLocalize('community_unnamed_memory'),
          caption:
              map['caption']?.toString() ??
              model['description']?.toString() ??
              '',
          placeName:
              map['place_name']?.toString() ??
              textLocalize('community_no_location'),
          latitude: (map['latitude'] as num?)?.toDouble() ?? 0,
          longitude: (map['longitude'] as num?)?.toDouble() ?? 0,
          authorName:
              map['user_id']?.toString() ?? textLocalize('community_anonymous'),
          modelName:
              map['model_name']?.toString() ??
              model['display_name']?.toString() ??
              model['scene_id']?.toString() ??
              '3D 模型',
          modelUrl: modelUrl,
          posesUrl: _posesUrlFromPath(model['ply_path']?.toString()),
          coverUrl: previewUrl,
          createdAt:
              DateTime.tryParse(map['created_at']?.toString() ?? '') ??
              DateTime.now(),
          tags: _extractTags(
            model['description']?.toString(),
            map['place_name']?.toString(),
          ),
        );
      }).toList();

      final merged = [..._localDrafts, ...posts];
      if (merged.isNotEmpty) {
        return merged;
      }
    } catch (_) {}

    return [..._localDrafts, ..._demoPosts];
  }

  Future<List<CommunityModelOption>> fetchShareableModels() async {
    try {
      final currentUserId =
          (_client.auth.currentUser?.id ?? '').trim();

      final response = await _client
          .from('model_assets')
          .select(
            'id, scene_id, display_name, description, tags, ply_path, preview_img_path, user_id',
          )
          .order('created_at', ascending: false)
          .limit(50);

      final models = <CommunityModelOption>[];
      for (final raw in response) {
        final map = Map<String, dynamic>.from(raw);

        // Client-side ownership check, same logic as recall's _isOwnModel
        if (currentUserId.isNotEmpty) {
          final ownerId = (map['user_id']?.toString() ?? '').trim();
          if (ownerId.isNotEmpty && ownerId != currentUserId) {
            continue;
          }
        }

        final path = map['ply_path']?.toString() ?? '';
        final publicUrl = _normalizeStorageUrl(path);
        // Same display name priority as recall's _modelDisplayName:
        // display_name → first tag → scene_id → fallback
        final displayName = _modelDisplayName(map);
        models.add(CommunityModelOption(
          id: map['id'].toString(),
          sceneId: displayName,
          description: map['description']?.toString() ?? '',
          modelUrl: publicUrl,
          posesUrl: _posesUrlFromPath(path),
          coverUrl: _normalizeStorageUrl(
            map['preview_img_path']?.toString() ?? '',
          ),
        ));
      }

      return models;
    } catch (_) {}

    return <CommunityModelOption>[];
  }

  Future<CommunityPost> createPost(CommunityComposerResult draft) async {
    final optimistic = CommunityPost(
      id: 'local-${DateTime.now().microsecondsSinceEpoch}',
      title: draft.title,
      caption: draft.caption,
      placeName: draft.placeName,
      latitude: draft.latitude,
      longitude: draft.longitude,
      authorName: _client.auth.currentUser?.email ?? '我',
      modelName: draft.model.sceneId,
      modelUrl: draft.model.modelUrl,
      posesUrl: draft.model.posesUrl,
      coverUrl: draft.model.coverUrl,
      createdAt: DateTime.now(),
      tags: _extractTags(draft.model.description, draft.placeName),
    );

    try {
      await _client.from('community_posts').insert({
        'user_id': _client.auth.currentUser?.id ?? 'local-user',
        'model_asset_id': draft.model.id,
        'model_name': draft.model.sceneId,
        'title': draft.title,
        'caption': draft.caption,
        'place_name': draft.placeName,
        'latitude': draft.latitude,
        'longitude': draft.longitude,
        'cover_image_url': draft.model.coverUrl,
      });
      return optimistic;
    } catch (_) {
      _localDrafts.insert(0, optimistic);
      return optimistic;
    }
  }

  /// Convert storage path / old server URL to a valid public URL.
  /// Same logic as recall's _normalizeStorageUrl + _toPublicUrl.
  String _normalizeStorageUrl(String raw) {
    if (raw.isEmpty) return '';

    // Strip old Supabase instance prefix if present
    const marker = '/storage/v1/object/public/braindance-assets/';
    final idx = raw.indexOf(marker);
    if (idx >= 0) {
      raw = raw.substring(idx + marker.length);
    }

    if (raw.startsWith('http://') || raw.startsWith('https://')) {
      return raw;
    }
    try {
      return _client.storage
          .from('braindance-assets')
          .getPublicUrl(raw);
    } catch (_) {
      return raw;
    }
  }

  String? _posesUrlFromPath(String? storagePath) {
    if (storagePath == null || storagePath.isEmpty) {
      return null;
    }
    final posesPath = storagePath.replaceAll(
      RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
      'webgl_poses.json',
    );
    if (posesPath == storagePath) {
      return null;
    }
    try {
      return _client.storage.from('braindance-assets').getPublicUrl(posesPath);
    } catch (_) {
      return null;
    }
  }

  /// Same display name priority as recall's _modelDisplayName:
  /// display_name → first tag → scene_id → fallback
  String _modelDisplayName(Map<String, dynamic> model) {
    final dn = model['display_name']?.toString().trim() ?? '';
    if (dn.isNotEmpty) return dn;

    final tags = model['tags'];
    if (tags is List) {
      for (final tag in tags) {
        final value = tag?.toString().trim() ?? '';
        if (value.isNotEmpty) return value;
      }
    }

    final sid = model['scene_id']?.toString().trim() ?? '';
    if (sid.isNotEmpty) return sid;

    return textLocalize('community_unnamed_model');
  }

  List<String> _extractTags(String? description, String? placeName) {
    final words = <String>[];
    if (placeName != null && placeName.isNotEmpty) {
      words.add(placeName);
    }
    if (description != null && description.isNotEmpty) {
      final tokens = description
          .split(RegExp(r'[\s,，。.!！？]+'))
          .where((token) => token.trim().length >= 2)
          .take(2);
      words.addAll(tokens);
    }
    if (words.isEmpty) {
      words.add('记忆');
    }
    return words.take(3).toList();
  }
}

final List<CommunityPost> _demoPosts = [
  CommunityPost(
    id: 'demo-post-1',
    title: '清晨刚亮时的断桥',
    caption: '薄雾和湖面的反光被一起留在模型里，适合从桥头慢慢推进看空间层次。',
    placeName: '杭州西湖',
    latitude: 30.258,
    longitude: 120.140,
    authorName: 'Lin',
    modelName: '西湖断桥晨雾',
    modelUrl: '',
    posesUrl: null,
    coverUrl: null,
    createdAt: DateTime.now().subtract(const Duration(hours: 2)),
    tags: const ['湖面', '晨雾', '桥'],
  ),
  CommunityPost(
    id: 'demo-post-2',
    title: '东京塔下的夜风',
    caption: '我把塔体、街道和人流一起收进去了，向下翻的时候会先看到塔身，再看到街区。',
    placeName: '东京塔',
    latitude: 35.659,
    longitude: 139.745,
    authorName: 'Aoi',
    modelName: '东京塔夜色',
    modelUrl: '',
    posesUrl: null,
    coverUrl: null,
    createdAt: DateTime.now().subtract(const Duration(hours: 6)),
    tags: const ['夜景', '城市', '塔体'],
  ),
];
