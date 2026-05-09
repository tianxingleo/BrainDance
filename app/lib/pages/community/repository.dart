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
        final modelUrl = _publicModelUrl(model['ply_path']?.toString() ?? '');
        final previewUrl = map['cover_image_url']?.toString().isNotEmpty == true
            ? map['cover_image_url']?.toString()
            : model['preview_img_path']?.toString();

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
      final userId = _client.auth.currentUser?.id;
      dynamic query = _client
          .from('model_assets')
          .select('id, scene_id, description, ply_path, preview_img_path')
          .order('created_at', ascending: false)
          .limit(20);
      if (userId != null && userId.isNotEmpty) {
        query = query.eq('user_id', userId);
      }

      final response = await query;
      final models = response.map<CommunityModelOption>((raw) {
        final map = Map<String, dynamic>.from(raw);
        final path = map['ply_path']?.toString() ?? '';
        final publicUrl = _publicModelUrl(path);
        return CommunityModelOption(
          id: map['id'].toString(),
          sceneId:
              map['display_name']?.toString() ??
              map['scene_id']?.toString() ??
              textLocalize('community_unnamed_model'),
          description: map['description']?.toString() ?? '',
          modelUrl: publicUrl,
          posesUrl: _posesUrlFromPath(path),
          coverUrl: map['preview_img_path']?.toString(),
        );
      }).toList();

      if (models.isNotEmpty) {
        return models;
      }
    } catch (_) {}

    return _demoModels;
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

  String _publicModelUrl(String storagePath) {
    if (storagePath.isEmpty) {
      return '';
    }
    if (storagePath.startsWith('http://') ||
        storagePath.startsWith('https://')) {
      return storagePath;
    }
    try {
      return _client.storage
          .from('braindance-assets')
          .getPublicUrl(storagePath);
    } catch (_) {
      return storagePath;
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

final List<CommunityModelOption> _demoModels = [
  CommunityModelOption(
    id: 'demo-1',
    sceneId: '西湖断桥晨雾',
    description: '湖面、柳树、桥面和低雾一起形成了安静的晨间空间。',
    modelUrl: '',
    posesUrl: null,
    coverUrl: null,
  ),
  CommunityModelOption(
    id: 'demo-2',
    sceneId: '东京塔夜色',
    description: '夜间城市灯光包围着塔体，适合做高对比空间浏览。',
    modelUrl: '',
    posesUrl: null,
    coverUrl: null,
  ),
];

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
