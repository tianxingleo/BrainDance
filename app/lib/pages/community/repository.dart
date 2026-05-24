import 'dart:convert';
import 'dart:io';

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/services/preview_image_resolver.dart';
import 'package:path_provider/path_provider.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'map_marker.dart';
import 'models.dart';

class CommunityRepository {
  static final List<CommunityPost> _localDrafts = [];

  SupabaseClient get _client => Supabase.instance.client;

  String get currentUserId => _client.auth.currentUser?.id ?? '';

  static const _postSelect = '''
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
            metadata,
            model_assets (
              scene_id,
              display_name,
              description,
              ply_path,
              preview_img_path,
              meta_info
            )
          ''';

  // ---- Posts ----

  Future<List<CommunityPost>> fetchPosts() async {
    try {
      final response = await _client
          .from('community_posts')
          .select(_postSelect)
          .order('created_at', ascending: false)
          .limit(24);

      final posts = response.map<CommunityPost>((raw) {
        final map = Map<String, dynamic>.from(raw);
        final model = map['model_assets'] is Map
            ? Map<String, dynamic>.from(map['model_assets'] as Map)
            : <String, dynamic>{};
        final modelUrl = _normalizeStorageUrl(
          model['ply_path']?.toString() ?? '',
        );
        final cover = _resolvePostCover(map, model);
        final metadata = _parseMetadata(map['metadata']);

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
          userId: map['user_id']?.toString() ?? '',
          authorName: metadata['author_email']?.toString() ??
              map['user_id']?.toString() ??
              textLocalize('community_anonymous'),
          modelName:
              map['model_name']?.toString() ??
              model['display_name']?.toString() ??
              model['scene_id']?.toString() ??
              '3D 模型',
          modelUrl: modelUrl,
          posesUrl: _posesUrlFromPath(model['ply_path']?.toString()),
          coverUrl: cover.primary,
          coverFallbackUrl: cover.fallback,
          createdAt:
              DateTime.tryParse(map['created_at']?.toString() ?? '') ??
              DateTime.now(),
          tags: _extractTags(
            model['description']?.toString(),
            map['place_name']?.toString(),
          ),
          isPublic: metadata['is_public'] != false,
          likeCount: (metadata['likes'] as List?)?.length ?? 0,
          favoriteCount: (metadata['favorites'] as List?)?.length ?? 0,
          viewCount: _readViewCount(metadata),
          isLikedByCurrentUser: _containsUser(metadata['likes'], currentUserId),
          isFavoritedByCurrentUser: _containsUser(
            metadata['favorites'],
            currentUserId,
          ),
          extraImages: List<Map<String, dynamic>>.from(
            metadata['images'] ?? [],
          ),
          commentCount: (metadata['comments'] as List?)?.length ?? 0,
        );
      }).toList();

      // Filter: public posts + own posts
      final visible = posts.where((p) {
        if (p.isPublic) return true;
        final uid = currentUserId;
        return uid.isNotEmpty && p.userId == uid;
      }).toList();

      final merged = [..._localDrafts, ...visible];
      if (merged.isNotEmpty) return merged;
    } catch (_) {
      rethrow;
    }

    return [];
  }

  // ---- Map markers ----

  /// Lightweight read path used by the community map. Independent of
  /// [fetchPosts] so the map isn't bound to the 24-row feed limit.
  Future<List<CommunityMapMarker>> fetchMapMarkers({
    int limit = 500,
    bool onlyPublic = true,
  }) async {
    final markers = <CommunityMapMarker>[];

    // Local optimistic drafts first — they have no metadata payload.
    for (final p in _localDrafts) {
      if (p.latitude == 0 && p.longitude == 0) continue;
      markers.add(
        CommunityMapMarker(
          id: p.id,
          title: p.title,
          placeName: p.placeName,
          latitude: p.latitude,
          longitude: p.longitude,
          coverUrl: p.coverUrl,
          createdAt: p.createdAt,
          likeCount: p.likeCount,
          viewCount: p.viewCount,
        ),
      );
    }

    try {
      final response = await _client
          .from('community_posts')
          .select(
            'id, title, place_name, latitude, longitude, cover_image_url, '
            'created_at, user_id, metadata',
          )
          .order('created_at', ascending: false)
          .limit(limit);

      final uid = currentUserId;
      for (final raw in response) {
        final map = Map<String, dynamic>.from(raw);
        final lat = (map['latitude'] as num?)?.toDouble() ?? 0;
        final lng = (map['longitude'] as num?)?.toDouble() ?? 0;
        if (lat == 0 && lng == 0) continue;

        final metadata = _parseMetadata(map['metadata']);
        final isPublic = metadata['is_public'] != false;
        if (onlyPublic && !isPublic) {
          final ownerId = map['user_id']?.toString() ?? '';
          if (uid.isEmpty || ownerId != uid) continue;
        }

        markers.add(
          CommunityMapMarker(
            id: map['id'].toString(),
            title:
                map['title']?.toString() ??
                textLocalize('community_unnamed_memory'),
            placeName:
                map['place_name']?.toString() ??
                textLocalize('community_no_location'),
            latitude: lat,
            longitude: lng,
            coverUrl: _normalizeStorageUrl(
              map['cover_image_url']?.toString() ?? '',
            ),
            createdAt:
                DateTime.tryParse(map['created_at']?.toString() ?? '') ??
                DateTime.now(),
            likeCount: (metadata['likes'] as List?)?.length ?? 0,
            viewCount: _readViewCount(metadata),
          ),
        );
      }
    } catch (_) {
      // Fall back to demo posts so the map isn't empty during offline dev.
      for (final p in _demoPosts) {
        if (p.latitude == 0 && p.longitude == 0) continue;
        markers.add(
          CommunityMapMarker(
            id: p.id,
            title: p.title,
            placeName: p.placeName,
            latitude: p.latitude,
            longitude: p.longitude,
            coverUrl: p.coverUrl,
            createdAt: p.createdAt,
            likeCount: p.likeCount,
            viewCount: p.viewCount,
          ),
        );
      }
    }

    // De-dup by id, keeping first occurrence (local drafts win).
    final seen = <String>{};
    return markers.where((m) => seen.add(m.id)).toList();
  }

  /// Loads a single post by id. Used by the map page to navigate into the
  /// existing detail view from a marker tap.
  Future<CommunityPost?> fetchPostById(String postId) async {
    if (postId.startsWith('local-')) {
      final hit = _localDrafts.where((p) => p.id == postId);
      return hit.isEmpty ? null : hit.first;
    }
    try {
      final response = await _client
          .from('community_posts')
          .select(_postSelect)
          .eq('id', postId)
          .maybeSingle();
      if (response == null) return null;
      final map = Map<String, dynamic>.from(response);
      final model = map['model_assets'] is Map
          ? Map<String, dynamic>.from(map['model_assets'] as Map)
          : <String, dynamic>{};
      final modelUrl = _normalizeStorageUrl(
        model['ply_path']?.toString() ?? '',
      );
      final cover = _resolvePostCover(map, model);
      final metadata = _parseMetadata(map['metadata']);
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
        userId: map['user_id']?.toString() ?? '',
        authorName: metadata['author_email']?.toString() ??
            map['user_id']?.toString() ??
            textLocalize('community_anonymous'),
        modelName: map['model_name']?.toString() ??
            model['display_name']?.toString() ??
            model['scene_id']?.toString() ??
            '3D 模型',
        modelUrl: modelUrl,
        posesUrl: _posesUrlFromPath(model['ply_path']?.toString()),
        coverUrl: cover.primary,
        coverFallbackUrl: cover.fallback,
        createdAt:
            DateTime.tryParse(map['created_at']?.toString() ?? '') ??
            DateTime.now(),
        tags: _extractTags(
          model['description']?.toString(),
          map['place_name']?.toString(),
        ),
        isPublic: metadata['is_public'] != false,
        likeCount: (metadata['likes'] as List?)?.length ?? 0,
        favoriteCount: (metadata['favorites'] as List?)?.length ?? 0,
        commentCount: (metadata['comments'] as List?)?.length ?? 0,
        viewCount: _readViewCount(metadata),
        isLikedByCurrentUser: _containsUser(metadata['likes'], currentUserId),
        isFavoritedByCurrentUser: _containsUser(
          metadata['favorites'],
          currentUserId,
        ),
        extraImages: List<Map<String, dynamic>>.from(metadata['images'] ?? []),
      );
    } catch (_) {
      return null;
    }
  }

  Future<List<CommunityPost>> fetchMyPosts() async {
    final uid = currentUserId;
    if (uid.isEmpty) return const [];

    try {
      final response = await _client
          .from('community_posts')
          .select(_postSelect)
          .eq('user_id', uid)
          .order('created_at', ascending: false);
      final posts = response.map<CommunityPost>((raw) {
        final map = Map<String, dynamic>.from(raw);
        final model = map['model_assets'] is Map
            ? Map<String, dynamic>.from(map['model_assets'] as Map)
            : <String, dynamic>{};
        final metadata = _parseMetadata(map['metadata']);
        final modelUrl = _normalizeStorageUrl(
          model['ply_path']?.toString() ?? '',
        );
        final cover = _resolvePostCover(map, model);

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
          userId: map['user_id']?.toString() ?? '',
          authorName: metadata['author_email']?.toString() ??
              map['user_id']?.toString() ??
              textLocalize('community_anonymous'),
          modelName: map['model_name']?.toString() ??
              model['display_name']?.toString() ??
              model['scene_id']?.toString() ??
              '3D 模型',
          modelUrl: modelUrl,
          posesUrl: _posesUrlFromPath(model['ply_path']?.toString()),
          coverUrl: cover.primary,
          coverFallbackUrl: cover.fallback,
          createdAt:
              DateTime.tryParse(map['created_at']?.toString() ?? '') ??
              DateTime.now(),
          tags: _extractTags(
            model['description']?.toString(),
            map['place_name']?.toString(),
          ),
          isPublic: metadata['is_public'] != false,
          likeCount: (metadata['likes'] as List?)?.length ?? 0,
          favoriteCount: (metadata['favorites'] as List?)?.length ?? 0,
          commentCount: (metadata['comments'] as List?)?.length ?? 0,
          viewCount: _readViewCount(metadata),
          isLikedByCurrentUser: _containsUser(metadata['likes'], uid),
          isFavoritedByCurrentUser: _containsUser(metadata['favorites'], uid),
          extraImages: List<Map<String, dynamic>>.from(
            metadata['images'] ?? [],
          ),
        );
      }).toList();

      return [
        ..._localDrafts.where((post) => post.id.startsWith('local-$uid-')),
        ...posts,
      ];
    } catch (_) {}

    return _localDrafts
        .where((post) => post.id.startsWith('local-$uid-'))
        .toList();
  }

  Future<List<CommunityPost>> fetchFavoritePosts() async {
    if (currentUserId.isEmpty) return const [];
    final posts = await fetchPosts();
    return posts.where((post) => post.isFavoritedByCurrentUser).toList();
  }

  Future<List<CommunityPost>> fetchLikedPosts() async {
    if (currentUserId.isEmpty) return const [];
    final posts = await fetchPosts();
    return posts.where((post) => post.isLikedByCurrentUser).toList();
  }

  Future<CommunityStats> fetchCommunityStats() async {
    final posts = await fetchMyPosts();
    final shareableModels = await fetchShareableModels();
    final draft = await loadDraft();

    return CommunityStats(
      postCount: posts.length,
      viewCount: posts.fold(0, (sum, post) => sum + post.viewCount),
      likeCount: posts.fold(0, (sum, post) => sum + post.likeCount),
      favoriteCount: posts.fold(0, (sum, post) => sum + post.favoriteCount),
      commentCount: posts.fold(0, (sum, post) => sum + post.commentCount),
      draftCount: draft.isEmpty ? 0 : 1,
      shareableModelCount: shareableModels.length,
    );
  }

  // ---- Shareable models ----

  Future<List<CommunityModelOption>> fetchShareableModels() async {
    try {
      final uid = currentUserId;

      final response = await _client
          .from('model_assets')
          .select(
            'id, scene_id, display_name, description, tags, ply_path, preview_img_path, meta_info, user_id',
          )
          .order('created_at', ascending: false)
          .limit(50);

      final models = <CommunityModelOption>[];
      for (final raw in response) {
        final map = Map<String, dynamic>.from(raw);

        if (uid.isNotEmpty) {
          final ownerId = map['user_id']?.toString() ?? '';
          if (ownerId.isNotEmpty && ownerId != uid) continue;
        }

        final path = map['ply_path']?.toString() ?? '';
        final publicUrl = _normalizeStorageUrl(path);
        final displayName = _modelDisplayName(map);
        final cover = resolvePreviewImagePaths(
          map,
          normalize: _normalizeStorageUrl,
        );
        models.add(
          CommunityModelOption(
            id: map['id'].toString(),
            sceneId: displayName,
            description: map['description']?.toString() ?? '',
            modelUrl: publicUrl,
            posesUrl: _posesUrlFromPath(path),
            coverUrl: cover.primary,
            coverFallbackUrl: cover.fallback,
          ),
        );
      }
      return models;
    } catch (_) {}
    return <CommunityModelOption>[];
  }

  // ---- Create / Publish ----

  Future<CommunityPost> createPost(CommunityComposerResult draft) async {
    if (draft.models.isEmpty) {
      throw ArgumentError('At least one model is required to create a post.');
    }
    final model = draft.models.first;
    final optimistic = CommunityPost(
      id: 'local-$currentUserId-${DateTime.now().microsecondsSinceEpoch}',
      title: draft.title,
      caption: draft.caption,
      placeName: draft.placeName,
      latitude: draft.latitude,
      longitude: draft.longitude,
      authorName: _client.auth.currentUser?.email ?? '我',
      modelName: model.sceneId,
      modelUrl: model.modelUrl,
      posesUrl: model.posesUrl,
      coverUrl: model.coverUrl,
      coverFallbackUrl: model.coverFallbackUrl,
      createdAt: DateTime.now(),
      tags: draft.tags,
      isPublic: draft.isPublic,
      extraImages: draft.models
          .skip(1)
          .map(
            (m) => {
              'coverUrl': m.coverUrl ?? '',
              'coverFallbackUrl': m.coverFallbackUrl ?? '',
              'modelName': m.sceneId,
              'modelUrl': m.modelUrl,
              'posesUrl': m.posesUrl ?? '',
              'tags': _extractTags(m.description, draft.placeName),
            },
          )
          .toList(),
    );

    // Build extra images metadata for multi-model posts
    final extraImagesMeta = draft.models
        .skip(1)
        .map(
          (m) => {
            'coverUrl': m.coverUrl ?? '',
            'coverFallbackUrl': m.coverFallbackUrl ?? '',
            'modelName': m.sceneId,
            'modelUrl': m.modelUrl,
            'posesUrl': m.posesUrl ?? '',
            'tags': _extractTags(m.description, draft.placeName),
          },
        )
        .toList();

    try {
      // Insert and get back the server-assigned id
      final insertResult = await _client.from('community_posts').insert({
        'user_id': _client.auth.currentUser?.id ?? 'local-user',
        'model_asset_id': model.id,
        'model_name': model.sceneId,
        'title': draft.title,
        'caption': draft.caption,
        'place_name': draft.placeName,
        'latitude': draft.latitude,
        'longitude': draft.longitude,
        'cover_image_url': model.coverUrl,
        'metadata': {
          'is_public': draft.isPublic,
          'author_email': _client.auth.currentUser?.email ?? '',
          'likes': <String>[],
          'favorites': <String>[],
          'comments': <Map<String, dynamic>>[],
          'images': extraImagesMeta,
        },
      }).select('id');

      if (insertResult is List && insertResult.isNotEmpty) {
        final realId = insertResult[0]['id']?.toString();
        if (realId != null && realId.isNotEmpty) {
          return optimistic.copyWithRealId(realId);
        }
      }
      return optimistic;
    } catch (_) {
      _localDrafts.insert(0, optimistic);
      return optimistic;
    }
  }

  Future<Map<String, dynamic>> fetchPostMetadata(String postId) async {
    try {
      final response = await _client
          .from('community_posts')
          .select('metadata')
          .eq('id', postId)
          .maybeSingle();
      if (response != null) {
        return _parseMetadata(response['metadata']);
      }
    } catch (_) {}
    return {};
  }

  // ---- Likes / Favorites ----

  Future<Map<String, dynamic>> toggleLike(
    String postId,
    Map<String, dynamic> currentMetadata,
  ) async {
    final uid = currentUserId;
    final likes = List<String>.from(currentMetadata['likes'] ?? []);
    if (likes.contains(uid)) {
      likes.remove(uid);
    } else {
      likes.add(uid);
    }
    final updated = {...currentMetadata, 'likes': likes};
    await _updateMetadata(postId, updated);
    return updated;
  }

  Future<Map<String, dynamic>> toggleFavorite(
    String postId,
    Map<String, dynamic> currentMetadata,
  ) async {
    final uid = currentUserId;
    final favorites = List<String>.from(currentMetadata['favorites'] ?? []);
    if (favorites.contains(uid)) {
      favorites.remove(uid);
    } else {
      favorites.add(uid);
    }
    final updated = {...currentMetadata, 'favorites': favorites};
    await _updateMetadata(postId, updated);
    return updated;
  }

  // ---- Comments ----

  Future<List<CommunityComment>> fetchComments(String postId) async {
    try {
      final response = await _client
          .from('community_posts')
          .select('metadata')
          .eq('id', postId)
          .maybeSingle();
      if (response == null) return [];
      final metadata = _parseMetadata(response['metadata']);
      final raw = List<Map<String, dynamic>>.from(metadata['comments'] ?? []);
      return _parseComments(raw, postId);
    } catch (_) {}
    return [];
  }

  Future<List<CommunityComment>> addComment(
    String postId,
    String text,
    Map<String, dynamic> currentMetadata,
  ) async {
    final uid = currentUserId;
    final userName = _client.auth.currentUser?.email ?? '匿名用户';
    final comments = List<Map<String, dynamic>>.from(
      currentMetadata['comments'] ?? [],
    );
    final newComment = {
      'id': 'c-$uid-${DateTime.now().microsecondsSinceEpoch}',
      'user_id': uid,
      'user_name': userName,
      'text': text,
      'created_at': DateTime.now().toIso8601String(),
    };
    comments.insert(0, newComment);
    final updated = {...currentMetadata, 'comments': comments};
    await _updateMetadata(postId, updated);
    return _parseComments(comments, postId);
  }

  List<CommunityComment> _parseComments(
    List<Map<String, dynamic>> raw,
    String postId,
  ) {
    return raw.map((c) {
      return CommunityComment(
        id: c['id']?.toString() ?? '',
        postId: postId,
        userId: c['user_id']?.toString() ?? '',
        userName: c['user_name']?.toString() ?? '',
        text: c['text']?.toString() ?? '',
        createdAt:
            DateTime.tryParse(c['created_at']?.toString() ?? '') ??
            DateTime.now(),
      );
    }).toList();
  }

  Future<void> setMetadata(String postId, Map<String, dynamic> metadata) async {
    try {
      await _client
          .from('community_posts')
          .update({'metadata': metadata})
          .eq('id', postId);
    } catch (_) {}
  }

  Future<void> recordPostView(String postId) async {
    if (postId.startsWith('local-')) return;
    final metadata = await fetchPostMetadata(postId);
    final uid = currentUserId;
    final viewers = List<String>.from(metadata['viewers'] ?? []);

    if (uid.isNotEmpty && viewers.contains(uid)) {
      return;
    }
    if (uid.isNotEmpty) {
      viewers.add(uid);
    }

    final updated = {
      ...metadata,
      'views': _readViewCount(metadata) + 1,
      'viewers': viewers,
    };
    await setMetadata(postId, updated);
  }

  Future<void> updatePost(
    String postId, {
    String? title,
    String? caption,
    String? placeName,
    double? latitude,
    double? longitude,
    bool? isPublic,
  }) async {
    final update = <String, dynamic>{
      'updated_at': DateTime.now().toIso8601String(),
    };
    if (title != null) update['title'] = title;
    if (caption != null) update['caption'] = caption;
    if (placeName != null) update['place_name'] = placeName;
    if (latitude != null) update['latitude'] = latitude;
    if (longitude != null) update['longitude'] = longitude;

    if (isPublic != null) {
      final metadata = await fetchPostMetadata(postId);
      update['metadata'] = {...metadata, 'is_public': isPublic};
    }

    try {
      await _client.from('community_posts').update(update).eq('id', postId);
    } catch (_) {}
  }

  Future<void> togglePostVisibility(CommunityPost post) async {
    await updatePost(post.id, isPublic: !post.isPublic);
  }

  Future<void> deletePost(String postId) async {
    _localDrafts.removeWhere((post) => post.id == postId);
    if (postId.startsWith('local-')) return;

    try {
      await _client.from('community_posts').delete().eq('id', postId);
    } catch (_) {}
  }

  Future<void> _updateMetadata(
    String postId,
    Map<String, dynamic> metadata,
  ) async {
    try {
      await _client
          .from('community_posts')
          .update({'metadata': metadata})
          .eq('id', postId);
    } catch (_) {}
  }

  // ---- Drafts ----

  Future<void> saveDraft(CommunityDraft draft) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/community_draft.json');
      await file.writeAsString(jsonEncode(draft.toJson()));
    } catch (_) {}
  }

  Future<CommunityDraft> loadDraft() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/community_draft.json');
      if (await file.exists()) {
        final content = await file.readAsString();
        final json = jsonDecode(content) as Map<String, dynamic>;
        return CommunityDraft.fromJson(json);
      }
    } catch (_) {}
    return const CommunityDraft();
  }

  Future<void> clearDraft() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/community_draft.json');
      if (await file.exists()) await file.delete();
    } catch (_) {}
  }

  // ---- Helpers ----

  Map<String, dynamic> _parseMetadata(dynamic raw) {
    if (raw is Map) return Map<String, dynamic>.from(raw);
    if (raw is String) {
      try {
        return Map<String, dynamic>.from(jsonDecode(raw));
      } catch (_) {}
    }
    return {};
  }

  bool _containsUser(dynamic raw, String uid) {
    if (uid.isEmpty || raw is! List) return false;
    return raw.map((value) => value?.toString() ?? '').contains(uid);
  }

  int _readViewCount(Map<String, dynamic> metadata) {
    final views = metadata['views'];
    if (views is num) return views.toInt();
    if (views is List) return views.length;

    final viewCount = metadata['view_count'];
    if (viewCount is num) return viewCount.toInt();

    final viewers = metadata['viewers'];
    if (viewers is List) return viewers.length;

    return 0;
  }

  PreviewImagePaths _resolvePostCover(
    Map<String, dynamic> post,
    Map<String, dynamic> model,
  ) {
    final modelCover = resolvePreviewImagePaths(
      model,
      normalize: _normalizeStorageUrl,
    );
    final explicitCover = post['cover_image_url']?.toString().trim() ?? '';
    if (explicitCover.isEmpty) return modelCover;

    final primary = _normalizeStorageUrl(explicitCover);
    final fallback = modelCover.fallback != primary
        ? modelCover.fallback
        : null;
    return PreviewImagePaths(primary: primary, fallback: fallback);
  }

  String _normalizeStorageUrl(String raw) {
    if (raw.isEmpty) return '';
    const marker = '/storage/v1/object/public/braindance-assets/';
    final idx = raw.indexOf(marker);
    if (idx >= 0) raw = raw.substring(idx + marker.length);
    if (raw.startsWith('http://') || raw.startsWith('https://')) return raw;
    try {
      return _client.storage.from('braindance-assets').getPublicUrl(raw);
    } catch (_) {
      return raw;
    }
  }

  String? _posesUrlFromPath(String? storagePath) {
    if (storagePath == null || storagePath.isEmpty) return null;
    final posesPath = storagePath.replaceAll(
      RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
      'webgl_poses.json',
    );
    if (posesPath == storagePath) return null;
    try {
      return _client.storage.from('braindance-assets').getPublicUrl(posesPath);
    } catch (_) {
      return null;
    }
  }

  String _modelDisplayName(Map<String, dynamic> model) {
    final dn = model['display_name']?.toString() ?? '';
    if (dn.isNotEmpty) return dn;
    final tags = model['tags'];
    if (tags is List) {
      for (final tag in tags) {
        final value = tag?.toString() ?? '';
        if (value.isNotEmpty) return value;
      }
    }
    final sid = model['scene_id']?.toString() ?? '';
    if (sid.isNotEmpty) return sid;
    return textLocalize('community_unnamed_model');
  }

  List<String> _extractTags(String? description, String? placeName) {
    final words = <String>[];
    if (placeName != null && placeName.isNotEmpty) words.add(placeName);
    if (description != null && description.isNotEmpty) {
      final tokens = description
          .split(RegExp(r'[\s,，。.!！？]+'))
          .where((token) => token.trim().length >= 2)
          .take(2);
      words.addAll(tokens);
    }
    if (words.isEmpty) words.add('记忆');
    return words.take(3).toList();
  }
}

final List<CommunityPost> _demoPosts = [
  CommunityPost(
    id: 'demo-post-1',
    title: '清晨刚亮时的断桥',
    caption: '薄雾和湖面的反光被一起留在模型里。',
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
    likeCount: 12,
    commentCount: 3,
  ),
  CommunityPost(
    id: 'demo-post-2',
    title: '东京塔下的夜风',
    caption: '塔体、街道和人流一起收进去了。',
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
    likeCount: 8,
    favoriteCount: 2,
  ),
];
