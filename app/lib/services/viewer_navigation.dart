import 'dart:async';

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../pages/webgl_viewer.dart';

// ── URL helpers ──────────────────────────────────────────────

/// 将 Storage 内的相对路径转为可访问的公开 URL。
String toPublicUrl(String storagePath) {
  try {
    return Supabase.instance.client.storage
        .from('braindance-assets')
        .getPublicUrl(storagePath);
  } catch (_) {
    return storagePath;
  }
}

/// 根据模型 ply_path 推导同场景的 webgl_poses.json 公开 URL。
String? toPosesUrl(String? plyPath) {
  if (plyPath == null || plyPath.isEmpty) return null;
  try {
    final posesPath = plyPath.replaceAll(
      RegExp(r'point_cloud\.(ply|splat|ksplat)$'),
      'webgl_poses.json',
    );
    if (posesPath == plyPath) return null;
    return Supabase.instance.client.storage
        .from('braindance-assets')
        .getPublicUrl(posesPath);
  } catch (_) {
    return null;
  }
}

// ── Sibling query ────────────────────────────────────────────

/// 根据 sceneId（display_name）自动查询兄弟模型（同名场景的不同时间版本）。
Future<List<Map<String, dynamic>>> _querySiblingModels(String sceneId) async {
  try {
    // 1. 查 processing_tasks 获取当前 sceneId 对应的 display_name
    //    sceneId 参数可能是 display_name，也可能是 scene_id
    final taskRows = await Supabase.instance.client
        .from('processing_tasks')
        .select('scene_id, display_name')
        .or('scene_id.eq.$sceneId,display_name.eq.$sceneId');
    final tasks = List<Map<String, dynamic>>.from(taskRows);
    if (tasks.isEmpty) return const [];

    // 确定 display_name（优先取已有值，否则回退到 sceneId）
    String? displayName;
    for (final t in tasks) {
      final dn = t['display_name']?.toString().trim() ?? '';
      if (dn.isNotEmpty) {
        displayName = dn;
        break;
      }
    }
    displayName ??= sceneId;

    // 2. 用 display_name 反查所有同名 scene_id
    final allTaskRows = await Supabase.instance.client
        .from('processing_tasks')
        .select('scene_id, display_name')
        .eq('display_name', displayName);
    final allTasks = List<Map<String, dynamic>>.from(allTaskRows);
    final siblingSceneIds = allTasks
        .map((t) => t['scene_id']?.toString())
        .where((s) => s != null && s.isNotEmpty)
        .toSet()
        .toList();
    if (siblingSceneIds.isEmpty) return const [];

    // 3. 查 model_assets 获取这些 scene_id 的完整模型数据
    final assetRows = await Supabase.instance.client
        .from('model_assets')
        .select(
          'id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info, created_at',
        )
        .inFilter('scene_id', siblingSceneIds)
        .order('created_at', ascending: false);
    final assets = List<Map<String, dynamic>>.from(assetRows);

    // 4. 合并 display_name
    final displayNameMap = <String, String>{};
    for (final t in allTasks) {
      final sid = t['scene_id']?.toString() ?? '';
      final dn = t['display_name']?.toString() ?? '';
      if (sid.isNotEmpty && dn.isNotEmpty) {
        displayNameMap[sid] = dn;
      }
    }
    for (final m in assets) {
      final sid = m['scene_id']?.toString() ?? '';
      if (displayNameMap.containsKey(sid)) {
        m['display_name'] = displayNameMap[sid];
      }
    }

    return assets;
  } catch (_) {
    return const [];
  }
}

// ── Unified navigation ──────────────────────────────────────

/// 统一的 3DGS Viewer 入口。自动查询兄弟模型并传入 timePeelingModels。
Completer<void>? _viewerLaunchInFlight;

Future<void> openViewer(
  BuildContext context, {
  required String initialModelUrl,
  String? posesUrl,
  required String sceneId,
  List<double>? initialPose,
  String? initialPoseId,
  bool initialMarkerArMode = false,
}) async {
  final inFlight = _viewerLaunchInFlight;
  if (inFlight != null) {
    return inFlight.future;
  }

  final guard = Completer<void>();
  _viewerLaunchInFlight = guard;

  try {
    final siblings = await _querySiblingModels(sceneId);
    if (!context.mounted) return;

    await Navigator.of(context).push(
      PageRouteBuilder(
        transitionDuration: const Duration(milliseconds: 320),
        reverseTransitionDuration: const Duration(milliseconds: 320),
        opaque: true,
        pageBuilder: (context, animation, secondaryAnimation) =>
            WebGLViewerPage(
              initialModelUrl: initialModelUrl,
              posesUrl: posesUrl,
              sceneId: sceneId,
              initialPose: initialPose,
              initialPoseId: initialPoseId,
              initialMarkerArMode: initialMarkerArMode,
              timePeelingModels: siblings,
            ),
        transitionsBuilder: (_, animation, secondaryAnimation, child) {
          final curved = CurvedAnimation(
            parent: animation,
            curve: Curves.easeInOutCubic,
          );
          return SlideTransition(
            position: Tween<Offset>(
              begin: const Offset(0, -1),
              end: Offset.zero,
            ).animate(curved),
            child: child,
          );
        },
      ),
    );
  } finally {
    if (!guard.isCompleted) {
      guard.complete();
    }
    if (identical(_viewerLaunchInFlight, guard)) {
      _viewerLaunchInFlight = null;
    }
  }
}
