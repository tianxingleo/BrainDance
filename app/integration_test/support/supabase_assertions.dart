import 'package:flutter_test/flutter_test.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class SupabaseAssertions {
  const SupabaseAssertions(this.client);

  final SupabaseClient client;

  Future<Map<String, dynamic>?> fetchTaskBySceneId(String sceneId) async {
    return client
        .from('processing_tasks')
        .select('*')
        .eq('scene_id', sceneId)
        .limit(1)
        .maybeSingle();
  }

  Future<Map<String, dynamic>?> fetchModelById(String modelId) async {
    return client
        .from('model_assets')
        .select('*')
        .eq('id', modelId)
        .limit(1)
        .maybeSingle();
  }

  Future<void> expectTaskExists(String sceneId) async {
    final row = await fetchTaskBySceneId(sceneId);
    expect(row, isNotNull, reason: '预期 processing_tasks 中存在 scene_id=$sceneId');
  }

  Future<void> expectModelDeleted(String modelId) async {
    final row = await fetchModelById(modelId);
    expect(row, isNull, reason: '预期 model_assets 中不存在 id=$modelId');
  }
}
