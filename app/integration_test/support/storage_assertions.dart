import 'package:flutter_test/flutter_test.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class StorageAssertions {
  const StorageAssertions(this.client);

  final SupabaseClient client;

  Future<void> expectObjectVisible({
    required String bucket,
    required String path,
  }) async {
    final normalizedPath = path.replaceAll('\\', '/');
    final slash = normalizedPath.lastIndexOf('/');
    final folder = slash == -1 ? '' : normalizedPath.substring(0, slash);
    final fileName = slash == -1 ? normalizedPath : normalizedPath.substring(slash + 1);
    final items = await client.storage.from(bucket).list(path: folder);
    final exists = items.any((item) => item.name == fileName);
    expect(exists, isTrue, reason: '预期 $bucket/$normalizedPath 在 Storage 中可见');
  }
}
