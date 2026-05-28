import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:braindance/services/local_model_catalog_service.dart';

void main() {
  setUpAll(() {
    dotenv.testLoad(
      fileInput:
          'SUPABASE_URL=http://127.0.0.1:54321\n'
          'SUPABASE_ANON_KEY=test-key\n',
    );
  });

  group('LocalModelCatalogService', () {
    test('解析仅包含 path/type 的 GGUF catalog 候选', () {
      const service = LocalModelCatalogService();
      final items = service.parseCatalogForTesting({
        'candidates': [
          {
            'path': 'releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf',
            'type': 'gguf',
            'notes': 'mobile release',
          },
          {
            'path': 'releases/qwen3-0.6b-braindance-q5-k-m.gguf',
            'type': 'gguf',
            'notes': 'small mobile release',
          },
          {
            'prefix': 'releases/qwen3-0.6b-braindance-round1/',
            'type': 'lora-release',
            'notes': 'adapter only',
          },
        ],
      });

      expect(items, hasLength(2));
      expect(
        items.map((item) => item.fileName),
        contains('qwen3-1.7b-braindance-q5-k-m-imatrix.gguf'),
      );
      expect(
        items.map((item) => item.fileName),
        contains('qwen3-0.6b-braindance-q5-k-m.gguf'),
      );
      expect(items.any((item) => item.downloadUrl.contains('round1')), isFalse);
    });
  });
}
