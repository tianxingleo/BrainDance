import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Local AI Catalog', () {
    for (final testName in const <String>[
      'BD-IT-LOCALAI-001 catalog 模式读取本地模型列表',
      'BD-IT-LOCALAI-002 catalog 缺失时回退 bucket 扫描',
    ]) {
      testWidgets(
        testName,
        (tester) async {
          await launchBrainDanceApp(tester);
        },
        skip: '依赖 braindance-models bucket 测试布局。',
      );
    }
  });
}
