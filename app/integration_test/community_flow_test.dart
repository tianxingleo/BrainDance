import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Community Flow', () {
    for (final testName in const <String>[
      'BD-IT-COMM-001 社区帖子读取与联表映射正确',
      'BD-IT-COMM-002 社区发帖成功',
      'BD-IT-COMM-003 社区发帖失败时本地草稿回退',
    ]) {
      testWidgets(
        testName,
        (tester) async {
          await launchBrainDanceApp(tester);
        },
        skip: '集成测试骨架已建立，待补充真实步骤与断言。',
      );
    }
  });
}
