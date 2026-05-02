import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Realtime Flow', () {
    for (final testName in const <String>[
      'BD-IT-REALTIME-001 Recall 通过 Realtime 显示 processing 任务',
      'BD-IT-REALTIME-002 Recall 通过 Realtime 移除完成任务',
      'BD-IT-REALTIME-003 全局任务通知去重与清零正确',
    ]) {
      testWidgets(
        testName,
        (tester) async {
          await launchBrainDanceApp(tester);
        },
        skip: true,
      );
    }
  });
}
