import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Task Submission', () {
    for (final testName in const <String>[
      'BD-IT-TASK-001 图片任务提交成功',
      'BD-IT-TASK-002 视频任务提交成功',
      'BD-IT-TASK-003 专用视频提交页 dual chain 写库成功',
      'BD-IT-TASK-004 未登录时先引导登录再提交',
      'BD-IT-TASK-005 任务列表分组与日志解析正确',
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
