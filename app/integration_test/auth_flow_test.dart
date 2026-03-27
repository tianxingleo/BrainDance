import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Auth Flow', () {
    testWidgets(
      'BD-IT-AUTH-001 普通用户登录成功',
      (tester) async {
        await launchBrainDanceApp(tester);
        // TODO: 填充真实登录动作与断言。
      },
      skip: '集成测试骨架已建立，待补充真实步骤与断言。',
    );

    testWidgets(
      'BD-IT-AUTH-002 错误密码登录失败',
      (tester) async {
        await launchBrainDanceApp(tester);
      },
      skip: '集成测试骨架已建立，待补充真实步骤与断言。',
    );

    testWidgets(
      'BD-IT-AUTH-003 Admin 模式直入首页',
      (tester) async {
        await launchBrainDanceApp(tester);
      },
      skip: '需要 admin 环境变量与独立运行口径。',
    );

    testWidgets(
      'BD-IT-AUTH-004 登出后任务页清空',
      (tester) async {
        await launchBrainDanceApp(tester);
      },
      skip: '依赖真实登录态和任务数据。',
    );
  });
}
