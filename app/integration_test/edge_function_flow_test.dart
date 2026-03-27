import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Edge Function Flow', () {
    for (final testName in const <String>[
      'BD-IT-EFUNC-001 search-models 返回结构与页面消费一致',
      'BD-IT-EFUNC-002 agent-recall 流式事件完整',
      'BD-IT-EFUNC-003 agent-recall 流式失败后 fallback',
      'BD-IT-EFUNC-004 confirm-text-image 端到端成功',
      'BD-IT-EFUNC-005 text-to-image 返回可用于后续确认',
      'BD-IT-EFUNC-006 Agent preview 与 execute 副作用隔离',
    ]) {
      testWidgets(
        testName,
        (tester) async {
          await launchBrainDanceApp(tester);
        },
        skip: '依赖函数服务、种子数据与部分外部模型服务。',
      );
    }
  });
}
