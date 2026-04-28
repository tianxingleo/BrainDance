import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'support/test_bootstrap.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Recall Flow', () {
    for (final testName in const <String>[
      'BD-IT-RECALL-001 Recall 首次加载模型列表并回填 display_name',
      'BD-IT-RECALL-002 Recall 读取 processing 状态任务',
      'BD-IT-RECALL-003 Recall 搜索走 search-models',
      'BD-IT-RECALL-004 Recall 重命名模型',
      'BD-IT-RECALL-005 Viewer 同名兄弟模型查询',
      'BD-IT-RECALL-006 Recall 删除当前用户云端模型',
      'BD-IT-RECALL-007 Recall 拦截删除他人模型',
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
