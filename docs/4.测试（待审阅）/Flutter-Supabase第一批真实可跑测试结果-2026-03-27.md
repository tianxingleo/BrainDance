# Flutter-Supabase 第一批真实可跑测试结果

更新时间：2026-03-27

执行分支：`feature/integration-test-skeleton`

执行命令：

```bash
python3 -m unittest tests.test_integration_skeleton -v
```

说明：

- 由于当前终端环境中缺少 `flutter` 与 `dart` 命令，本轮“真实可跑测试”优先覆盖已落地的 `integration_test` 骨架、`pubspec.yaml` 依赖声明以及 `tests/scripts` 的参数校验与 `dry-run` 编排行为。
- 这批测试已经在当前分支真实执行，结果如下。

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 读取 `app/pubspec.yaml`，检查是否声明 `integration_test` 依赖 | `dev_dependencies` 中存在 `integration_test`，且使用 `sdk: flutter` | 通过，检测到 `integration_test` 依赖声明 | 验证测试基础依赖已接入 |
| 2 | 检查 `app/integration_test/` 与 `support/` 下关键骨架文件是否存在 | 预期测试分组文件与支撑层文件存在，无缺失 | 通过，关键文件全部存在 | 覆盖 `auth/task/recall/realtime/community/edge/local_ai` 与核心 support 文件 |
| 3 | 执行 `tests/scripts/run_flutter_integration_tests.sh --group auth --env admin`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 正确映射到 `integration_test/auth_flow_test.dart`，并以 dry-run 成功退出 | 通过，输出包含 `group=auth`、`env=admin`、`integration_test/auth_flow_test.dart`、`dry-run enabled` | 验证 Flutter 测试分组入口映射正确 |
| 4 | 执行 `tests/scripts/run_flutter_integration_tests.sh --group unknown`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 返回非 0，提示不支持的 group | 通过，脚本返回失败并输出 `unsupported group` | 验证非法参数防御 |
| 5 | 执行 `tests/scripts/mutate_processing_task_status.sh`，不传参数 | 返回非 0，并输出 usage 提示 | 通过，脚本返回失败并输出 `usage:` | 验证 Realtime 状态改写脚本的参数校验 |
| 6 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh`，不传 target | 返回非 0，并输出 usage 提示 | 通过，脚本返回失败并输出 `usage:` | 验证函数冒烟脚本入口参数校验 |
| 7 | 执行 `tests/scripts/run_full_integration_suite.sh --mode local`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 按编排顺序串起 cleanup、bootstrap、seed、Flutter 分组和 Edge smoke，并以 dry-run 成功退出 | 通过，输出包含 `[suite] dry-run enabled`、`[bootstrap] dry-run enabled`、`[seed] profile=minimal`、`[flutter-it] group=auth`、`[edge-smoke] target=search-models` | 验证全量编排入口已能在无真实后端变更条件下跑通流程 |

## 汇总

| 指标 | 结果 |
| --- | --- |
| 总测试数 | 7 |
| 通过 | 7 |
| 失败 | 0 |
| 结论 | 第一批基于骨架与脚本入口的真实可跑测试全部通过 |

## 限制

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `flutter pub get`、`flutter test`、`dart format` | 运行 Flutter/Dart 级测试与格式化 | 当前环境缺少 `flutter`、`dart` 命令，未执行 | 下一批若要实现真正的 Flutter 页面级集成测试，需要先补齐 Flutter SDK 环境 |
