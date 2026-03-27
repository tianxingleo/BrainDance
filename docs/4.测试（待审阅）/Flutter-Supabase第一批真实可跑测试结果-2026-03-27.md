# Flutter-Supabase 第一批真实可跑测试结果

更新时间：2026-03-27

执行分支：`feature/integration-test-skeleton`

执行命令：

```bash
python3 -m unittest tests.test_integration_skeleton -v
```

说明：

- 由于当前终端环境中缺少 `flutter` 与 `dart` 命令，本轮“真实可跑测试”优先覆盖已落地的 `integration_test` 骨架、`pubspec.yaml` 依赖声明以及 `tests/scripts` 的参数校验与 `dry-run` 编排行为。
- 这批测试已经在当前分支真实执行；随着分支继续推进，当前结果已扩展为 20 条可跑测试，结果如下。

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 读取 `app/pubspec.yaml`，检查是否声明 `integration_test` 依赖 | `dev_dependencies` 中存在 `integration_test`，且使用 `sdk: flutter` | 通过，检测到 `integration_test` 依赖声明 | 验证测试基础依赖已接入 |
| 2 | 检查 `app/integration_test/` 与 `support/` 下关键骨架文件是否存在 | 预期测试分组文件与支撑层文件存在，无缺失 | 通过，关键文件全部存在 | 覆盖 `auth/task/recall/realtime/community/edge/local_ai` 与核心 support 文件 |
| 3 | 执行 `tests/scripts/run_flutter_integration_tests.sh --group auth --env admin`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 正确映射到 `integration_test/auth_flow_test.dart`，并以 dry-run 成功退出 | 通过，输出包含 `group=auth`、`env=admin`、`integration_test/auth_flow_test.dart`、`dry-run enabled` | 验证 Flutter 测试分组入口映射正确 |
| 4 | 执行 `tests/scripts/run_flutter_integration_tests.sh --group unknown`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 返回非 0，提示不支持的 group | 通过，脚本返回失败并输出 `unsupported group` | 验证非法参数防御 |
| 5 | 执行 `tests/scripts/mutate_processing_task_status.sh`，不传参数 | 返回非 0，并输出 usage 提示 | 通过，脚本返回失败并输出 `usage:` | 验证 Realtime 状态改写脚本的参数校验 |
| 6 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh`，不传 target | 返回非 0，并输出 usage 提示 | 通过，脚本返回失败并输出 `usage:` | 验证函数冒烟脚本入口参数校验 |
| 7 | 执行 `tests/scripts/run_full_integration_suite.sh --mode local`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 按编排顺序串起 cleanup、bootstrap、seed、Flutter 分组和 Edge smoke，并以 dry-run 成功退出 | 通过，输出包含 `[suite] dry-run enabled`、`[bootstrap] dry-run enabled`、`[seed] profile=minimal`、`[flutter-it] group=auth`、`[edge-smoke] target=search-models` | 验证全量编排入口已能在无真实后端变更条件下跑通流程 |
| 8 | 检查 `tests/fixtures/`、`tests/http/`、`tests/output/.gitkeep`、`app/.env.test.example` 是否存在 | 预期 fixtures、HTTP 请求脚本、输出目录占位文件和测试环境示例文件全部存在 | 通过，相关资产文件均存在 | 验证第二批测试支撑资产已经补齐 |
| 9 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh search-models`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 分发到 `tests/http/search_models_smoke.sh`，生成 dry-run 输出文件并成功退出 | 通过，输出包含 `[http-search-models] dry-run wrote` | 验证 Edge smoke 分发链路已打通 |
| 10 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh unknown-target`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 返回非 0，并输出不支持的 target | 通过，脚本返回失败并输出 `unsupported target` | 验证 Edge smoke 非法 target 防御 |
| 11 | 执行 `tests/scripts/seed_supabase_test_data.sh --profile invalid`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 返回非 0，并输出不支持的 profile | 通过，脚本返回失败并输出 `unsupported profile` | 验证种子数据脚本 profile 校验 |
| 12 | 执行 `tests/scripts/seed_supabase_test_data.sh --profile agent`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 输出 agent 对应 fixture 路径，并在 dry-run 下安全退出 | 通过，输出包含 `supabase_seed_agent.sql` 与 `dry-run enabled` | 验证种子脚本已能按 profile 绑定 fixture |
| 13 | 执行 `tests/scripts/cleanup_supabase_test_data.sh`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 输出清理 fixture 路径，并在 dry-run 下安全退出 | 通过，输出包含 `cleanup_integration.sql` 与 `dry-run enabled` | 验证清理脚本已绑定统一 fixture 入口 |
| 14 | 执行 `tests/http/search_models_smoke.sh <临时输出文件>`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 成功写出 dry-run JSON 文件，内容标识 target 为 `search-models` | 通过，脚本写出输出文件且内容正确 | 验证 search-models HTTP 冒烟脚本可独立执行 |
| 15 | 执行 `tests/http/confirm_text_image_smoke.sh <临时输出文件>`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 成功写出 dry-run JSON 文件，内容标识 target 为 `confirm-text-image` | 通过，脚本写出输出文件且内容正确 | 验证 confirm-text-image HTTP 冒烟脚本可独立执行 |
| 16 | 执行 `tests/http/agent_recall_stream_smoke.sh <临时输出文件>`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 成功写出 dry-run JSONL 文件，内容标识 target 为 `agent-recall` | 通过，脚本写出输出文件且内容正确 | 验证 agent-recall 流式冒烟脚本可独立执行 |
| 17 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh confirm-text-image`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 分发到 `tests/http/confirm_text_image_smoke.sh` 并成功退出 | 通过，输出包含 `[http-confirm-text-image] dry-run wrote` | 验证 confirm-text-image 分发链路 |
| 18 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh agent-recall`，并开启 `BRAINDANCE_IT_DRY_RUN=1` | 分发到 `tests/http/agent_recall_stream_smoke.sh` 并成功退出 | 通过，输出包含 `[http-agent-recall] dry-run wrote` | 验证 agent-recall 分发链路 |
| 19 | 执行 `tests/scripts/bootstrap_supabase_test_env.sh`，不启用 dry-run | 本地 Supabase 可连接，且 `braindance-assets`、`braindance-models` bucket 被确保存在 | 通过，脚本真实执行并完成 bucket 确认 | 验证 bootstrap 已进入真实本地执行分支 |
| 20 | 执行 `cleanup -> seed minimal -> 查库 -> cleanup`，不启用 dry-run | `processing_tasks`、`model_assets`、`community_posts` 与 `storage.objects` 在 seed 后存在，在 cleanup 后清空 | 通过，真实本地 DB/Storage 闭环执行成功 | 验证最小真实种子与清理链路已经打通 |

## 汇总

| 指标 | 结果 |
| --- | --- |
| 总测试数 | 20 |
| 通过 | 20 |
| 失败 | 0 |
| 结论 | 当前分支上基于骨架、fixtures、HTTP 冒烟脚本、脚本分发、dry-run 编排以及最小真实本地 DB/Storage 闭环的可跑测试全部通过 |

## 限制

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `flutter pub get`、`flutter test`、`dart format` | 运行 Flutter/Dart 级测试与格式化 | 当前环境缺少 `flutter`、`dart` 命令，未执行 | 下一批若要实现真正的 Flutter 页面级集成测试，需要先补齐 Flutter SDK 环境 |
