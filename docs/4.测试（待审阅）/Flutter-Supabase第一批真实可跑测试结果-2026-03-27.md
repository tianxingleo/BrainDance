# Flutter-Supabase 第一批真实可跑测试结果

更新时间：2026-03-27

执行分支：`feature/integration-test-skeleton`

执行命令：

```bash
python3 -m unittest tests.test_integration_skeleton -v
```

说明：

- 由于当前终端环境中缺少 `flutter` 与 `dart` 命令，本轮“真实可跑测试”优先覆盖已落地的 `integration_test` 骨架、`pubspec.yaml` 依赖声明以及 `tests/scripts` 的参数校验与 `dry-run` 编排行为。
- 这批测试已经在当前分支真实执行；随着分支继续推进，当前结果已扩展为 28 条可跑测试。
- 按本轮输出规范，以下结果改为“每个测试用例一张独立表”，且每个用例内的“具体步骤”均从 `1` 开始重新编号。

## 用例 01：`pubspec.yaml` 集成测试依赖声明校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 打开 `app/pubspec.yaml` | 文件可读取 | 通过，文件可正常读取 | 验证测试入口配置文件存在 |
| 2 | 检查 `dev_dependencies` 下是否包含 `integration_test` | 存在 `integration_test` 依赖项 | 通过，检测到 `integration_test:` | 验证集成测试依赖已接入 |
| 3 | 检查 `integration_test` 是否使用 `sdk: flutter` | 依赖来源为 Flutter SDK | 通过，检测到 `sdk: flutter` | 验证依赖声明形式正确 |

## 用例 02：`integration_test` 骨架文件存在性校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 检查 `app/integration_test/` 目录是否存在 | 目录存在 | 通过，目录存在 | 验证 Flutter 集成测试目录已创建 |
| 2 | 检查分组测试文件 | `auth/task/recall/realtime/community/edge/local_ai` 对应用例文件全部存在 | 通过，分组测试文件全部存在 | 覆盖主要业务测试分组 |
| 3 | 检查 `support/` 支撑文件 | `test_bootstrap.dart`、`test_env.dart`、`supabase_assertions.dart` 全部存在 | 通过，support 文件齐全 | 验证支撑层骨架完整 |

## 用例 03：Flutter 分组入口 `auth` dry-run 映射校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_flutter_integration_tests.sh --group auth --env admin`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功进入 dry-run | 通过，命令成功退出 | 验证入口可执行 |
| 2 | 检查输出中的 group 与 env | 输出包含 `group=auth`、`env=admin` | 通过，输出匹配 | 验证参数传递正确 |
| 3 | 检查目标测试文件映射 | 输出包含 `integration_test/auth_flow_test.dart` | 通过，映射到正确文件 | 验证分组映射正确 |
| 4 | 检查 dry-run 标识 | 输出包含 `dry-run enabled` | 通过，输出匹配 | 验证未触发真实 Flutter 执行 |

## 用例 04：Flutter 分组入口非法 group 防御校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_flutter_integration_tests.sh --group unknown`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本返回非 0 | 通过，脚本执行失败 | 验证非法 group 被拒绝 |
| 2 | 检查错误输出 | 输出包含 `unsupported group` | 通过，错误提示匹配 | 验证防御信息明确 |

## 用例 05：`mutate_processing_task_status.sh` 参数校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 直接执行 `tests/scripts/mutate_processing_task_status.sh`，不传参数 | 脚本返回非 0 | 通过，脚本执行失败 | 验证必填参数生效 |
| 2 | 检查错误输出 | 输出包含 `usage:` | 通过，usage 已输出 | 验证脚本参数说明存在 |

## 用例 06：`run_edge_function_smoke_tests.sh` 参数校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 直接执行 `tests/scripts/run_edge_function_smoke_tests.sh`，不传 target | 脚本返回非 0 | 通过，脚本执行失败 | 验证 target 为必填参数 |
| 2 | 检查错误输出 | 输出包含 `usage:` | 通过，usage 已输出 | 验证入口脚本提示完整 |

## 用例 07：全量编排入口 dry-run 顺序校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_full_integration_suite.sh --mode local`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功进入 dry-run | 通过，命令成功退出 | 验证全量入口可执行 |
| 2 | 检查 bootstrap 与 seed 输出 | 输出包含 `[bootstrap] dry-run enabled`、`[seed] profile=minimal` | 通过，输出匹配 | 验证前置阶段已编排 |
| 3 | 检查 Flutter 与 Edge 输出 | 输出包含 `[flutter-it] group=auth`、`[edge-smoke] target=search-models` | 通过，输出匹配 | 验证后续阶段已编排 |
| 4 | 检查 suite 级 dry-run 标记 | 输出包含 `[suite] dry-run enabled` | 通过，输出匹配 | 验证整条链路未落真实变更 |

## 用例 08：fixtures 与 HTTP 支撑资产存在性校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 检查 `tests/fixtures/` 目录中的 SQL 文件 | `minimal/realtime/agent/cleanup` 四类 fixture 存在 | 通过，相关文件均存在 | 验证 SQL 支撑资产已补齐 |
| 2 | 检查 `tests/http/` 中的 HTTP 冒烟脚本 | `search_models`、`confirm_text_image`、`agent_recall` 三类脚本存在 | 通过，相关文件均存在 | 验证 HTTP 冒烟脚本齐备 |
| 3 | 检查 `tests/output/.gitkeep` 与 `app/.env.test.example` | 占位文件与示例环境文件存在 | 通过，相关文件均存在 | 验证输出目录与环境模板存在 |

## 用例 09：`search-models` 分发入口 dry-run 校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh search-models`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 成功分发到 `search_models_smoke.sh` | 通过，命令成功退出 | 验证分发入口可达 |
| 2 | 检查输出文件写入提示 | 输出包含 `[http-search-models] dry-run wrote` | 通过，输出匹配 | 验证目标脚本被调用 |

## 用例 10：Edge smoke 非法 target 防御校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh unknown-target`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本返回非 0 | 通过，脚本执行失败 | 验证非法 target 被拒绝 |
| 2 | 检查错误输出 | 输出包含 `unsupported target` | 通过，错误提示匹配 | 验证防御信息明确 |

## 用例 11：种子脚本非法 profile 防御校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/seed_supabase_test_data.sh --profile invalid`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本返回非 0 | 通过，脚本执行失败 | 验证 profile 校验生效 |
| 2 | 检查错误输出 | 输出包含 `unsupported profile` | 通过，错误提示匹配 | 验证非法 profile 被拦截 |

## 用例 12：种子脚本 `agent` profile dry-run 绑定校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/seed_supabase_test_data.sh --profile agent`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功退出 | 通过，命令成功退出 | 验证 `agent` profile 可识别 |
| 2 | 检查 fixture 路径输出 | 输出包含 `supabase_seed_agent.sql` | 通过，输出匹配 | 验证 fixture 绑定正确 |
| 3 | 检查 dry-run 标识 | 输出包含 `dry-run enabled` | 通过，输出匹配 | 验证未落真实数据 |

## 用例 13：清理脚本 dry-run 绑定校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/cleanup_supabase_test_data.sh`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功退出 | 通过，命令成功退出 | 验证清理入口可执行 |
| 2 | 检查 fixture 路径输出 | 输出包含 `cleanup_integration.sql` | 通过，输出匹配 | 验证清理 fixture 绑定正确 |
| 3 | 检查 dry-run 标识 | 输出包含 `dry-run enabled` | 通过，输出匹配 | 验证未落真实删除 |

## 用例 14：`search-models` HTTP 脚本 dry-run 校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/http/search_models_smoke.sh <临时输出文件>`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功写出输出文件 | 通过，输出文件生成成功 | 验证脚本可独立执行 |
| 2 | 检查输出文件内容 | JSON 中包含 `target=search-models` | 通过，内容匹配 | 验证 dry-run 输出结构正确 |

## 用例 15：`confirm-text-image` HTTP 脚本 dry-run 校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/http/confirm_text_image_smoke.sh <临时输出文件>`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功写出输出文件 | 通过，输出文件生成成功 | 验证脚本可独立执行 |
| 2 | 检查输出文件内容 | JSON 中包含 `target=confirm-text-image` | 通过，内容匹配 | 验证 dry-run 输出结构正确 |

## 用例 16：`agent-recall` HTTP 脚本 dry-run 校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/http/agent_recall_stream_smoke.sh <临时输出文件>`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 脚本成功写出输出文件 | 通过，输出文件生成成功 | 验证脚本可独立执行 |
| 2 | 检查输出文件内容 | JSONL 中包含 `target=agent-recall` | 通过，内容匹配 | 验证 dry-run 流式输出结构正确 |

## 用例 17：`confirm-text-image` 分发入口 dry-run 校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh confirm-text-image`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 成功分发到 `confirm_text_image_smoke.sh` | 通过，命令成功退出 | 验证分发入口可达 |
| 2 | 检查输出提示 | 输出包含 `[http-confirm-text-image] dry-run wrote` | 通过，输出匹配 | 验证目标脚本已执行 |

## 用例 18：`agent-recall` 分发入口 dry-run 校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh agent-recall`，并设置 `BRAINDANCE_IT_DRY_RUN=1` | 成功分发到 `agent_recall_stream_smoke.sh` | 通过，命令成功退出 | 验证分发入口可达 |
| 2 | 检查输出提示 | 输出包含 `[http-agent-recall] dry-run wrote` | 通过，输出匹配 | 验证目标脚本已执行 |

## 用例 19：本地 Supabase bootstrap 真实执行校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/bootstrap_supabase_test_env.sh` | 本地 Supabase 栈可连接 | 通过，脚本真实执行成功 | 验证本地 Supabase 环境可用 |
| 2 | 检查 bucket 创建结果 | `braindance-assets` 与 `braindance-models` 均存在 | 通过，bucket 确认完成 | 验证 bootstrap 已进入真实执行 |

## 用例 20：最小 DB/Storage 闭环真实校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `cleanup` | 测试数据被预清理 | 通过，清理成功 | 验证起始环境干净 |
| 2 | 执行 `seed minimal` | 插入最小任务、资产、社区帖子与存储文件 | 通过，种子成功写入 | 验证最小种子可落库 |
| 3 | 查库检查 `processing_tasks`、`model_assets`、`community_posts`、`storage.objects` | 四类数据均存在 | 通过，数据数量符合预期 | 验证最小闭环种子完整 |
| 4 | 再次执行 `cleanup` 并复查 | 四类测试数据全部清空 | 通过，清理后数量归零 | 验证最小闭环可重复执行 |

## 用例 21：`search-models` 本地 HTTP 冒烟真实校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/http/search_models_smoke.sh <临时输出文件>` | 请求本地 `search-models` 函数 | 通过，脚本成功执行 | 验证真实 HTTP 链路可达 |
| 2 | 检查返回状态 | 返回 `400` 校验错误 | 通过，脚本输出 `status=400` | 验证缺少 `query` 时函数防御正确 |
| 3 | 检查响应体 | 输出文件中包含 `success=false` 与 `query` 校验错误 | 通过，内容匹配 | 验证真实错误体符合预期 |

## 用例 22：`search-models` 分发入口真实校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh search-models` | 成功分发到真实 `search_models_smoke.sh` | 通过，命令成功退出 | 验证分发入口与真实脚本已接通 |
| 2 | 检查输出提示 | 输出包含 `[http-search-models] wrote` | 通过，输出匹配 | 验证目标脚本已真实执行 |
| 3 | 检查响应文件 | 文件中存在真实 `400` 错误体 | 通过，内容匹配 | 验证分发后的真实结果正确 |

## 用例 23：Realtime 任务状态改写真实闭环校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `cleanup` | 起始环境干净 | 通过，清理成功 | 验证前置环境 |
| 2 | 执行 `seed realtime` | 插入待测 `processing_tasks` 记录 | 通过，种子成功写入 | 验证 Realtime 样例可落库 |
| 3 | 执行 `mutate_processing_task_status.sh` | 指定任务状态更新为 `processing` 且追加日志 | 通过，脚本真实更新成功 | 验证状态改写脚本已接通 |
| 4 | 查库验证 `status` 与 `logs` | 状态从 `pending` 变为 `processing`，日志数量从 `0` 增为 `1` | 通过，查询结果符合预期 | 验证 DB 真实变更正确 |
| 5 | 再次执行 `cleanup` | 测试记录清空 | 通过，清理完成 | 验证闭环结束状态正确 |

## 用例 24：`agent-recall` NDJSON 流式真实校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/http/agent_recall_stream_smoke.sh <临时输出文件>` | 以 NDJSON 方式请求本地 `agent-recall?stream=1` | 通过，脚本成功执行 | 验证 NDJSON 流式链路可达 |
| 2 | 检查事件类型 | 输出包含 `ping`、`status`、`done` 事件 | 通过，事件齐全 | 验证流式事件序列完整 |
| 3 | 检查回答内容 | 输出包含“BrainDance 的空间记忆智能管理助手” | 通过，内容匹配 | 验证稳定通用回答返回正确 |

## 用例 25：`agent-recall` 分发入口真实流式校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/scripts/run_edge_function_smoke_tests.sh agent-recall` | 成功分发到真实 `agent_recall_stream_smoke.sh` | 通过，命令成功退出 | 验证分发入口与流式脚本已接通 |
| 2 | 检查输出提示 | 输出包含 `[http-agent-recall] wrote` | 通过，输出匹配 | 验证目标脚本已执行 |
| 3 | 检查事件流文件 | 文件中存在 `done` 事件与最终载荷 | 通过，内容匹配 | 验证分发后的真实流式结果正确 |

## 用例 26：`agent-recall` SSE 流式协议真实校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `tests/http/agent_recall_stream_smoke.sh <临时输出文件>`，并设置 `BRAINDANCE_IT_AGENT_STREAM_FORMAT=sse` | 以 SSE 方式请求本地 `agent-recall?stream=1` | 通过，脚本成功执行 | 验证 SSE 流式协议可达 |
| 2 | 检查事件格式 | 输出包含 `event: ping`、`event: status`、`event: done` | 通过，事件格式匹配 | 验证 SSE 协议正确 |
| 3 | 检查回答内容 | 输出包含“BrainDance 的空间记忆智能管理助手” | 通过，内容匹配 | 验证 SSE 下回答稳定 |

## 用例 27：`agent-recall` preview 防误写真值校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 直接请求 `agent-recall` JSON 接口，传入 `query=确认执行`、`executionMode=preview` 与 `sessionState.lastOperationPreview` | 命中 preview 保护分支 | 通过，HTTP `200` 返回成功响应 | 验证接口可处理该保护场景 |
| 2 | 检查响应模式与结论 | 返回 `mode=asset_metadata`，并提示“当前请求还是 preview 模式” | 通过，字段与提示匹配 | 验证防误写逻辑命中 |
| 3 | 检查响应分辨率 | `response_resolution.kind=tool_success` | 通过，字段匹配 | 验证保护分支的标准化响应正确 |

## 用例 28：`agent-recall` 多轮续聊执行闭环真实校验

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `seed minimal` | 生成待操作模型资产样例 | 通过，最小种子成功写入 | 验证续聊测试前置环境 |
| 2 | 第一轮发送“把这个模型改名为宿舍-归档版”，模式为 `preview` | 返回 `follow_up.confirm_write` 与 `session_state.lastOperationPreview` | 通过，响应命中 `rename_model_asset` 预览 | 验证多轮续聊第一轮状态承接正确 |
| 3 | 第二轮携带上一轮 `conversationSummary + sessionState` 发送“确认执行”，模式为 `execute` | 正式执行改名并返回成功响应 | 通过，响应中包含“已正式执行改名” | 验证多轮续聊第二轮正式执行成功 |
| 4 | 查库验证 `model_assets.display_name` | `display_name` 更新为“宿舍-归档版” | 通过，数据库结果匹配 | 验证真实写库成功 |
| 5 | 执行 `cleanup` | 测试数据清空 | 通过，清理完成 | 验证闭环结束状态正确 |

## 汇总

| 指标 | 结果 |
| --- | --- |
| 总测试数 | 28 |
| 通过 | 28 |
| 失败 | 0 |
| 结论 | 当前分支上基于骨架、fixtures、HTTP 冒烟脚本、脚本分发、dry-run 编排，以及真实本地 DB/Storage/Edge Function/任务状态改写/agent-recall NDJSON 与 SSE 流式链路、preview 防误写保护、多轮续聊执行闭环的可跑测试全部通过 |

## 限制

| 具体步骤（即编号从1开始） | 输入 | 期望输出 | 实际输出 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 执行 `flutter pub get`、`flutter test`、`dart format` | 运行 Flutter/Dart 级测试与格式化 | 当前环境缺少 `flutter`、`dart` 命令，未执行 | 下一批若要实现真正的 Flutter 页面级集成测试，需要先补齐 Flutter SDK 环境 |
