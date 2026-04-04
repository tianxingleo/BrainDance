# Flutter-Supabase 集成测试脚本设计与执行方案

更新时间：2026-03-27

## 1. 文档目标

本文对应 [Flutter-Supabase详细集成测试用例表-2026-03-27.md](/home/ltx/projects/BrainDance/docs/4.测试（待审阅）/Flutter-Supabase详细集成测试用例表-2026-03-27.md)，用于设计这些测试的脚本组织方式、测试环境、数据准备、执行顺序和清理方案。

目标不是只描述“怎么测”，而是把后续要落地的测试资产结构也提前定好，减少实现时的分歧。

## 2. 设计原则

## 2.1 分层执行

所有测试按 4 层组织：

1. `测试环境与种子层`
2. `Flutter 集成测试层`
3. `函数协议与联调层`
4. `清理与回收层`

这样拆的原因是：

- Flutter 适合验证页面交互与状态变化
- Shell / SQL / HTTP 脚本更适合做建数、改数、查数
- Agent 与搜索函数需要协议级断言，单靠 UI 测试不够精确

## 2.2 测试与生产逻辑解耦

不建议直接在页面代码里塞测试分支。应把测试控制能力放在：

- `.env.test`
- `integration_test/support/*`
- `tests/scripts/*`
- `tests/fixtures/*`

## 2.3 数据可重复

所有测试数据都必须满足：

- 可重复创建
- 可按前缀清理
- 不依赖手工点选 Studio
- 不污染开发数据

建议所有测试实体统一使用前缀：

```text
it_20260327_*
```

例如：

- `scene_id = it_20260327_scene_completed_001`
- `display_name = IT-杭州西湖-001`
- `collection title = IT-专题-001`

## 2.4 把外部不稳定依赖隔离出去

对 `text-to-image` 和部分 Agent 上游模型调用，建议区分：

- `default integration`
- `extended online regression`

默认集成测试只验证 Supabase 边界和 Flutter 消费逻辑，尽量不把每次 CI 都绑死在外部模型平台上。

## 3. 推荐目录结构

建议新增如下测试目录：

```text
app/
  integration_test/
    auth_flow_test.dart
    task_submission_test.dart
    recall_flow_test.dart
    realtime_flow_test.dart
    community_flow_test.dart
    edge_function_flow_test.dart
    local_ai_catalog_test.dart
    support/
      test_env.dart
      test_accounts.dart
      test_driver.dart
      test_ids.dart
      supabase_assertions.dart
      storage_assertions.dart
      ui_robot.dart
      waiters.dart

tests/
  fixtures/
    supabase_seed_minimal.sql
    supabase_seed_realtime.sql
    supabase_seed_agent.sql
    cleanup_integration.sql
  scripts/
    bootstrap_supabase_test_env.sh
    seed_supabase_test_data.sh
    cleanup_supabase_test_data.sh
    mutate_processing_task_status.sh
    run_flutter_integration_tests.sh
    run_edge_function_smoke_tests.sh
    run_full_integration_suite.sh
  http/
    search_models_smoke.sh
    confirm_text_image_smoke.sh
    agent_recall_stream_smoke.sh
```

## 4. 各类脚本职责设计

## 4.1 环境引导脚本

脚本：`tests/scripts/bootstrap_supabase_test_env.sh`

职责：

- 启动本地 Supabase
- 检查必要 bucket 是否存在
- 检查关键函数是否可访问
- 校验 `.env.test` 配置是否齐全

建议流程：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd supabase
supabase start

cd ..
tests/scripts/seed_supabase_test_data.sh --phase bootstrap
```

必须检查：

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `braindance-assets`
- `braindance-models`

## 4.2 种子数据脚本

脚本：`tests/scripts/seed_supabase_test_data.sh`

职责：

- 创建测试用户
- 插入 `processing_tasks`
- 插入 `model_assets`
- 插入 `community_posts`
- 插入 `memory_poses`
- 插入 `memory_collections`
- 上传 Storage 测试文件

建议入参：

```bash
tests/scripts/seed_supabase_test_data.sh --profile minimal
tests/scripts/seed_supabase_test_data.sh --profile realtime
tests/scripts/seed_supabase_test_data.sh --profile agent
tests/scripts/seed_supabase_test_data.sh --phase bootstrap
```

推荐实现方式：

- 优先使用 `psql` 执行 SQL fixture
- Storage 上传用 `curl` 或 `supabase storage` 相关命令
- 用户创建可通过测试专用 service-role 脚本完成

推荐拆分 profile：

- `minimal`：Auth + Task + Recall 基础链路
- `realtime`：增加可变状态任务与日志
- `agent`：增加 `memory_poses`、专题、关系表数据

## 4.3 清理脚本

脚本：`tests/scripts/cleanup_supabase_test_data.sh`

职责：

- 删除所有 `it_*` 前缀测试数据
- 清理 bucket 中的测试目录
- 恢复可能被改名或改元数据的测试模型

建议顺序：

1. 删除 `community_posts`
2. 删除 `memory_collection_items`
3. 删除 `memory_collections`
4. 删除 `memory_poses`
5. 删除 `related_model_links`
6. 删除 `model_assets`
7. 删除 `processing_tasks`
8. 删除 Storage 文件

原因：

- 需要先删除外键或引用下游，再删上游

## 4.4 状态变更脚本

脚本：`tests/scripts/mutate_processing_task_status.sh`

职责：

- 在 Realtime 测试中异步修改 `processing_tasks.status`
- 可同时写入 `logs`

建议入参：

```bash
tests/scripts/mutate_processing_task_status.sh \
  --task-id xxx \
  --status processing \
  --append-log "开始处理"
```

这类脚本是 `BD-IT-REALTIME-001`、`BD-IT-REALTIME-002`、`BD-IT-REALTIME-003` 的关键辅助工具。

## 4.5 Edge Function 冒烟脚本

这些脚本用于把函数自身协议先跑通，再交给 Flutter 页面消费。

### `tests/http/search_models_smoke.sh`

职责：

- 直接请求 `search-models`
- 保存原始 JSON 返回

### `tests/http/confirm_text_image_smoke.sh`

职责：

- 直接请求 `confirm-text-image`
- 验证返回 `scene_id`

### `tests/http/agent_recall_stream_smoke.sh`

职责：

- 直接以 NDJSON 或 SSE 方式请求 `agent-recall`
- 保存全部事件流到文件

建议输出：

```text
tests/output/agent_recall_stream.jsonl
tests/output/search_models_response.json
tests/output/confirm_text_image_response.json
```

## 5. Flutter 集成测试文件拆分建议

## 5.1 `auth_flow_test.dart`

覆盖用例：

- `BD-IT-AUTH-001`
- `BD-IT-AUTH-002`
- `BD-IT-AUTH-003`
- `BD-IT-AUTH-004`

建议结构：

```dart
group('Auth Flow', () {
  testWidgets('BD-IT-AUTH-001 普通用户登录成功', ...);
  testWidgets('BD-IT-AUTH-002 错误密码登录失败', ...);
  testWidgets('BD-IT-AUTH-003 Admin 模式直入首页', ...);
  testWidgets('BD-IT-AUTH-004 登出后任务页清空', ...);
});
```

## 5.2 `task_submission_test.dart`

覆盖用例：

- `BD-IT-TASK-001`
- `BD-IT-TASK-002`
- `BD-IT-TASK-003`
- `BD-IT-TASK-004`
- `BD-IT-TASK-005`

建议这里补一个 `TestDriver` 能力，专门负责：

- 选择图片/视频测试资源
- 等待上传进度结束
- 从数据库查询最新 `scene_id`

## 5.3 `recall_flow_test.dart`

覆盖用例：

- `BD-IT-RECALL-001`
- `BD-IT-RECALL-002`
- `BD-IT-RECALL-003`
- `BD-IT-RECALL-004`
- `BD-IT-RECALL-005`
- `BD-IT-RECALL-006`
- `BD-IT-RECALL-007`

建议把以下动作封成 `ui_robot.dart`：

- 进入 Recall
- 搜索模型
- 打开模型详情
- 打开重命名弹窗
- 删除模型
- 点击已完成任务并进入 Viewer

## 5.4 `realtime_flow_test.dart`

覆盖用例：

- `BD-IT-REALTIME-001`
- `BD-IT-REALTIME-002`
- `BD-IT-REALTIME-003`

这组测试的关键不在按钮点击，而在“前台 Flutter 正在等待，后台脚本异步改库”。

建议流程：

1. Flutter 页面进入监听状态
2. 测试驱动调用 `mutate_processing_task_status.sh`
3. Flutter 端 `pump` 并等待 UI 变化
4. 数据库回查确认状态变化已生效

## 5.5 `community_flow_test.dart`

覆盖用例：

- `BD-IT-COMM-001`
- `BD-IT-COMM-002`
- `BD-IT-COMM-003`

这里建议同时提供数据库校验和 UI 校验，不要只看页面卡片数量。

## 5.6 `edge_function_flow_test.dart`

覆盖用例：

- `BD-IT-EFUNC-001`
- `BD-IT-EFUNC-002`
- `BD-IT-EFUNC-003`
- `BD-IT-EFUNC-004`
- `BD-IT-EFUNC-005`
- `BD-IT-EFUNC-006`

建议拆成两类断言：

- Flutter UI 消费结果
- 测试辅助层直接请求函数看原始返回

## 5.7 `local_ai_catalog_test.dart`

覆盖用例：

- `BD-IT-LOCALAI-001`
- `BD-IT-LOCALAI-002`

这组测试更偏向数据源与展示逻辑，不需要复杂页面路径，但需要稳定控制 bucket 文件布局。

## 6. `integration_test/support` 支撑层设计

## 6.1 `test_env.dart`

职责：

- 加载 `.env.test`
- 暴露 URL、anon key、service-role key
- 提供是否运行在线扩展测试的开关

建议字段：

```dart
class TestEnv {
  static String get supabaseUrl => ...;
  static String get anonKey => ...;
  static String get serviceRoleKey => ...;
  static bool get enableOnlineModelTests => ...;
}
```

## 6.2 `test_accounts.dart`

职责：

- 封装 `user_a`、`user_b`、`admin` 三种账户
- 提供登录、登出和恢复初始态的工具

## 6.3 `test_ids.dart`

职责：

- 统一生成测试 ID 与前缀

示例：

```dart
class TestIds {
  static const runPrefix = 'it_20260327';
  static String scene(String suffix) => '${runPrefix}_scene_$suffix';
}
```

## 6.4 `supabase_assertions.dart`

职责：

- 查询 `processing_tasks`
- 查询 `model_assets`
- 查询 `community_posts`
- 查询 `memory_collections`

典型方法：

- `expectProcessingTaskExists(...)`
- `expectModelAssetDeleted(...)`
- `fetchLatestTaskBySceneId(...)`

## 6.5 `storage_assertions.dart`

职责：

- 验证 Storage 对象是否存在
- 验证目录是否已清空

典型方法：

- `expectStorageObjectExists(bucket, path)`
- `expectStoragePrefixEmpty(bucket, prefix)`

## 6.6 `waiters.dart`

职责：

- 等待 UI 出现
- 等待数据库状态更新
- 等待 Realtime 传播

不要在测试里写大量硬编码 `Future.delayed()`，统一放进这里做带超时控制的轮询等待。

## 7. 各用例的脚本映射关系

## 7.1 Auth 组

| 用例编号 | Flutter 测试文件 | 辅助脚本 | 备注 |
| --- | --- | --- | --- |
| `BD-IT-AUTH-001` | `auth_flow_test.dart` | `bootstrap_supabase_test_env.sh` | 主要走 UI + Session 断言 |
| `BD-IT-AUTH-002` | `auth_flow_test.dart` | 无 | 不需要建数，只需现成用户 |
| `BD-IT-AUTH-003` | `auth_flow_test.dart` | `run_flutter_integration_tests.sh --env admin` | 需切换环境变量 |
| `BD-IT-AUTH-004` | `auth_flow_test.dart` | 无 | 需要任务样本，可复用 seed |

## 7.2 Task 组

| 用例编号 | Flutter 测试文件 | 辅助脚本 | 备注 |
| --- | --- | --- | --- |
| `BD-IT-TASK-001` | `task_submission_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 上传图片 + DB 写入 |
| `BD-IT-TASK-002` | `task_submission_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 上传视频 + DB 写入 |
| `BD-IT-TASK-003` | `task_submission_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 旧入口兼容性 |
| `BD-IT-TASK-004` | `task_submission_test.dart` | 无 | 需测试登录导航 |
| `BD-IT-TASK-005` | `task_submission_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 需要多状态任务 |

## 7.3 Recall / Realtime 组

| 用例编号 | Flutter 测试文件 | 辅助脚本 | 备注 |
| --- | --- | --- | --- |
| `BD-IT-RECALL-001` | `recall_flow_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 需要模型与任务显示名数据 |
| `BD-IT-RECALL-002` | `recall_flow_test.dart` | `seed_supabase_test_data.sh --profile realtime` | 需要 processing 样本 |
| `BD-IT-RECALL-003` | `recall_flow_test.dart` | `run_edge_function_smoke_tests.sh search-models` | 最好先确认函数本身 |
| `BD-IT-RECALL-004` | `recall_flow_test.dart` | `cleanup_supabase_test_data.sh --restore-names` | 结束后需恢复原名 |
| `BD-IT-RECALL-005` | `recall_flow_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 需要同名版本 |
| `BD-IT-RECALL-006` | `recall_flow_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 使用专门待删样本 |
| `BD-IT-RECALL-007` | `recall_flow_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 需要 user_b 数据 |
| `BD-IT-REALTIME-001` | `realtime_flow_test.dart` | `mutate_processing_task_status.sh` | 后台改状态为 processing |
| `BD-IT-REALTIME-002` | `realtime_flow_test.dart` | `mutate_processing_task_status.sh` | 后台改状态为 completed/failed |
| `BD-IT-REALTIME-003` | `realtime_flow_test.dart` | `mutate_processing_task_status.sh` | 校验通知与去重 |

## 7.4 Community / Edge Function / Local AI 组

| 用例编号 | Flutter 测试文件 | 辅助脚本 | 备注 |
| --- | --- | --- | --- |
| `BD-IT-COMM-001` | `community_flow_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 需要帖子样本 |
| `BD-IT-COMM-002` | `community_flow_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 需要当前用户模型 |
| `BD-IT-COMM-003` | `community_flow_test.dart` | `run_flutter_integration_tests.sh --fault community_insert_fail` | 需要受控失败注入 |
| `BD-IT-EFUNC-001` | `edge_function_flow_test.dart` | `tests/http/search_models_smoke.sh` | 函数与 UI 双断言 |
| `BD-IT-EFUNC-002` | `edge_function_flow_test.dart` | `tests/http/agent_recall_stream_smoke.sh` | 保存事件流 |
| `BD-IT-EFUNC-003` | `edge_function_flow_test.dart` | `run_flutter_integration_tests.sh --fault agent_stream_fail` | 需要流式失败注入 |
| `BD-IT-EFUNC-004` | `edge_function_flow_test.dart` | `tests/http/confirm_text_image_smoke.sh` | 建议使用固定图片 URL |
| `BD-IT-EFUNC-005` | `edge_function_flow_test.dart` | `run_edge_function_smoke_tests.sh text-to-image` | 默认可转为扩展回归 |
| `BD-IT-EFUNC-006` | `edge_function_flow_test.dart` | `run_edge_function_smoke_tests.sh agent-preview-execute` | 需做数据库快照比对 |
| `BD-IT-LOCALAI-001` | `local_ai_catalog_test.dart` | `seed_supabase_test_data.sh --profile minimal` | 要上传 catalog |
| `BD-IT-LOCALAI-002` | `local_ai_catalog_test.dart` | `seed_supabase_test_data.sh --phase local_ai_fallback` | 要去掉 catalog |

## 8. 推荐执行编排

## 8.1 单机手工联调编排

适合开发者本地：

```bash
tests/scripts/bootstrap_supabase_test_env.sh
tests/scripts/run_full_integration_suite.sh --mode local
```

## 8.2 CI 编排

建议拆成 3 个 Job：

### Job 1：基础链路

执行：

- Auth
- Task
- Recall 静态读取
- Community
- Local AI

### Job 2：Realtime

执行：

- `BD-IT-REALTIME-*`

原因：

- Realtime 测试更容易受时序影响，单独拆出来更容易定位问题

### Job 3：Edge Functions

执行：

- `search-models`
- `agent-recall`
- `confirm-text-image`

扩展：

- `text-to-image` 可按条件触发

## 8.3 全量编排脚本

建议新增：

```bash
tests/scripts/run_full_integration_suite.sh
```

职责：

1. 清理旧测试数据
2. 启动 Supabase
3. 创建种子数据
4. 执行 Flutter 集成测试
5. 执行函数级冒烟测试
6. 汇总报告
7. 清理测试数据

建议伪代码：

```bash
#!/usr/bin/env bash
set -euo pipefail

tests/scripts/cleanup_supabase_test_data.sh || true
tests/scripts/bootstrap_supabase_test_env.sh
tests/scripts/seed_supabase_test_data.sh --profile minimal
tests/scripts/seed_supabase_test_data.sh --profile realtime
tests/scripts/seed_supabase_test_data.sh --profile agent

tests/scripts/run_flutter_integration_tests.sh --group auth
tests/scripts/run_flutter_integration_tests.sh --group task
tests/scripts/run_flutter_integration_tests.sh --group recall
tests/scripts/run_flutter_integration_tests.sh --group realtime
tests/scripts/run_flutter_integration_tests.sh --group community
tests/scripts/run_flutter_integration_tests.sh --group local_ai

tests/scripts/run_edge_function_smoke_tests.sh search-models
tests/scripts/run_edge_function_smoke_tests.sh agent-recall
tests/scripts/run_edge_function_smoke_tests.sh confirm-text-image

tests/scripts/cleanup_supabase_test_data.sh
```

## 9. Flutter 测试执行脚本设计

脚本：`tests/scripts/run_flutter_integration_tests.sh`

建议入参：

```bash
tests/scripts/run_flutter_integration_tests.sh --group auth
tests/scripts/run_flutter_integration_tests.sh --group realtime
tests/scripts/run_flutter_integration_tests.sh --env admin
tests/scripts/run_flutter_integration_tests.sh --fault agent_stream_fail
```

职责：

- 切换 `.env.test` 或传递 dart-defines
- 指定运行某个测试文件或某组测试
- 接收故障注入开关

建议映射：

```text
auth -> app/integration_test/auth_flow_test.dart
task -> app/integration_test/task_submission_test.dart
recall -> app/integration_test/recall_flow_test.dart
realtime -> app/integration_test/realtime_flow_test.dart
community -> app/integration_test/community_flow_test.dart
edge -> app/integration_test/edge_function_flow_test.dart
local_ai -> app/integration_test/local_ai_catalog_test.dart
```

## 10. 故障注入方案

为覆盖异常路径，建议支持如下故障注入开关。

## 10.1 `community_insert_fail`

实现建议：

- 测试环境中临时让 `community_posts` 插入失败
- 或用 HTTP 拦截代理返回 500

用途：

- 验证 `BD-IT-COMM-003`

## 10.2 `agent_stream_fail`

实现建议：

- 在测试代理层截断 `agent-recall?stream=1`
- 或在测试环境中让流式请求超时，但保留非流式可用

用途：

- 验证 `BD-IT-EFUNC-003`

## 10.3 `catalog_missing`

实现建议：

- 移除 `catalog/model_catalog.json`
- 保留 bucket 中 `.gguf` 文件

用途：

- 验证 `BD-IT-LOCALAI-002`

## 10.4 `storage_partial_failure`

实现建议：

- 在专门扩展测试中人为让 DB 写入失败或上传失败

用途：

- 后续补充“上传成功但写库失败”的事务一致性测试

注意：

- 当前这项建议先保留为扩展项，因为需要较稳定的故障控制能力

## 11. 报告与产物设计

建议所有脚本统一输出到：

```text
tests/output/
  flutter/
  edge/
  sql/
  storage/
```

推荐产物：

- Flutter 集成测试日志
- Edge Function 原始响应
- Agent 流事件 JSONL
- SQL 快照
- Storage 列表快照

示例：

```text
tests/output/flutter/auth_flow.log
tests/output/edge/agent_recall_stream.jsonl
tests/output/sql/before_agent_preview.json
tests/output/sql/after_agent_execute.json
tests/output/storage/it_scene_folder_listing.json
```

## 12. 推荐的实现先后顺序

建议按以下顺序落地脚本和测试：

1. `bootstrap_supabase_test_env.sh`
2. `seed_supabase_test_data.sh`
3. `cleanup_supabase_test_data.sh`
4. `integration_test/support/test_env.dart`
5. `integration_test/support/supabase_assertions.dart`
6. `auth_flow_test.dart`
7. `task_submission_test.dart`
8. `recall_flow_test.dart`
9. `mutate_processing_task_status.sh`
10. `realtime_flow_test.dart`
11. `community_flow_test.dart`
12. `search_models_smoke.sh`
13. `agent_recall_stream_smoke.sh`
14. `edge_function_flow_test.dart`
15. `local_ai_catalog_test.dart`
16. `run_full_integration_suite.sh`

这样安排的原因是：

- 先把环境和数据控制住
- 再做页面主链路
- 再做 Realtime 与 Edge Function 这类时序更复杂的用例

## 13. 风险与限制

## 13.1 当前仓库尚无 `integration_test/`

这意味着：

- 测试基础设施需要先补齐
- 不能期待立刻执行本文中的全部脚本

## 13.2 `text-to-image` 和部分 Agent 依赖外部模型服务

这意味着：

- 默认 CI 不宜强依赖真实联网调用
- 最好分成 stub 回归与在线扩展回归两档

## 13.3 当前前端存在直连 Storage REST 与 SDK 混用

这意味着：

- 测试断言不能只看一个 SDK 返回
- 需要分别检查 REST 上传结果和数据库写入结果

## 14. 结论

如果后续按本文方案落地，BrainDance 的 Flutter-Supabase 集成测试将形成三层闭环：

1. Flutter 端真实页面交互
2. Supabase 数据库与 Storage 状态断言
3. Edge Function 协议级回归

这样可以覆盖当前项目最关键的风险点：

- 登录态漂移
- 上传成功但写库失败
- Recall 列表与显示名来源不一致
- Realtime 不稳定
- Agent 流式协议或写工具副作用失控

这两份文档配合起来，已经足够直接进入测试脚本实现阶段。
