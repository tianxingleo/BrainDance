# 跨模块契约审查报告

## 审计概要
- **审计时间**：2026-04-30
- **审查范围**：app/ (Flutter) / ai_engine/ (Python) / supabase/ (SQL + TS) / dashboard/ (Vue) / 3dgs_viewer/
- **发现契约不一致数**：14 个

---

## 契约 1: task_type 枚举

### 定义位置

| 模块 | 文件 | 定义的值 |
|------|------|---------|
| Supabase Migration | `migrations/20260121000000_add_task_type_and_params.sql:29` 注释 | `video_3dgs`, `single_image_sam3d`, `single_image_sharp`（仅 3 种） |
| Supabase Migration | `migrations/20260121000000_add_task_type_and_params.sql:40-42` | CHECK 约束被注释掉，无实际约束 |
| Worker Factory | `ai_engine/3dgs/src/core/factory.py:13-26` | `video_3dgs`, `video_dual_chain`, `multi_image`, `single_image_sam3d`, `single_image_sharp`, `da3_feed_forward_3dgs`, `da3_sugar`, `da3+sugar`, `da3_2dgs`, `da3+2dgs`, `sparse2dgs`（11 种） |
| Flutter Generate | `app/lib/pages/generate.dart:124-138` | `video_3dgs`, `video_dual_chain`, `single_image_sam3d`, `single_image_sharp`, `da3_feed_forward_3dgs`, `da3_sugar`, `da3_2dgs`, `sparse2dgs`（8 种） |
| Flutter TaskList | `app/lib/pages/task_list/category_section.dart:338-350` | `_getTaskTypeLabel` 中定义了展示映射 |

### 生产者
- Flutter `generate_submission.dart` — 创建任务时写入 task_type
- Flutter `video_submit.dart` — 写入 `video_dual_chain`

### 消费者
- Worker `factory.py` — 根据 task_type 选择 Pipeline
- Worker `worker.py:788` — 读取 task_type 分支处理
- Flutter `category_section.dart` — 读取 task_type 显示图标和标签
- Dashboard `task-insights.ts` — 不直接使用 task_type

### 不一致点

1. **Migration 注释仅列 3 种，实际有 11 种**。注释已过时，会给后续开发者误导。
2. **Worker 支持但 Flutter 不展示的 task_type**：`multi_image`、`da3+sugar`（加号格式）、`da3+2dgs`（加号格式）在 Flutter 的 switch 中会走到 default 分支显示原始字符串。
3. **命名风格不统一**：Worker 用 `da3+sugar` 和 `da3_sugar` 两种格式指向同一个 Pipeline，Flutter 只使用下划线格式 `da3_sugar`。

### 风险
- Flutter 提交 `da3_sugar` 类型任务，Worker 能正确路由。但如果前端传了 `da3+sugar` 也能工作。两种写法的存在增加了混淆。
- 新增 task_type 时，需要在 Migration 注释、Worker Factory、Flutter Generate、Flutter TaskList 四处同步更新，遗漏任何一处都会出问题。

### 建议修复
1. 建立一个 `task_types.yaml` 或常量文件作为唯一真相源，各模块从此处生成枚举。
2. 启用 Migration 中的 CHECK 约束。
3. 统一使用下划线格式，废弃加号格式。

---

## 契约 2: processing_tasks 表字段读写路径

### 字段读写矩阵

| 字段 | Migration 定义 | Worker 写入 | Flutter 读取 | Dashboard 读取 |
|------|:---:|:---:|:---:|:---:|
| id | ✅ | — | ✅ | ✅ |
| user_id | ✅ (text) | — | ✅ (创建时写入) | ✅ |
| scene_id | ✅ | — | ✅ (创建时写入) | — |
| status | ✅ (text, default 'pending') | ✅ (pending→processing→completed/failed) | ✅ (Realtime 监听) | ✅ |
| created_at | ✅ | — | — | ✅ |
| updated_at | ✅ (无自动更新) | ❌ 不更新 | — | ✅ |
| logs | ✅ (jsonb[]) | ✅ (追加日志) | — | ✅ |
| tags | ✅ | — | ✅ (创建时写入) | — |
| quality_score | ✅ (int, default 0) | ✅ | ✅ | ✅ |
| quality_reason | ✅ | ✅ | — | — |
| subject | ✅ | — | — | — |
| category | ✅ | — | — | — |
| description | ✅ | — | — | — |
| keywords | ✅ | — | — | — |
| task_type | Migration 新增 | — | ✅ (创建时写入) | — |
| task_params | Migration 新增 | ✅ (读取) | ✅ (创建时写入) | — |
| display_name | Migration 新增 | — | ✅ | — |

### 不一致点

1. **updated_at 不被自动更新**：Worker 修改 status 时不会更新 updated_at，导致 Dashboard 的时间线排序不准确。
2. **subject / category / description / keywords 四个字段**：Migration 中定义了，但在代码中几乎未见使用，可能是遗留字段。
3. **quality_score 默认值 0**：Dashboard 审计已指出，0 分被计入平均分会拉低统计。Flutter 端对 0 分也没有特殊处理。

### 风险
中低。主要是 updated_at 不准确和废弃字段造成的混淆。

---

## 契约 3: model_assets 表字段读写路径

### 字段读写矩阵

| 字段 | Migration 定义 | Worker 写入 | Flutter 读取 | Dashboard 读取 |
|------|:---:|:---:|:---:|:---:|
| id | ✅ | — | ✅ | ✅ |
| scene_id | ✅ (unique) | ✅ | ✅ | — |
| user_id | ✅ (text) | ✅ | ✅ | ✅ |
| source_task_id | ✅ | ✅ | — | — |
| description | ✅ | ✅ (AI 生成) | ✅ | — |
| objects | ✅ (text[]) | ✅ (AI 生成) | ✅ | — |
| tags | ✅ (text[]) | ✅ (AI 生成) | ✅ | — |
| embedding | ✅ (vector(1536)) | ✅ | — (通过搜索函数) | — |
| ply_path | ✅ | ✅ | ✅ | — |
| preview_img_path | ✅ | ✅ | ✅ | — |
| meta_info | ✅ (jsonb) | ✅ | — | — |
| created_at | ✅ | — | ✅ | ✅ |
| display_name | Migration 新增 | ✅ | ✅ | — |
| place_id | Migration 新增 | — | — | — |
| memory_thread_id | Migration 新增 | — | — | — |
| version_label | Migration 新增 | — | — | — |
| summary_title | Migration 新增 | — | — | — |
| event_label | Migration 新增 | — | — | — |
| agent_meta | Migration 新增 | — | — | — |

### 不一致点

1. **place_id / memory_thread_id / version_label / summary_title / event_label / agent_meta**：这些 Agent 记忆字段已在 Migration 中定义（2026-03-26），但 Worker 和 Flutter 都未实际写入或读取。属于"schema 已定义，代码未对接"的状态。
2. **ply_path 与 preview_img_path 的路径格式**：Worker 写入的是 Storage path（如 `{user_id}/{scene_id}/output/point_cloud.ply`），Flutter 直接拼接为公开 URL 读取。如果 Storage 不是 public bucket，需要使用 signed URL。
3. **display_name 同步问题**：model_assets 和 processing_tasks 各自有 display_name 字段，但 Worker 只写入 model_assets 的 display_name，processing_tasks 的 display_name 由 Flutter 创建时写入。两者可能不一致。

### 风险
- Agent 记忆字段的 schema 已就绪但未使用，说明 Agent 高级功能（时间对比、弱图谱）的前端消费链路尚未完成。
- ply_path 的公开读取依赖 Storage policy，前面已指出 Storage 全开策略的安全风险。

---

## 契约 4: Storage bucket 和 object path

### Bucket 定义

| Bucket | 用途 | 访问策略 |
|--------|------|---------|
| `braindance-assets` | 任务素材、中间结果、输出模型 | 有 user folder 策略 + 全开策略（冲突） |
| `braindance-models` | 端侧 AI 模型发布 | 未在 migration 中定义策略 |

### Path 约定

| 路径 | 写入者 | 读取者 | 约定来源 |
|------|--------|--------|---------|
| `{user_id}/{scene_id}/raw/video.mp4` | Flutter 上传 | Worker 下载 | README |
| `{user_id}/{scene_id}/raw/image.png` | Flutter 上传 | Worker 下载 | README |
| `{user_id}/{scene_id}/raw/images.zip` | Flutter 上传 | Worker 下载 | README |
| `{user_id}/{scene_id}/raw/thumbnail.jpg` | Worker 生成 | Flutter/Dashboard 展示 | README |
| `{user_id}/{scene_id}/output/point_cloud.ply` | Worker 生成 | Flutter/Viewer 加载 | README |
| `{user_id}/{scene_id}/output/point_cloud.splat` | Worker 生成 | Viewer 加载 | README |
| `catalog/model_catalog.json` | 手动维护 | Flutter 本地 AI 读取 | README |
| `releases/*.gguf` | 手动上传 | Flutter 本地 AI 下载 | README |

### 不一致点

1. **braindance-models bucket 缺少 RLS 策略**：Migration 中只为 braindance-assets 定义了 Storage 策略，braindance-models 的访问完全依赖 Supabase 默认行为（可能开放也可能不可访问）。
2. **thumbnail.jpg 路径约定 vs Worker 实际输出**：README 写的是 `raw/thumbnail.jpg`，但实际 Worker 可能输出到 `output/preview.jpg` 或其他路径。Flutter 端通过 `preview_img_path` 字段获取实际路径，不直接拼接 thumbnail 路径，所以如果 preview_img_path 正确则不影响功能。
3. **point_cloud.ksplat 格式**：README 列出了 `ksplat` 格式，但 Worker 代码中未找到 ksplat 导出逻辑。可能是 Viewer 支持但当前 Pipeline 不生成。

### 风险
- braindance-models 的访问策略不明确，端侧模型下载可能随时失败。
- ksplat 格式承诺了但未实现。

---

## 契约 5: Edge Function 请求/响应 Schema

### agent-recall

| 方面 | Flutter 发送 | Edge Function 期望 | 一致性 |
|------|------------|-------------------|--------|
| query | `query` (string) | `query` (string) | ✅ |
| executionMode | `execution_mode` (string) | `execution_mode` (string) | ✅ |
| selectedModelIds | `selected_model_ids` (string[]) | `selectedModelIds` (string[]) | ⚠️ 命名风格不一致 |
| sessionState | `session_state` (json) | `sessionState` (json) | ⚠️ 命名风格不一致 |
| conversationSummary | `conversation_summary` (string) | `conversationSummary` (string) | ⚠️ 命名风格不一致 |

**不一致**：Flutter 使用 snake_case，Edge Function 内部使用 camelCase。需要确认中间是否有转换层。如果 Flutter 直接发送 snake_case，Edge Function 可能无法正确解析。

### search-models

| 方面 | Flutter 发送 | Edge Function 返回 | 一致性 |
|------|------------|-------------------|--------|
| query | `q` (string) | — | ✅ |
| 时间范围 | `from`/`to` | — | ✅ |
| 返回结果 | — | `results[]` 含 id, scene_id, ply_path, similarity | ✅ |

### spatial-search-agent / time-compare-agent

这两个 Function 使用 `_shared/agent-core/spatialAgent.ts` 共享 Core，与 agent-recall 共享大部分逻辑。

### 不一致点

1. **命名风格不一致**：Flutter 和 Edge Function 之间的字段命名混用 snake_case 和 camelCase。虽然 JSON 不强制大小写，但容易导致字段名拼写错误而静默失败。
2. **响应格式不统一**：agent-recall 返回 SSE/NDJSON 流式事件，search-models 返回普通 JSON。Flutter 需要针对不同 Function 使用不同的解析逻辑。

---

## 契约 6: 状态枚举

### processing_tasks.status

| 值 | Worker 写入 | Flutter 读取 | Dashboard 读取 |
|----|:---:|:---:|:---:|
| `pending` | ✅ (查询条件) | ✅ (显示排队中) | ✅ |
| `processing` | ✅ (update status) | ✅ (显示处理中) | ✅ |
| `completed` | ✅ (update status) | ✅ (显示已完成) | ✅ |
| `failed` | ✅ (update status) | ✅ (显示失败) | ✅ |

**一致性**：较好，四个状态在所有模块中一致。但缺少其他可能的状态值（如 `cancelled`、`retrying`）。

### worker_nodes.status

| 值 | Worker 写入 | Dashboard 读取 |
|----|:---:|:---:|
| `starting` | ✅ | ✅ |
| `idle` | ✅ | ✅ |
| `busy` | ✅ | ✅ |
| `stopping` | ✅ | ✅ |
| `offline` | ✅ | ✅ |
| `error` | ✅ | ⚠️ Dashboard 可能有特殊处理 |

**一致性**：Worker 写入的 status 值与 Migration 注释一致。

### worker_nodes.desired_state

| 值 | Dashboard 写入 | Worker 读取 |
|----|:---:|:---:|
| `run` | ✅ | ✅ |
| `pause` | ✅ | ✅ |

**一致性**：一致，但缺少 `stop`/`restart` 等可能需要的状态。

---

## 契约 7: Worker 输出文件 vs 前端消费

### 3DGS 产物格式

| 产物 | Worker 生成 | Flutter Viewer 支持 | 3DGS Viewer 支持 |
|------|:---:|:---:|:---:|
| `.ply` | ✅ | ✅ (通过 WebView) | ✅ |
| `.splat` | ✅ | ✅ (通过 WebView) | ✅ |
| `.ksplat` | ❌ (未生成) | ✅ (Viewer 支持) | — |

### 缩略图

| 产物 | Worker 生成 | Flutter 消费 |
|------|:---:|:---:|
| `preview_img_path` | ✅ (写入 model_assets) | ✅ (从 model_assets 读取) |

**一致性**：ply 和 splat 格式在 Worker 和 Viewer 之间一致。ksplat 是 Viewer 支持但 Worker 未生成的格式，不影响当前功能。

---

## 契约 8: 错误状态与重试

### 任务失败处理

| 方面 | Worker 行为 | Flutter 行为 | Dashboard 行为 |
|------|-----------|------------|--------------|
| 失败状态 | 写入 `status=failed` + `logs` | 显示失败标签 + 日志 | 显示失败卡片 |
| 重试机制 | ❌ 无自动重试 | ❌ 用户无法手动重试 | ❌ 无重试按钮 |
| 失败原因 | 写入 `quality_reason` | 不显示 | 不显示 |

### 不一致点

1. **Worker 写入 `quality_reason` 但 Flutter 和 Dashboard 都不展示**。用户不知道任务为什么失败。
2. **没有任何模块提供重试能力**。任务失败后只能重新提交。
3. **Worker 的 "dual chain" 模式**：快链失败后仍会尝试慢链，但这条降级路径用户不可见。

---

## 不一致汇总表

| # | 契约 | 不一致描述 | 严重程度 | 影响模块 | 建议修复 |
|---|------|-----------|:---:|---------|---------|
| 1 | task_type | Migration 注释仅列 3 种，实际 11 种 | P2 | 全模块 | 更新注释 + 启用 CHECK |
| 2 | task_type | Worker 支持加号格式 `da3+sugar`，Flutter 只用下划线 | P2 | Worker/Flutter | 统一格式 |
| 3 | task_type | CHECK 约束被注释掉 | P1 | Supabase | 启用约束 |
| 4 | processing_tasks | updated_at 不被自动更新 | P2 | Worker/Dashboard | 添加触发器 |
| 5 | model_assets | Agent 记忆字段已定义但未使用 | P2 | Supabase/Agent | 对接或移除 |
| 6 | model_assets | display_name 在两表间可能不一致 | P3 | Flutter/Worker | 统一写入源 |
| 7 | Storage | braindance-models bucket 缺少策略 | P1 | Flutter | 添加策略 |
| 8 | Storage | "Enable all" 策略覆盖 user folder 策略 | P0 | 全模块 | 删除全开策略 |
| 9 | Edge Function | 请求字段命名风格混用 snake_case/camelCase | P2 | Flutter/Edge | 统一风格 |
| 10 | Edge Function | agent-recall 流式 vs search-models JSON，解析逻辑不统一 | P3 | Flutter | 抽取统一客户端 |
| 11 | 状态枚举 | 缺少 cancelled/retrying 等状态 | P3 | 全模块 | 评估后补充 |
| 12 | 错误处理 | quality_reason 写入但前端不展示 | P2 | Flutter/Dashboard | 添加展示 |
| 13 | 错误处理 | 无重试机制 | P2 | 全模块 | 添加重试按钮 |
| 14 | 输出格式 | ksplat 在 README 列出但 Worker 未生成 | P3 | README/Worker | 更新 README 或实现 |

---

## 建议新建 Issue 清单

- [ ] [P0] 删除 Storage "Enable all" 策略，修复 user folder 策略被覆盖
- [ ] [P1] 为 braindance-models bucket 添加访问策略
- [ ] [P1] 启用 task_type CHECK 约束，统一枚举定义
- [ ] [P2] 统一 Edge Function 请求字段命名风格
- [ ] [P2] 添加 processing_tasks.updated_at 自动更新触发器
- [ ] [P2] Flutter/Dashboard 展示 task 失败原因 (quality_reason)
- [ ] [P2] 统一 task_type 命名格式（废弃加号格式）
- [ ] [P3] 补充 cancelled/retrying 状态枚举
- [ ] [P3] 添加任务重试按钮（Flutter + Dashboard）
- [ ] [P3] 更新 README 中 ksplat 相关说明
