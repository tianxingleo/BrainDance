# LangChain 实现现状 (2026-03-26)

## 本轮目标

本轮工作不是继续堆新入口，而是把 `agent-recall` 真正收口成统一入口，并把时间对比、记忆整理、多模态创作、长期记忆摘要这几类能力接进共享 Core，避免能力散落在独立函数和旧文档描述里。

## 当前真实实现

### 1. 统一入口

- 正式入口仍然是 [agent-recall/index.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/index.ts)。
- `agent-recall` 现在直接把请求上下文透传给共享 Core [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)。
- 当前请求已支持：
  - `query`
  - `selectedModelIds`
  - `executionMode`
  - `currentSceneId`
  - `currentModelId`
  - `currentMode`
  - `candidateSceneIds`
  - `sessionId`
  - `conversationSummary`

### 2. 共享 Core 已扩成多模式

[spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts) 当前可路由到：

- `spatial_search`
  - 继续负责空间检索、候选打分、多轮补工具与最终动作生成。
- `asset_metadata`
  - 在原有资产工具基础上，新增专题归档、线程归组、pose 摘要、相关模型查找。
- `time_compare`
  - 复用 [time-compare-agent/agent.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/agent.ts) 的时间窗口比较能力，并统一返回到 `agent-recall` 协议。
- `creative`
  - 基于选中模型生成创作上下文与导览大纲；`execute` 时会向 `processing_tasks` 入队异步创作任务。
- `memory_graph`
  - 基于当前模型生成最近趋势、缺失模式、变化时间线与弱图谱摘要。

### 3. 正式返回协议

当前 `agent-recall` / 共享 Core 的正式返回结构已补齐：

- `mode`
- `answer`
- `evidence`
- `actions`
- `top_candidates`
- `selected_candidate_reason`
- `asset_context`
- `compare_context`
- `collection_context`
- `creative_context`
- `memory_graph_context`

兼容性处理：

- 仍保留 `candidates` 字段，作为 `top_candidates` 的兼容别名，避免 Flutter 端旧解析立即失效。
- 正式动作协议只保留：
  - `open_scene`
  - `fly_to_pose`

### 4. 资产与记忆工具

新增共享工具文件 [memoryTools.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/memoryTools.ts)，当前已实现：

- `get_pose_summary`
- `find_related_models`
- `list_place_versions`
- `create_memory_collection`
- `add_models_to_collection`
- `summarize_collection`
- `group_models_into_thread`
- `prepare_story_context`
- `generate_story_outline`
- `enqueue_creative_task`
- `get_recent_place_trend`
- `find_missing_object_pattern`
- `summarize_place_change_timeline`
- `build_personal_memory_graph_summary`

其中：

- `asset_metadata` 模式主要消费前 7 类工具。
- `creative` 模式消费创作上下文、大纲和异步入队能力。
- `memory_graph` 模式消费趋势、缺失、时间线和关系摘要能力。

### 5. 数据库迁移

本轮新增了两组迁移：

- [20260326121000_add_agent_memory_fields.sql](/home/ltx/projects/BrainDance/supabase/migrations/20260326121000_add_agent_memory_fields.sql)
  - 为 `model_assets` 增加 `place_id / memory_thread_id / version_label / summary_title / event_label / agent_meta`
- [20260326122000_create_memory_links_and_collections.sql](/home/ltx/projects/BrainDance/supabase/migrations/20260326122000_create_memory_links_and_collections.sql)
  - 新增 `related_model_links`
  - 新增 `memory_collections`
  - 新增 `memory_collection_items`

额外修复：

- 清理了被运行日志污染的 [20260225020151_create_memory_poses_table.sql](/home/ltx/projects/BrainDance/supabase/migrations/20260225020151_create_memory_poses_table.sql)，避免新环境重放迁移时失败。

### 6. 前端消费层

[agent_recall_service.dart](/home/ltx/projects/BrainDance/app/lib/services/agent_recall_service.dart) 当前已补齐：

- `mode`
- `top_candidates`
- `selected_candidate_reason`
- `asset_context`
- `compare_context`
- `collection_context`

[recall.dart](/home/ltx/projects/BrainDance/app/lib/pages/recall.dart) 当前已做最小展示增强：

- 展示 `mode`
- 展示 `selected_candidate_reason`
- 展示前 3 个候选项摘要

## 已完成 / 未完成

### 已完成

- 统一入口上下文透传
- 统一正式动作协议为 `open_scene / fly_to_pose`
- `model_assets.display_name` 继续作为正式资产名称来源
- 共享 Core 多模式扩展
- 记忆专题与线程归组的数据库基础
- 创作任务异步入队
- 长期记忆的弱图谱摘要能力

### 仍未完成

- `place_id`、`memory_thread_id` 的历史数据回填策略还没有做
- `related_model_links` 仍缺少后台批量构建任务，当前以启发式检索为主
- Flutter Recall 页还没有做 mode-aware 的完整卡片 UI，只做了最小展示
- `creative` 模式目前只入队任务，没有新增 Worker 端创作产出流水线
- `memory_graph` 模式目前仍是弱图谱摘要，不是完整图数据库能力

## 验证

本轮已通过：

- `deno test supabase/functions/agent-recall/test.ts supabase/functions/spatial-search-agent/test.ts supabase/functions/time-compare-agent/test.ts`
- `deno check supabase/functions/agent-recall/index.ts supabase/functions/spatial-search-agent/index.ts supabase/functions/_shared/agent-core/spatialAgent.ts supabase/functions/_shared/agent-core/memoryTools.ts`

未完成的验证：

- 本地未执行 Flutter `dart format`，原因是当前环境缺少 `dart` 命令。
- 未执行需要真实 Supabase 环境的 `agent-recall` smoke 集成调用，因为当前会话未注入可用的线上/本地服务配置。
