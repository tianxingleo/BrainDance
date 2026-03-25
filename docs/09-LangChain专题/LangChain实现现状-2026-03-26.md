# LangChain 实现现状 (2026-03-26)

## 本轮目标

本轮工作不是继续堆新入口，而是把 `agent-recall`
真正收口成统一入口，并把时间对比、记忆整理、多模态创作、长期记忆摘要这几类能力接进共享
Core，避免能力散落在独立函数和旧文档描述里。

## 当前真实实现

### 1. 统一入口

- 正式入口仍然是
  [agent-recall/index.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/index.ts)。
- `agent-recall` 现在直接把请求上下文透传给共享 Core
  [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)。
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

[spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)
当前可路由到：

- `spatial_search`
  - 继续负责空间检索、候选打分、多轮补工具与最终动作生成。
- `asset_metadata`
  - 在原有资产工具基础上，新增专题归档、线程归组、pose 摘要、相关模型查找。
- `time_compare`
  - 复用
    [time-compare-agent/agent.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/agent.ts)
    的时间窗口比较能力，并统一返回到 `agent-recall` 协议。
- `creative`
  - 基于选中模型生成创作上下文与导览大纲；`execute` 时会向 `processing_tasks`
    入队异步创作任务。
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

本轮进一步收紧：

- 共享 Core 已新增按 `mode` 区分的判别联合响应 schema，不再只是单一“大对象 +
  mode 字段”。
- `spatial_search / asset_metadata / time_compare / creative / memory_graph`
  五类返回现在都有各自约束的 `evidence` / `*_context` 结构。
- `visualizationActionSchema` 已从 `payload: record` 改为按
  `open_scene / fly_to_pose` 区分的 typed payload union。
- 工具边界已开始在消费侧即校验：
  - 空间候选结果通过 `poseSearchResultSchema / sceneSearchResultSchema` 解析。
  - 资产工具结果通过 `kind` 判别联合 schema 解析。
- 会话上下文已支持 `sessionState`，包含：
  - `lastMode`
  - `lastSelectedModelIds`
  - `lastCandidateRefs`
  - `lastOperationPreview`
- `buildAgentContextBlock()`
  现在会把候选引用列表按编号展开，便于模型理解“上一个”“第二个”这类指代。

兼容性处理：

- 仍保留 `candidates` 字段，作为 `top_candidates` 的兼容别名，避免 Flutter
  端旧解析立即失效。
- 正式动作协议只保留：
  - `open_scene`
  - `fly_to_pose`

### 4. 资产与记忆工具

新增共享工具文件
[memoryTools.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/memoryTools.ts)，当前已实现：

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
  - 为 `model_assets` 增加
    `place_id / memory_thread_id / version_label / summary_title / event_label / agent_meta`
- [20260326122000_create_memory_links_and_collections.sql](/home/ltx/projects/BrainDance/supabase/migrations/20260326122000_create_memory_links_and_collections.sql)
  - 新增 `related_model_links`
  - 新增 `memory_collections`
  - 新增 `memory_collection_items`

额外修复：

- 清理了被运行日志污染的
  [20260225020151_create_memory_poses_table.sql](/home/ltx/projects/BrainDance/supabase/migrations/20260225020151_create_memory_poses_table.sql)，避免新环境重放迁移时失败。

### 6. 前端消费层

[agent_recall_service.dart](/home/ltx/projects/BrainDance/app/lib/services/agent_recall_service.dart)
当前已补齐：

- `mode`
- `top_candidates`
- `selected_candidate_reason`
- `asset_context`
- `compare_context`
- `collection_context`
- `creative_context`
- `memory_graph_context`
- 兼容正式动作协议 `actions[].payload`
- 兼容 `matrix` 的二维数组/扁平数组解析
- 请求侧支持传入结构化 `sessionState`
- 将 Supabase `FunctionException` 的 5xx / upstream
  错误归一化为更可读的前端提示，避免 Flutter 侧直接暴露
  `An invalid response was received from the upstream server`

[recall.dart](/home/ltx/projects/BrainDance/app/lib/pages/recall.dart)
当前已做最小展示增强：

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

- `deno test supabase/functions/agent-recall/test.ts`
- `deno check supabase/functions/_shared/agent-core/assetTools.ts supabase/functions/_shared/agent-core/spatialAgent.ts supabase/functions/agent-recall/index.ts supabase/functions/agent-recall/schemas/request.ts`

未完成的验证：

- 未执行需要真实 Supabase 环境的 `agent-recall` smoke
  集成调用，因为当前会话未注入可用的线上/本地服务配置。
- 当前环境缺少 `dart` 命令，因此本轮无法实际执行 `dart format` /
  `dart analyze`；前端改动仅完成代码层同步，仍需在具备 Dart/Flutter SDK
  的环境补一次格式化和分析。

补充：

- 2026-03-26 针对 Flutter Recall Agent 消费层又做了一次兼容修复，静态检查已确认
  `agent_recall_service.dart` 可通过分析。
- 这次修复主要解决前端仍按旧协议读取 `sceneId / ply / poses / matrix`
  顶层字段的问题；当前正式协议应从 `actions[].payload` 读取。
- 若后续仍出现 upstream 类 502/504，需要继续从 Supabase Edge Function 日志与
  DashScope / OpenAI 网关侧排查，而不是再把问题归因到 Flutter JSON 解析层。

## 2026-03-26 第二次补充：Flutter Agent 流式过程体验

### 背景

- Flutter Recall 页原先虽然有 `queryStream()` 入口，但实际上仍然退化为一次性
  `invoke`，前端只能看到非常粗的中间步骤，无法像 Codex 一样持续看到最新进展。
- 用户侧核心诉求不是再加一个 loading，而是要让 Agent
  在执行过程中持续输出“当前在做什么”，并在完成后自动把过程收起，只强调最终答案。

### 本次实现

- 更新
  [agent-recall/index.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/index.ts)
  - 新增 `stream=1` 流式分支，返回 `application/x-ndjson`。
  - Edge Function 现在会把 `runSpatialSearchAgent()`
    内部进度事件逐条下发，并在最终结果生成后继续把 `answer` 切成增量 `message`
    事件，再发送 `done`。
- 更新
  [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)
  - 新增 `AgentProgressEvent` / `AgentRuntimeCallbacks`。
  - 在模式路由、空间意图解析、工具轮次、工具调用、工具结果、最终候选定稿等节点发出结构化状态事件。
  - 这样 Flutter 不再只能依赖最后的 `tool_trace`
    事后回放，而能真正消费执行中的过程态。
- 更新
  [agent_recall_service.dart](/home/ltx/projects/BrainDance/app/lib/services/agent_recall_service.dart)
  - `queryStream()` 改为直接请求 Supabase Edge Function 流式端点并逐行解析
    NDJSON。
  - 保留普通 `query()`
    兜底；若流式链路失败，仍会自动回退到一次性结果，避免功能不可用。
  - `ChatMessage` 新增
    `liveStatus`、`summaries`、`isProcessCollapsed`，为“实时进展 +
    自动折叠”提供状态承载。
- 更新 [recall.dart](/home/ltx/projects/BrainDance/app/lib/pages/recall.dart)
  - 新增实时状态卡，优先展示 Agent 当前最新摘要。
  - 将 `status/tool_call/tool_result/message/done/error` 统一收口到
    `_consumeAgentEvent()`，让前端逻辑不再散落在监听回调里。
  - 完成后默认自动折叠“执行过程”，只突出最终答案；用户如需排查，再手动展开过程和工具调用明细。

### 当前效果口径

- 现在的“流式”重点是：
  - 阶段摘要实时推进
  - 工具调用/结果实时追加
  - 最终回答以增量 `message` 形式写入前端
- 需要明确的是：
  - 目前最终回答的 token 级生成仍不是直接透传底层模型的原生 token
    stream，而是基于服务端阶段事件 + 最终答案切片输出。
  - 这已经能显著改善 Recall 页体验，但如果后续要进一步做到真正 LLM token
    级实时生成，仍需要继续改 LangChain / 模型 SDK 的原生 streaming 接入方式。

### 本次涉及文件

- `supabase/functions/agent-recall/index.ts`
- `supabase/functions/_shared/agent-core/spatialAgent.ts`
- `app/lib/services/agent_recall_service.dart`
- `app/lib/pages/recall.dart`
