# BrainDance LangChain Agent 实现路径与方式详解

> 最后更新：2026-04-25

---

## 一、做了什么

BrainDance 项目实现了一套**面向空间记忆的智能 Agent 系统**，核心目标是让用户通过自然语言与 3D 空间扫描资产进行交互。用户可以用一句话检索场景中的物体、对比不同时间点的空间变化、管理模型元数据、创建记忆专题，甚至触发创意性任务（如生成空间导览大纲）。

### 核心能力

| 能力 | 说明 |
|------|------|
| **空间检索** | 用户说"桌子上的花瓶在哪"，Agent 检索 3D 场景并返回具体位置和视角 |
| **资产元数据管理** | 改名、批量打标签、对比模型差异、查重名模型等 |
| **时间对比** | 对同一地点不同时间点的扫描结果做结构化 diff（物体增减、标签变化） |
| **记忆图谱** | 构建模型间的关系图、追踪地点变化趋势、发现物体缺失模式 |
| **创意任务** | 生成导览大纲、旁白脚本，下发给异步任务队列 |
| **记忆专题** | 创建/管理模型集合，自动生成摘要和标签建议 |
| **会话状态** | 多轮对话上下文保持，支持操作确认流程（dry run → 用户确认 → 正式写入） |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Flutter 端 (客户端)                        │
│  recall_agent_runtime.dart  ←→  agent_recall_service.dart   │
│  (UI 状态 + 事件消费)             (HTTP/SSE 流式请求)         │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTPS (SSE / NDJSON 流式)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Supabase Edge Function (Deno)                   │
│              agent-recall/index.ts                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           spatialAgent.ts (Agent 编排核心)            │    │
│  │                                                       │    │
│  │  1. 意图路由 (classifyAgentMode)                      │    │
│  │     → 5 种模式分流                                    │    │
│  │                                                       │    │
│  │  2. 各模式编排                                        │    │
│  │     ├── spatial_search → 向量检索 + 关键词检索        │    │
│  │     ├── asset_metadata → 工具链编排                   │    │
│  │     ├── time_compare   → 时间对比子 Agent             │    │
│  │     ├── creative       → 故事大纲 + 异步任务          │    │
│  │     └── memory_graph   → 关系图谱分析                 │    │
│  │                                                       │    │
│  │  3. 工具层 (DynamicStructuredTool)                     │    │
│  │     ├── assetTools.ts   (资产 CRUD + 对比)            │    │
│  │     └── memoryTools.ts  (记忆关系 + 专题 + 线程归组)  │    │
│  │                                                       │    │
│  │  4. LLM 层                                            │    │
│  │     ├── ChatOpenAI (Qwen3.5-plus via DashScope)       │    │
│  │     └── OpenAIEmbeddings (text-embedding-v2)          │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│              Supabase PostgreSQL (模型资产表)                 │
│              model_assets / memory_poses / ...               │
└─────────────────────────────────────────────────────────────┘
```

### 关键文件一览

| 层级 | 文件路径 | 职责 |
|------|---------|------|
| **Edge Function 入口** | `supabase/functions/agent-recall/index.ts` | HTTP 请求处理、流式/SSE 通道建立、参数校验 |
| **Agent 编排核心** | `supabase/functions/_shared/agent-core/spatialAgent.ts` | 意图分类、模式路由、工具编排循环、候选排序、最终回答生成 |
| **资产工具** | `supabase/functions/_shared/agent-core/assetTools.ts` | 模型资产 CRUD、对比、批量修改、语义检索 |
| **记忆工具** | `supabase/functions/_shared/agent-core/memoryTools.ts` | 关系模型、地点版本链、记忆专题、故事大纲、创作任务 |
| **Prompt** | `supabase/functions/agent-recall/prompts/recallSystemPrompt.ts` | Agent 系统提示词 |
| **请求 Schema** | `supabase/functions/agent-recall/schemas/request.ts` | Zod 请求参数校验 |
| **Flutter 服务层** | `app/lib/services/agent_recall_service.dart` | SSE 流解析、数据模型、HTTP 请求 |
| **Flutter UI 运行时** | `app/lib/pages/recall/recall_agent_runtime.dart` | Agent 事件消费、步骤展示、实时状态更新 |

---

## 三、怎么做的

### 3.1 意图路由：分层分类策略

Agent 并非把所有请求直接扔给 LLM 做自由编排，而是采用了**分层路由**策略，先做确定性判断，再兜底到 LLM 结构化输出：

```
用户输入
  │
  ├─ 1. 闲聊/致谢检测 (isDirectReplyQuery)
  │     直接返回预设回答，不调用 LLM
  │
  ├─ 2. 前端模式驱动 (currentMode)
  │     compare → time_compare
  │     batch_edit / collection → asset_metadata
  │
  ├─ 3. 资产发现查询检测 (isAssetDiscoveryQuery)
  │     正则匹配"改名/标签/对比/专题/推荐"等关键词
  │     → asset_metadata 模式
  │
  ├─ 4. 启发式空间路由 (shouldPreferHeuristicSpatialRoute)
  │     正则匹配"在哪/位置/最近/今天"等关键词
  │     → spatial_search 模式（跳过 LLM 调用）
  │
  └─ 5. LLM 意图分类 (classifyAgentMode)
        调用 Qwen 的 structured output，输出 mode + tool_policy + reasoning
        → 5 种模式之一
```

**设计意图**：常见查询走确定性路径，减少 LLM 调用次数和延迟；只有真正模糊的请求才消耗 LLM token 做分类。

### 3.2 五种 Agent 模式

#### 模式一：`spatial_search` — 空间检索

用户问的是"某物在哪"、"最近拍过什么场景"等空间定位问题。

**执行路径**：
1. **意图解析**：先尝试 LLM 结构化输出（`spatialIntentSchema`），超时 8 秒则回退到正则启发式解析（`parseSpatialIntentHeuristically`）
2. **多路召回**：
   - **语义检索**：用 `OpenAIEmbeddings` 将 query 向量化，调用 Supabase RPC `match_model_assets` 做向量相似度搜索
   - **关键词检索**：用 `ilike` 在 description/objects/tags 上做模糊匹配
   - **Pose 检索**：在 `memory_poses` 表中搜索匹配帧
3. **候选融合与排序**：将多路结果按 `sourceScores` 加权融合，截取 top-N 候选
4. **最终选择**：LLM 从候选中选出最佳匹配，输出 `selection`（包含 sceneId、modelId、poseImageId、confidence）
5. **动作生成**：返回 `open_scene` 或 `fly_to_pose` 前端可执行动作

#### 模式二：`asset_metadata` — 资产元数据管理

用户要管理模型资产（改名、打标签、查重名、对比模型等）。

**执行路径**：
1. 检测确定性意图（如"改名"操作的精确参数提取），能不走 LLM 就不走
2. 构建 LangChain `DynamicStructuredTool` 工具集：
   - `read_model_assets`：通用读库（支持语义检索、标签过滤、时间范围）
   - `get_model_asset_bundle`：读取完整摘要（含 pose_count）
   - `compare_model_assets`：结构化对比（共同标签、差异物体、时间排序）
   - `rename_model_asset`：改名（默认 dry run）
   - `write_model_assets`：逐条修改
   - `batch_patch_model_metadata`：批量模板化修改
3. **工具编排循环**：最多 3 轮（`MAX_AGENT_TOOL_ROUNDS`），每轮 LLM 选择调用工具 → 执行 → 结果反馈 → 决定下一步
4. **安全机制**：所有写操作默认 `dryRun: true`，只返回预览；用户确认后才正式写入
5. **确认流程**：通过 `sessionState.lastOperationPreview` 记住上一轮预览操作，用户说"确认执行"时重放

#### 模式三：`time_compare` — 时间对比

用户要对比同一地点不同时间点的变化。

**执行路径**：
1. 委托给独立子 Agent `runTimeCompareAgent`（`time-compare-agent/agent.ts`）
2. 子 Agent 负责：解析时间窗口 → 选取 baseline/target → 结构化 diff（物体增减、标签变化）→ 生成对比摘要
3. 返回 `timeCompareContext` 包含完整的 diff 数据和可视化动作

#### 模式四：`creative` — 创意任务

用户要生成导览大纲、旁白脚本等创意内容。

**执行路径**：
1. `prepareStoryContext`：收集选中模型的时空上下文
2. `generateStoryOutlineFromContext`：基于上下文生成大纲
3. `enqueueCreativeTask`：将创作任务插入 `processing_tasks` 表，异步执行

#### 模式五：`memory_graph` — 记忆图谱

用户想了解模型间的关系、地点变化趋势。

**执行路径**：
1. 并行调用多个分析工具：
   - `getRecentPlaceTrend`：追踪地点变化趋势
   - `findMissingObjectPattern`：发现物体缺失模式
   - `summarizePlaceChangeTimeline`：时间线摘要
   - `buildPersonalMemoryGraphSummary`：关系图谱摘要
2. LLM 综合分析后输出结构化的 `memoryGraphContext`

### 3.3 工具系统：LangChain DynamicStructuredTool

所有 Agent 工具都通过 LangChain 的 `DynamicStructuredTool` 构建，核心模式：

```typescript
// assetTools.ts 示例
new DynamicStructuredTool({
  name: "read_model_assets",
  description: "通用模型资产读库工具...",
  schema: z.object({  // Zod schema 做参数校验
    mode: z.enum(["list", "duplicate_display_name"]).default("list"),
    modelIds: z.array(z.string().uuid()).default([]),
    query: z.string().default(""),
    // ...
  }),
  func: async (input) => {
    // 业务逻辑：查询 Supabase → 返回 JSON 字符串
    return JSON.stringify({ kind: "list_model_assets", rows });
  },
});
```

**特点**：
- **Zod Schema 强校验**：所有工具输入都通过 Zod schema 校验，LLM 产生的参数格式错误会被自动拦截
- **模型 ID 安全校验**：`restrictModelIds` 函数确保工具只能操作用户已选中的模型，防止越权
- **安全沙箱**：写操作默认 `dryRun`，必须用户二次确认才生效

### 3.4 流式事件系统

Agent 通过自定义事件流向前端实时推送执行进度：

```
事件类型:
  ping       → 连接保活（含 2KB 填充以冲破代理缓冲）
  status     → 阶段状态（如 "request_received", "intent_classified"）
  plan       → 执行计划（标题 + 步骤列表）
  thought    → Agent 思考过程
  tool_call  → 工具调用开始（工具名 + 参数）
  tool_result→ 工具执行结果摘要
  message    → 最终回答增量片段（delta）
  done       → 完成（含完整结果）
  error      → 错误信息
```

**传输协议**：支持 SSE（`text/event-stream`）和 NDJSON 两种格式，根据客户端 `Accept` 头自动选择。

**前端消费**（Flutter 端 `_consumeAgentEvent`）：
- 每个 `status`/`thought`/`tool_call` 事件都映射为 UI 步骤卡片
- `message` 事件做增量拼接（`_mergeAgentAnswerDelta`），支持增量片段和累计片段两种模式
- `done` 事件触发最终结果解析和 UI 完成

### 3.5 会话状态管理

Agent 维护跨轮次的状态以支持多轮交互：

```typescript
sessionState: {
  lastMode: "asset_metadata" | ...,           // 上一轮模式
  lastSelectedModelIds: [...],                 // 上一轮操作的模型
  lastCandidateRefs: [...],                    // 上一轮的候选结果
  lastOperationPreview: {                      // 上一轮的写操作预览
    toolName: "rename_model_asset",
    affectedCount: 1,
    modelIds: [...],
    args: { ... },
  },
}
```

**用途**：
- 用户说"把它改名"时，Agent 从 `lastSelectedModelIds` 推断"它"指代哪个模型
- 用户说"确认执行"时，Agent 从 `lastOperationPreview` 重放操作

---

## 四、有什么特点

### 4.1 确定性优先的混合路由

不是所有请求都走 LLM 推理。对于明确的意图（问候、改名、空间检索），用正则和规则快速路由，只有模糊请求才消耗 LLM token。这显著降低了延迟和成本。

### 4.2 Dry-Run + 确认流程的写安全模型

所有资产修改操作默认走预览模式：
1. Agent 调用写工具时 `dryRun: true`，只返回"将要做什么"的预览
2. 预览结果存入 `sessionState.lastOperationPreview`
3. 用户说"确认执行"时，Agent 重放该操作并传入 `dryRun: false`

这个设计避免了 LLM 幻觉导致的数据破坏。

### 4.3 多路召回融合

空间检索不依赖单一检索方式，而是语义向量检索、关键词检索、Pose 检索三路并行，按来源加权融合后排序。每路检索都有独立的容错，某路失败不影响整体。

### 4.4 实时进度可视化

通过自定义事件流，前端能在 Agent 执行过程中实时展示：
- 当前处于哪个阶段
- 正在调用什么工具
- 工具返回了什么结果
- Agent 在"想"什么

这使得长时间运行的 Agent 任务不会让用户觉得"卡住了"。

### 4.5 Supabase Edge Function 运行时

整个 Agent 后端运行在 Supabase Edge Function（Deno runtime）上，无需自建服务器。LangChain 的 npm 包通过 Deno 的 npm 兼容层直接导入：

```typescript
import { ChatOpenAI } from "npm:@langchain/openai@0.6";
import { DynamicStructuredTool } from "npm:@langchain/core@0.3/tools";
import { z } from "npm:zod@3.25";
```

### 4.6 模型无关的 LLM 接入

通过 LangChain 的 `ChatOpenAI` 抽象层，底层使用 DashScope（阿里通义千问）的 OpenAI 兼容接口，默认模型为 `qwen3.5-plus`，嵌入模型为 `text-embedding-v2`。只需修改环境变量即可切换模型。

---

## 五、有什么困难

### 5.1 意图分类的不确定性

**问题**：用户查询可能同时包含空间检索和资产管理的意图（如"找到那个会议室的场景，给它打个标签"），分类边界模糊。

**应对**：
- 分层路由：先用规则拦截明确意图，LLM 只处理模糊部分
- 前端模式驱动：当前页面处于 `compare` 模式时，直接走时间对比
- `sessionState.lastMode` 作为上下文辅助分类

### 5.2 LLM 结构化输出的稳定性

**问题**：LLM 的 structured output 并不总是可靠——工具参数可能不符合 Zod schema，意图分类可能输出非法 mode。

**应对**：
- 所有 LLM 输出都经过 Zod schema 校验，解析失败时回退到启发式路径
- `spatialIntentSchema` 的 LLM 解析设了 8 秒超时，超时后自动切换到正则解析
- 工具编排循环设了 `MAX_AGENT_TOOL_ROUNDS = 3` 的硬上限，防止无限循环

### 5.3 流式传输的代理缓冲

**问题**：Nginx 等反向代理会缓冲 SSE 响应，导致前端长时间收不到事件。

**应对**：
- 首个 `ping` 事件携带 2KB 空白填充，强制冲破代理缓冲区
- 设置 `X-Accel-Buffering: no` 和 `Cache-Control: no-cache` 响应头
- 前端设置 300 秒超时（`receiveTimeout`），并设计 bootstrap 占位状态避免白屏等待

### 5.4 工具调用的安全性

**问题**：LLM 可能生成超出用户授权范围的工具调用参数（如操作未选中的模型）。

**应对**：
- `restrictModelIds` 函数在每个工具调用前校验请求的模型 ID 是否在用户已选范围内
- 越界直接抛错，LLM 收到错误信息后自行调整
- 写操作必须经过 dry run → 确认两步流程

### 5.5 多轮对话的指代消解

**问题**：用户说"把它改名"，Agent 需要知道"它"是谁。

**应对**：
- `sessionState.lastSelectedModelIds` 记录上一轮操作的模型
- `sessionState.lastCandidateRefs` 记录上一轮的候选结果
- `parseDeterministicAssetRenameIntent` 通过正则解析"当前模型/最新模型/这些模型"等指代表达
- 当前方案仍有局限：复杂指代（如"刚才那个不是，用旁边那个"）难以处理

### 5.6 Edge Function 冷启动和执行时限

**问题**：Supabase Edge Function 有冷启动延迟和执行时间限制，Agent 编排涉及多轮 LLM 调用，容易超时。

**应对**：
- 流式响应保持连接活跃，避免被代理判定超时
- 意图解析设 8 秒超时
- 各 LLM 调用通过 `withTimeout` 包装
- 工具调用轮次硬上限 3 轮

### 5.7 向量检索与关键词检索的互补性

**问题**：纯向量检索对精确匹配（如特定模型名）效果不佳，纯关键词检索对语义相似性（如"有花的地方"）效果不佳。

**应对**：
- 多路召回融合策略：语义检索 + 关键词检索并行，结果按加权分融合
- `computeKeywordScore` 做独立的 token 匹配评分
- 候选排序综合考虑语义相似度、关键词命中、来源权重

---

## 六、数据流总结

以一次典型的空间检索为例：

```
用户: "桌子上有没有花瓶？"
  │
  ▼ Flutter: AgentRecallService.queryStream()
  │  构建 SSE 请求 → POST /agent-recall?stream=1
  │
  ▼ Edge Function: index.ts
  │  参数校验 → 创建 ReadableStream → 调用 runSpatialSearchAgent()
  │
  ▼ spatialAgent.ts: runSpatialSearchAgent()
  │
  ├─ 1. classifyAgentMode("桌子上有没有花瓶？")
  │     → shouldPreferHeuristicSpatialRoute = true
  │     → mode: spatial_search, toolPolicy: tool_chain
  │
  ├─ 2. parseSpatialIntentHeuristically()
  │     → targetType: "object", objectHint: "桌子上有没有花瓶"
  │
  ├─ 3. 多路召回
  │     ├─ embeddings.embedQuery("桌子上有没有花瓶") → 向量
  │     ├─ supabase.rpc("match_model_assets", ...) → 语义召回
  │     └─ supabase.from("model_assets").select(...).ilike(...) → 关键词召回
  │
  ├─ 4. 候选融合排序 → top_candidates
  │
  ├─ 5. LLM 从候选中选择最佳匹配 → selection
  │     { modelId, sceneId, poseImageId, confidence: 0.87 }
  │
  ├─ 6. 生成 actions: [{ type: "fly_to_pose", payload: { ... } }]
  │
  └─ 7. 返回最终结果 → done event

  ▼ Flutter: _consumeAgentEvent()
     解析 SSE 事件 → 更新 UI 步骤卡片 → 显示最终回答 → 执行飞行动作
```
