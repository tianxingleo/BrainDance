# LangChain 实现现状（2026-03-25）

本文档用于记录截至 2026-03-25，BrainDance 仓库中已经落地的 LangChain 相关代码现状。这里强调“以代码为准”，不只记录规划，还记录当前实现到底做到了什么。

## 1. 总体结论

当前仓库里的 LangChain 相关工作，主要集中在 Supabase Deno Edge Functions 一侧，而不是 Python Worker 一侧。

现状可以概括为三层：

1. `search-models`
   - 仍然是共享检索底座。
   - 提供时间解析、Embedding 生成、向量检索等基础能力。
2. `agent-recall`
   - 是当前稳定协议入口。
   - 不直接使用 LangChain tool calling，而是复用共享搜索逻辑后返回稳定的 `answer + evidence + actions`。
3. `spatial-search-agent`
   - 是当前真正采用 LangChain TS tool calling 的实验链路。
   - 既能做空间检索，也已经扩展出模型资产元数据操作 / 分析模式。

另外，当前还新增了一个不直接依赖 LangChain 的 `time-compare-agent`，它复用了共享搜索底座，用于补足时间对比的最小可用能力。

## 2. 共享底座：`search-models`

相关文件：

- [index.ts](/home/ltx/projects/BrainDance/supabase/functions/search-models/index.ts)
- [shared.ts](/home/ltx/projects/BrainDance/supabase/functions/search-models/shared.ts)

它当前承担的角色不是 Agent，而是 Agent 的共享检索底层，主要包括：

- 解析用户查询中的时间语义。
- 生成 Embedding。
- 调用 `match_memory_poses` 做向量检索。
- 返回基础检索结果结构。

当前 `agent-recall` 和 `time-compare-agent` 都直接复用了这层逻辑，因此后续如果要调整时间解析、检索阈值或者召回结构，优先应该收敛到这一层，而不是分别在多个入口重复修改。

## 3. 稳定入口：`agent-recall`

相关文件：

- [index.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/index.ts)
- [recallAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/agent/recallAgent.ts)
- [searchSpace.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/tools/searchSpace.ts)
- [getSceneAsset.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/tools/getSceneAsset.ts)
- [buildViewAction.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/tools/buildViewAction.ts)

它当前已经做到：

- 使用 `searchSpace()` 复用共享搜索逻辑。
- 从搜索结果中提取证据，生成 `evidence`。
- 从搜索结果中生成稳定动作协议，输出 `open_scene`、`fly_to_pose`。
- 组织稳定响应结构：`answer + evidence + actions`。

它当前没有做到的事：

- 不使用 LangChain tool calling。
- 不做多轮 tool 调度。
- 不处理模型资产元数据操作。
- 不承担时间对比逻辑。

因此可以把它理解为：

- 它是当前前端更容易消费的稳定 Recall 协议层。
- 它不是当前仓库里最“agentic”的那条链路。

## 4. LangChain TS 主实验链路：`spatial-search-agent`

相关文件：

- [index.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/index.ts)
- [agent.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/agent.ts)
- [assetTools.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/assetTools.ts)
- [test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/test.ts)
- [assetTools.test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/assetTools.test.ts)

这是当前仓库里真正落地了 LangChain TS tool calling 的主链路。

### 4.1 当前技术形态

代码中已经明确使用：

- `ChatOpenAI`
- `OpenAIEmbeddings`
- `DynamicStructuredTool`
- `model.bindTools(tools)`
- `withStructuredOutput(...)`

这说明当前实现不是简单 prompt 拼接，而是：

- 先用结构化输出做路由或意图解析。
- 再把工具注册给模型。
- 再通过多轮 tool calling 拉取结果。
- 最后做候选裁决和动作生成。

### 4.2 当前已经实现的两种模式

`spatial-search-agent` 当前不是单一模式，而是先路由到两种模式之一：

1. `spatial_search`
   - 面向空间检索。
   - 处理物体、位置、时间、场景类请求。
2. `asset_metadata`
   - 面向模型资产元数据操作和分析。
   - 处理改名、批量标签、批量描述、摘要读取和结构化对比。

路由逻辑位于 [agent.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/agent.ts) 中的 `classifyAgentMode(...)`。

### 4.3 空间检索模式已做内容

当前空间检索模式已经完成：

- 意图解析：把查询归一到 `object / location / time / scene` 四类。
- 时间归一：支持“今天 / 昨天 / 最近 / 最新 / 刚才”等相对时间。
- 3 个核心检索 tools：
  - `pose_semantic_search`
  - `scene_metadata_search`
  - `recent_scene_search`
- 候选聚合：
  - 把不同 tool 返回的结果合并到同一个 `SceneCandidate`。
- 候选打分：
  - 根据不同目标类型，对 `pose / scene / time / lexical` 四类得分做加权。
- 证据不足时强制补工具轮次：
  - 当候选数太少、最高分太低或证据来源过单时，会强制补下一轮 tool。
- 最终裁决：
  - 使用结构化输出从候选中选择最可信的 `scene / pose`。
- Viewer 动作生成：
  - `open_model`
  - `fly_to_pose`
  - `highlight_hotspot`

这说明当前 LangChain 试验链路已经不只是“能调用一下 tool”，而是具备了比较完整的“意图解析 → tool 调度 → 候选融合 → 结果裁决 → 前端动作”的闭环。

### 4.4 模型资产元数据模式已做内容

这部分是当前代码里新增且非常关键的一块，主要集中在 [assetTools.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/assetTools.ts)。

当前已经存在的资产工具包括：

- `list_model_assets`
  - 按模型 ID、场景 ID、标签、关键词、时间范围筛选候选模型。
- `rename_model_asset`
  - 修改单个模型的 `display_name`。
- `batch_patch_model_metadata`
  - 批量修改 `display_name`、`description`、`tags`。
- `get_model_asset_bundle`
  - 拉取一个或多个模型的完整摘要。
- `compare_model_assets`
  - 输出多个模型之间的共同标签、差异对象、时间顺序和 `pose_count`。

当前这套资产工具已经体现出几个工程约束：

- 默认 `dry_run`，优先预览，不直接写库。
- 只有显式允许写入时才正式更新数据库。
- 可通过 `selectedModelIds` 把 Agent 的操作范围收敛到前端已选模型集合内。
- 明确限制不能改 `ply_path`、`scene_id`、`embedding`、`user_id` 等系统字段。

这意味着当前 LangChain 实验链路已经开始从“空间检索 Agent”扩展到“资产管理 Agent”，而且不是停留在规划里，是已经写进代码的。

### 4.5 当前输入协议已扩展

`spatial-search-agent` 的 HTTP 请求当前除了 `query` 之外，还支持：

- `selectedModelIds`
- `executionMode`

其中：

- `selectedModelIds`
  - 用于前端多选后，把 Agent 操作范围限制在某批模型内。
- `executionMode`
  - `preview` 时只做 dry run。
  - `execute` 时才允许真正写入。

这说明当前实验链路已经不再只是“问答接口”，而是在向“带安全边界的操作代理”演进。

## 5. 时间对比最小链路：`time-compare-agent`

相关文件：

- [index.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/index.ts)
- [agent.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/agent.ts)
- [request.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/schemas/request.ts)
- [response.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/schemas/response.ts)
- [test.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/test.ts)

它当前不是 LangChain tool calling 链路，但属于当前 Agent 能力扩展的一部分。

已经完成的能力：

- 解析时间对比意图。
- 自动补齐基线窗口和目标窗口。
- 对两个时间窗口分别做搜索。
- 补全场景 display name、objects、tags、pose tag 等信息。
- 生成差分摘要：
  - `commonObjects`
  - `addedObjects`
  - `removedObjects`
  - `commonTags`
  - `addedTags`
  - `removedTags`
- 返回双侧动作：
  - `open_scene`
  - `fly_to_pose`

当前明确限制：

- 它基于搜索命中和元数据差分工作。
- 还不是基于 `place_id / memory_thread_id / alignment_*` 的严格跨扫描比较。

## 6. 当前测试覆盖情况

当前仓库中，和 LangChain / Agent 相关的测试已经至少覆盖到：

- [agent-recall/test.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/test.ts)
  - 检查 evidence 和稳定动作协议。
- [spatial-search-agent/test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/test.ts)
  - 检查时间归一、候选打分、Viewer 动作生成。
- [spatial-search-agent/assetTools.test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/assetTools.test.ts)
  - 检查模板改名和资产对比结果。
- [time-compare-agent/test.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/test.ts)
  - 检查双窗口补全、对象 / 标签差分和动作生成。

## 7. 当前最值得注意的现实状态

截至当前代码，BrainDance 的 LangChain 相关实现已经不是“只做了一个搜索 demo”，而是同时存在以下几条线：

- 一条稳定入口：
  - `agent-recall`
- 一条 LangChain TS 试验主线：
  - `spatial-search-agent`
- 一个时间对比的最小扩展入口：
  - `time-compare-agent`
- 一个共享检索底座：
  - `search-models/shared.ts`

但也必须明确：

- 稳定前端协议和实验链路协议还没有完全统一。
- `agent-recall` 与 `spatial-search-agent` 的动作命名仍不一致。
- 时间对比能力目前仍是近似版。
- 资产元数据 Agent 已经进入代码，但是否完成联调、是否适合作为稳定入口，还要继续验证。

## 8. 后续文档维护建议

从现在开始，LangChain 相关变更建议按下面方式维护：

1. 路线和长期规划
   - 继续写在 [Agent规划与LangChain实践路线.md](../02-架构设计/Agent规划与LangChain实践路线.md)。
2. 当前代码现状
   - 继续更新本目录下的“实现现状”文档。
3. 每次阶段性停顿或未完成收尾
   - 新开一篇“阶段总结”或“联调记录”，不要把所有历史都堆到同一篇里。

## 9. 2026-03-26 代码核对补充

以下内容是 2026-03-26 按真实代码逐个文件核对后的补充说明。这里不推翻上面的历史表述，而是把当前已经确认的漂移点、真实实现位置和真实进度补充清楚。

### 9.1 当前真正的主实现位置已经下沉到 `_shared/agent-core`

`spatial-search-agent` 的 HTTP 入口仍在：

- [index.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/index.ts)

但核心实现不在旧文档写的：

- [agent.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/agent.ts)
- [assetTools.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/assetTools.ts)

而是已经抽到了：

- [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)
- [assetTools.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/assetTools.ts)

当前 `spatial-search-agent/index.ts` 只是参数校验和 HTTP 包装，真正的 LangChain 路由、多轮工具调用、候选融合、裁决和资产工具编排，全部都在共享核心里。

这意味着当前仓库的真实结构不是“每个 Edge Function 各自维护一套 Agent 逻辑”，而是：

- `spatial-search-agent` 作为实验入口直接调用共享 Agent 核心。
- `agent-recall` 也复用同一个共享 Agent 核心，再映射成稳定协议。

### 9.2 `agent-recall` 的真实状态已经比旧文档更“agentic”

旧文档里把 `agent-recall` 描述成“直接复用共享搜索逻辑后返回稳定 `answer + evidence + actions`”，这在当前代码里已经不完全准确。

当前真实实现位于：

- [recallAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/agent/recallAgent.ts)
- [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)

当前 `agent-recall` 的真实链路是：

- `index.ts` 支持普通 JSON 响应，也支持 `text/event-stream` 的 SSE 输出。
- `runRecallAgent()` 不再直接调用 `searchSpace()` 作为主执行路径，而是默认调用 `runSpatialSearchAgent()`。
- 它会先发出 `plan`、`thinking`、`tool_call`、`tool_result`、`message`、`done` 这类结构化流事件。
- 它始终以 `executionMode: "preview"` 调用统一 Agent 核心，避免在 Recall 协议入口直接执行写入副作用。
- 拿到统一核心返回的 `open_model` / `fly_to_pose` 后，再映射成前端稳定消费的 `open_scene` / `fly_to_pose`。

因此，当前 `agent-recall` 更准确的定位应该是：

- 它是“稳定协议层 + 流式包装层”。
- 它底层已经复用 LangChain Agent 核心，而不是只复用旧的搜索 helper。
- 它仍然不直接暴露资产写入协议，但已经能承接统一 Agent 核心的路由与执行结果。

补充说明：

- [searchSpace.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/tools/searchSpace.ts)
- [getSceneAsset.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/tools/getSceneAsset.ts)
- [buildViewAction.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/tools/buildViewAction.ts)

这些文件当前仍然存在，也有测试覆盖，但它们更像是保留的协议转换/辅助函数，不再是 `agent-recall` 的主执行路径。

### 9.3 `spatial-search-agent` 真实已完成进度

按当前 [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts) 的代码，已经可以明确认为下面这些能力是“已实现，不是规划”。

已实现：

- 使用 `ChatOpenAI`、`OpenAIEmbeddings`、`DynamicStructuredTool`、`withStructuredOutput(...)`、`bindTools(...)` 形成完整 LangChain TS 调用链。
- 先通过 `classifyAgentMode()` 在 `spatial_search` 与 `asset_metadata` 两种模式之间路由。
- 在 `spatial_search` 模式下，先通过 `parseSpatialIntent()` 产出结构化意图，再进入最多 3 轮的工具调度。
- 空间检索工具已经明确落地为：
  - `pose_semantic_search`
  - `scene_metadata_search`
  - `recent_scene_search`
- 候选会聚合到统一的 `SceneCandidate`，并以 `pose / scene / time / lexical` 混合加权打分。
- 当候选数量不足、最高分不足，或者证据来源过单时，会通过 `shouldForceAnotherToolRound()` 强制补一轮工具调用，而不是直接结束。
- 最终会用 `withStructuredOutput(selectionSchema)` 对前 5 个候选做结构化裁决，得到 `scene_id / model_id / pose_image_id / confidence / answer`。
- 已生成三类 Viewer 动作：
  - `open_model`
  - `fly_to_pose`
  - `highlight_hotspot`
- 返回结果里已经包含：
  - `tool_trace`
  - `viewer_payload`
  - `candidates`
  - `selection`

当前限制：

- 轮数上限仍是 3 轮，不是开放式多轮 Agent。
- `tool_trace` 目前是工具名、参数和摘要，适合排查，不是完整的 reasoning transcript。
- 最终动作仍由共享核心输出 `open_model`，而稳定协议入口 `agent-recall` 会再映射成 `open_scene`，两个接口口径还未统一。

### 9.4 `asset_metadata` 模式真实已完成进度

这部分当前不是实验草稿，而是已经有完整工具集合和状态汇总逻辑。

真实文件：

- [assetTools.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/assetTools.ts)

当前已经确认落地：

- `list_model_assets`
  - 可按 `modelIds / sceneIds / tags / query / startTime / endTime / limit` 过滤模型候选。
- `rename_model_asset`
  - 支持单模型改名，默认 `dryRun: true`。
- `batch_patch_model_metadata`
  - 支持批量修改 `display_name`、`description`、`tags`。
  - 支持 `displayNameTemplate / displayNamePrefix / displayNameSuffix / tagsAdd / tagsRemove / descriptionReplace / descriptionAppend`。
- `get_model_asset_bundle`
  - 会补齐 `pose_count`，返回完整摘要。
- `compare_model_assets`
  - 已返回共同标签、共同对象、每个模型独有标签/对象、时间顺序、`pose_count_by_model`。

当前工程约束也已经明确写进代码：

- `selectedModelIds` 会真正限制可操作范围，越界 ID 会直接报错。
- `executionMode !== "execute"` 时会强制转成 dry run。
- 即使是执行态，也只允许更新 `display_name`、`description`、`tags`。
- 统一通过 `AssetToolState` 汇总 `list / bundle / comparison / operation`，最后拼成 `asset_context` 返回。

当前还没有看到的能力：

- 没有删除模型、移动模型、修改系统字段之类的操作。
- 没有看到审批流、审计日志或多步确认机制，当前确认语义主要通过 `dry_run` 和 `requires_confirmation` 表达。
- `asset_metadata` 模式返回 `actions: []`，说明它当前更偏结构化资产操作/分析，不负责 Viewer 联动动作。

### 9.5 `search-models` 的真实定位需要补充两点

旧文档把它描述为共享检索底座，这个判断仍然成立，但当前代码还有两个值得补充的现实点：

真实文件：

- [shared.ts](/home/ltx/projects/BrainDance/supabase/functions/search-models/shared.ts)

补充结论：

- 它不仅做时间解析、Embedding 和 `match_memory_poses` 检索，还会额外补齐 `display_name`。
- 它当前使用的是 DashScope 兼容接口：
  - chat 默认模型是 `qwen-turbo` 或环境变量指定值。
  - embedding 默认模型是 `text-embedding-v2`。

同时也要注意，当前真正复用 `search-models/shared.ts` 的主要是：

- `search-models` 自身入口。
- `time-compare-agent`。

而 `spatial-search-agent` 的主链路已经改成自己在共享 Agent 核心里直接调用 `ChatOpenAI`、`OpenAIEmbeddings` 和 Supabase，不再走 `runSearchModelsQuery()` 这条旧路径。

### 9.6 `time-compare-agent` 的真实进度

旧文档整体判断基本成立，但按真实代码还可以表述得更具体。

真实文件：

- [agent.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/agent.ts)
- [request.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/schemas/request.ts)
- [response.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/schemas/response.ts)

当前已实现：

- 通过大模型先解析 `search_text / compare_focus / baseline_* / target_* / reasoning`。
- 如果只给了一侧窗口，会用 `normalizeCompareWindows()` 自动补另一侧。
- 如果两侧都没给，会默认构造“最近 7 天”与“更早 7 天”的双窗口。
- 会先做向量检索，再补：
  - `processing_tasks.display_name`
  - `model_assets` 中的 `description / objects / tags / ply_path / created_at`
  - `memory_poses.tag`
- 输出协议中已经明确包含：
  - `intent`
  - `comparison.baseline`
  - `comparison.target`
  - `comparison.diff`
  - `toolTrace`
  - `actions`
- 双侧动作会带 `slot: baseline | target`，不是简单返回两个无标签动作。

当前限制：

- 它依赖检索命中的最佳候选，不是同一空间严格配准后的扫描对齐。
- 差分基于 `objects / tags / matched frame tag` 等结构化线索，不能等同于几何变化检测。
- 它不是 LangChain tool calling 实现，仍然是专用流程型 Agent。

### 9.7 当前测试覆盖的真实口径

测试文件仍然是下面这些：

- [agent-recall/test.ts](/home/ltx/projects/BrainDance/supabase/functions/agent-recall/test.ts)
- [spatial-search-agent/test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/test.ts)
- [spatial-search-agent/assetTools.test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/assetTools.test.ts)
- [time-compare-agent/test.ts](/home/ltx/projects/BrainDance/supabase/functions/time-compare-agent/test.ts)

但真实覆盖点比旧文档写得更细：

- `agent-recall/test.ts`
  - 除了旧 helper 的动作与 evidence 提取，还验证了 `runRecallAgent()` 的 SSE/流式事件序列。
- `spatial-search-agent/test.ts`
  - 实际测试的是 `_shared/agent-core/spatialAgent.ts` 导出的公共函数，而不只是 HTTP 入口。
- `assetTools.test.ts`
  - 实际测试目标也是 `_shared/agent-core/assetTools.ts`，只是测试文件仍挂在 `spatial-search-agent/` 目录下。
- `time-compare-agent/test.ts`
  - 已覆盖窗口补全、对象/标签差分、双槽位动作生成。

当前仍未看到的测试：

- 没有看到直接跑真实 Supabase / DashScope 的联调测试。
- 没有看到 `asset_metadata` 写入分支的端到端测试。
- 没有看到 `spatial_search` 与 `asset_metadata` 两种模式切换的端到端回归测试。

### 9.8 截至 2026-03-26 的真实进度总结

按当前代码，更准确的状态应该表述为：

- 已完成一套共享 LangChain TS Agent 核心，位置在 [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)。
- 已完成一个实验入口 `spatial-search-agent`，可处理空间检索和资产元数据两类请求。
- 已完成一个稳定协议入口 `agent-recall`，并且已经具备 SSE 事件流和统一核心结果映射能力。
- 已完成一个专用的 `time-compare-agent`，能输出双时间窗口差分和双槽位动作。
- 已完成资产元数据的最小可执行操作链路，但目前更适合受控场景，不宜直接表述为“完全稳定可对外开放”。

仍处于未完全收口状态的点：

- `agent-recall` 与 `spatial-search-agent` 的动作协议仍有两套命名。
- 共享 Agent 核心已经落地，但目录和测试文件名还保留部分旧结构，文档和代码位置容易继续漂移。
- 资产操作虽然可执行，但正式执行分支的联调和回归证据还不充分。
- 时间对比仍是检索+元数据差分，不是严格的跨扫描时序对齐能力。
