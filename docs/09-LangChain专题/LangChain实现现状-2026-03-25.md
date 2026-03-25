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
