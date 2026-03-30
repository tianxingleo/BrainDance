# Supabase Deno LangChain Agent 开发汇总（基于 dev 分支）

## 1. 文档目的

本文档只讨论 BrainDance 在 `dev` 分支上已经落地的 `Supabase + Deno + LangChain Agent` 相关工作，不再展开 3DGS、Qwen3 微调、Flutter 本地 GGUF 推理等其它主题。

目标是把这条线讲清楚：

- 这套 Agent 到底是为了解决什么问题。
- 它在系统里放在哪一层。
- 从 `search-models` 到 `spatial-search-agent` 再到 `agent-recall` 是怎么演进的。
- 现在真实代码里已经做到了什么。
- 当前主线、实验线、稳定协议和剩余问题分别是什么。

本文档以 `dev` 分支代码、文档和提交记录为准。

---

## 2. 先给结论

截至 `dev` 分支当前状态，BrainDance 的 Agent 线已经不是“做了一个能搜东西的 Demo”，而是已经形成了一套比较完整的在线编排体系：

1. 在线能力明确放在 `Supabase Edge Functions (Deno / TypeScript)` 侧，而不是 Python Worker 侧。
2. `search-models` 被收敛成共享检索底座，负责时间解析、Embedding、向量检索和基础结果结构。
3. `agent-recall` 已经成为正式统一入口，前端应优先面向它联调。
4. 共享核心能力已经收敛到 `supabase/functions/_shared/agent-core/spatialAgent.ts`。
5. 共享 Core 已不再只处理空间搜索，而是扩展成多模式 Agent：
   - `spatial_search`
   - `asset_metadata`
   - `time_compare`
   - `creative`
   - `memory_graph`
6. Flutter Recall 页已经能消费这套流式 Agent 协议，包括状态事件、工具轨迹、候选结果、最终回答、多轮续聊状态和 follow-up。
7. 当前最大进展，不是又加了多少 prompt，而是把协议、工具边界、状态续传、流式反馈、调试 CLI 和批量回归都补齐了。

如果只用一句话概括：

> `dev` 分支上的这条线，已经从“搜索接口”升级成了“围绕三维记忆资产进行检索、整理、比较和后续动作编排的在线 Agent Runtime”。

---

## 3. 这条线要解决什么问题

BrainDance 不是一个纯聊天应用，它的核心对象是：

- `model_assets`
- `memory_poses`
- 场景描述
- 对象标签
- 时间范围
- 视角证据
- 3D 场景动作

所以它需要的不是“一个泛用聊天机器人”，而是一个能围绕这些真实资产工作的 Agent。

从业务上看，用户问题大致有几类：

1. 空间检索
   - “黑色耳机在哪”
   - “窗边那个台灯还在吗”
   - “最像厨房角落堆着纸箱的空间”
2. 时间对比
   - “和上周相比有什么变化”
   - “这个空间前后有什么差异”
3. 资产整理
   - “把最新三个模型改名为宿舍合集”
   - “整理一下最近的模型”
4. 记忆总结和延展
   - “这些模型能组成什么专题”
   - “近期有什么变化趋势”

普通搜索接口最多只能返回结果列表，但无法稳定表达：

- 当前到底命中了哪一个场景。
- 为什么选它。
- 前端接下来该执行什么动作。
- 当前对话是不是还需要用户补充信息。
- 这次请求是不是已经进入“可执行写操作”的阶段。

因此，这条 Agent 线的本质任务是把已有空间记忆底座升级为：

- 可路由
- 可调工具
- 可返回动作
- 可多轮续聊
- 可做状态管理
- 可流式反馈

的在线编排层。

---

## 4. 总体架构判断

这条线的核心架构判断在 [Agent规划与LangChain实践路线.md](/ltx-data/BrainDance/docs/02-架构设计/Agent规划与LangChain实践路线.md) 里写得很明确，真实代码也基本按这个方向推进。

### 4.1 为什么 Agent 放在 Supabase Deno，而不是 Python Worker

原因很直接：

1. Python Worker 主要负责重计算：
   - 3DGS
   - 理解
   - 打标
   - Embedding 写入
2. 前端在线交互本质上更像 BFF / orchestration：
   - 接收用户查询
   - 读数据库
   - 调检索
   - 组织响应
   - 返回动作
3. Supabase Edge Functions 已经天然处在前端直连层。

所以在当前架构下：

- Python Worker 负责“生产资产”
- Supabase Deno Agent 负责“消费资产并编排响应”

这是这条线能够持续推进的前提。

### 4.2 为什么不把 `search-models` 直接做成大 Agent

因为 `search-models` 的职责应该是清晰、稳定、可复用的工具底层，而不是把所有逻辑都堆进去。

所以最终形成了两层：

```text
search-models   -> 共享检索底座
agent-recall    -> 正式统一 Agent 入口
```

这让后续修改时间解析、阈值、Embedding、召回逻辑时，可以优先在共享层收口，而不是多处重复维护。

---

## 5. 时间线：这条线是怎么长出来的

按 `dev` 分支提交看，这条线的演进非常清楚。

### 5.1 第一阶段：确立 All-in-Supabase 方向

关键提交：

- `3496967` / `32f598d` `docs: 重构 README 以反映 All-in-Supabase 架构演进 (移除 Go/Redis)`

这一阶段的重要意义不是功能，而是路线定了：

- 后端交互层以 Supabase 为中心
- 在线服务走 Edge Functions
- 不再走过去那种多中间件堆叠思路

### 5.2 第二阶段：先有规划，再有第一版 Agent

关键提交：

- `2ea6bcc` `docs(agent): 补充 Agent 三步规划与 LangChain 实践路线`
- `f3cb573` `feat(supabase): 新增空间检索 Agent`

这时的核心工作是：

1. 把 Agent 定位成“空间检索 Agent”，不是通用大 Agent。
2. 落地第一版 `spatial-search-agent`。

这说明团队当时的策略是对的：先做最有价值的空间检索，再逐步扩。

### 5.3 第三阶段：引入稳定入口 `agent-recall`

关键提交：

- `16de256` `feat(agent): 新增 agent-recall 稳定入口并固化动作协议`
- `a936e07` `refactor(agent): 让 agent-recall 直接复用 search-models 共享逻辑`
- `8dd1e48` `refactor(search): 收口 search-models 共享入口并补联调清单`

这一阶段的关键变化是：

1. 不再让前端直接面向实验链路。
2. 增加稳定入口 `agent-recall`。
3. 把 `search-models` 改造成共享逻辑底座。

也就是从“实验 Agent”迈向“可接前端的正式协议”。

### 5.4 第四阶段：扩展多模式能力

关键提交：

- `844653d` `feat(supabase): 新增时间对比 Agent 并同步架构文档`
- `4fc3f7c` `feat(supabase): 为模型资产 Agent 增加元数据工具链`
- `862c231` `feat(supabase): 接入模型资产元数据 Agent 路由`
- `d19e93a` `refactor(agent): 升级 agent-recall 为统一多工具编排入口`

这一阶段，Agent 已经从单一空间搜索扩展成多模式：

- 空间搜索
- 资产元数据操作
- 时间对比

这一步是整个系统真正开始“像 Agent”而不是“像搜索接口”的转折点。

### 5.5 第五阶段：统一共享 Core，做状态和协议收口

关键提交非常密集，集中在 2026-03-26：

- `705bc1d` `彻底收口 agent-recall 入口，消除协议漂移`
- `7ae3aec` `改进与扩充 Agent Context，引入 Few-Shot 提示词模块化管理`
- `34703d3` `完成 P0 与 P1 阶段残留对齐`
- `b9c33d0` / `eb8e1e7` `统一 Recall 多模式入口并补齐记忆工具链`
- `9441752` `收紧多模式 Agent 协议与会话上下文`
- `cd830ac` `实现 Agent 空间检索的流式响应与实时反馈`
- `5bad8c2` `打通 Recall Agent 多轮续聊链路`

这里的本质，不是又多了几个工具，而是：

1. 共享 Core 真正集中起来了。
2. 请求/响应协议被系统化。
3. 会话状态开始可续传。
4. 流式事件被前后端一起接通。

### 5.6 第六阶段：调试、回归和稳定性打磨

关键提交：

- `934438a` `新增桌面端 agent-recall 调试 CLI`
- `c16087a` `增强调试 CLI 的分析与日志能力`
- `7461416` `新增 agent-recall 批量测试脚本与发散数据集`
- `1664f2e` `接通 agent-recall 本地流式冒烟`
- 一系列 `fix(agent)`、`fix(agent-recall)`、`test(integration)` 提交

这说明工程进入了一个很关键的阶段：

- 主体架构不再频繁重来
- 开始大量补联调工具、批量回归和协议兼容

这通常意味着系统已经从“设计期”进入“稳定化期”。

---

## 6. 真实代码结构

当前 `dev` 分支上，这条线的核心代码主要分成五层。

### 6.1 第一层：共享检索底座

文件：

- [search-models/index.ts](/ltx-data/BrainDance/supabase/functions/search-models/index.ts)
- [search-models/shared.ts](/ltx-data/BrainDance/supabase/functions/search-models/shared.ts)

它负责：

1. 解析用户查询中的时间语义。
2. 调 DashScope 生成 Embedding。
3. 调用 Supabase RPC `match_memory_poses` 做向量搜索。
4. 对结果补齐 `display_name`。
5. 返回基础结构化结果。

它不是 Agent，但它是所有 Agent 的检索底层。

### 6.2 第二层：正式统一入口 `agent-recall`

文件：

- [agent-recall/index.ts](/ltx-data/BrainDance/supabase/functions/agent-recall/index.ts)
- [agent-recall/schemas/request.ts](/ltx-data/BrainDance/supabase/functions/agent-recall/schemas/request.ts)

职责：

1. 作为正式 HTTP 入口。
2. 校验请求参数。
3. 判断是否走流式协议。
4. 调用共享 Core。
5. 把事件按 `SSE / NDJSON` 两种格式发回前端。

它本身不承载全部业务逻辑，而是负责接线和协议。

### 6.3 第三层：共享 Agent Core

文件：

- [spatialAgent.ts](/ltx-data/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)

这是整条线的核心。

当前它已经承担：

1. 路由当前请求属于哪种模式。
2. 维护统一上下文。
3. 调用空间检索、资产工具、时间对比、创作、弱图谱等能力。
4. 生成统一响应结构。
5. 发送结构化进度事件。
6. 管理会话状态和 follow-up。

换句话说，真正的 Agent 大脑已经收口到这里了。

### 6.4 第四层：具体工具与模式能力

文件包括：

- [assetTools.ts](/ltx-data/BrainDance/supabase/functions/_shared/agent-core/assetTools.ts)
- [memoryTools.ts](/ltx-data/BrainDance/supabase/functions/_shared/agent-core/memoryTools.ts)
- [time-compare-agent/agent.ts](/ltx-data/BrainDance/supabase/functions/time-compare-agent/agent.ts)

这些文件提供了：

1. 资产读取、改名、批量修改、对比等工具。
2. 记忆专题、模型分组、pose 摘要、趋势分析等工具。
3. 双时间窗口对比能力。

### 6.5 第五层：前端消费

文件：

- [agent_recall_service.dart](/ltx-data/BrainDance/app/lib/services/agent_recall_service.dart)
- [recall.dart](/ltx-data/BrainDance/app/lib/pages/recall.dart)
- [recall_agent_runtime.dart](/ltx-data/BrainDance/app/lib/pages/recall/recall_agent_runtime.dart)

职责：

1. 构造请求体。
2. 解析流式事件。
3. 展示状态、工具调用、工具结果、最终答案。
4. 保存并续传 `session_state` 和 `conversation_summary`。
5. 消费 `follow_up` 并展示快捷回复。

---

## 7. Agent 的模式设计

目前共享 Core 已经支持五种模式。

### 7.1 `spatial_search`

这是最初也是最核心的模式。

解决的问题：

- 找某个物体
- 找某个位置
- 带时间过滤找场景
- 找和某种描述最像的空间

当前能力：

1. 意图解析。
2. 检索工具调用。
3. 候选合并和排序。
4. 生成空间证据。
5. 返回前端动作。

### 7.2 `asset_metadata`

这是从“只搜”升级到“能整理资产”的关键模式。

当前能力：

1. 读取模型资产列表。
2. 读取模型 bundle。
3. 单模型改名。
4. 批量改名或修改元数据。
5. 模型间对比。
6. 对“最新模型改名”“最新 N 个模型批量改名”做确定性兜底。

它的工程意义非常大，因为它把 Agent 从“回答问题”扩展成“帮你管理模型资产”。

### 7.3 `time_compare`

这是时间对比的最小可用模式。

当前能力：

1. 解析 baseline / target 时间窗口。
2. 分别搜索两个时间窗的候选。
3. 对比 `objects / tags / description` 差异。
4. 返回双侧场景与动作建议。

当前限制也很明确：

- 它还不是几何对齐后的强对比。
- 更接近“基于搜索命中和元数据的时间窗口差分”。

### 7.4 `creative`

当前已经能：

1. 基于选中模型生成创作上下文。
2. 生成故事或导览提纲。
3. 在 `execute` 模式下把任务入队到 `processing_tasks`。

但还没有完整创作产物流水线，所以它现在更像“创作任务编排入口”。

### 7.5 `memory_graph`

当前能做的，是弱图谱摘要，而不是完整图数据库。

比如：

1. 最近趋势。
2. 缺失模式。
3. 某个地点的变化时间线。
4. 个人记忆摘要。

这代表项目已经开始思考长期记忆层，但还没有完全做完。

---

## 8. 协议设计：这条线成熟的关键

这条线真正成熟的地方，不只是代码多，而是协议开始稳定。

### 8.1 请求协议

`agent-recall` 当前已支持：

- `query`
- `selectedModelIds`
- `executionMode`
- `currentSceneId`
- `currentModelId`
- `currentMode`
- `candidateSceneIds`
- `sessionId`
- `conversationSummary`
- `sessionState`

这说明它已经考虑了：

1. 前端当前上下文。
2. 多选模型操作范围。
3. 预览和执行两种模式。
4. 多轮对话续传。

### 8.2 正式响应协议

当前正式返回至少包含：

- `mode`
- `answer`
- `evidence`
- `actions`
- `top_candidates`
- `selected_candidate_reason`
- `tool_trace`
- `session_state`
- `conversation_summary`
- `follow_up`

这意味着它不只是聊天返回，而是同时服务于：

1. 用户读答案。
2. 前端执行动作。
3. 调试和诊断。
4. 多轮续聊。

### 8.3 动作协议

当前正式动作协议已经收口成两种：

1. `open_scene`
2. `fly_to_pose`

这是非常正确的工程收口。

早期一些实验名字，比如 `highlight_hotspot`，现在被明确限制在实验链路，不作为正式稳定协议。

### 8.4 会话协议

当前最有价值的一个补丁，是把会话状态协议做出来了。

包括：

- `session_state.lastMode`
- `lastSelectedModelIds`
- `lastCandidateRefs`
- `lastOperationPreview`

这让如下场景可以稳定成立：

1. 上一轮预览改名。
2. 用户下一轮说“确认执行”。
3. Agent 不需要重新猜目标，而是直接重放上一轮参数。

这类能力，是很多看起来“像 Agent”的系统真正不好做的地方。

---

## 9. 流式事件：从“有结果”到“有过程”

这是 2026-03-26 之后非常明显的强化点。

### 9.1 后端流式协议

`agent-recall/index.ts` 当前支持：

- `text/event-stream`
- `application/x-ndjson`

并会下发事件：

- `ping`
- `status`
- `plan`
- `thought`
- `tool_call`
- `tool_result`
- `message`
- `done`
- `error`

### 9.2 为什么这很重要

因为之前的问题不是“没有返回答案”，而是：

- 前端不知道 Agent 在干什么。
- 用户只看到一个 loading。
- 出错时不知道卡在哪一步。

流式事件补齐后：

1. 用户可以看到阶段推进。
2. 前端可以展示工具轨迹。
3. 调试时能区分是意图解析卡住、工具失败、还是回答阶段出问题。

### 9.3 Flutter 消费层的作用

Flutter 侧不是简单把文本流拼起来，而是把事件做成：

1. 状态卡片。
2. 步骤时间线。
3. 工具调用记录。
4. 结果自动折叠。

这说明 Agent 体验已经从“后端好不好用”走到了“前端怎么呈现更合理”。

---

## 10. 当前已经落地的关键工程能力

这里把最值得关注的实质成果单独列出来。

### 10.1 共享检索逻辑收口

`search-models/shared.ts` 成为共享底座后，时间解析、Embedding、RPC 检索不再散落多处。

这解决的是长期维护问题。

### 10.2 从单一空间检索扩展成多模式 Agent

不是单纯“再加几个 prompt”，而是已经形成模式路由和模式化输出。

### 10.3 资产操作具备预览与执行分离

`executionMode = preview / execute` 是一个非常重要的安全边界。

这让写库类操作不至于一上来就真的改数据。

### 10.4 最新模型改名和批量改名的确定性兜底

这类看似小功能，其实最能体现系统是否真正可用。

因为完全靠 LLM 自由推理时，这类请求最容易出现范围漂移和误写。

### 10.5 会话状态与操作预览重放

这是多轮 Agent 真正好用的关键，不然“确认执行”永远会变成重新问一遍。

### 10.6 无候选时的通用 fallback

2026-03-27 专门修了“你是谁”这类问题。

重要的不是这几个问句本身，而是共享 Core 终于具备了：

- 没有候选时不直接抛错
- 同一个 Agent 自己自然收口

这是一种成熟度体现。

### 10.7 桌面调试 CLI

`agent_recall_debug_cli.py` 的价值非常高：

1. 不用起 Flutter 就能复现流式链路。
2. 能直接看到所有事件。
3. 能看候选、工具轨迹、最终结果和中断原因。

这大幅降低了联调成本。

### 10.8 批量回归脚本和发散数据集

`run_agent_recall_batch_suite.py` 和 `agent_recall_batch_suite.json` 说明这条线已经不再只靠人手点，而开始具备批量回归能力。

这是非常关键的工程跃迁。

---

## 11. 当前还没有做完的地方

虽然这条线已经很完整，但并不是“全部收尾”。

### 11.1 历史数据层还没完全回填

比如：

- `place_id`
- `memory_thread_id`

这些字段已经进入迁移设计，但历史数据回填策略还没完成。

### 11.2 `related_model_links` 还缺后台构建

现在更多还是启发式检索和即时组织，不是真正长期维护好的关联图。

### 11.3 `creative` 模式只有入队，没有完整生产流水线

它能排任务，但还不是完整“创作 Agent 产品功能”。

### 11.4 `memory_graph` 还是弱图谱摘要

目前还是总结和趋势，不是强关系数据库。

### 11.5 前端展示仍偏“最小可用”

Flutter Recall 已经能用，但 mode-aware 的完整 UI 和更精细的交互仍有继续打磨空间。

### 11.6 实验链和正式链仍要继续统一

虽然共享 Core 已大幅收口，但：

- `spatial-search-agent`
- `agent-recall`

这两条线的历史包袱还没完全消失，文档里也多次提到口径漂移和协议修复。

---

## 12. 当前主线、实验线和稳定入口应该怎么理解

这是最容易混淆的地方。

### 12.1 正式稳定入口

当前应视为正式稳定入口的是：

- `supabase/functions/agent-recall/index.ts`

原因：

1. 前端逐步转向消费它。
2. 协议最完整。
3. 会话状态、follow-up、流式事件、多模式都在这里统一承接。

### 12.2 共享核心主线

真正的主线实现不是 `agent-recall/index.ts` 本身，而是：

- `supabase/functions/_shared/agent-core/spatialAgent.ts`

这是整个 Agent 线最应该持续维护和收口的文件。

### 12.3 实验链或历史链

`spatial-search-agent` 更像：

- LangChain TS 实验起点
- 早期主实验场
- 某些测试仍依赖的链路

它仍然重要，但不该再被视作未来唯一前端入口。

---

## 13. 对当前进度的判断

如果按工程成熟度分级，我会这样看。

### 13.1 已经完成的

1. 架构路线清晰。
2. 共享检索底座清晰。
3. 正式统一入口已形成。
4. 多模式共享 Core 已形成。
5. 正式请求/响应协议已形成。
6. 流式协议已打通。
7. 多轮会话状态已打通。
8. 桌面联调工具已落地。
9. 批量回归脚本和发散数据集已落地。

### 13.2 正在稳定化的

1. 协议兼容性。
2. Flutter 流式消费体验。
3. 资产类工具的退出策略。
4. 无候选和非检索直答场景。
5. 批量回归覆盖率。

### 13.3 还在继续建设的

1. 更完整的记忆图谱。
2. 更强的时间对比。
3. 创作模式的下游工作流。
4. 历史数据回填与长期治理。

---

## 14. 最终判断

只看 `Supabase + Deno + LangChain Agent` 这条线，`dev` 分支当前的真实状态可以概括成下面几句话：

1. 它已经不再是“想做 Agent”，而是已经做出了一个可工作的在线 Agent Runtime。
2. 它最重要的成果不是某个单独接口，而是 `search-models -> agent-recall -> shared agent core -> Flutter consumer` 这整条链路成型了。
3. 它最成熟的地方，不是 prompt，而是协议、状态、工具边界、流式事件和回归体系。
4. 当前主线应该继续围绕 `agent-recall + shared agent core` 收口，而不是重新分叉出更多入口。
5. 这条线已经进入“稳定化和扩展化并行”的阶段，不再是早期概念验证。

如果后续还要继续写这条线，建议优先围绕以下三件事推进：

1. 继续收口共享 Core，减少历史入口和协议漂移。
2. 把现有批量回归真正跑起来，形成基线结果。
3. 把 `time_compare / creative / memory_graph` 从“可调用”继续推进到“可稳定交付”。

---

## 15. 本文依据的关键文件

架构与说明：

- [Agent规划与LangChain实践路线.md](/ltx-data/BrainDance/docs/02-架构设计/Agent规划与LangChain实践路线.md)
- [Agent联调与回归清单.md](/ltx-data/BrainDance/docs/02-架构设计/Agent联调与回归清单.md)
- [LangChain专题 README](/ltx-data/BrainDance/docs/09-LangChain专题/README.md)
- [LangChain实现现状-2026-03-25.md](/ltx-data/BrainDance/docs/09-LangChain专题/LangChain实现现状-2026-03-25.md)
- [LangChain实现现状-2026-03-26.md](/ltx-data/BrainDance/docs/09-LangChain专题/LangChain实现现状-2026-03-26.md)
- [LangChain实现现状-2026-03-27.md](/ltx-data/BrainDance/docs/09-LangChain专题/LangChain实现现状-2026-03-27.md)

后端代码：

- [search-models/index.ts](/ltx-data/BrainDance/supabase/functions/search-models/index.ts)
- [search-models/shared.ts](/ltx-data/BrainDance/supabase/functions/search-models/shared.ts)
- [agent-recall/index.ts](/ltx-data/BrainDance/supabase/functions/agent-recall/index.ts)
- [spatialAgent.ts](/ltx-data/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)
- [time-compare-agent/agent.ts](/ltx-data/BrainDance/supabase/functions/time-compare-agent/agent.ts)

前端代码：

- [agent_recall_service.dart](/ltx-data/BrainDance/app/lib/services/agent_recall_service.dart)
- [recall.dart](/ltx-data/BrainDance/app/lib/pages/recall.dart)
- [recall_agent_runtime.dart](/ltx-data/BrainDance/app/lib/pages/recall/recall_agent_runtime.dart)

调试与回归：

- [agent_recall_debug_cli.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py)
- [run_agent_recall_batch_suite.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_agent_recall_batch_suite.py)
