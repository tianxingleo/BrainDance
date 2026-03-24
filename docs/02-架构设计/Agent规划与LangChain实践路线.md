# BrainDance Agent 规划与 LangChain 实践路线

> 本文档用于沉淀 BrainDance 在 Agent 方向上的近期规划、工程边界与三步实施路径，避免后续讨论停留在概念层。

## 1. 目标与背景

BrainDance 当前已经具备一条较完整的空间记忆主链路：

- Flutter/Web 前端直接连接 Supabase。
- `supabase/functions/search-models` 已承担自然语言检索入口。
- `model_assets`、`memory_poses`、`match_memory_poses` 已提供空间资产与帧级锚点能力。
- Python Worker 负责 3DGS 重建、理解、打标、Embedding 写入等重计算任务。

因此，BrainDance 要做的不是从零搭一个“万能 Agent”，而是把现有的空间检索链路升级为“可调用工具 + 可返回动作 + 可持续演进”的 Agent Runtime。

本文档对应的核心判断是：

- 第一阶段优先做 `空间检索 Agent`，而不是通用大 Agent。
- 在线 Agent 更适合放在 Supabase Edge Functions / TypeScript 一侧，而不是 Python Worker 一侧。
- Agent 的价值不只是回答文本，而是返回 `空间证据 + 空间动作`。

## 2. 架构判断

### 2.1 为什么优先 LangChain TS，而不是 LangChain Python

当前 BrainDance 的系统分层已经比较明确：

- Python 侧主要承担 3DGS、离线理解、Embedding 写入和批处理任务，属于重计算层。
- Supabase 是前端直接交互的在线接口层，Edge Functions 已承载 `search-models` 等实时能力。
- Recall 搜索、Viewer 打开、视角跳转这些在线交互，本质上更接近 BFF / orchestration，而不是训练或推理服务。

因此，现阶段更合理的职责划分是：

- Python Worker：负责生产空间记忆资产。
- Supabase Edge Functions：负责消费空间记忆资产，并对前端提供在线 Agent 能力。
- Flutter / Web：负责执行 Agent 返回的结构化动作。

这个判断的直接结论是：

- 第一版 Agent 优先采用 TypeScript 方案。
- LangChain TS 可以作为在线 Agent 编排框架，但不应越过现有工具层直接操作底层数据。
- Python 侧暂不承担在线 Agent Runtime。

### 2.2 Agent 不是替代 `search-models`

现有 `search-models` 应继续保留为专用搜索工具，而不是直接膨胀成大而全的 Agent。

建议在 Supabase 层形成如下职责分离：

```text
supabase/functions/
├── search-models/     # 专用搜索工具，负责时间解析、embedding、向量检索
└── agent-recall/      # 新增总入口，负责路由、工具调用、动作组织
```

这样可以保持：

- `search-models` 的工具职责清晰。
- `agent-recall` 的接口语义稳定。
- 前端不需要感知内部的工具编排细节。

## 3. Agent 设计原则

### 3.1 先做可控 Agent，再做复杂 Agent

BrainDance 当前最缺的不是“更聪明的推理”，而是以下基础设施：

- 稳定的工具契约
- 稳定的动作协议
- 明确的查询分类
- 可回归的评测集
- 可追踪的调用日志

因此，第一阶段不追求自由发挥，而追求：

- 可解释
- 可测试
- 可前端消费
- 可持续扩展

### 3.2 Agent 负责决策，工具负责事实

Agent 层只负责：

- 判定用户意图
- 选择工具
- 组织最终响应

工具层负责：

- 搜索空间
- 读取资产
- 返回证据视角
- 生成动作描述

禁止让 Agent 在第一阶段直接：

- 拼接 SQL
- 越过既有接口直接查底表
- 直接操作 Viewer 的低层状态
- 在无证据时编造空间、时间或物体位置

### 3.3 BrainDance 的回答必须带动作

BrainDance 的 Agent 不应只返回一句文本，而应返回：

- `answer`：人类可读回答
- `evidence`：空间证据
- `actions[]`：前端可执行动作

这也是 BrainDance 与普通聊天 Agent 的核心差异。

建议统一响应结构：

```ts
type AgentRecallResponse = {
  answer: string;
  evidence: {
    sceneId: string;
    similarity: number;
    matchedFrames: Array<{
      imageName: string;
      similarity: number;
      transformMatrix: unknown;
    }>;
  } | null;
  actions: Array<
    | { type: "open_scene"; sceneId: string }
    | { type: "fly_to_pose"; sceneId: string; imageName?: string }
    | { type: "highlight_region"; sceneId: string; label?: string }
  >;
};
```

## 4. 第一阶段目标排序

结合现有仓库能力，Agent 方向建议按下面顺序推进：

### P0：空间检索 Agent

最先落地，直接复用现有 `search-models`、`memory_poses`、`match_memory_poses`。

核心体验：

- 用户输入自然语言
- 系统完成搜索与候选选择
- 返回答案、证据视角与 Viewer 动作

### P1：记忆整理 Agent

在已有资产、标签、描述、Embedding 的基础上做内容组织与归档。

适合的能力包括：

- 自动命名空间
- 自动生成摘要标题
- 归档专题集合
- 合并同一地点的多次扫描

### P2：时间对比 Agent

产品辨识度高，但对数据层要求更高，不建议抢在 P0 前面做。

在缺少稳定 `place_id / memory_thread_id / alignment` 抽象前，当前能力更接近“按时间过滤搜索”，还不足以稳定支持“同一空间跨时间比较”。

### P3：第二大脑 / 长期记忆图谱

属于长期方向，需要建立跨空间、跨时间的长期状态层，不作为近期交付目标。

### P4：多模态创作 Agent

适合展示，但不应优先于检索和整理主链路。

## 5. 三步实施路线

### 第一步：搭好 Agent 地基

### 5.1 目标

先把现有空间检索链路包装成稳定工具层和动作协议，不急着追求复杂推理。

### 5.2 本阶段要完成的事

1. 新增 `supabase/functions/agent-recall/` 作为总入口。
2. 保留 `search-models` 为专用搜索工具，不直接改造成大 Agent。
3. 固定响应协议，统一返回 `answer + evidence + actions`。
4. 定义第一批工具：
   - `searchSpace(query, threshold?)`
   - `getSceneAsset(sceneId)`
   - `buildViewAction(sceneId, matchedFrame?)`
5. 建立一套最小评测集，例如 `tests/agent_recall_cases.jsonl`：
   - 找物体
   - 找位置
   - 带时间过滤
   - 模糊空间描述

### 5.3 本阶段的完成标志

- 前端可以调用 `agent-recall`。
- `agent-recall` 可以复用 `search-models`。
- 接口返回结构化证据与动作。
- 至少有一套回归题集可重复验证。

### 5.4 本阶段不要做什么

- 不上来就做 LangGraph 长流程。
- 不直接做长期记忆图谱。
- 不做跨时空复杂对齐。
- 不把业务逻辑散落到前端。

### 第二步：做最小可用空间检索 Agent

### 5.5 目标

把“搜索接口”升级成“搜索 + 决策 + 动作输出”的最小可用 Agent。

### 5.6 推荐推进方式

建议分两小步做：

- v2.0：先手写轻量 router，验证业务链路。
- v2.1：再接入 LangChain TS，让框架负责路由、工具调用和结构化输出。

### 5.7 第一版意图分类

第一版只建议支持 4 类意图：

- `object_lookup`
- `location_lookup`
- `time_filtered_search`
- `scene_similarity_search`

示例：

- “黑色耳机在哪” → `object_lookup`
- “窗边那个台灯还在吗” → `location_lookup`
- “上周拍到的红色杯子” → `time_filtered_search`
- “最像厨房角落堆纸箱的三个空间” → `scene_similarity_search`

### 5.8 LangChain TS 在第二步中的职责

LangChain TS 只负责三件事：

- 判断 query 属于哪类
- 决定要不要调用工具
- 组织最终响应

不负责：

- 拼数据库查询
- 直接写表
- 绕过 `search-models`
- 直接操控 Viewer 内部状态

### 5.9 本阶段的产出

前端至少打通两类动作：

- `open_scene`
- `fly_to_pose`

用户输入类似：

> 帮我找去年那次扫描里书桌上的黑色耳机

系统输出应包含：

- 一句基于证据的回答
- Top 结果对应的空间证据
- 可直接执行的打开场景和飞行动作

### 5.10 本阶段的完成标志

- 用户单轮输入可触发完整 Agent 链路。
- Agent 能稳定走到 `search-models`。
- 能返回 Top scene 与 matched frame。
- 前端可以消费 `open_scene` / `fly_to_pose`。
- 至少能演示 5 个真实查询样例。

### 第三步：扩展到时间对比与记忆整理

### 5.11 目标

把 Agent 从“单次空间检索”扩展到“跨时间比较”和“内容归档整理”。

### 5.12 先补数据层，再补 Agent

时间对比能力需要先补以下概念：

- `place_id`：同一物理空间的稳定标识
- `memory_thread_id`：同一地点多次扫描的归档组
- `scan_version`：版本信息
- `alignment_status / alignment_transform`：跨扫描对齐结果

在这层没补齐前，不建议直接承诺强时序比较体验。

### 5.13 第三步建议分成两条线

A. 时间对比 Agent

- `listPlaceVersions(placeId)`
- `compareTwoScans(scanA, scanB)`
- `summarizeSpatialDiff(diffResult)`

返回内容应包括：

- 变化摘要
- 新增 / 消失 / 位移的对象
- 对应时间点
- 证据视角

B. 记忆整理 Agent

基于现有 `model_assets`、标签、描述、Embedding，先做更容易落地的内容组织能力，例如：

- 自动命名空间
- 自动生成摘要标题
- 自动归档专题
- “2024 搬家记忆集”
- “大学四年的宿舍变化集”

### 5.14 本阶段的完成标志

满足下列任意两项即可视为进入可用状态：

- 能回答“这个房间两个月前和现在有什么变化”
- 能返回变化摘要与证据视角
- 能把一组相关空间整理成专题
- 能自动生成一条 memory collection

## 6. 推荐目录与接口边界

建议后续在 Supabase 层采用如下目录：

```text
supabase/functions/
├── search-models/
│   └── index.ts
└── agent-recall/
    ├── index.ts
    ├── tools/
    │   ├── searchSpace.ts
    │   ├── getSceneAsset.ts
    │   └── buildViewAction.ts
    ├── prompts/
    │   └── recallSystemPrompt.ts
    ├── schemas/
    │   ├── request.ts
    │   └── response.ts
    └── agent/
        └── recallAgent.ts
```

对应职责边界如下：

- `search-models`：时间解析、Embedding、向量检索
- `agent-recall`：意图路由、工具调用、结果组织、动作输出
- Python Worker：重建、打标、Embedding 写入、离线任务
- Flutter/Web：渲染场景、执行动作、展示证据

## 7. Prompt 与工具约束

第一版系统 Prompt 应保持克制，重点强调“只根据工具结果回答”：

```text
你是 BrainDance Recall Agent。
你的任务不是自由聊天，而是：
1. 理解用户是否在查找空间记忆中的物体、位置、时间或场景。
2. 必要时调用工具。
3. 只根据工具结果回答。
4. 如果存在空间证据，输出前端可执行动作。
5. 找不到时明确说明，不要编造。
```

工具输出必须优先采用稳定 JSON，而不是松散自然语言，避免后续评测与前端联调失控。

## 8. 工程实践要求

为避免 Agent 方向失控，后续落地应同时满足以下要求：

- 先固定工具契约，再讨论复杂 Prompt。
- 先做 deterministic skeleton，再逐步增加 agentic flexibility。
- 第一阶段必须建立评测集和 trace 机制。
- 文档描述必须与真实工具行为一致，避免注释与实现漂移。
- 对涉及 RLS、用户身份透传和服务角色密钥的链路，必须先做安全核查。

建议尽早补充：

- `agent_recall_cases.jsonl`
- 接口调用日志 / trace
- Top-K rerank 与证据摘要验证集
- Viewer 动作执行结果检查

## 9. 近期实施建议

近期建议按下面顺序推进：

1. 清理现有搜索链路的工具契约与文档口径。
2. 新增 `agent-recall`，先跑通普通 TS 编排。
3. 固定 `answer + evidence + actions` 响应协议。
4. 补最小评测集与回归脚本。
5. 在第二阶段再接 LangChain TS，避免过早被框架牵着走。
6. 待 `place_id / memory_thread_id / alignment` 补齐后，再进入时间对比 Agent。

## 10. 总结

BrainDance 的 Agent 方向，不应围绕“我也有一个聊天 Agent”来设计，而应围绕下面这条原则：

> 答案本身还不够，Agent 必须返回空间证据，并驱动空间动作。

只要坚持这个原则，BrainDance 的 Agent 就不是普通 RAG 的变体，而是能够把回答落回真实空间界面的空间记忆代理。
