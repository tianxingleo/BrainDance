# LangChain 方案：按用户要求控制模型卡片展示数量

## 背景

当前 `agent-recall` / `spatial-search-agent` 共享 `runSpatialSearchAgent` 核心，但模型数量限制分散在多处：

- `read_model_assets` 默认 `limit = 10`，最大允许 50。
- 资产模式回答文本只列出前 5 个模型。
- `spatial_search` 后端 `top_candidates` 最多返回 5 个候选。
- Flutter 普通空间候选卡片只展示前 3 个。
- Flutter `asset_metadata` 模式从 `asset_context.bundle/list` 中展示前 5 个模型卡片。

这会导致用户说“展示 8 个模型”“列出最新 12 个模型”时，Agent 可能已经读到了更多模型，但前端仍按固定数量裁剪，最终展示数量和用户意图不一致。

## 目标

- 用户明确要求展示 N 个模型时，最终前端模型卡片尽量展示 N 个。
- 后端读取数量、回答文本数量、响应候选数量、前端卡片数量使用同一份展示意图。
- 保留安全上限，避免一次返回过多模型导致响应过大或移动端 UI 过载。
- 默认行为保持兼容：用户没有明确数量时，资产模式仍默认展示 5 个，空间检索模式仍默认展示 3 个候选卡片。

## 非目标

- 不把“全部模型”解释为无限制全库返回。
- 不为了展示数量新增特化 Agent；优先沿用通用工具、通用响应协议和前端卡片组件。
- 不改变 `read_model_assets` 的最大查询能力上限，仍保留 `1..50` 的工具级约束。

## 建议方案

### 1. 引入展示数量协议字段

在共享 Core 的最终响应中新增一个轻量展示协议字段，例如：

```json
{
  "presentation": {
    "requested_model_count": 8,
    "effective_model_count": 8,
    "default_model_count": 5,
    "max_model_count": 20,
    "source": "user_explicit"
  }
}
```

字段含义：

- `requested_model_count`：从用户 query 中识别出的原始数量；没有明确数量时为 `null`。
- `effective_model_count`：实际允许展示的数量，经过默认值和上限裁剪。
- `default_model_count`：当前模式默认展示数量。
- `max_model_count`：当前模式最大展示数量。
- `source`：`user_explicit` / `default` / `clamped`。

建议上限：

- `asset_metadata`：默认 5，最大 20。
- `spatial_search`：默认 3，最大 10。

原因：资产列表卡片是主要结果，允许更多；空间检索候选通常还伴随打开场景、飞到视角等动作，过多候选会稀释主结果。

### 2. 后端统一解析用户展示数量

在 `spatialAgent.ts` 增加一个确定性解析函数，放在进入工具调用前执行：

```ts
type ModelPresentationRequest = {
  requestedModelCount: number | null;
  effectiveModelCount: number;
  defaultModelCount: number;
  maxModelCount: number;
  source: "user_explicit" | "default" | "clamped";
};
```

解析范围只处理明确数量：

- “展示 8 个模型”
- “列出最新 12 个”
- “推荐 6 个”
- “给我看三个模型”
- “前 10 个候选”

不明确的表达继续走默认值：

- “多推荐几个”
- “多找一些”
- “有哪些模型”

中文数字可以先支持 `一` 到 `二十`，阿拉伯数字直接支持。`全部` / `所有` 不直接全量返回，建议映射为当前模式最大值，并在回答中说明“先展示最多 N 个”。

### 3. 让工具读取数量跟随展示数量

资产模式下，构造 Agent 上下文时把展示数量写清楚：

- 用户要求 N 个模型时，优先调用 `read_model_assets` 且 `limit = effective_model_count`。
- 如果后续需要 `get_model_asset_bundle`，也只对展示范围内的模型展开。

这里不建议新增 `read_model_assets_display` 之类的专用工具。现有 `read_model_assets.limit` 已经足够，只需要让提示词和运行时上下文明确：用户要求展示数量时，读库数量应与展示数量对齐。

可在 `unified_agent` / `asset_tool_loop` prompt 中补充：

```text
如果上下文提供 requested_model_count / effective_model_count，
且用户是在要求列出、展示、推荐模型列表，
调用 read_model_assets 时应把 limit 设置为 effective_model_count。
```

### 4. 后端响应按 effective 数量裁剪

需要调整这些固定裁剪点：

- `buildListAnswer(rows)`：从 `rows.slice(0, 5)` 改为 `rows.slice(0, displayCount)`。
- `buildBundleAnswer(rows)`：同上。
- 推荐类回答 `buildListRecommendationAnswer` / `buildBundleRecommendationAnswer`：同上。
- `spatial_search` 的 `top_candidates`：从 `deduplicatedCandidates.slice(0, 5)` 改为 `slice(0, spatialEffectiveCount)`。

建议把展示数量作为参数传入 answer builder，而不是在 `assetTools.ts` 内部重复解析 query。

### 5. 前端改为读取后端展示协议

Flutter 端当前有两类硬编码：

- 普通候选：`result?.candidates.take(3).toList()`
- 资产模式：`source.take(5)`

建议 `AgentRecallResponse` 解析新增字段：

```dart
final int? effectiveModelCount;
final int? requestedModelCount;
```

渲染时：

- `spatial_search`：`take(result.effectiveModelCount ?? 3)`
- `asset_metadata`：`take(result.effectiveModelCount ?? 5)`

如果后端旧版本没有 `presentation` 字段，继续使用现有默认值，保证兼容。

### 6. 回答文本与卡片数量保持一致

后端最终回答里的“找到 N 个 / 展示前 N 个”应使用 `effective_model_count`，不要再硬编码 5 或 3。

当用户要求数量超过上限时：

- 卡片展示 `max_model_count` 个。
- 回答中说明已经按上限展示，例如：“你要求展示 50 个，我先展示前 20 个，避免一次加载过多。”

这样用户看到的文本、卡片和响应协议一致。

## 推荐改动范围

后端：

- `supabase/functions/_shared/agent-core/spatialAgent.ts`
  - 新增展示数量解析。
  - 在最终响应中加入 `presentation`。
  - 将 `top_candidates` 裁剪数量改为动态值。
  - 将展示数量传入资产回答构建。
- `supabase/functions/_shared/agent-core/assetTools.ts`
  - `buildAssetAnswer` 增加展示数量参数。
  - 列表、详情包、推荐回答从固定 5 改为动态值。
- `supabase/functions/_shared/agent-core/prompts/unified_agent.ts`
  - 增加 `effective_model_count` 使用规则。
- `supabase/functions/_shared/agent-core/prompts/asset_tool_loop.ts`
  - 增加 `read_model_assets.limit` 与展示数量对齐规则。

前端：

- `app/lib/services/agent_recall_service.dart`
  - 解析 `presentation`。
- `app/lib/services/agent_recall_models.dart`
  - 如仍被使用，也同步解析 `presentation`。
- `app/lib/pages/recall/recall_search.dart`
  - 普通候选和资产卡片展示数量改为读取 `effectiveModelCount`。
- `app/lib/pages/agent_chat/chat_view.dart`
  - 同步改为读取 `effectiveModelCount`。

测试：

- `supabase/functions/_shared/agent-core/assetTools.test.ts`
- `supabase/functions/spatial-search-agent/test.ts`
- Flutter 侧至少补充响应解析单元测试，或通过现有页面最小联调验证。

## 验证建议

后端单元测试：

- “推荐 3 个模型”时，`read_model_assets.limit` 应为 3，回答最多列 3 个。
- “展示 8 个模型”时，`asset_context.list` 和回答展示数量应为 8。
- “展示 50 个模型”时，`presentation.source = clamped`，`effective_model_count = 20`。
- 空间检索“给我 6 个候选”时，`top_candidates.length <= 6`，且不超过 10。
- 没有明确数量时，资产模式保持 5，空间模式保持 3。

桌面调试 CLI：

```bash
python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py \
  --query "请推荐 8 个模型" \
  --execution-mode preview \
  --show-request \
  --show-response-meta \
  --show-event-timeline \
  --show-full-result
```

前端验证：

- Recall 页输入“推荐 8 个模型”，确认最终展示 8 张资产卡片。
- Agent Chat 页输入同样请求，确认卡片数量一致。
- 输入“找一下电脑，给我 6 个候选”，确认空间候选卡片最多 6 张。
- 输入“有什么推荐的模型”，确认仍是默认 5 张。

## 风险与注意事项

- 如果只改前端 `take(N)`，但后端 `top_candidates` 仍最多 5，用户要求 8 个空间候选时仍无法满足。因此后端响应裁剪也必须同步改。
- 如果只改 `read_model_assets.limit`，但回答 builder 和 Flutter 仍固定 5，用户仍只能看到 5 个。
- 如果完全依赖 LLM 自己理解数量，稳定性不足。展示数量应由确定性解析兜底，再通过 prompt 告知 Agent 如何设置工具参数。
- `asset_context` 可能随展示数量变大而增大，建议最大值先设为 20，后续根据真机性能再调整。

## 分阶段落地

第一阶段：

- 后端新增 `presentation` 字段。
- 资产模式回答和 `asset_context` 卡片展示从固定 5 改为动态数量。
- Flutter 资产卡片读取 `effectiveModelCount`。

第二阶段：

- 空间检索 `top_candidates` 和 Flutter 普通候选卡片改为动态数量。
- 补充空间候选相关测试。

第三阶段：

- 优化数量解析，支持更多中文数量表达和“全部/所有”的上限提示。
- 根据联调结果决定是否把资产模式最大展示数量从 20 调整为可配置环境变量。
