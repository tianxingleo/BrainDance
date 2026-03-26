# LangChain 阶段总结 - 2026-03-27 - 资产查找误路由

## 已完成

- 已定位 Flutter Agent 中“找一个会议室资产”返回动漫插画描述的问题根因。
- 已确认问题不在 Flutter 端拼装，而在共享 Core 的首轮路由捷径：
  - `classifyAgentMode(...)` 会把包含“找/查/搜”的短句优先送进 `spatial_search`。
  - `找一个会议室资产` 因命中这条捷径，根本没有进入 `asset_metadata` 的资产工具链。
- 已在共享 Core 中新增资产查找识别：
  - `isAssetDiscoveryQuery(...)` 用于识别“找会议室资产 / 办公室模型 / 某类场景资产”这类资产级请求。
  - 这类请求不再进入 `shouldPreferHeuristicSpatialRoute(...)` 的空间快路径。
  - 路由阶段会优先判为 `asset_metadata`。
- 已补充确定性资产查找兜底：
  - 新增 `parseDeterministicAssetLookupIntent(...)`。
  - 新增 `runDeterministicAssetLookupFlow(...)`。
  - 命中后会直接调用 `list_model_assets`，而不是继续走空间检索。
- 已补充单测，覆盖：
  - 资产级查找识别。
  - “找一个会议室资产”不再触发空间快路径。
  - 原有空间意图启发式行为未被误删。

## 还未完成

- 还未在真实 Supabase 数据和 Flutter 真机页面上做端到端联调。
- 还未进一步收紧 Flutter 端对 `spatial_search` 候选描述的展示策略；当前主要修后端误路由。

## 当前阻塞点

- 当前环境没有直接联到用户线上数据的安全 smoke 条件，因此只能先完成共享 Core 逻辑修复与本地测试。

## 下一步建议

- 优先在 Flutter Recall 页用真实账号复测以下问句：
  - `找一个会议室资产`
  - `帮我找个办公室模型`
  - `会议室里的投影仪在哪`
- 预期结果：
  - 前两句进入 `asset_metadata`，会看到 `list_model_assets` 的工具轨迹。
  - 第三句仍进入 `spatial_search`，不会误被当成资产列表查询。
- 如果线上仍有误判，再继续把资产查找识别与 `display_name / tags / description` 的排序规则收紧。

## 已修改文件

- [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)
- [test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/test.ts)
