# LangChain 阶段总结 - 2026-03-27 - 资产查找误路由

## 已完成

- 已定位 Flutter Agent 中“找一个会议室资产”返回动漫插画描述的问题根因。
- 已确认问题不在 Flutter 端拼装，而在共享 Core 的早期路由和资产快路径策略：
  - `找一个会议室资产` 一类问句曾因为“找/查/搜”启发式误入 `spatial_search`。
  - “找初音未来相关的”“找手办相关的”“请你找一下有没有重名的模型”这类请求即使路由到了 `asset_metadata`，也仍可能被旧的确定性资产快路径或关键词匹配限制住，没真正回到 Agent 工具回路。
- 已在共享 Core 中新增资产查找识别：
  - `isAssetDiscoveryQuery(...)` 用于识别“找会议室资产 / 办公室模型 / 某类场景资产”这类资产级请求。
  - 这类请求不再进入 `shouldPreferHeuristicSpatialRoute(...)` 的空间快路径。
  - 路由阶段会优先判为 `asset_metadata`。
- 已把空间快路径从“带找/查/搜就直接进 `spatial_search`”收紧为“必须具备明确空间定位线索”：
  - 明确位置/存在性问句，如“在哪 / 桌上 / 场景里 / 有没有”仍可直走空间检索。
  - “找初音未来相关的”“找某种风格/主题相关的”这类主题相关性检索不再误进空间链路。
- 已把资产链路进一步收敛到“通用读写工具 + Agent 规划”：
  - `read_model_assets` 现在优先走向量召回，不再依赖中文关键词清洗后的死板文本匹配。
  - `write_model_assets` 已作为通用写库工具进入主链路，支持“分别改名”“逐条改描述/标签”这类一对一写入。
  - `asset_metadata` 主链路不再调用确定性资产查找快路径，资产查找与资产写入统一回到 Agent 工具回路里规划。
  - “把最新两个模型分别改名为 test1 和 test2”这类请求现在要求 Agent 先读出最近两个模型，再按模型 ID 分别写入，而不是套用硬编码模板。
- 已补齐写操作确认执行闭环：
  - `session_state.lastOperationPreview` 记录 `write_model_assets` 预览参数后，用户后续说“确认执行”时，共享 Core 可以直接重放该工具并正式写库。
- 已补充单测，覆盖：
  - 资产级查找识别。
  - “找一个会议室资产”不再触发空间快路径。
  - “找初音未来相关的”这类主题检索不再误路由到空间检索。
  - `write_model_assets` 预览结果会被正确收集。
  - “重名模型”回答文案仍可从通用读库结果生成。

## 还未完成

- 还未在真实 Supabase 数据和 Flutter 真机页面上做端到端联调。
- 还未继续收敛资产工具集合；当前主链路已经优先使用通用读写工具，但历史上的专用工具实现仍保留在代码库里以兼容旧能力。

## 当前阻塞点

- 当前环境没有直接联到用户线上数据的安全 smoke 条件，因此只能先完成共享 Core 逻辑修复、本地类型检查和单测。

## 下一步建议

- 优先在 Flutter Recall 页用真实账号复测以下问句：
  - `找一个会议室资产`
  - `帮我找一下洛天依相关的模型`
  - `搜索找一下手办相关的`
  - `请你找一下有没有重名的模型`
  - `请你帮我修改最新两个模型的名字分别为 test1 和 test2`
  - `会议室里的投影仪在哪`
- 预期结果：
  - 前四句进入 `asset_metadata`，会看到 `read_model_assets` / `write_model_assets` 的工具轨迹，而不是“绕过空间检索但没找到”的固定回复。
  - 最后一句仍进入 `spatial_search`，不会误被当成资产列表查询。
- 如果线上仍有召回缺失，再重点排查 `match_model_assets` 与 embedding 生成链路，而不是重新退回关键词匹配。

## 已修改文件

- [assetTools.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/assetTools.ts)
- [assetTools.test.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/assetTools.test.ts)
- [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)
- [asset_tool_loop.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/prompts/asset_tool_loop.ts)
- [test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/test.ts)
