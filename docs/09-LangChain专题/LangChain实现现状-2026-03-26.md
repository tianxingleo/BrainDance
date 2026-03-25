# LangChain 实现现状 (2026-03-26)

## 本次工作核心：收口、统一与补强

基于 3 月 25 日的代码审查建议，本次提交对 `agent-recall` 及其核心实现进行了全面重构收口，解决了多个模块间的协议漂移问题。

### 1. 入口统一 (agent-recall 变成真正的 Façade)
- **已完成**：将 `agent-recall` 中的独立 `runRecallAgent` 废弃，现已直接接入 `_shared/agent-core/spatialAgent.ts` 里的 `runSpatialSearchAgent`。
- **已完成**：移除 `agent-recall` 下的多余 `agent/` 与 `tools/` 目录，使其成为轻量级入口。
- **现状**：真正实现了“强能力写在共享 Core，正式入口走新逻辑”的架构统一，前端、测试和动作协议全部收口。

### 2. 补全正式产品能力 (支持选中及执行模式)
- **已完成**：`agent-recall` 的请求 Schema 全面升级。
- **现状**：现已支持接受 `selectedModelIds` 和 `executionMode: "preview" | "execute"` 参数。资产相关的选中模型批量操作、安全预览现已向前端彻底开放。

### 3. 解决三处协议漂移
- **漂移 A (名称来源)**：已修改 `search-models/shared.ts`，彻底移除从 `processing_tasks` 中回填 `display_name` 的 fallback 逻辑。现**严格以 `model_assets.display_name` 为唯一名称来源**。
- **漂移 B (动作协议)**：在共享 Core 及 `agent-recall` 的测试中，统一下游的返回协议。**只保留了 `open_model` 和 `fly_to_pose`，正式从核心逻辑中移除了不稳定的 `highlight_hotspot`**，避免因前端缺乏热点数据层导致的消费异常。
- **漂移 C (测试错位)**：重写了 `agent-recall/test.ts` 和 `agent-recall/smoke.ts`，专注于验证代理接口层 (Façade) 对请求协议 (`selectedModelIds` / `executionMode`) 的防腐解析，并且在集成测试阶段检测正确的动作协议。

## 待办与下一步计划 (下周工作)
1. **新增解释型工具 `get_pose_summary`**：需要通过 `memory_poses` 提取摘要，以强化模型对比解释能力。
2. **新增资产工具 `find_related_models`**：实现同空间场景的模糊搜索匹配工具 (基于 tags、时间和 scene_id)，加强版本整理与相关性比对能力。


### 4. 解决上下文管理不足与 Prompt 模块化
- **已完成**：将请求 Schema 进一步升级，添加 UI Context (`currentSceneId`, `currentModelId`, `candidateSceneIds`) 以及会话记忆 (`conversationSummary`, `sessionId`)。
- **已完成**：抽取原内联于 `spatialAgent.ts` 的各类系统提示词至独立的 `prompts/` 目录下（如 `route.ts`、`spatial_intent.ts`、`asset_tool_loop.ts`、`spatial_tool_loop.ts`、`selection.ts`，新增 `context.ts`）。
- **已完成**：在提示词中补全了大量 Few-Shot 示例并实现了标准化的统一上下文注入块 (`buildAgentContextBlock`)，强化了多轮执行连贯性、以及基于预览执行写操作的逻辑稳健性。
- **已补强**：在空间结果组织的 Prompt 中补上了“面向 UI”的三段式回答模板要求（一句结论、一句证据、一句下一步建议）。
