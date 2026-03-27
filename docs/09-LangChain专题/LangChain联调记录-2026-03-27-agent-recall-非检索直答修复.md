# LangChain 联调记录 - 2026-03-27 - agent-recall 非检索直答修复

## 背景

- Flutter Recall 页询问 `你是谁` 时，`agent-recall` 最终返回 `Exception: An invalid response was received from the upstream server`。
- 这类问句本质上不是空间检索、时间对比或资产操作，而是 Agent 的身份/能力说明问题。
- 共享 Core 之前只把 `你好 / 谢谢` 识别为直答，`你是谁 / 你能做什么 / 你的 system prompt 是什么` 仍会继续依赖上游结构化模型做路由或编排，因此会把上游不稳定直接暴露到 Flutter。

## 本轮实现

- 在 [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts) 中补充通用 `buildGeneralAssistantFallbackAnswer()`，让共享 Core 在“没有可信候选/当前不适合进入检索工具链”时，回退成同一个 Agent 的自然语言回答，而不是直接抛错。
- `runSpatialSearchAgent()` 不再在 `rankedCandidates.length === 0` 时直接 `throw new Error("No candidates found")`；现在会显式发送 `no_candidate_fallback` 状态，再调用通用自由回答 fallback 生成最终答复。
- 参考 OpenCode 的可审计会话语义，本轮还新增了 `response_resolution` 元数据，用于显式标注当前回答是 `retrieval_success`、`direct_reply` 还是 `general_fallback`，便于 Flutter、调试 CLI 与后端日志统一判断“这次是怎么收口的”。
- Flutter 侧 [agent_recall_service.dart](/home/ltx/projects/BrainDance/app/lib/services/agent_recall_service.dart) 补充上游异常归一化，若 SDK 返回固定文案 `An invalid response was received from the upstream server`，页面会统一展示既有本地化文案 `agent_error_upstream`，避免把英文底层报错直接暴露给用户。

## 进一步的 Prompt / Agent 架构收口

- 在 [route.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/prompts/route.ts) 中把路由器从“模式分类器”改成“能力边界判断器”，明确要求先判断是否真的需要工具链，而不是看到问题就硬路由。
- 在 [spatial_tool_loop.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/prompts/spatial_tool_loop.ts) 中补充“工具能力认知 / 行动原则 / 停止条件”，强调空间 Agent 是通用 Agent 的一个工作模式，不是纯工具调用器。
- 在 [asset_tool_loop.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/prompts/asset_tool_loop.ts) 中补充资产工具边界、非工具场景、自主停止条件，并明确“解释、澄清、引导”优先于机械调工具。
- 在 [context.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/prompts/context.ts) 中增加统一 Agent 工作约束，让 route、spatial、asset 三条 prompt 共享同一套基础行为原则。
- 在 [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts) 中新增 `shouldStopAssetToolLoop()`，把“已经拿到预览/对比/摘要/关系结果就停止”“多轮只剩列表读取就停止”收敛为共享 Core 的系统级退出策略，而不是只靠模型自觉。
- 在 [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts) 中进一步补上 `tool_policy` 执行分叉：route 不再只返回 mode，还会显式声明 `direct_answer / tool_chain`。共享 Core 现在会在 `spatial_search + direct_answer` 时直接跳过 `parseSpatialIntent` 和检索 tools，避免出现“reasoning 说不该调工具，但执行上还是继续检索”的控制流错位。

## 外部设计借鉴

- OpenClaw 的公开文档强调 system prompt 不是强制约束，真正稳定的是 failover、边界和执行策略这些系统能力；本轮映射到 BrainDance 上，就是把“无候选时不要失败”“工具循环何时停止”下沉到共享 Core。来源：
  - https://docs.openclaw.ai/faq
  - https://docs.openclaw.ai/advanced/system-prompt
- OpenCode 的文档和仓库强调 `client-server-agents` 分层、隐藏系统 agent、会话压缩与显式状态管理；本轮映射到 BrainDance 上，就是把 `response_resolution` 和 Agent 工作约束做成稳定协议，而不是只藏在日志里。来源：
  - https://github.com/sst/opencode
  - https://opencode.ai/docs/agents/
  - https://opencode.ai/docs/config/

## 为什么这不是“特殊路由优化”

- 这次没有给 `你是谁` 单独加入口路由，也没有在前端写问句特判。
- 调整点落在共享 Core 的失败收口能力层：只要当前检索/工具链没有给出可信候选，就由同一个 Agent 退回自然语言回答，而不是把无候选异常直接上抛。
- 同一套逻辑不仅覆盖 `你是谁`，也覆盖其它非检索闲聊、模糊追问、元问题，以及未来可能出现的“没有候选但仍应继续对话”的场景。

## 验证

- 已补充 [spatial-search-agent/test.ts](/home/ltx/projects/BrainDance/supabase/functions/spatial-search-agent/test.ts) 回归测试，覆盖：
  - 问候/致谢仍维持原有直答能力。
  - 无候选时的通用 Agent fallback 可返回自然语言说明。
- 已执行：
  - `deno test supabase/functions/spatial-search-agent/test.ts supabase/functions/agent-recall/test.ts`
  - `deno test supabase/functions/_shared/agent-core/assetTools.test.ts`
  - `deno check supabase/functions/_shared/agent-core/spatialAgent.ts supabase/functions/agent-recall/index.ts`
  - `deno check supabase/functions/_shared/agent-core/prompts/route.ts supabase/functions/_shared/agent-core/prompts/spatial_tool_loop.ts supabase/functions/_shared/agent-core/prompts/asset_tool_loop.ts supabase/functions/_shared/agent-core/prompts/context.ts`
  - `deno test --allow-env supabase/functions/agent-recall/smoke.ts`
- 已执行桌面联调：
  - `python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py --query '你可以干什么' --execution-mode preview`
  - 本轮结果已确认不再进入 `intent` / `tool_call` / `tool_result`，而是直接走 `general_answer`，最终 `tool_trace_count=0`、`candidate_count=0`。
- 当前环境未提供 `SUPABASE_SERVICE_ROLE_KEY`，因此 `smoke.ts` 本轮只验证了闭环入口脚本可执行，并把 `你是谁` 纳入真实请求用例列表；未实际打到远端函数服务。

## 影响与风险

- 影响范围覆盖 `agent-recall` 与 `spatial-search-agent` 两条共享 Core 链路。
- 这次主要收敛的是“无候选即抛错”的边界问题；如果后续仍有其它自然语言对话场景异常，应继续增强共享 Core 的通用 fallback 能力，而不是回到上层做硬编码路由。
- 当前环境未执行 Flutter 真机联调；但由于前端修改仅限错误文案归一化，主要风险仍集中在后端是否还有其它未收敛的上游依赖点。
