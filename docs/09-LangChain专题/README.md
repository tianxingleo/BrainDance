# LangChain 专题文档

本目录专门存放 BrainDance 仓库内与 LangChain / Agent 编排相关的设计、实现现状、阶段总结和后续迭代记录。

## 当前文档

- [LangChain实现现状-2026-03-27.md](./LangChain实现现状-2026-03-27.md)
  - 新增电脑端 `agent-recall` 调试 CLI 说明。
  - 重点记录 Flutter 流式协议复现入口、事件打印和候选/工具轨迹调试方式。
- [LangChain联调记录-2026-03-27-agent-recall-非检索直答修复.md](./LangChain联调记录-2026-03-27-agent-recall-非检索直答修复.md)
  - 记录 `你是谁` 等无候选对话场景直接上抛异常的根因与修复。
  - 覆盖共享 Core 通用自然语言 fallback、Flutter 错误归一化与回归验证结果。
- [LangChain实现现状-2026-03-25.md](./LangChain实现现状-2026-03-25.md)
  - 说明当前仓库中已经落地的 LangChain 相关代码。
  - 区分稳定入口、实验链路、最小可用扩展和正在演进中的能力。
- [LangChain实现现状-2026-03-26.md](./LangChain实现现状-2026-03-26.md)
  - 记录 2026-03-26 这轮收口后的真实代码状态。
  - 覆盖统一入口、多模式共享 Core、数据库迁移、动作协议与剩余待办。

## 当前实现总览

- 共享 Core 已收敛到 [spatialAgent.ts](/home/ltx/projects/BrainDance/supabase/functions/_shared/agent-core/spatialAgent.ts)。
- `agent-recall` 与 `spatial-search-agent` 现在都只是入口壳层，强能力集中在共享 Core 与相关工具文件中。
- 正式稳定动作协议以 `open_scene`、`fly_to_pose` 为准，不再把 `highlight_hotspot` 视为正式前端能力。
- 2026-03-26 已补充纯问候/致谢直答模式：`你好`、`谢谢` 这类闲聊不再误进 `spatial_search` 工具链，Flutter 端也会在 `done` 事件兜底显示最终 `answer`，避免只剩选择理由和工具调试信息。
- 2026-03-26 已补充一次 `worker failed to boot` 故障归因：若前端收到 503，且 Edge Function 日志表现为 worker 启动失败，应优先检查共享 Core 文件是否残留 Git 冲突标记或其它语法错误。
- 2026-03-26 已补充“最新模型改名”确定性兜底：当用户表达“修改/重命名最新模型名字”时，共享 Core 会先锁定最新模型，再根据是否提供新名字返回缺参提示或直接生成改名预览/执行结果，避免只停在 `list_model_assets` 候选列表。
- 2026-03-26 已补充资产写操作重放协议：`session_state.lastOperationPreview` 现在会保存目标模型与工具参数，前端在确认执行时会显式切到 `execute`，共享 Core 可直接重放上一轮预览，不再让 LLM 重新猜测写入范围。
- 2026-03-26 已补充“最新 N 个模型批量改名”确定性兜底：像“把最新三个模型改名为 xxx”这类请求会先按 `created_at` 倒序锁定最近 N 个模型，再走 `batch_patch_model_metadata` 生成预览或正式执行。
- 2026-03-27 已把资产主链路继续收口到“通用读写工具 + Agent 规划”：`asset_metadata` 模式不再依赖确定性资产查找快路径，`read_model_assets` 走 embedding 语义召回，`write_model_assets` 支持“分别改名/逐条修改”，确认执行时也可重放该通用写工具。
- 2026-03-26 已补充 Agent 多轮续聊协议：共享 Core 现在会返回 `session_state / conversation_summary / follow_up`，Flutter Recall 页会保留最近一轮 Agent 会话，并把“继续输入什么”或快捷回复显式展示出来。
- 2026-03-26 已补充简单空间问句的确定性兜底：像 `查一下电脑`、`看一下桌上的杯子` 这类短句会优先走规则路由与固定工具顺序，避免因为上游模型 503 导致整个 Agent 看起来像“超时”。
- 2026-03-26 已补充 Agent 编排可解释性与续轮收敛：共享 Core 现在会额外发送 `plan / thought` 事件，把模式判断、意图判断、每轮为什么继续/停止说清楚；同时会拦截重复工具参数，并在高分候选证据已经足够时提前停止，不再机械追求“至少 3 个候选”。
- 2026-03-26 已补充 Flutter Agent 流式可视化修复：`agent-recall` 现支持 `text/event-stream` 与 `application/x-ndjson` 双协议，Flutter Recall 页会把 `status / tool_call / tool_result` 固化成步骤时间线，并在 `done` 事件用 `tool_trace` 补齐最终工具轨迹，避免用户只看到最后一句回答。
- 2026-03-26 已修复 `asset_metadata` 读取类回答过于机械的问题：当用户询问“有什么推荐的模型”这类资产概览问题时，最终回答不再直接返回“已读取 N 个模型资产摘要”，而会整理模型名称、描述、标签和 pose 数形成用户可读的概览。
- 2026-03-26 已进一步收紧“推荐模型”链路：后端在推荐类问句下只输出前 5 个推荐项，并按信息完整度与时间做简化排序；Flutter Recall 页在 Agent 模式下不再继续渲染底部全量模型列表，避免出现“上面推荐 10 个、下面又展开全库 109 个模型”的混淆展示。
- 2026-03-26 已补充 Flutter 首包等待态修复：Recall 页现在会在请求发出后立即展示本地引导状态，消费后端 `ping` 事件把“流式连接已建立”显式展示出来，并在首个远端阶段事件到达前持续更新等待态，避免用户在第一次刷新阶段长时间只看到静止的“连接中”。
- 2026-03-26 已补充 `spatial_search` 链路级提速：空间检索不再依赖“多轮 LLM 工具调度 + 最终 LLM 裁决”的串行结构，改为“单次意图解析 + 并行检索工具 + 确定性评分选优”，同时去掉 `pose_semantic_search` 内部按行补标签的 N+1 查询。
- 2026-03-26 已补充空间意图解析超时兜底：`parseSpatialIntent` 现在有 8 秒超时保护，若结构化解析超时或失败，会自动切到规则版意图解析并继续检索，不再长期卡在“正在解析空间意图和时间约束”。
- 2026-03-27 已新增电脑端调试 CLI：`ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py` 可以直接按 Flutter 的请求体和流式协议调 `agent-recall`，并打印 `status / plan / thought / tool_call / tool_result / message / done`，最终再汇总 `top_candidates`、`tool_trace`、`follow_up` 与 `session_state`，用于不启动 Flutter 时的桌面联调。
- 2026-03-27 已补充调试 CLI 断流诊断：当 `agent-recall` 在 `done` 前提前断开连接，或只返回 `error` 事件未返回 `done` 时，CLI 不再直接抛 Python `ChunkedEncodingError` 栈，而会保留已收到的事件、时间线、响应元信息与中断原因，方便继续归因是“后端主动报错”还是“HTTP 流被中途截断”。
- 2026-03-27 已补充共享 Core 的通用自然语言 fallback：当当前检索/工具链没有产出可信候选时，同一个 Agent 会退回自由回答，而不是把 `No candidates found` 继续上抛成前端的上游异常。
- 2026-03-27 已补充 Flutter 上游异常归一化：若 Supabase SDK 返回固定英文报错 `An invalid response was received from the upstream server`，Recall 页会统一映射到现有本地化上游异常提示，避免把底层英文错误直接显示给用户。
- 2026-03-27 已补充 Flutter 最终回答去重兼容：Recall 页消费 `agent-recall` 的 `message.delta` 时，会同时兼容“真正增量片段”和“累计全文片段”两种上游流式正文格式，避免在“你是谁”这类 direct answer 场景里把同一段回答重复拼成两到三遍。

## 2026-03-26 修复记录

- 已修复共享 Agent Core 中 `isDirectReplyQuery is not defined` 的运行时风险。
- 处理方式是把纯问候/致谢判定逻辑前移到共享常量和纯函数，避免在 `classifyAgentMode` 首次路由时出现未定义引用。
- 影响范围是 `agent-recall` 与 `spatial-search-agent` 两个依赖 `runSpatialSearchAgent` 的 Supabase Edge Function。
- 已在本地执行 `deno test supabase/functions/agent-recall/test.ts supabase/functions/spatial-search-agent/test.ts`，当前 17 个测试全部通过。
- Flutter / Dart 工具链在当前环境中仍不可用，Recall 页相关改动尚未完成端侧 `analyze` 或真机联调；合并后应在具备 Flutter 环境的机器补跑最小页面验证。
- 本轮另外补充了 `agent-recall` 流式协议兼容层：后端会在进入编排和整理最终回答时显式发送 `status`，Flutter 优先走 `SSE` 并兼容旧 `NDJSON` 解析。
- 本轮另外补充了 Recall 页运行态展示：工具调用中间步骤不再只停留在顶部状态文案，而会落成可见时间线；如果中途漏掉部分事件，也会在最终 `done` 阶段依据 `tool_trace` 自动补齐。
- 本轮另外补充了 Recall 页首包等待可视化：前端会先插入“请求已提交”的本地状态步骤，收到 `ping` 后立即切到“流式连接已建立”，若远端阶段事件仍未到达则用本地定时阶段文案兜底，直到真正收到 `status / tool_call / tool_result / done` 为止。
- 本轮另外补充了 `spatial_search` 执行链重构：保留一次意图解析，但移除了 LangChain 多轮 `bindTools` 调度和 `selectBestResult` 终裁调用，改为按意图选择必要工具并行执行，再使用现有 `scoreSceneCandidate` 与模板化回答直接产出结果。
- 本轮另外补充了 `pose_semantic_search` 的数据库访问优化：原实现会按每个候选模型循环查询 `memory_poses` 补标签，现已改成一次批量读取 `model_id + image_name` 映射，显著减少 RPC 后的二次往返。
- 本轮另外补充了空间意图解析超时 fallback：若上游结构化模型在意图解析阶段超时或返回异常，系统会明确发送 `intent_fallback` 状态，并用规则版 `scene/location/object/time` 解析继续下游检索。

## 维护规则

- 只要涉及 LangChain 相关实现修改，就优先在本目录补充记录，而不是把信息继续散落到别的目录里。
- 若一次改动还没做完，也必须先写阶段总结，说明“已完成 / 未完成 / 风险 / 下一步”。
- 当单篇文档已经过长、主题开始混杂时，不要继续无上限追加，应新开一篇文档承接下一阶段内容。
- 建议文件命名采用：
  - `LangChain实现现状-YYYY-MM-DD.md`
  - `LangChain阶段总结-YYYY-MM-DD-序号.md`
  - `LangChain联调记录-YYYY-MM-DD-序号.md`

## 与其他文档的关系

- [Agent 规划与 LangChain 实践路线](../02-架构设计/Agent规划与LangChain实践路线.md)
  - 负责规划、路线和长期边界。
- 本目录文档
  - 负责记录“代码现在实际做到了什么”“哪些还在实验中”“本轮改动停在了哪里”。
