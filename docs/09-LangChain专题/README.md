# LangChain 专题文档

本目录专门存放 BrainDance 仓库内与 LangChain / Agent 编排相关的设计、实现现状、阶段总结和后续迭代记录。

## 当前文档

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
- 2026-03-26 已补充 Agent 多轮续聊协议：共享 Core 现在会返回 `session_state / conversation_summary / follow_up`，Flutter Recall 页会保留最近一轮 Agent 会话，并把“继续输入什么”或快捷回复显式展示出来。
- 2026-03-26 已补充简单空间问句的确定性兜底：像 `查一下电脑`、`看一下桌上的杯子` 这类短句会优先走规则路由与固定工具顺序，避免因为上游模型 503 导致整个 Agent 看起来像“超时”。

## 2026-03-26 修复记录

- 已修复共享 Agent Core 中 `isDirectReplyQuery is not defined` 的运行时风险。
- 处理方式是把纯问候/致谢判定逻辑前移到共享常量和纯函数，避免在 `classifyAgentMode` 首次路由时出现未定义引用。
- 影响范围是 `agent-recall` 与 `spatial-search-agent` 两个依赖 `runSpatialSearchAgent` 的 Supabase Edge Function。
- 本地环境当前缺少 `deno` 可执行文件，未能在仓库内直接完成 Deno 侧自动化验证；部署前需在具备 Deno/Supabase CLI 的环境补跑对应测试或最小联调。

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
