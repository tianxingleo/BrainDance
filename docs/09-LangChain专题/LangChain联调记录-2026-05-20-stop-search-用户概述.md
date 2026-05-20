# LangChain 联调记录：stop_search 后用户概述

日期：2026-05-20

## 背景

- 共享 Agent Core 中 `stop_search` 原本只负责停止工具循环。
- LLM 主动判断“当前信息足够”后，最终回答仍可能回到确定性模板，前端用户看不到 LLM 对当前工具结果的自然语言整理。

## 实现

- 在 `executeUnifiedAgentLoop` 的 `stop_search` 分支中，停止继续调用工具后追加一次无工具 LLM 调用。
- 新增总结提示词，只允许基于当前 `tool_trace`、空间候选和资产上下文生成 2 到 4 句用户可读中文概述。
- 将该概述作为最后一条 `AIMessage` 写回消息列表，最终响应优先采用这段概述。
- 如果总结调用失败，会记录 warning 并保留原有确定性回答路径，不中断主链路。

## 验证

- 新增 `pickSpatialSearchAnswerAfterStop` 单元测试，覆盖：
  - 存在 `stop_search` 且有总结时，优先使用用户可读总结。
  - 没有 `stop_search` 时，保留原有确定性回答。

## 影响

- 影响范围为依赖 `runSpatialSearchAgent` 的 `agent-recall` 与 `spatial-search-agent`。
- `stop_search` 后会多一次 LLM 调用，换取更自然的前端最终回答。
- 写操作 preview/execute 的安全边界不变，仍由 `executionMode`、`dryRun` 和 `requires_confirmation` 控制。
