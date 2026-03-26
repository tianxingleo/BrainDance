# LangChain 实现现状 - 2026-03-27

## 背景

这轮补的是一个“电脑端可直接复现 Flutter Recall Agent 行为”的调试入口。

问题不在于仓库里完全没有调试能力，而在于现有脚本大多偏：

- 本地 QA 检索链路验证
- benchmark / 批处理
- Edge Function 单次 smoke

它们都不能直接回答一个前端联调里最实际的问题：

> Flutter 现在到底收到了哪些流式事件，候选结果、工具轨迹、最终回答分别长什么样？

## 本轮新增

- 新增脚本：`ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py`
- 新增测试：`tests/test_agent_recall_debug_cli.py`

这个 CLI 的目标不是重写一套 Agent，而是直接复用正式 `agent-recall` 的流式协议，按 Flutter 当前消费方式把过程摊开：

- 发送与 Flutter 一致的请求体字段：
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
- 支持消费 `text/event-stream` 与 `application/x-ndjson`
- 按事件打印：
  - `ping`
  - `status`
  - `plan`
  - `thought`
  - `tool_call`
  - `tool_result`
  - `message`
  - `done`
- 在 `done` 后额外汇总输出：
  - 最终回答
  - 动作 `actions`
  - 候选 `top_candidates`
  - 工具轨迹 `tool_trace`
  - `follow_up`
  - `session_state`
- 2026-03-27 后续补强了更适合分析的问题归因信息：
  - 请求摘要（endpoint / headers 摘要 / payload）
  - HTTP 响应元信息（状态码、关键响应头）
  - 事件时间线（事件序号、相对耗时、摘要）
  - 调试统计（首包、首个状态、首次工具调用、首字、done 耗时与事件计数）
  - `evidence` 摘要输出
  - 更完整的 JSON 落盘与可选 JSONL 事件时间线落盘

## 当前用途

这个入口适合做下面几类事情：

- 桌面联调时复现 Flutter Agent 面板看到的实际流
- 验证某条 query 为什么停在候选、为什么没有出最终回答
- 检查多轮续聊时 `session_state` / `follow_up` 是否符合预期
- 不起 Flutter 的情况下先看后端流式事件质量

## 使用方式

单轮调试：

```bash
python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py \
  --query "把最新三个模型改名为宿舍合集" \
  --execution-mode preview
```

带多轮上下文调试：

```bash
python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py \
  --query "确认执行" \
  --execution-mode execute \
  --conversation-summary "上一轮已经确认改名范围" \
  --session-state-file /path/to/session_state.json
```

落盘完整事件：

```bash
python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py \
  --query "黑色耳机在哪" \
  --log-file ai_engine/finetune_qwen3/logs/agent_recall_debug/earphone.json
```

打印更详细的分析信息：

```bash
python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py \
  --query "请你找一下洛天依相关的模型" \
  --show-request \
  --show-response-meta \
  --show-event-timeline \
  --show-full-result \
  --event-log-file ai_engine/finetune_qwen3/logs/agent_recall_debug
```

环境要求：

- 默认复用仓库已有 `SUPABASE_URL` / `SUPABASE_KEY` 读取逻辑
- 若目标函数需要登录态，可额外提供 `SUPABASE_ACCESS_TOKEN`

## 验证状态

已补单测覆盖：

- SSE 事件切分
- NDJSON 事件切分与尾包保留
- 请求体字段构造与 Flutter 对齐
- 事件时间线摘要生成
- 核心耗时统计汇总
- 目录模式日志路径生成

本轮只补了 CLI 和解析测试，没有在当前文档里声称已经完成真实线上联调结论；真实链路是否可跑仍取决于：

- Supabase Edge Function 当前是否可访问
- 目标环境里 `agent-recall` 是否已部署
- 当前 query 是否需要登录态数据
