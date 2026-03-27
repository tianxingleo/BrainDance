# LangChain 联调记录：2026-03-27 agent-recall 批量测试脚本与数据集

## 背景

- 当前仓库已经有电脑端单轮调试入口 [agent_recall_debug_cli.py](/home/ltx/projects/BrainDance/ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py)，适合复现单个问题。
- 但在 Supabase Deno `agent-recall` 逐渐被当作“面向当前项目场景微调过的通用智能体”后，只靠单轮手点已经不够覆盖实际链路。
- 本轮新增一套批量测试 runner 和发散数据集，用来批量压 `agent-recall` 的流式 LangChain 编排链路，重点观察请求协议、事件时间线、工具轨迹、多轮续聊和预览/执行切换是否还能保持稳定。

## 本轮新增

### 1. 批量测试脚本

- 新增脚本 [run_agent_recall_batch_suite.py](/home/ltx/projects/BrainDance/ai_engine/finetune_qwen3/scripts/run_agent_recall_batch_suite.py)。
- 设计原则是“复用真实 CLI，而不是重写一套假客户端”：
  - 直接复用 [agent_recall_debug_cli.py](/home/ltx/projects/BrainDance/ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py) 的 `run_single_turn`。
  - 继续走真实 `SSE / NDJSON` 流式解析逻辑。
  - 继续使用现有 Supabase 配置解析方式，避免引入第二套 endpoint / header 拼装代码。
- 脚本支持：
  - 通过 JSON suite 批量跑单轮与多轮 case。
  - 按 `case_id / category / tag` 过滤。
  - 输出每个 case 的完整调试结果 JSON。
  - 生成整体 `summary.json`，统计通过率、分类通过率、工具覆盖和事件覆盖。
  - 对多轮 scenario 自动继承上一轮真实返回的 `session_state / conversation_summary / session_id`，不需要把假上下文写死在数据集里。

### 2. 发散测试数据集

- 新增数据集 [agent_recall_batch_suite.json](/home/ltx/projects/BrainDance/ai_engine/finetune_qwen3/data/agent_recall_batch_suite.json)。
- 当前共有 29 个 scenario，展开后是 33 个 turn。
- 覆盖维度包括：
  - `persona / chitchat`
  - `inventory / recommendation`
  - `spatial_search / temporal_search / time_compare`
  - `asset_metadata / collection`
  - `multi_turn`
  - `protocol`
  - `ambiguous / composite / robustness`
- 具体问法刻意做了发散：
  - 正常中文问法
  - 英文问法
  - 中英混合
  - 口语化噪声
  - 极短 query
  - 长复合指令
  - 预览后确认执行
  - 检索后继续追问
  - 对比后继续收缩问题范围

## 结果判定策略

- 默认成功约束比较克制，避免把测试集做成脆弱的 prompt snapshot：
  - HTTP 状态码为 `200`
  - 能收到 `done`
  - `answer` 非空
  - `mode` 非空
  - 返回 `session_state`
- 个别用例会额外检查：
  - 是否存在 `follow_up`
  - 是否切到 `execute`
  - 是否覆盖 `NDJSON`
- 对链路测试来说，这一层优先验证“协议、编排和状态流是否通”，而不是把最终回答写死成唯一标准答案。

## 推荐用法

### 1. 先跑一条最轻量 case，确认环境连通

```bash
python ai_engine/finetune_qwen3/scripts/run_agent_recall_batch_suite.py \
  --cases persona_identity_001 \
  --quiet-cli \
  --summary-only
```

### 2. 跑全量发散集

```bash
python ai_engine/finetune_qwen3/scripts/run_agent_recall_batch_suite.py \
  --quiet-cli \
  --print-failures
```

### 3. 只看某类问题

```bash
python ai_engine/finetune_qwen3/scripts/run_agent_recall_batch_suite.py \
  --categories multi_turn,time_compare \
  --quiet-cli
```

### 4. 只看写操作和协议问题

```bash
python ai_engine/finetune_qwen3/scripts/run_agent_recall_batch_suite.py \
  --tags write,protocol \
  --quiet-cli
```

## 输出位置

- 汇总文件默认写到：
  - [summary.json](/home/ltx/projects/BrainDance/ai_engine/finetune_qwen3/logs/agent_recall_batch_suite/agent_recall_batch_suite/summary.json)
- 单 case 调试结果默认写到：
  - `/home/ltx/projects/BrainDance/ai_engine/finetune_qwen3/logs/agent_recall_batch_suite/agent_recall_batch_suite/cases/`

每条 case 会保留：

- 原始请求信息
- 事件列表
- 事件时间线
- 最终 `result`
- 运行期异常或断流信息
- 该条 case 的自动校验结果

## 当前已完成与未完成

### 已完成

- 已有批量 runner。
- 已有高发散 JSON 数据集。
- 已支持多轮场景继承上一轮真实返回状态。
- 已补单元测试，覆盖 suite 展开、继承逻辑和基础成功判定。

### 暂未完成

- 还没有接入更细粒度的“回答质量评估器”，例如针对不同 category 定制语义 judge。
- 还没有把这套 suite 接进 CI，也还没有做基线版本 A/B 对比。
- 还没有补“预期工具覆盖率阈值”或“预期事件顺序模板”的更强约束。

## 下一步建议

- 第一优先级：在真实环境跑一轮全量 33 turn，沉淀失败 case。
- 第二优先级：按失败类型给 `spatial_search / asset_metadata / time_compare / multi_turn` 分别加更细的断言。
- 第三优先级：如果后续继续演化 `agent-recall`，可把这套 suite 拆成：
  - `smoke`
  - `regression`
  - `exploratory`
  三层，以便在联调、回归和发版前分别使用。
