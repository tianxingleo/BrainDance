# Qwen3-1.7B 微调实践记录 Part 16+

## Part 16-A：检索路由可观测性基础设施补齐

### 时间

- 2026-03-21

### 本 part 目标

- 暂停继续开新一轮 LoRA 训练
- 把当前 debug 主线切到“检索层专项观察 + 交互日志积累”
- 先补齐 route 级可观测性，而不是继续靠主观体验判断问题归因

### 本次判断

当前 `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 已经把关键生成层问题压到可接受范围。新的主要矛盾不再是“模型不会说”，而是：

- 某些 `object_lookup` query 的召回不稳
- 某些 query 依赖 `lexical_fallback`
- 某些回答更适合 deterministic formatter，而不是继续交给 LoRA 自由生成

因此本 part 不开 `round4.2`，先把真实交互日志、路由统计和 answer route 边界做清楚。

### 本次改动

#### 1. real-chain 检索结果补充 `answer_route`

修改文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增/调整：

- 为 greeting / persona 显式区分 `query_class`
  - `greeting`
  - `persona`
- 不再把这两类统一记成笼统的 `non_retrieval`
- 新增 `answer_route` 字段
  - `fixed_response`
  - `inventory_formatter`
  - `semantic_summary_formatter`
  - `lora_generation`
- `retrieve_real_chain_case()` 现在会把 `answer_route` 同时写入：
  - 顶层 chain
  - `retrieval` 字段

这样后续可以明确区分：

- 问题是检索 miss
- 还是 formatter 路由承担了主要修复作用

#### 2. interactive debug 日志补充 Part 16 所需字段

修改文件：

- `ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py`

新增日志字段：

- `turn_id`
- `answer_route`
- `user_feedback_label`
- `issue_bucket`

交互时新增可选人工标注：

- `user_feedback_label`
  - `good`
  - `acceptable`
  - `bad`
- `issue_bucket`
  - `retrieval_miss`
  - `retrieval_low_relevance`
  - `answer_style`
  - `focus_drift`
  - `formatter_needed`

同时 summary 里新增聚合项：

- `query_class_counts`
- `retrieval_route_counts`
- `fallback_reason_counts`
- `answer_route_counts`
- `user_feedback_label_counts`
- `issue_bucket_counts`

这一步完成后，interactive session 不再只是留原始问答，而是具备 route-level 统计条件。

#### 3. 新增 interactive route 聚合脚本

新增文件：

- `ai_engine/finetune_qwen3/scripts/summarize_interactive_debug_routes.py`

用途：

- 聚合一个或多个 interactive session JSONL
- 输出 route summary 的 JSON 和 Markdown

输出目标：

- `ai_engine/finetune_qwen3/logs/interactive_route_summary.json`
- `ai_engine/finetune_qwen3/logs/interactive_route_summary.md`

当前汇总维度包括：

- overall
  - `query_class_counts`
  - `retrieval_route_counts`
  - `fallback_reason_counts`
  - `answer_route_counts`
  - `user_feedback_label_counts`
  - `issue_bucket_counts`
- `by_query_class`
- `by_retrieval_route`
- `by_answer_route`

每个子维度至少统计：

- `count`
- `avg_hit_count`
- `fallback_rate`
- `avg_retrieval_latency_sec`
- `avg_generation_latency_sec`

这意味着 Part 16 后续可以不再手工数 route，而是直接出报表。

#### 4. 新增 query routing strategy 文档

新增文件：

- `ai_engine/finetune_qwen3/docs/query_routing_strategy.md`

当前显式策略表已经写明：

- `greeting -> fixed_response`
- `persona -> fixed_response`
- `inventory -> inventory_formatter`
- `object_lookup -> lora_generation`
- `partial_coverage -> lora_generation`
- `abstract semantic query -> semantic_summary_formatter`

这样后续再看 bad case 时，能更快定位问题是在：

- query classification
- retrieval route
- answer route

#### 5. 新增 Part 16 调试案例模板

新增文件：

- `ai_engine/finetune_qwen3/data/interactive_debug_cases_part16_template.json`

当前先落一版 32 条模板，覆盖：

- `recent_hit`
- `no_hit`
- `object_lookup`
- `partial_coverage`
- `must_answer`
- `multi_hit_must_answer`
- `inventory`
- `abstract_semantic_query`

每条 case 预留字段：

- `expected_route_hint`
- `expected_focus`
- `priority`
- `notes`

这样后面人工 triage 不需要再临时补上下文。

### 当前结论

Part 16-A 完成后，系统已经从“只能看单条回答好不好”升级为“可以按 query class / retrieval route / answer route 做聚合归因”。

当前最重要的下一步不再是训练，而是：

1. 运行 interactive debug，累计 30 到 50 条真实交互
2. 用 summary 脚本观察 `lexical_fallback` 占比
3. 看 `object_lookup` 是否是 fallback 最高的 query class
4. 看 bad case 主要落在 retrieval 还是 formatter / answer style

### 验证情况

本 part 代码侧目标是“补齐观测与汇总基础设施”，不涉及新的训练任务。

已完成：

- 路由字段补齐
- interactive 日志结构扩展
- route 聚合脚本落地
- 策略文档与题集模板落地

待下一步执行：

- 用真实 interactive session 跑出第一版 route summary
- 根据统计结果决定是否进入 Part 17 检索专项优化

### 本 part 一句话结论

Part 16-A 先不继续训模型，先把“query_class -> retrieval_route -> answer_route -> 人工归因”这条观测链补齐，让下一阶段的优化依据从主观感觉变成真实路由统计。

## Part 16-B：历史交互日志回填与首版路由统计清洗

### 时间

- 2026-03-21

### 本 part 目标

- 把已有 interactive session 的历史日志补齐到 Part 16 的字段规范
- 先清掉 legacy session 里的缺失 route 元数据，再看第一版 route summary
- 验证当前阶段的主要瓶颈是否继续指向检索与路由，而不是生成层

### 本次判断

Part 16-A 完成后虽然已经有了 route summary 脚本，但首版汇总里仍然混有 `unknown`，原因不是统计脚本本身，而是旧 session 里缺：

- `turn_id`
- `query_class`
- `retrieval_route`
- `answer_route`

如果不先把历史日志清洗掉，Part 16 的分布判断会失真。

### 本次改动

#### 1. 新增历史 session 回填脚本

新增文件：

- `ai_engine/finetune_qwen3/scripts/backfill_interactive_session_fields.py`

功能：

- 扫描 `interactive_sessions/*.jsonl`
- 给旧 session 回填：
  - `turn_id`
  - `query_class`
  - `retrieval_route`
  - `answer_route`
  - `user_feedback_label`
  - `issue_bucket`
- 支持两种模式：
  - `heuristic`
  - `rehydrated`

其中 `heuristic` 用于不依赖外部检索链路的轻量回填；`rehydrated` 尝试用当前真实 retrieval chain 重算路由元数据。

#### 2. 新增 conda 包装脚本，统一 Part 16 执行入口

新增文件：

- `ai_engine/finetune_qwen3/scripts/run_backfill_interactive_routes.sh`

作用：

- 自动进入 `qwen3_ft`
- 先执行历史日志回填
- 再执行 route summary 汇总

这样后面就不需要每次手动拼：

- `conda run -n qwen3_ft ...`
- 回填
- 汇总

#### 3. 回填脚本增加失败降级

原本 `--rehydrate_route` 一旦遇到外部链路错误会整批中断。

现在改为：

- 优先尝试 `rehydrated`
- 如果真实链路失败，则自动降级为 `heuristic`
- 并把失败原因记录到：
  - `route_backfill_method`
  - `route_backfill_error`

这样 Part 16 的日志清洗不会再被单次 RPC 故障全部阻断。

### 运行结果

#### 1. conda 环境确认

本链路应统一使用：

- `conda activate qwen3_ft`

之前默认 `python` 缺少 `torch`，无法直接 import `run_real_chain_debug.py`。这说明 Part 16 相关脚本不能再默认依赖系统 Python。

#### 2. rehydrate 路径被外部 RPC 阻塞

在 `qwen3_ft` 中尝试：

- `backfill_interactive_session_fields.py --rehydrate_route`

时，真实检索链路的 Supabase RPC：

- `match_memory_poses`

返回了：

- `530 Server Error`

这说明当前阻塞点不在 conda，而在外部 retrieval 服务可用性。

#### 3. heuristic 回填成功

随后改用不依赖 RPC 的 heuristic 回填，结果：

- `files=8`
- `updated_files=8`
- `updated_rows=21`

即当前 8 个 session、21 条历史交互已经全部补到 Part 16 所需字段结构。

### 清洗后首版 summary

重新生成：

- `ai_engine/finetune_qwen3/logs/interactive_route_summary.json`
- `ai_engine/finetune_qwen3/logs/interactive_route_summary.md`

当前整体统计为：

- `session_count = 8`
- `turn_count = 21`
- `error_count = 0`

#### query_class 分布

- `object_lookup = 8`
- `inventory = 6`
- `non_retrieval = 4`
- `greeting = 1`
- `persona = 1`
- `partial_coverage = 1`

#### retrieval_route 分布

- `vector_plus_filter = 6`
- `inventory_special_case = 6`
- `non_retrieval = 6`
- `lexical_fallback = 3`

#### answer_route 分布

- `lora_generation = 9`
- `inventory_formatter = 6`
- `fixed_response = 6`

#### 当前直接可读出的信号

1. 当前样本里 `object_lookup` 仍是主类，且 fallback rate 约为 `0.375`
2. `inventory` 已经稳定转到 `inventory_special_case + inventory_formatter`
3. 问候/身份类已经从“误走 no-hit”清洗到固定路由
4. 当前交互量只有 `21` 条，仍不足以决定是否直接进入 Part 17

### 当前结论

Part 16-B 的价值不在于“又修了一个回答”，而在于把已有历史交互正式纳入 Part 16 的统计口径。

目前可以明确：

- 当前主瓶颈仍然不是 LoRA 生成层
- `inventory` 和社交问法的路由边界已经清楚
- 后续最值得继续观察的是：
  - `object_lookup`
  - `lexical_fallback`
  - 外部 retrieval RPC 稳定性

### 下一步

接下来优先做：

1. 继续用 Part 16 题集和真实交互把样本扩到 `30 ~ 50` 条
2. 开始实际填写：
   - `user_feedback_label`
   - `issue_bucket`
3. 单独观察外部 RPC `530` 是否持续出现
4. 如果 `object_lookup` 的 fallback rate 继续偏高，再进入 Part 17 检索专项优化

### 本 part 一句话结论

Part 16-B 已经把历史 session 清洗进统一日志口径，也证明了当前系统的真实不稳定点更偏向检索服务和 object lookup 路由，而不是继续训练 LoRA。

## Part 16-C：真实交互扩样与首轮人工归因

### 时间

- 2026-03-21

### 本 part 目标

- 把 interactive debug 样本从当前 `21` 条扩到 `30 ~ 50` 条
- 正式开始填写 `user_feedback_label` 与 `issue_bucket`
- 输出第一版 route-level 人工归因结论
- 在不新开 LoRA、也不直接进入 Part 17 的前提下，先判断当前主瓶颈是否稳定指向 retrieval

### 本次判断

Part 16-A / 16-B 已经证明当前方向是对的，系统也已经进入“该做统计观察、暂不做训练”的阶段。

现阶段已经可以明确两点：

- `query_class -> retrieval_route -> answer_route -> 人工归因` 的观测链已经补齐
- 当前不稳定点更偏向：
  - `object_lookup`
  - `lexical_fallback`
  - 外部 RPC 可用性

而不是继续指向 LoRA 生成层本身。

但当前样本量仍然偏小：

- `session_count = 8`
- `turn_count = 21`

这个量级足够证明“方向没错”，但还不足以支撑：

- 直接开启新一轮 LoRA
- 直接进入 Part 17 检索专项大改
- 直接把产品默认链路切到当前策略

因此 Part 16-C 的核心不是“继续改模型”，而是“补样本、做归因、形成第一版可执行判断”。

### 本 part 不做的事

- 不开新一轮 LoRA
- 不急着做 Part 17 大改检索
- 不急着把产品默认链路切过去

### 本次执行重点

#### 1. 用 Part 16 题集模板继续扩样

优先补以下 query class：

- `object_lookup`
- `partial_coverage`
- `inventory`
- `abstract_semantic_query`
- 口语化 `recent`

扩样目标不是泛泛“测更多”，而是针对当前最可疑的 retrieval 路由多打几轮，尽快看清：

- `object_lookup` 的 fallback 是偶发还是持续偏高
- `inventory_special_case` 是否已经稳定
- 口语 recent 问法是否还会误分流

#### 2. 开始正式填写人工标注字段

每条真实交互至少补齐：

- `user_feedback_label`
- `issue_bucket`

Part 16-C 最需要回答的，不再只是“系统答出来了没有”，而是：

- 用户是否接受这条回答
- 问题属于 retrieval、focus、style，还是 formatter 边界问题

#### 3. 单独盯 `object_lookup` 的 3 个核心指标

建议在 summary 中单独给出：

- `object_lookup_count`
- `object_lookup_lexical_fallback_rate`
- `object_lookup_bad_rate`

以及其中 bad case 落到：

- `retrieval_miss`
- `retrieval_low_relevance`

的占比。

如果这一组继续偏高，那么 Part 17 的主线就基本明确应为 retrieval 专项优化，而不是训练层继续加料。

#### 4. 单独记录外部 RPC `530` 频率

当前已经确认 `match_memory_poses` 存在 `530 Server Error` 风险。

如果该问题持续出现，那么在进入 Part 17 之前，还需要先补一层更靠前的服务可用性兜底，否则会污染 Part 16-C 对 retrieval 路由本身的判断。

建议在日志或 summary 中额外统计：

- `rpc_error_count`
- `rpc_error_rate`
- `fallback_after_rpc_error_count`

#### 5. 输出第一版 route summary 人工归因报告

当累计样本达到 `30 ~ 50` 条后，直接产出一版简短结论，至少回答以下 4 个问题：

1. 当前 `object_lookup` 是否是主问题类
2. `lexical_fallback` 依赖是否偏高
3. 哪些 query class 更适合 formatter 路由
4. 是否已经值得进入 Part 17

### 本部分重点观察

- `object_lookup` 的 fallback rate
- `inventory_special_case` 的稳定性
- `abstract_semantic_query` 是否需要进一步 formatter 化
- `match_memory_poses` 的 `530` 是否持续影响 real-chain

### 本部分验收标准

- 至少累计 `30 ~ 50` 条真实交互
- 完成首轮人工 triage
- 能明确回答：
  - 当前主瓶颈是不是 retrieval
  - 是否值得进入 Part 17 检索专项优化

### 本次实际执行

#### 1. 新增 Part 16-C 批量扩样脚本

新增文件：

- `ai_engine/finetune_qwen3/scripts/run_part16c_batch_probe.py`

作用：

- 直接读取 `interactive_debug_cases_part16_template.json`
- 批量执行真实 retrieval chain
- 自动写入 interactive 风格 JSONL
- 自动补：
  - `user_feedback_label`
  - `issue_bucket`
  - `triage_label`
  - `triage_reason`

这样 Part 16-C 不再依赖纯手工输入，可以重复跑题集扩样。

#### 2. 给 real-chain 增加 RPC 级可观测字段与降级记录

修改文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增/调整：

- 增加 `safe_rpc_match_memory_poses()`
- 在 retrieval 结果中补充：
  - `rpc_error_count`
  - `rpc_errors`
  - `fallback_after_rpc_error`
- 当 RPC 抛异常时，不再直接让调用方失去上下文，而是保留 route 级错误信息，便于 Part 16-C 单独统计

#### 3. route summary 增加 Part 16-C 核心指标

修改文件：

- `ai_engine/finetune_qwen3/scripts/summarize_interactive_debug_routes.py`

新增统计：

- overall
  - `rpc_error_count`
  - `rpc_error_rate`
  - `fallback_after_rpc_error_count`
- `object_lookup_summary`
  - `object_lookup_count`
  - `object_lookup_lexical_fallback_rate`
  - `object_lookup_bad_rate`
  - `object_lookup_retrieval_miss_bad_count`
  - `object_lookup_retrieval_low_relevance_bad_count`
- 各子分组额外补：
  - `bad_rate`
  - `rpc_error_count`
  - `rpc_error_rate`
  - `fallback_after_rpc_error_count`

#### 4. interactive 调试脚本补齐 RPC 字段，并修复旧 bug

修改文件：

- `ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py`

新增：

- `rpc_error` 相关日志字段
- summary 中的 RPC 聚合项

同时修复一个旧问题：

- `print(json.dumps(preview...))` 原本缩进错误
- 在默认不打开 `--show_evidence` 时会触发：
  - `UnboundLocalError: local variable 'preview' referenced before assignment`

这也是全量 summary 中 `error_count = 5` 的来源之一，不应被误判为 retrieval 问题。

### 本次运行

#### 1. 冒烟运行

执行：

- `part16c_smoke_20260321`

结果：

- `turn_count = 5`
- `object_lookup_count = 4`
- `object_lookup_lexical_fallback_rate = 0.25`
- `rpc_error_count = 0`

这一步确认了：

- 批量脚本可跑
- 自动 triage 可落日志
- 新版 summary 可正常出报表

#### 2. Part 16-C 批量扩样

执行：

- `part16c_batch_20260321`

覆盖：

- `recent`
- `no_hit`
- `object_lookup`
- `partial_coverage`
- `must_answer`
- `multi_hit_must_answer`
- `inventory`
- `abstract_semantic_query`

本轮新增：

- `turn_count = 34`

其中：

- `good = 22`
- `acceptable = 8`
- `bad = 4`

首轮 issue 归因为：

- `retrieval_miss = 4`
- `focus_drift = 4`
- `answer_style = 3`

#### 3. 全量汇总

重新汇总：

- `ai_engine/finetune_qwen3/logs/interactive_route_summary.json`
- `ai_engine/finetune_qwen3/logs/interactive_route_summary.md`

当前全量统计为：

- `session_count = 10`
- `turn_count = 60`
- `error_count = 5`

说明：

- 已经超过 Part 16-C 预设的 `30 ~ 50` 条门槛
- `error_count = 5` 来自旧 interactive 脚本 bug，不是本轮新的 retrieval 结论

### 当前统计结论

#### 1. `object_lookup` 仍然是主问题类

当前：

- `object_lookup_count = 22`
- `object_lookup_lexical_fallback_rate = 0.2727`
- `object_lookup_bad_rate = 0.1818`
- `object_lookup_retrieval_miss_bad_count = 4`
- `object_lookup_retrieval_low_relevance_bad_count = 0`

这说明：

- `object_lookup` 仍是当前最值得继续盯的 query class
- 当前 bad case 更偏向直接 `retrieval_miss`
- 还没有出现大规模 `retrieval_low_relevance`

#### 2. `partial_coverage` 主要问题不是 retrieval miss，而是回答边界

当前：

- `partial_coverage_count = 11`
- `bad_rate = 0`
- `acceptable = 5`
- `issue_bucket` 主要落在：
  - `focus_drift = 4`
  - `answer_style = 1`

这说明：

- `partial_coverage` 现在更像 formatter / answer policy 问题
- 不像 `object_lookup` 那样直接体现为 retrieval 主瓶颈

#### 3. `inventory_special_case` 已经稳定

当前：

- `inventory_count = 9`
- `fallback_rate = 0`
- `bad_rate = 0`

这说明：

- `inventory_special_case + inventory_formatter` 已经是稳定路径
- 这部分不需要进入 Part 17 作为主优化对象

#### 4. 口语化 `recent` 总体稳定

当前：

- `recent_capture_count = 8`
- `bad_rate = 0`
- `user_feedback_label = good` 为主

说明 recent 问法目前不是最紧急问题。

#### 5. 外部 RPC `530` 本轮未复现为持续主因

当前 summary 为：

- `rpc_error_count = 0`
- `rpc_error_rate = 0`
- `fallback_after_rpc_error_count = 0`

同时 `fallback_reason_counts` 中：

- `rpc_empty = 24`

这说明：

- 本轮没有复现需要单独记为 `rpc_error` 的硬故障
- 但 `rpc_empty` 仍然较多，说明向量召回为空仍是当前链路中的高频现象
- 现阶段更像“召回为空 + lexical fallback 介入”，而不是“530 连续打断实验”

### 当前判断

Part 16-C 经过实际跑样本后，现在可以明确回答：

1. 当前主瓶颈更偏向 retrieval，尤其是 `object_lookup`
2. `lexical_fallback` 依赖存在，但目前更像中等偏高，而不是已经失控
3. `inventory` 已稳定适合 formatter
4. `partial_coverage` 更值得做 answer/formatter 层打磨，而不是先做 retrieval 大改

### 是否进入 Part 17

当前结论是：

- 可以开始准备 Part 17
- 但不建议直接做“大而全”的检索重构

更合理的 Part 17 入口应是：

- 先做 `object_lookup` 专项 retrieval 优化
- 聚焦减少：
  - `rpc_empty`
  - `object_lookup` no-hit bad case
  - 对 `lexical_fallback` 的依赖

而不是把所有 query class 一起重写。

### 进入 Part 17 的条件

如果后续扩样后出现以下任一信号，就进入 Part 17：

- `object_lookup` 的 fallback rate 持续偏高
- `lexical_fallback` 成为高频救命路径
- `530` 持续影响 real-chain 稳定性
- 同类 bad case 多次落到：
  - `retrieval_miss`
  - `retrieval_low_relevance`

### 暂不进入 Part 17 的条件

如果上述信号没有持续出现，则先不急着大改 retrieval，而是继续积累真实交互，再考虑体验层或 formatter 层打磨。

### 当前结论

Part 16-C 的首要任务已经明确：

- 先把交互样本补到 `30 ~ 50` 条
- 先把 `user_feedback_label` / `issue_bucket` 填起来
- 先拿到第一版 route-level 人工归因结论

在这之前，不值得继续训，也不值得直接进入 Part 17 大改。

### 本 part 一句话结论

Part 16-C 先做真实交互扩样和首轮人工归因，用更稳定的 route-level 统计判断当前主瓶颈是否真的落在 retrieval，再决定是否进入 Part 17。

## Part 16-D：interactive debug 运行时修复与 persona 问法补齐

### 时间

- 2026-03-21

### 本 part 背景

在手工执行：

- `bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu1.sh`

时，出现了一组非链路本身的问题：

1. 每轮回答打印后都会抛出：
   - `UnboundLocalError: local variable 'preview' referenced before assignment`
2. `你有什么用处` 这类明显属于助手能力说明的问题，没有命中 non-retrieval / persona 分支，而是误走了检索链路，答成了无关的“暂无相关记录”。

这两个问题会直接污染 Part 16 的 interactive session：

- 前者会把本来已经成功产出的回答整轮记成 error
- 后者会把 persona 能力问法错误计入 retrieval 质量问题

因此需要先修 runtime 与 query guardrail，再继续看 route-level 统计。

### 本次改动

#### 1. 修复 interactive debug 的 `preview` 作用域错误

修改文件：

- `ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py`

问题原因：

- `preview` 只在 `--show_evidence` 分支里定义
- 但 `print(json.dumps(preview, ...))` 的缩进落在了分支外
- 导致默认不带 `--show_evidence` 时，每轮回答后都会访问未定义变量

修复方式：

- 把 `print(json.dumps(preview, ...))` 一并放回 `if args.show_evidence:` 代码块内部

修复后预期：

- 默认模式下不再因为 `preview` 崩溃
- 只有显式开启 `--show_evidence` 时才打印检索摘要
- interactive session 不再把正常回答误记成 error

#### 2. 补齐 persona / capability 问法模式

修改文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增匹配短语：

- `你有什么用`
- `你有什么用处`
- `你有什么作用`
- `你能帮我做什么`
- `你可以帮我做什么`

修复目标：

- 让这类问法直接走：
  - `query_class = persona`
  - `retrieval_route = non_retrieval`
  - `answer_route = fixed_response`

这样它们不会再误入 retrieval / object lookup 统计。

### 当前判断

这次问题的主因不是模型退化，也不是 Supabase retrieval 本身异常，而是：

- interactive 调试壳子里存在一个明确的 Python 运行时 bug
- non-retrieval guardrail 对“能力说明”口语变体覆盖不够

也就是说，`interactive_debug_20260321T140506Z` 这 5 条 error 不能直接当成 retrieval 坏例使用，后续汇总时应视作一批已知脚本缺陷造成的脏样本。

### 下一步

1. 对修复后的脚本做一次最小回归验证
2. 复跑一轮 interactive debug，确认：
   - 不再出现 `preview` 异常
   - `你是谁` / `你有什么用处` 能稳定走 persona fixed response
3. 再决定是否需要清理或单独标记这批历史脏样本

### 本 part 一句话结论

Part 16-D 先修掉 interactive debug 壳子层的运行时 bug，并补齐 persona 问法 guardrail，避免把脚本缺陷误判成 retrieval 或模型质量问题。

## Part 16-E：Part 17 基线固化与历史脏样本标记

### 时间

- 2026-03-21

### 本 part 背景

进入 Part 17 前，先把 `object_lookup` 的基线和评估口径固定下来，避免后面只能凭感觉判断“似乎变好了”。

同时，Part 16-D 已确认：

- `interactive_debug_20260321T140506Z` 的 5 条 error 来自 `preview` 作用域 bug
- 这批样本不应继续混进 retrieval 分析

### 本次改动

#### 1. 固化 object 专项题集

新增：

- `ai_engine/finetune_qwen3/data/object_lookup_eval_cases_part17.json`

当前固定 `12` 条 `object_lookup` 题目，覆盖：

- 实体名：
  - `帮我找一下洛天依模型`
  - `找一下洛天依`
- 泛后缀：
  - `显示器内容`
  - `笔记本电脑记录`
  - `有没有台面相关场景？`
- 具体物体：
  - `地球仪`
  - `沙发`
  - `书架`
  - `手办`

#### 2. 新增 object 专项评估脚本

新增：

- `ai_engine/finetune_qwen3/scripts/evaluate_object_lookup_part17.py`

输出：

- `ai_engine/finetune_qwen3/logs/object_lookup_before_summary.json`
- `ai_engine/finetune_qwen3/logs/object_lookup_after_summary.json`
- `ai_engine/finetune_qwen3/logs/object_lookup_before_after_compare.md`

当前对比指标固定为：

- `object_lookup_count`
- `object_lookup_hit_rate`
- `object_lookup_bad_rate`
- `object_lookup_lexical_fallback_rate`
- `object_lookup_retrieval_miss_bad_count`
- `object_lookup_retrieval_low_relevance_bad_count`
- `object_lookup_rpc_empty_count`

#### 3. 给历史脏样本加显式标记

新增：

- `ai_engine/finetune_qwen3/scripts/mark_part16d_invalid_samples.py`

执行后：

- `interactive_debug_20260321T140506Z.jsonl` 中的 5 条 `preview` 崩溃样本已写入：
  - `sample_valid_for_retrieval_analysis = false`
  - `sample_invalid_reason = part16d_preview_scope_bug`

#### 4. route summary 支持排除无效样本

更新：

- `ai_engine/finetune_qwen3/scripts/summarize_interactive_debug_routes.py`

现在 summary 会：

- 显式统计 `excluded_invalid_count`
- 默认排除 `sample_valid_for_retrieval_analysis = false`
- 在 `object_lookup` section 中补：
  - `object_lookup_hit_rate`
  - `object_lookup_rpc_empty_count`

### 固化后的基线

来自当前 route summary：

- `turn_count = 62`
- `excluded_invalid_count = 5`
- `object_lookup_count = 22`
- `object_lookup_hit_rate = 1.0`
- `object_lookup_lexical_fallback_rate = 0.2727`
- `object_lookup_bad_rate = 0.1818`
- `object_lookup_retrieval_miss_bad_count = 4`
- `object_lookup_retrieval_low_relevance_bad_count = 0`
- `object_lookup_rpc_empty_count` 已纳入后续统计口径

专项题集 before 基线：

- `object_lookup_count = 12`
- `object_lookup_hit_rate = 1.0`
- `object_lookup_bad_rate = 0.0`
- `object_lookup_lexical_fallback_rate = 0.4167`
- `object_lookup_retrieval_miss_bad_count = 0`
- `object_lookup_rpc_empty_count = 4`

### 当前判断

这一小段完成后，Part 17 至少具备了两个条件：

1. 有一套固定 object 专项题集
2. 有一套能排除历史脏样本的 route-level 统计口径

后续优化不再只是“看起来更顺”，而是可以直接比较 before / after。

### 本 part 一句话结论

Part 16-E 先把 Part 17 的评估地基补齐：固定 object 专项题集、补 before/after 脚本、标记历史脏样本，并让 route summary 能显式排除这些无效样本。

## Part 16-F：object_lookup 检索专项优化首轮落地与回归

### 时间

- 2026-03-21

### 本次改动

#### 1. 强化 `normalize_lookup_terms()`

更新文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增能力：

- 支持并列拆分：
  - `和`
  - `及`
  - `与`
  - `、`
  - `，`
- 支持 query 前缀清洗：
  - `帮我找一下`
  - `找一下`
  - `最近有没有`
  - `关于`
- 支持尾部语气与泛后缀清洗：
  - `吗`
  - `呢`
  - `模型`
  - `内容`
  - `记录`
  - `画面`
  - `相关`

现在日志中会保留：

- `raw_target_objects`
- `normalized_lookup_terms`

便于后面直接看到“原始 target 是什么，最后拿什么做检索”。

#### 2. 把 object path 改成显式候选构建

新增 helper：

- `build_object_lookup_candidates()`
- `merge_object_candidates()`
- `score_object_candidate()`

这轮没有重写全链路，只对 object-like 查询做了更清晰的分层：

- 向量主召回
- lexical candidate supplement
- merge / dedup / rerank

新引入 route：

- `merged_vector_lexical`

同时保留：

- `vector_only`
- `vector_plus_filter`
- `lexical_fallback`

#### 3. recent/time 的 object-like 问法复用 object candidate builder

修复点：

- 某些问法会被 parser 判成 `recent_capture / time_qa`
- 但本质仍然是在问具体实体

处理方式：

- 若 `recent_capture / time_qa` 同时带有明确 `search_text + normalized_lookup_terms`
- 则仍走 object candidate builder

这样可以避免“看起来像 recent，但其实是 object query”的问法完全跳过 lexical supplement。

#### 4. 增强日志与回填

同步更新：

- `interactive_debug_chat.py`
- `run_part16c_batch_probe.py`
- `backfill_interactive_session_fields.py`

新增记录：

- `raw_target_objects`
- `normalized_lookup_terms`
- `route_reasons`
- `sample_valid_for_retrieval_analysis`

### 自测结果

#### 1. object 专项 before / after

文件：

- `ai_engine/finetune_qwen3/logs/object_lookup_before_summary.json`
- `ai_engine/finetune_qwen3/logs/object_lookup_after_summary.json`
- `ai_engine/finetune_qwen3/logs/object_lookup_before_after_compare.md`

当前 before / after 结果：

- `object_lookup_count: 12 -> 12`
- `object_lookup_hit_rate: 1.0 -> 1.0`
- `object_lookup_bad_rate: 0.0 -> 0.0`
- `object_lookup_retrieval_miss_bad_count: 0 -> 0`
- `object_lookup_retrieval_low_relevance_bad_count: 0 -> 0`
- `object_lookup_rpc_empty_count: 4 -> 4`
- `object_lookup_lexical_fallback_rate: 0.4167 -> 0.5833`

after route 分布：

- `lexical_fallback = 5`
- `vector_plus_filter = 4`
- `vector_only = 1`
- `merged_vector_lexical = 2`

#### 2. 结果解读

这一轮的结论不是“object retrieval 明显变强了”，而是：

- 没有引入新的 bad case
- 没有引入 `retrieval_low_relevance`
- object 路径的可观测性明显增强
- 词面补充不再只是一个黑盒兜底，而是有了显式 merge route

但同时也要明确承认：

- `rpc_empty` 没下降
- full lexical fallback 没下降
- 如果把 `merged_vector_lexical` 也算作 lexical dependency，则 fallback rate 反而升高

这说明当前瓶颈并不只是 term normalization 不够好，更像：

- Supabase 向量召回对某些实体词本身就偏弱
- 当前 object 提升更多体现在“补救更透明”，而不是“向量主路更强”

#### 3. 回归检查

额外跑了两组 smoke：

1. `part17_regression_subset_20260321`
2. `part17_inventory_smoke_20260321`

观察结果：

- `recent_hit`
  - `recent_list` 路由保持稳定
  - 未被 object 分支抢走
- `partial_coverage`
  - 复查样例仍走 `vector_plus_filter`
  - 未出现新的 route 混乱
- `inventory`
  - `inventory_special_case = 2/2`
  - `inventory_formatter = 2/2`
  - 两条都保持 `good`

也就是说，这轮 Part 17 首批 object 改动目前没有把：

- `recent_capture`
- `partial_coverage`
- `inventory`

这些旁路逻辑带坏。

### 当前判断

Part 17 第一轮已经完成了“窄范围检索专项优化”的最小闭环：

1. object path 的 normalization 更稳定
2. object path 有了显式 merge / rerank
3. route-level 评估与脏样本排除已就位
4. 回归上没有观察到其他 query class 的明显退化

但它还没有满足“明显压低 lexical 依赖”的目标。

因此当前最准确的判断是：

- Part 17 已经完成第一轮工程化落地
- 但从结果看，还不值得宣称 retrieval 已明显改善
- 下一步如果继续做 Part 17.1，更应该聚焦：
  - alias / synonym 表
  - object term 到 assets 字段的更系统映射
  - 向量阈值 / post-filter 调整
  - 更强的 lexical + vector rerank

### 本 part 一句话结论

Part 16-F 完成了 object_lookup 检索专项优化的首轮工程化落地，并证明这轮改动没有破坏其他 query class；但从 before / after 看，当前收益主要体现在可观测性和 merge 透明度，`rpc_empty` 与 lexical 依赖还没有被真正压下去。
