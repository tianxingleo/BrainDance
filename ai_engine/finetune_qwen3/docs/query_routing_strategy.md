# Query Routing Strategy

## Scope

这份策略表用于 Part 16 期间统一约束 debug 链路里的 `query_class -> retrieval_route -> answer_route`。目标不是追求最终产品形态，而是让问题归因稳定、日志可聚合、后续检索优化有明确边界。

## Strategy Table

| query_class | retrieval_route | answer_route | 说明 |
| --- | --- | --- | --- |
| `greeting` | `non_retrieval` | `fixed_response` | 问候类不进入检索，直接固定回复 |
| `persona` | `non_retrieval` | `fixed_response` | 身份/能力说明不进入检索，直接固定回复 |
| `inventory` | `inventory_special_case` | `inventory_formatter` | 模型资产盘点走专门资产列表查询，再用 deterministic formatter 汇总 |
| `recent_capture` | `recent_list` / `vector_only` / `vector_plus_filter` | `lora_generation` | recent/time 问法优先 recent list，否则按普通检索链路回答 |
| `time_qa` | `recent_list` / `vector_only` / `vector_plus_filter` | `lora_generation` | 带时间窗的检索仍由检索链路返回证据，生成侧只做人话压缩 |
| `object_lookup` | `vector_only` / `vector_plus_filter` / `lexical_fallback` | `lora_generation` | 具体实体检索默认交给 LoRA 把证据压成短句 |
| `object_lookup` + semantic expansion | `lexical_fallback` | `semantic_summary_formatter` | 抽象概念问法命中扩展词时，优先 deterministic summary |
| `partial_coverage` | `vector_plus_filter` / `lexical_fallback` | `lora_generation` | partial 仍依赖 support map 和 LoRA 生成显式命中/未命中 |

## Current Logging Contract

交互日志至少保留这些字段：

- `session_name`
- `turn_id`
- `question`
- `query_class`
- `retrieval_route`
- `fallback_trigger_reason`
- `answer_route`
- `hit_count`
- `retrieval_latency_sec`
- `generation_latency_sec`
- `user_feedback_label`
- `issue_bucket`
- `feedback`

## Part 16 Decision Rule

如果后续 route summary 里出现以下信号，就进入 Part 17 检索专项优化：

- `lexical_fallback` 长期占比偏高
- `object_lookup` 的 fallback rate 显著高于其他 query class
- `retrieval_miss` / `retrieval_low_relevance` 成为主要 `issue_bucket`

如果 route summary 显示 retrieval 已经比较稳，而 `formatter_needed` / `answer_style` 占主导，则优先进入体验层打磨，而不是新一轮 LoRA。
