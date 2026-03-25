# Qwen3-1.7B 微调实践记录 Part 24

## Part 24：把 0.6B / 1.7B merged / 1.7B Q4 GGUF 并入两份总评测报告

### 本 part 目标

把此前分散在：

- `Part21`
- `Part23`
- 各自 benchmark 日志

里的本地部署变体结果，真正并入两份总报告：

- `Qwen3-1.7B-LoRA-对标评测报告-2026-03-22.md`
- `Qwen3-1.7B-LoRA-严格无泄漏对标评测报告-2026-03-22.md`

目标不是简单补文字，而是把以下 4 个本地版本放进同一张比较框架：

1. `1.7B LoRA`
2. `0.6B LoRA`
3. `1.7B merged`
4. `1.7B Q4 GGUF`

---

## 一、本轮新增实测

为了避免严格无泄漏报告里只补“主观推断”，本轮额外补跑了 strict v3：

- `0.6B LoRA`
- `1.7B merged`
- `1.7B Q4 GGUF`

对应日志：

- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_0p6b_lora_gpu1.json`
- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_merged_gpu1.json`
- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_q4_gguf_gpu1.json`

### 1. strict v3 结果摘要

| 本地版本 | false_no_answer | partial_hallucination | evidence_utilization | partial_precision | partial_false_negative | partial_missing_negation | must_answer_focus | natural_style |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.7B LoRA | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8889 | 0.8281 |
| 0.6B LoRA | 0.0000 | 0.0556 | 1.0000 | 0.9474 | 0.0000 | 0.0556 | 1.0000 | 0.8281 |
| 1.7B merged | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6667 | 0.7812 |
| 1.7B Q4 GGUF | 0.0000 | 0.0556 | 1.0000 | 0.9474 | 0.0000 | 0.0556 | 0.6667 | 0.7188 |

### 2. Q4 strict v3 的额外结论

这轮 strict 跑 `Q4 GGUF` 时又暴露了一个真实工程问题：

- `llama.cpp` 子进程偶发启动失败
- 失败根因不是模型损坏，而是共享 GPU 环境下偶发 `cudaMalloc failed: out of memory`
- 失败发生在模型加载阶段，不是生成阶段

本轮处理方式：

1. 先把 `batch / ubatch / threads` 降到更保守的配置
2. 给 `evaluate_gguf_benchmark.py` 增加有限重试
3. 最终完成 strict v3 全量 64 题评测

这说明：

- `Q4 GGUF` 已具备实跑能力
- 但在共享 GPU 环境下，`llama.cpp` 这条部署链路对可用显存波动比 HF 版本更敏感

---

## 二、写入总报告的内容

### 1. 原始 benchmark 总报告

已新增：

- “本地部署变体补充总表（新增）”
- `1.7B LoRA / 0.6B LoRA / 1.7B merged / 1.7B Q4 GGUF` 四版本同表对比
- 一段面向部署选择的解释文字

### 2. 严格无泄漏总报告

已新增：

- “本地部署变体补充总表（严格集新增）”
- strict v3 下四版本同表对比
- 一段面向部署选择的解释文字
- strict 相关新增日志路径

---

## 三、本轮结论

把四个本地版本真正并入两份总报告后，当前项目里的模型选择关系更清楚了：

1. `1.7B LoRA`：当前本地质量上界
2. `0.6B LoRA`：最强轻量端侧候选之一
3. `1.7B merged`：最稳的独立 HF 部署版本
4. `1.7B Q4 GGUF`：最接近端侧落地链路，但仍需继续平衡质量与资源

进一步说：

- 在原始 benchmark 上，`Q4` 的质量回退比 strict 集更明显
- 在 strict 集上，`Q4` 的任务正确性没有继续明显恶化，但自然度与聚焦度仍偏弱
- `0.6B LoRA` 在 strict 集上的稳定性比预期更强，已经足够值得继续推进端侧对接

---

## 四、下一步

接下来建议顺序：

1. 测 `Q5_K_M`，确认是否能显著改善 `Q4` 的回退
2. 做 Flutter 实际调用口径的小样本延迟/显存测试
3. 把 `0.6B LoRA` 与 `1.7B Q4 GGUF` 做成端侧选型结论页
