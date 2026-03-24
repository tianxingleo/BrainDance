# Qwen3-1.7B 微调实践记录 Part 27

## Part 27：定位 Q5_K_M 在 strict 集退化的真实原因

### 本 part 目标

不是停留在“`Q5` strict 集分数更低”这个表层现象，而是把退化模式拆到 case 级别，确认它到底退化在哪里。

本次新增了正式分析脚本：

- `ai_engine/finetune_qwen3/scripts/analyze_q4_q5_regression.py`

输入：

- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_q4_gguf_gpu1.json`
- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_q5_gguf_gpu1.json`

输出：

- `ai_engine/finetune_qwen3/logs/q4_vs_q5_strict_regression_analysis.json`
- `ai_engine/finetune_qwen3/logs/q4_vs_q5_strict_regression_analysis.md`

---

## 一、先看结论

`Q5_K_M` 在 strict 集上的退化，不是“整体变傻”或“普遍胡说”，而是更集中、更有模式：

1. 共享 `64` 个 case 里，正式脚本定位到 `9` 个回退 case
2. 其中 `7` 个集中在 `partial_coverage`
3. 回退 flag 里最明显的是：
   - `partial_missing_negation`: `7` 次
   - `partial_hallucination`: `3` 次
4. 也就是说，`Q5` 最容易犯的错不是没读到证据，而是：
   - 只回答命中的对象
   - 漏掉对未命中对象的明确否定

---

## 二、指标层面的硬结论

strict v3 / 64 题：

| 版本 | partial_hallucination | partial_precision | partial_missing_negation | must_answer_focus | natural_style |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 0.0556 | 0.9474 | 0.0556 | 0.6667 | 0.7188 |
| Q5_K_M | 0.2222 | 0.8182 | 0.4444 | 1.0000 | 0.9062 |

这里最关键的是：

- `partial_missing_negation`: `0.0556 -> 0.4444`
- `partial_precision`: `0.9474 -> 0.8182`
- `partial_hallucination`: `0.0556 -> 0.2222`

但它的另一面也很清楚：

- `must_answer_focus`: `0.6667 -> 1.0000`
- `natural_style`: `0.7188 -> 0.9062`

所以 `Q5` 的真实行为变化不是“纯退化”，而是：

- 更自然
- 更像在直接回答用户
- 但对 BrainDance 这套 `partial coverage + 必须显式否定未命中对象` 的规则遵守变差

---

## 三、case 级模式

### 1. 典型模式一：只回答命中项，不补否定

案例：

- `partial_coverage_001_rw`
- `partial_coverage_003_rw`
- `partial_coverage_009_rw`
- `partial_coverage_013_rw`

典型对比：

- `Q4`：`发现过地毯，但没见到打印机；暂无打印机相关记录。`
- `Q5`：`目前只查到地毯这条记录。`

这类回答从“自然人类语言”角度看未必差，但对 BrainDance 的评测规则来说，缺失了对未命中对象的明确否定，因此会被算作：

- `partial_missing_negation = true`

### 2. 典型模式二：为了自然表达，把未命中对象带成了命中项

案例：

- `partial_coverage_005_rw`
- `partial_coverage_007_rw`

典型对比：

- `Q4`：`电视柜在客厅场景中，咖啡机暂无相关记录。`
- `Q5`：`电视柜里放着咖啡机。`

这一类不是简单漏答，而是把未命中对象编进了正向描述里，因此同时触发：

- `partial_hallucination = true`
- `partial_missing_negation = true`

---

## 四、当前工程判断

关于“为什么 `Q5` strict 集明显退化”，当前最准确的判断是：

1. 不是 `Q5` 读不懂 retrieval
2. 不是 `Q5` 普遍答错
3. 而是 `Q5` 更偏向自然、简洁、聚焦命中对象的回答风格
4. 这种风格和 BrainDance 当前 strict 集的任务规则有冲突

所以后续要修，不应该盲目换回 `Q4` 或直接放弃 `Q5`，而应该优先尝试：

1. `importance matrix` 量化
2. 更贴近 `partial coverage` 规则的校准语料
3. 如果仍不稳，再考虑量化类型或模板进一步约束

---

## 五、本 part 产出

- 新增分析脚本：`ai_engine/finetune_qwen3/scripts/analyze_q4_q5_regression.py`
- 生成正式分析结果：
  - `ai_engine/finetune_qwen3/logs/q4_vs_q5_strict_regression_analysis.json`
  - `ai_engine/finetune_qwen3/logs/q4_vs_q5_strict_regression_analysis.md`
- 给后续 `imatrix` 实验建立了明确验证目标：
  - 优先看 `partial_missing_negation`
  - 其次看 `partial_hallucination`
  - 不再只看综合分
