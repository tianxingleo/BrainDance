# Qwen3-1.7B 微调实践记录 Part 26

## Part 26：补测 Q5_K_M，确认是否真的优于 Q4_K_M

### 本 part 目标

验证一个非常具体的问题：

- `Q5_K_M` 能不能显著改善 `Q4_K_M` 的质量回退

这里不做空想，直接走真实链路：

1. 从已有 `model-f16.gguf` 量化出 `model-f16-q5_k_m.gguf`
2. 在 `gpu1` 上跑原始 80 题 benchmark
3. 在 `gpu1` 上跑 strict v3 / 64 题 benchmark
4. 和当前 `Q4_K_M` 做正面对比

---

## 一、真实量化结果

输入：

- `ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/model-f16.gguf`

输出：

- `ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/model-f16-q5_k_m.gguf`

量化工具：

- `ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda/bin/llama-quantize`

结果：

- `Q4_K_M`：约 `1.1G`
- `Q5_K_M`：约 `1.2G`

也就是说：

- `Q5` 比 `Q4` 多占一点空间
- 但仍处于端侧可接受体积区间

---

## 二、原始 80 题 benchmark：Q5 是否改善了 Q4

### 1. 结果对比

| 量化版本 | false_no_answer | partial_hallucination | partial_precision | partial_missing_negation | must_answer_focus | natural_style | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0.0000 | 0.2500 | 0.8000 | 0.2500 | 0.5000 | 0.8375 | 84.50 |
| Q5_K_M | 0.0156 | 0.1250 | 0.8824 | 0.2500 | 0.8750 | 0.9500 | 90.82 |

### 2. 判断

如果只看原始 80 题 benchmark，答案是：

- `Q5_K_M` 确实比 `Q4_K_M` 更好

改善最明显的是：

- `partial_hallucination`
- `partial_precision`
- `must_answer_focus`
- `natural_style`

但它也不是单向改进：

- `false_no_answer_rate` 从 `0` 退到了 `0.0156`
- `partial_missing_negation_rate` 没有改善

---

## 三、strict v3：Q5 是否仍然改善 Q4

### 1. 结果对比

| 量化版本 | false_no_answer | partial_hallucination | partial_precision | partial_missing_negation | must_answer_focus | natural_style | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0.0000 | 0.0556 | 0.9474 | 0.0556 | 0.6667 | 0.7188 | 94.21 |
| Q5_K_M | 0.0000 | 0.2222 | 0.8182 | 0.4444 | 1.0000 | 0.9062 | 88.38 |

### 2. 判断

如果看 strict v3，结论反过来了：

- `Q5_K_M` 没有改善 `Q4_K_M`
- 而且在关键任务指标上更差

明显变差的是：

- `partial_hallucination`
- `partial_precision`
- `partial_missing_negation`

变好的主要是体验侧：

- `must_answer_focus`
- `natural_style`

---

## 四、最终结论

所以对于“`Q5_K_M` 能不能显著改善 `Q4_K_M` 的回退”这个问题，当前最准确的答案是：

- **不能下结论说“显著改善”**

更细一点说：

1. 原始 benchmark 上，`Q5` 看起来优于 `Q4`
2. strict v3 上，`Q5` 反而劣于 `Q4`
3. 因此当前 `Q5` 的收益不稳定，不适合直接替换 `Q4` 作为默认量化版本

所以当前工程判断应该是：

- `Q5_K_M` 不是本轮的确定升级
- 它只是一个“在原始集上更好、在严格集上更差”的不稳定候选

---

## 五、当前建议

接下来更值得做的是：

1. 不要立刻把 `Q5_K_M` 提升为默认端侧版本
2. 继续保留 `Q4_K_M` 作为当前 GGUF 主线
3. 如果继续量化探索，应优先排查：
   - 当前 `llama.cpp` / GGUF 转换链路是否影响了 `Q5` 在 strict 集上的行为
   - 是否需要 importance matrix 或更稳定的量化设置
4. 同时把 `0.6B LoRA` 保持为更稳的小模型端侧候选
