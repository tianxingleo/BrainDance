# Qwen3-1.7B 微调实践记录 Part 28

## Part 28：试 importance matrix 量化，并验证能否修复 Q5 strict 回退

### 本 part 目标

延续 Part 27 的结论，验证一个更具体的问题：

- `importance matrix` 量化，能不能把 `Q5_K_M` 在 strict 集上的 `partial_missing_negation` 回退拉回来

本次新增：

- 语料导出脚本：`ai_engine/finetune_qwen3/scripts/export_imatrix_corpus.py`
- GPU1 量化执行脚本：`ai_engine/finetune_qwen3/scripts/run_imatrix_quantization_gpu1.sh`

---

## 一、校准语料与量化产物

### 1. 校准语料

输入源：

- `ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl`
- `ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark_strict_no_leak_ood_20260322_v3.jsonl`
- `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl`

导出方式：

- `chat` 模式
- 共 `256` 条样本

产物：

- `ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/imatrix_v1/calibration_corpus_chat_256.txt`

### 2. imatrix 生成

模型：

- `model-f16.gguf`

参数：

- `gpu1`
- `CUDA_VISIBLE_DEVICES=1`
- `--device CUDA0`
- `-ngl 999`
- `--chunks 128`

产物：

- `imatrix_chat_256_chunks128.gguf`

### 3. imatrix 量化产物

- `model-f16-q4_k_m-imatrix.gguf`
- `model-f16-q5_k_m-imatrix.gguf`

---

## 二、原始 80 题 benchmark

| 量化版本 | partial_hallucination | partial_precision | partial_missing_negation | must_answer_focus | natural_style | 综合分 |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0.2500 | 0.8000 | 0.2500 | 0.5000 | 0.8375 | 84.50 |
| Q5_K_M | 0.1250 | 0.8824 | 0.2500 | 0.8750 | 0.9500 | 90.82 |
| Q4_K_M + imatrix | 0.1250 | 0.8889 | 0.1875 | 0.4375 | 0.7250 | 88.33 |
| Q5_K_M + imatrix | 0.0625 | 0.9412 | 0.0625 | 0.6875 | 0.8375 | 94.12 |

原始集下的判断非常直接：

1. `Q5_K_M + imatrix` 是这 4 个量化版本里最强的
2. 它把 `partial_missing_negation` 从 `0.2500` 压到 `0.0625`
3. `partial_hallucination` 也继续下降到 `0.0625`
4. `natural_style` 比原始 `Q5` 略降，但仍在可接受区间

---

## 三、strict v3 / 64 题 benchmark

| 量化版本 | partial_hallucination | partial_precision | partial_false_negative | partial_missing_negation | must_answer_focus | natural_style | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0.0556 | 0.9474 | 0.0000 | 0.0556 | 0.6667 | 0.7188 | 94.21 |
| Q5_K_M | 0.2222 | 0.8182 | 0.0000 | 0.4444 | 1.0000 | 0.9062 | 88.38 |
| Q4_K_M + imatrix | 0.1111 | 0.8947 | 0.0556 | 0.2222 | 0.7778 | 0.6562 | 91.20 |
| Q5_K_M + imatrix | 0.0556 | 0.9474 | 0.0000 | 0.0556 | 0.6667 | 0.7969 | 94.21 |

这里的结论比原始集更有价值：

1. `Q4_K_M + imatrix` 没有变好，反而退步
2. `Q5_K_M + imatrix` 把原始 `Q5` 的 strict 回退基本全部修回来了
3. `Q5_K_M + imatrix` 在核心任务指标上已经追平 `Q4_K_M`
4. 同时它的 `natural_style` 还高于 `Q4_K_M`

也就是说，本轮最重要的实验结果不是“imatrix 普遍有效”，而是：

- **imatrix 对 `Q5` 有效**
- **imatrix 对 `Q4` 无效，甚至有害**

---

## 四、Q5 strict 回退是否被修复

对比：

- baseline：`Q5_K_M`
- candidate：`Q5_K_M + imatrix`

指标变化：

- `partial_hallucination`: `0.2222 -> 0.0556`
- `partial_precision`: `0.8182 -> 0.9474`
- `partial_missing_negation`: `0.4444 -> 0.0556`

这说明 Part 27 里识别出的主问题，确实被 `imatrix` 量化大幅修复了。

但代价也存在：

- `must_answer_focus`: `1.0000 -> 0.6667`
- `natural_style`: `0.9062 -> 0.7969`

所以更准确的工程结论是：

- `Q5 + imatrix` 用一部分“更自然、更聚焦”的风格收益
- 换回了对 BrainDance strict 规则更稳定的遵守

对于当前任务，这个交换是值得的。

---

## 五、当前建议

本轮量化探索后的建议更新为：

1. 不要把 `Q4_K_M + imatrix` 作为升级方向
2. 可以把 `Q5_K_M + imatrix` 作为新的 GGUF 候选主线
3. 后续继续验证时，优先关注：
   - strict 集稳定性
   - `partial_missing_negation`
   - `must_answer_focus` 是否还能进一步拉回

如果继续迭代，下一步值得做的是：

1. 调整校准语料，进一步提高 `partial coverage` 样本占比
2. 保持 `Q5 + imatrix` 主线，再观察 Flutter / 端侧真实调用链下的小样本表现

---

## 六、本 part 产出

- 新增脚本：
  - `ai_engine/finetune_qwen3/scripts/export_imatrix_corpus.py`
  - `ai_engine/finetune_qwen3/scripts/run_imatrix_quantization_gpu1.sh`
- 新增量化产物：
  - `.../imatrix_v1/model-f16-q4_k_m-imatrix.gguf`
  - `.../imatrix_v1/model-f16-q5_k_m-imatrix.gguf`
- 新增 benchmark 日志：
  - `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_q4_gguf_imatrix_v1_round4_1_patch_mixed_gpu1.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_q5_gguf_imatrix_v1_round4_1_patch_mixed_gpu1.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_q4_gguf_imatrix_v1_gpu1.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_q5_gguf_imatrix_v1_gpu1.json`
- 新增分析结果：
  - `ai_engine/finetune_qwen3/logs/q5_vs_q5_imatrix_strict_analysis.json`
  - `ai_engine/finetune_qwen3/logs/q4_vs_q5_imatrix_strict_analysis.json`
  - `ai_engine/finetune_qwen3/logs/q4_vs_q4_imatrix_strict_analysis.json`
