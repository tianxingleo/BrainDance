# Qwen3-1.7B 微调实践记录 Part 31

## Part 31-A：1.7B full SFT 支线落地与 gpu1 执行入口补齐

### 时间

- 2026-03-23

### 本 part 目标

- 在现有 `ai_engine/finetune_qwen3` 骨架上补齐 `Qwen3-1.7B full SFT` 支线
- 保持与现有 `LoRA / merged / strict benchmark` 同口径
- 保证可以直接在 `conda run -n qwen3_ft` + `gpu1` 上训练与评测

### 本次改动

新增脚本：

- `ai_engine/finetune_qwen3/scripts/run_train_qwen3_1p7b_full_gpu1.sh`
- `ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_1p7b_full_gpu1.sh`
- `ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_1p7b_full_gpu1.sh`
- `ai_engine/finetune_qwen3/scripts/run_benchmark_strict_qwen3_1p7b_full_gpu1.sh`

兼容性修正：

- `ai_engine/finetune_qwen3/scripts/run_smoke_eval.py`
  - 把 `AutoModelForCausalLM.from_pretrained()` 的参数从 `dtype=` 修正为 `torch_dtype=`
  - 这样 full HF 模型目录可以与当前 transformers 版本正常对齐

### 本 part 执行约束

本轮执行统一使用：

- 环境：`qwen3_ft`
- GPU：`CUDA_VISIBLE_DEVICES=1`
- 模型：`Qwen/Qwen3-1.7B`
- 训练长度：`cutoff_len=1536`
- 精度：`auto_bf16`
- 梯度累积：`gradient_accumulation_steps=8`

### 本 part 一句话结论

`1.7B full SFT` 的训练、smoke eval、benchmark、strict benchmark 四个入口已全部补齐，可以作为独立实验支线稳定复现。

## Part 31-B：mini smoke 训练验证 full 支线可行性

### 本 part 目标

- 先验证 `1.7B full` 在 `gpu1` 上是否会 OOM
- 验证完整 HF 模型目录是否能保存成功
- 验证保存后的模型能否直接接入 smoke eval 与 benchmark

### 本次做法

为避免一上来直接跑完整 900/100：

- 从主训练集抽取 `200` 条 train
- 从验证集抽取 `50` 条 val
- 运行 `max_steps=20` 的 smoke full train

实际输出目录：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_full_sft_smoke_gpu1`

实际评测日志：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_full_smoke_gpu1.json`
- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_1p7b_full_smoke_gpu1.json`

### 关键观察

- `gpu1` 显存稳定在约 `18G`
- `20 step` 可正常 forward / backward / eval / save
- full 模型目录可以直接被 `run_smoke_eval.py` 和 `evaluate_benchmark.py` 加载

smoke 训练结果：

- `train_loss`: `2.8611`
- `eval_loss`: `2.1784`
- `steps_per_epoch_estimate`: `25`

smoke benchmark（主集）：

- `partial_false_negative_rate`: `0.0625`
- `partial_missing_negation_rate`: `0.8125`
- `must_answer_focus_rate`: `0.9375`

smoke benchmark（strict）：

- `partial_false_negative_rate`: `0.0556`
- `partial_missing_negation_rate`: `0.6667`
- `must_answer_focus_rate`: `1.0`

### 本次判断

这一步的结论不是“smoke 质量很好”，而是：

- `1.7B full` 在当前硬件上可训
- full 产物可直接进入现有评测流水线
- 可以放心进入完整 round1

### 本 part 一句话结论

`1.7B full` 在 `gpu1` 上单卡可训且链路可闭环，实验可进入正式轮次。

## Part 31-C：round1 正式全量微调（lr=8e-6）与主结果评测

### 本 part 目标

- 用完整 `900 / 100` 数据跑完第一轮 full SFT
- 和当前 `1.7B LoRA current best / 1.7B merged` 做同口径 benchmark 对照
- 判断 full 是否具备继续推进价值

### 训练配置

- 输出目录：`ai_engine/finetune_qwen3/outputs/qwen3_1p7b_full_sft_round1_gpu1`
- `num_train_epochs=1`
- `learning_rate=8e-6`
- `per_device_train_batch_size=1`
- `per_device_eval_batch_size=1`
- `gradient_accumulation_steps=8`

### 训练结果

训练规格写入：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_full_sft_round1_gpu1/training_spec.json`

最终评估写入：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_full_sft_round1_gpu1/final_metrics.json`

关键数据：

- `trainable_params`: `1,720,574,976`
- `trainable_ratio`: `100.0%`
- `steps_per_epoch_estimate`: `113`
- `train_runtime`: `156.40s`
- `train_loss`: `1.1663`
- `eval_loss`: `0.6181`

产物体量：

- `model.safetensors`: 约 `3.3G`
- 输出目录总占用：约 `13G`

### smoke eval 抽查

正式 round1 的 smoke 输出已恢复正常自然句式：

- `recent_hit`: 能概括最近拍到的内容
- `no_hit`: 能稳定返回“暂无相关记录”
- `partial_hit`: 能回答“有触控笔相关记录”

说明这轮 full 至少没有出现“训练后不会说话”或 HF 目录不可用的问题。

### benchmark 对照

主集对照：

| 模型 | false_no_answer | partial_hallucination | natural_style | evidence_utilization | partial_hit_precision | partial_false_negative | partial_missing_negation | must_answer_focus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1.7B LoRA round4_1_patch_mixed` | `0.0` | `0.0` | `0.85` | `1.0` | `1.0` | `0.0` | `0.0` | `0.8125` |
| `1.7B merged round4_1_patch_mixed` | `0.0` | `0.0625` | `0.825` | `1.0` | `0.9375` | `0.0625` | `0.0625` | `0.6875` |
| `1.7B full round1 lr=8e-6` | `0.0` | `0.0625` | `0.9125` | `1.0` | `0.9375` | `0.0625` | `0.125` | `0.75` |

strict 集对照：

| 模型 | false_no_answer | partial_hallucination | natural_style | evidence_utilization | partial_hit_precision | partial_false_negative | partial_missing_negation | must_answer_focus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1.7B LoRA round4_1_patch_mixed` | `0.0` | `0.0` | `0.8281` | `1.0` | `1.0` | `0.0` | `0.0` | `0.8889` |
| `1.7B merged round4_1_patch_mixed` | `0.0` | `0.0` | `0.7812` | `1.0` | `1.0` | `0.0` | `0.0` | `0.6667` |
| `1.7B full round1 lr=8e-6` | `0.0` | `0.1111` | `0.8438` | `1.0` | `0.8889` | `0.1111` | `0.1111` | `0.8889` |

### 本次判断

这轮 `full round1` 的信号是：

- 优点
  - `natural_style_rate` 比当前 LoRA 更高
  - `evidence_utilization_rate` 维持 `1.0`
  - `must_answer_focus_rate` 在 strict 集与 LoRA 持平
- 缺点
  - `partial_false_negative_rate` 比 LoRA 明显退化
  - `partial_missing_negation_rate` 比 LoRA 明显退化
  - strict 集 `partial_hallucination_rate` 也出现退化

所以这轮不能证明 `1.7B full` 优于当前 `LoRA current best`，最多只能说明：

- full 可训练
- full 风格自然度有潜在优势
- 但在当前高约束任务中，纪律性指标没有赢

### 本 part 一句话结论

`lr=8e-6` 的 `1.7B full round1` 可行但不优，尚不足以替代当前 LoRA 主线。

## Part 31-D：保守学习率复验（lr=5e-6）与停止条件确认

### 本 part 目标

- 排除“第一轮只是学习率偏大”的可能
- 用更保守的 `lr=5e-6` 再做一轮 full 验证
- 如果仍无明确收益，则终止 full 深挖

### 训练配置

- 输出目录：`ai_engine/finetune_qwen3/outputs/qwen3_1p7b_full_sft_round1_lr5e6_gpu1`
- 其余配置保持与 round1 一致，仅修改：
  - `learning_rate=5e-6`

### 训练结果

关键数据：

- `train_runtime`: `165.92s`
- `train_loss`: `1.6661`
- `eval_loss`: `1.0381`

从 `eval_loss` 看，这一轮已经明显弱于 `lr=8e-6`。

### benchmark 结果

主集：

- `false_no_answer_rate`: `0.0156`
- `partial_hallucination_rate`: `0.1875`
- `natural_style_rate`: `0.95`
- `evidence_utilization_rate`: `0.9688`
- `partial_hit_precision`: `0.8333`
- `partial_false_negative_rate`: `0.0625`
- `partial_missing_negation_rate`: `0.3125`
- `must_answer_focus_rate`: `0.75`

strict 集：

- `false_no_answer_rate`: `0.0`
- `partial_hallucination_rate`: `0.1667`
- `natural_style_rate`: `0.9219`
- `evidence_utilization_rate`: `0.9818`
- `partial_hit_precision`: `0.8333`
- `partial_false_negative_rate`: `0.1667`
- `partial_missing_negation_rate`: `0.3889`
- `must_answer_focus_rate`: `0.6667`

### 本次判断

`lr=5e-6` 的结论非常明确：

- 风格自然度继续提高
- 但纪律性、命中精度、partial 覆盖质量进一步恶化
- 已经不是“略差于 LoRA”，而是整体不适合作为当前主线候选

这说明：

- 当前问题不只是 `8e-6` 偏大
- 至少在这套 `900/100` 数据与当前任务约束下，`1.7B full` 暂时没有体现出高成本对应的质量收益

### 最终结论

本轮 `1.7B full SFT` 可行性实验已经完整回答了两个问题：

1. `1.7B full` 能不能在 `gpu1` 上稳定全量微调？
   - 能，链路已跑通
2. `1.7B full` 值不值得替代当前 `1.7B LoRA current best`？
   - 目前不值得

当前建议保持：

- 训练主线：`1.7B LoRA round4_1_patch_mixed`
- 质量基线：`1.7B merged`
- 部署主线：`1.7B Q5_K_M + imatrix GGUF`

`1.7B full` 当前更适合作为：

- 已验证可训练的备用研究方向
- 后续若要继续，只建议在新的数据配方或更强 patch 目标下再重开实验

### 本 part 一句话结论

`1.7B full` 已完成可行性验证，但两轮结果都未显示出足够 ROI，结论是“链路通过、方向暂缓，不进入主线”。
