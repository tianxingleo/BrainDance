# Qwen3-0.6B 微调实践记录 Part 31

## Part 31：Qwen3-0.6B 全量微调首轮可行性实验

### 时间

- 2026-03-23

### 本 part 目标

基于当前已经跑通的 `Qwen3-0.6B LoRA` 链路，补一条最小改动的 `full SFT` 支线，并在 `gpu1` 上完成：

- `train_full_sft.py` 落地
- `0.6B full` smoke train
- `0.6B full` 正式 1 epoch 训练
- smoke eval
- 原始 80 题 benchmark
- strict v3 / 64 题 benchmark

这一轮不回答“是否直接替换主线部署”，只回答：

1. `0.6B full` 是否真实可训、可评测、可复现
2. 它相对 `0.6B LoRA` 的收益究竟落在“任务正确性”还是“输出风格”

---

## 一、本轮新增支线

### 1. 新增 full SFT 训练脚本

新增文件：

- [train_full_sft.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/train_full_sft.py)

实现原则：

- 复用现有 LoRA 版本的数据格式与 assistant-only loss mask
- 保留 `Qwen3` chat template 处理逻辑
- 去掉 `PEFT/LoRA` 注入，改为全参数训练
- 训练配置改成更保守的 `full SFT` 参数

本轮正式训练参数：

- 模型：`Qwen/Qwen3-0.6B`
- `cutoff_len = 1536`
- `num_train_epochs = 1`
- `learning_rate = 1e-5`
- `per_device_train_batch_size = 2`
- `per_device_eval_batch_size = 2`
- `gradient_accumulation_steps = 8`
- 精度：`auto_bf16`

另外新增了 `--max_steps`，便于先做 smoke train。

### 2. 新增 gpu1 包装脚本

新增文件：

- [run_train_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_full_gpu1.sh)
- [run_smoke_eval_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_full_gpu1.sh)
- [run_benchmark_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_full_gpu1.sh)
- [run_benchmark_strict_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_strict_qwen3_0p6b_full_gpu1.sh)

这些脚本统一固定：

- `conda run -n qwen3_ft`
- `CUDA_VISIBLE_DEVICES=1`
- `HF_ENDPOINT=https://hf-mirror.com`

### 3. 顺手修了 workflow 脚本的顶层导入问题

修改文件：

- [train_lora_sft.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/train_lora_sft.py)
- [train_full_sft.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/train_full_sft.py)
- [merge_lora_adapter.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/merge_lora_adapter.py)
- [test_qwen3_workflow_scripts.py](/ltx-data/BrainDance/tests/test_qwen3_workflow_scripts.py)

原因：

- 默认 Python 环境下，`bitsandbytes/triton` 组合会在模块导入阶段干扰 `peft/Trainer`
- 这会导致“只测 helper 函数”的 workflow test 也被环境噪声拖死

本轮把重依赖挪进 `main()` 路径后，`CUDA_VISIBLE_DEVICES='' pytest -q tests/test_qwen3_workflow_scripts.py` 已恢复通过。

---

## 二、训练执行情况

### 1. smoke train

输出目录：

- [qwen3_0p6b_full_sft_smoke_gpu1](/ltx-data/BrainDance/ai_engine/finetune_qwen3/outputs/qwen3_0p6b_full_sft_smoke_gpu1)

执行方式：

- `gpu1`
- `max_steps = 5`
- `save_strategy = no`
- `eval_strategy = no`

关键观察：

- `trainable_ratio = 100%`
- 5 步内 loss 从 `2.7907` 降到 `1.8739`
- 训练过程无 OOM、无保存异常、无 tokenizer/chat template 异常

结论：

- `0.6B full SFT` 支线在当前环境里可正常 forward / backward

### 2. 正式 1 epoch 训练

训练脚本：

- [run_train_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_full_gpu1.sh)

训练日志：

- `ai_engine/finetune_qwen3/logs/train_qwen3_0p6b_full_round1_gpu1.log`

训练产物：

- [qwen3_0p6b_full_sft_round1](/ltx-data/BrainDance/ai_engine/finetune_qwen3/outputs/qwen3_0p6b_full_sft_round1)

关键结果：

- `trainable_params = 596,049,920`
- `total_params = 596,049,920`
- `steps_per_epoch_estimate = 57`
- 训练耗时 `78.039s`
- 最终 `eval_loss = 0.49685850739479065`

显存观察：

- `gpu1` 正式训练期间显存约 `22GB`
- 明显高于 `0.6B LoRA`，但仍在当前 `L20 48GB` 范围内稳定运行

### 3. 本阶段总结

这一阶段已经回答了第一个问题：

> `Qwen3-0.6B full SFT` 在 BrainDance 当前数据、脚本和 `gpu1 + qwen3_ft` 环境里是可以真实跑通的。

---

## 三、评测结果

### 1. smoke eval

输出文件：

- `ai_engine/finetune_qwen3/logs/smoke_eval_qwen3_0p6b_full_round1_gpu1.json`

代表性输出：

- `recent_hit`：`最近拍到的是触控笔桌面采集 01，3月19日。`
- `no_hit`：`目前没有找到与自行车相关的记录。`
- `partial_hit`：`最近拍到过触控笔，没有拍到冰箱。`

结论：

- hit / no-hit / partial 三类基础行为正常

### 2. 原始 80 题 benchmark

脚本：

- [run_benchmark_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_full_gpu1.sh)

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_full_round1_gpu1.json`

与 `0.6B LoRA` 对比如下：

| 指标 | 0.6B LoRA | 0.6B full |
|---|---:|---:|
| false_no_answer_rate | 0.0000 | 0.0000 |
| partial_hallucination_rate | 0.0625 | 0.1250 |
| natural_style_rate | 0.8000 | 0.9250 |
| evidence_utilization_rate | 0.9844 | 1.0000 |
| partial_hit_precision | 0.9375 | 0.8824 |
| partial_false_negative_rate | 0.0625 | 0.0625 |
| partial_missing_negation_rate | 0.1875 | 0.0625 |
| must_answer_focus_rate | 0.6875 | 0.8125 |

原始 benchmark 的直接结论：

1. `full` 在风格与聚焦度上明显更稳
2. `full` 把 `partial_missing_negation_rate` 从 `0.1875` 压到了 `0.0625`
3. 但 `partial_hallucination_rate` 反而从 `0.0625` 升到了 `0.1250`
4. 因此不能把这轮 `full` 简单判断为“全方位优于 LoRA”

### 3. strict v3 / 64 题 benchmark

脚本：

- [run_benchmark_strict_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_strict_qwen3_0p6b_full_gpu1.sh)

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_0p6b_full_gpu1.json`

与 `0.6B LoRA` 对比如下：

| 指标 | 0.6B LoRA | 0.6B full |
|---|---:|---:|
| false_no_answer_rate | 0.0000 | 0.0000 |
| partial_hallucination_rate | 0.0556 | 0.0556 |
| natural_style_rate | 0.8281 | 0.9375 |
| evidence_utilization_rate | 1.0000 | 1.0000 |
| partial_hit_precision | 0.9474 | 0.9474 |
| partial_false_negative_rate | 0.0000 | 0.0000 |
| partial_missing_negation_rate | 0.0556 | 0.0556 |
| must_answer_focus_rate | 1.0000 | 1.0000 |

strict 集的直接结论：

1. 在核心任务正确性口径上，`full` 和 `LoRA` 基本打平
2. 这轮 `full` 最清晰的收益仍然是 `natural_style_rate`
3. strict 集没有出现比 LoRA 更差的任务纪律退化

---

## 四、case 级观察

### 1. 原始 benchmark 的主要退化点

本轮 `0.6B full` 在原始集上主要新增了两个问题 case：

- `partial_coverage_006`
  - 回答：`目前只找到了写字台，没有看到椅子。`
  - 真实应答：支持对象是 `椅子`，不支持对象是 `写字台`
  - 这是典型的 supported / unsupported 对调
- `partial_coverage_008`
  - 回答：`最近拍到海边的风景画，暂无相关记录。`
  - 真实应答：支持对象是 `风景画`，不支持对象是 `海边`
  - 这里把 unsupported 词 `海边` 正向说出来了

这两个 case 说明：

- `full` 虽然更自然、更像一句完整短答
- 但在一小部分 `partial_coverage` 样本上，仍会出现 unsupported term 被正向带出的现象

### 2. strict 集的单点退化

strict v3 中的主要问题 case 是：

- `partial_coverage_006_rw`
  - `LoRA`：`目前只找到椅子相关内容，未见写字台。`
  - `full`：`有椅子和写字台。`

这说明：

- strict 集下 `full` 的总体指标没有变差
- 但它并没有完全解决 partial coverage 的 unsupported leakage 风险

### 3. 风格提升的真实来源

原始集上有 `10` 个 case 出现了 `natural_style: false -> true` 的改善，主要集中在：

- `recent_hit`
- `stability`
- 部分 `must_answer / partial_coverage`

典型变化不是“任务规则突然更强”，而是：

- 从偏列表、偏堆砌的答法
- 变成更短、更自然、更像最终产品输出的句子

因此这轮 `full` 的最大收益，更接近：

> 保持正确性的同时，把回答形式往产品化短句方向再推了一步。

---

## 五、本轮结论

Part 31 的结论可以明确写成三条：

1. `Qwen3-0.6B full SFT` 已经在 BrainDance 当前链路上真实跑通，训练、smoke eval、原始 benchmark、strict benchmark 全部可复现。
2. 相比 `0.6B LoRA`，这轮 `full` 的主要收益是 `natural_style`、`must_answer_focus` 和 `partial_missing_negation`，不是核心任务正确性的全面跃迁。
3. 由于原始 benchmark 上 `partial_hallucination_rate` 出现回升，这一轮还不适合直接判定 `full` 应该替代当前 `0.6B LoRA` 主线。

更直接一点说：

- 如果目标是“更自然、更像产品句子”，这轮 `full` 有价值
- 如果目标是“严格替代 LoRA，成为新的默认 0.6B 主线”，证据还不够

---

## 六、下一步建议

下一轮如果继续做 `0.6B full`，建议顺序如下：

1. 先把学习率从 `1e-5` 下调到 `8e-6` 或 `5e-6`
2. 继续盯 `partial_coverage` 的 unsupported leakage，而不是只看总体分数
3. 保持同一份 train/val，不要立刻混入 patch-only 数据
4. 如果 raw benchmark 的 hallucination 仍压不住，就把 `full` 定位为“风格探索支线”，不要替换 `0.6B LoRA`

一句话总结：

> `0.6B full` 这轮不是失败，但它当前更像“风格更强、纪律未必更强”的对照实验版本，而不是已经可以无争议接管主线的版本。
