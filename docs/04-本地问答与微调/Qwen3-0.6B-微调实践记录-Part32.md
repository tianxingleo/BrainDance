# Qwen3-0.6B 微调实践记录 Part 32

## Part 32：Qwen3-0.6B full SFT 第二轮与第三轮保守学习率对照实验

### 时间

- 2026-03-23

### 本 part 目标

基于 Part 31 的首轮 `0.6B full SFT` 结果，继续验证一个更聚焦的问题：

> 如果不改数据、不改 benchmark、不混 patch-only，只把 `learning_rate` 从 `1e-5` 下调，能不能在保住风格收益的同时，把 `partial_hallucination` 压回去？

本轮只做最小变量控制：

- 数据不变
- `epoch = 1`
- batch / grad accum 不变
- 只调 `learning_rate`

对照轮次：

1. Round 1：`lr = 1e-5`（Part 31 基线）
2. Round 2：`lr = 8e-6`
3. Round 3：`lr = 5e-6`

---

## 一、本轮改动

### 1. 调整 `0.6B full` 训练包装脚本

修改文件：

- [run_train_qwen3_0p6b_full_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_full_gpu1.sh)

本轮把脚本改成支持通过环境变量或位置参数覆盖：

- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `TRAIN_BATCH_SIZE`
- `EVAL_BATCH_SIZE`
- `GRAD_ACCUM_STEPS`

目的很简单：

- 不再为每个 learning rate 复制一份新脚本
- 保持后续 `round2 / round3` 的变量控制干净明确

### 2. 实验边界保持不变

本轮没有改：

- 训练数据
- 验证数据
- tokenizer / chat template
- eval 脚本
- 原始 benchmark
- strict v3 benchmark

因此这轮结果可以直接解释成：

> 仅仅下调 learning rate，会不会改善 `0.6B full` 的 partial coverage 纪律。

---

## 二、Round 2：`lr = 8e-6`

### 1. 训练执行

训练日志：

- `ai_engine/finetune_qwen3/logs/train_qwen3_0p6b_full_round2_lr8e6_gpu1.log`

训练产物：

- [qwen3_0p6b_full_sft_round2_lr8e6](/ltx-data/BrainDance/ai_engine/finetune_qwen3/outputs/qwen3_0p6b_full_sft_round2_lr8e6)

关键训练结果：

- `eval_loss = 0.708493709564209`

对比首轮：

- Round 1 `1e-5`：`0.4969`
- Round 2 `8e-6`：`0.7085`

单看 loss，这已经不是改善，而是明显变差。

### 2. smoke eval

输出文件：

- `ai_engine/finetune_qwen3/logs/smoke_eval_qwen3_0p6b_full_round2_lr8e6_gpu1.json`

代表性现象：

- `no_hit` 退回成：`暂无相关记录。`
- `partial_hit` 退回成：`目前没有记录触控笔和冰箱的相关信息。`

这说明模型开始更保守，但保守得不够精确，已经有把 partial coverage 一起打成 no-hit 的倾向。

### 3. 原始 80 题 benchmark

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_full_round2_lr8e6_gpu1.json`

核心指标：

| 指标 | Round1 `1e-5` | Round2 `8e-6` |
|---|---:|---:|
| false_no_answer_rate | 0.0000 | 0.0000 |
| partial_hallucination_rate | 0.1250 | 0.1250 |
| evidence_utilization_rate | 1.0000 | 1.0000 |
| partial_hit_precision | 0.8824 | 0.8889 |
| partial_false_negative_rate | 0.0625 | 0.0000 |
| partial_missing_negation_rate | 0.0625 | 0.3125 |
| must_answer_focus_rate | 0.8125 | 0.6875 |
| natural_style_rate | 0.9250 | 0.9250 |

原始集的结论：

1. `partial_hallucination` 没有被压回去
2. `partial_missing_negation` 从 `0.0625` 反弹到 `0.3125`
3. `must_answer_focus` 从 `0.8125` 回落到 `0.6875`
4. 风格收益虽然还在，但任务纪律已经明显变差

### 4. strict v3 / 64 题 benchmark

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_0p6b_full_round2_lr8e6_gpu1.json`

核心指标：

| 指标 | Round1 `1e-5` | Round2 `8e-6` |
|---|---:|---:|
| false_no_answer_rate | 0.0000 | 0.0182 |
| partial_hallucination_rate | 0.0556 | 0.0556 |
| evidence_utilization_rate | 1.0000 | 0.9636 |
| partial_hit_precision | 0.9474 | 0.9333 |
| partial_false_negative_rate | 0.0000 | 0.2222 |
| partial_missing_negation_rate | 0.0556 | 0.1667 |
| must_answer_focus_rate | 1.0000 | 0.8889 |
| natural_style_rate | 0.9375 | 1.0000 |

strict 集的结论更直接：

1. `8e-6` 虽然把 `natural_style` 拉到了 `1.0000`
2. 但它同时引入了：
   - `false_no_answer`
   - `partial_false_negative`
   - `partial_missing_negation` 回升
   - `evidence_utilization` 回退
3. 这已经不是“风格更好但任务打平”，而是任务纪律开始实质退化

### 5. 本阶段总结

Round 2 可以直接下结论：

> `lr = 8e-6` 没有修复首轮 `0.6B full` 的 partial hallucination，反而明显破坏了 strict 集上的 partial coverage 纪律。

---

## 三、Round 3：`lr = 5e-6`

### 1. 训练执行

训练日志：

- `ai_engine/finetune_qwen3/logs/train_qwen3_0p6b_full_round3_lr5e6_gpu1.log`

训练产物：

- [qwen3_0p6b_full_sft_round3_lr5e6](/ltx-data/BrainDance/ai_engine/finetune_qwen3/outputs/qwen3_0p6b_full_sft_round3_lr5e6)

关键训练结果：

- `eval_loss = 1.0966296195983887`

这已经明显差于：

- Round 1 `1e-5`：`0.4969`
- Round 2 `8e-6`：`0.7085`

说明继续降 lr，不是“更稳”，而是在当前训练设置下明显训不满。

### 2. smoke eval

输出文件：

- `ai_engine/finetune_qwen3/logs/smoke_eval_qwen3_0p6b_full_round3_lr5e6_gpu1.json`

代表性现象：

- `partial_hit` 输出变成：`根据证据，我最近拍过触控笔。`

这已经开始复述规则口吻，且完全漏掉了 unsupported object 的否定。

### 3. 原始 80 题 benchmark

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_full_round3_lr5e6_gpu1.json`

核心指标：

| 指标 | Round1 `1e-5` | Round3 `5e-6` |
|---|---:|---:|
| false_no_answer_rate | 0.0000 | 0.0469 |
| partial_hallucination_rate | 0.1250 | 0.3750 |
| evidence_utilization_rate | 1.0000 | 0.9375 |
| partial_hit_precision | 0.8824 | 0.7143 |
| partial_false_negative_rate | 0.0625 | 0.0625 |
| partial_missing_negation_rate | 0.0625 | 0.5625 |
| must_answer_focus_rate | 0.8125 | 0.8750 |
| natural_style_rate | 0.9250 | 0.9750 |

原始集结论：

1. 风格仍然更自然
2. 但任务正确性已经大幅崩坏
3. `partial_hallucination / partial_missing_negation / partial_precision` 全线恶化

### 4. strict v3 / 64 题 benchmark

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_0p6b_full_round3_lr5e6_gpu1.json`

核心指标：

| 指标 | Round1 `1e-5` | Round3 `5e-6` |
|---|---:|---:|
| false_no_answer_rate | 0.0000 | 0.1273 |
| partial_hallucination_rate | 0.0556 | 0.3889 |
| evidence_utilization_rate | 1.0000 | 0.8364 |
| partial_hit_precision | 0.9474 | 0.6957 |
| partial_false_negative_rate | 0.0000 | 0.1111 |
| partial_missing_negation_rate | 0.0556 | 0.6667 |
| must_answer_focus_rate | 1.0000 | 0.8889 |
| natural_style_rate | 0.9375 | 0.9062 |

strict 集结论：

1. `5e-6` 不是“更稳”，而是明显失稳
2. 它不仅没有修复 partial coverage 问题，反而把 strict 集主干纪律一起打坏了
3. 这轮已经足够证明“继续下调 lr”不是当前正确方向

### 5. 本阶段总结

Round 3 的结论非常明确：

> `lr = 5e-6` 是失败配置，不应继续沿这个方向投入。

---

## 四、case 级观察

### 1. `partial_coverage_006`

这是最典型的 partial coverage 对照：

- `0.6B LoRA`：
  - `目前只找到椅子相关内容，未见写字台。`
- `full round1 / 1e-5`：
  - `目前只找到了写字台，没有看到椅子。`
- `full round2 / 8e-6`：
  - `最近记录里有椅子，没有写字台。`
- `full round3 / 5e-6`：
  - `最近记录里有椅子和写字台。`

这里能看出三件事：

1. `8e-6` 确实修掉了 round1 的 supported/unsupported 对调
2. 但这种局部修复没有泛化成整体指标改善
3. `5e-6` 则直接退回成 unsupported leakage

### 2. `partial_coverage_008`

这个 case 对 learning rate 更敏感：

- `0.6B LoRA`：
  - `最近拍到海边的记录暂时没有。`
- `full round1 / 1e-5`：
  - `最近拍到海边的风景画，暂无相关记录。`
- `full round2 / 8e-6`：
  - `最近拍过风景画，海边暂无相关记录。`
- `full round3 / 5e-6`：
  - `最近拍过风景画，海边的风景可能在其他地方有记录。`

说明：

- `8e-6` 在个别 case 上能修回 unsupported term 的正向带出
- 但 `5e-6` 明显把 unsupported leakage 放大

### 3. strict 的 `partial_coverage_006_rw`

strict 里的代表问题更稳定：

- `0.6B LoRA`：
  - `目前只找到椅子相关内容，未见写字台。`
- `full round1 / 1e-5`：
  - `有椅子和写字台。`
- `full round2 / 8e-6`：
  - `有椅子和写字台。`
- `full round3 / 5e-6`：
  - `有椅子和写字台。`

这说明：

- strict 集里最关键的 unsupported leakage case，并没有因为降 lr 而被修复
- 反而在其它 case 上又引入了新的 false negative / false no-answer

### 4. case 级总体判断

case 级对照指向的结论和总体指标一致：

- 低 learning rate 的确会让回答更“收”
- 但这种“收”并没有稳定转化成更好的 partial coverage 纪律
- 一旦继续往下压，模型会开始：
  - 漏答 supported object
  - 不再显式否定 unsupported object
  - 甚至把 hit case 误打成 no-hit

---

## 五、三轮 full 对照结论

### 1. 原始集结论

三轮 `0.6B full` 原始集结果可以概括为：

- `1e-5`：
  - 风格提升最明显
  - `partial_missing_negation` 有显著改善
  - 但 `partial_hallucination` 仍高于 LoRA
- `8e-6`：
  - 没有压回 `partial_hallucination`
  - 反而让 `partial_missing_negation` 回升
- `5e-6`：
  - 明显失稳

### 2. strict 集结论

strict v3 下结论更明确：

- `1e-5` 是当前三轮里唯一还能维持“任务正确性基本打平 + 风格提升”的点
- `8e-6` 已经开始破坏 strict 集纪律
- `5e-6` 则是明显不可用

### 3. 对主线的影响

这一轮最需要明确写下来的组织结论是：

> 当前 `0.6B full` 仍定位为风格探索支线，不替代 `0.6B LoRA` 主线，也不影响现有 `1.7B` 部署主线。

并且这轮对照实验还补充回答了一个很重要的问题：

> 至少在当前数据规模、epoch 和训练设置下，`0.6B full` 继续下调 learning rate 不是正确优化方向。

---

## 六、下一步建议

现在最合理的下一步，不是继续往 `8e-6 / 5e-6` 方向试，而是：

1. 把 `1e-5` 视为当前 `0.6B full` 的暂时最佳点
2. 如果要继续优化，只盯 `partial_coverage` 的 unsupported leakage
3. 下一轮优先考虑：
   - 数据层针对 partial coverage 做更干净的对照补样
   - 或训练策略层做更细的 loss / sample weighting
4. 不建议继续做“只降 learning rate”的盲试

一句话总结：

> Part 32 证明了 `0.6B full` 的问题不是“learning rate 还不够低”，而是当前优化矛盾已经不在单纯 lr 上；在现有设置里，`1e-5` 仍然是三轮 full 里最稳的点。
