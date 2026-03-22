# Qwen3-1.7B 微调实践记录 Part 21

## Part 21：Qwen3-0.6B 首轮 LoRA 训练与对标验证

### 本 part 目标

把 Part 20 刚补好的 `Qwen3-0.6B` 实验入口真正跑起来，完成：

- 首轮 LoRA 训练
- smoke eval
- 固定 benchmark
- release adapter 导出

这里的目标不是证明 `0.6B` 已经优于 `1.7B`，而是确认“小模型路线”在当前工程里已经从“设想”变成“真实可复现”。

---

## 一、执行前判断

本轮先检查 GPU 状态：

- `gpu0` 空闲
- `gpu1` 仍被外部任务长期占用约 `36GB` 显存

因此后续实验统一切到 `gpu0`。

同时确认本地缓存：

- `Qwen/Qwen3-0.6B` 已在 Hugging Face 本地缓存
- `Qwen/Qwen3-1.7B` 已在 Hugging Face 本地缓存

这样训练时不需要额外下载底模。

---

## 二、本轮执行内容

### 1. 在 gpu0 上启动 0.6B 首轮训练

使用脚本：

- [run_train_qwen3_0p6b_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_gpu0.sh)

训练日志：

- `ai_engine/finetune_qwen3/logs/train_qwen3_0p6b_round1_gpu0.log`

训练产物：

- [qwen3_0p6b_lora_sft_round1](/ltx-data/BrainDance/ai_engine/finetune_qwen3/outputs/qwen3_0p6b_lora_sft_round1)

### 2. 训练过程观察

实际观察：

- `gpu0` 显存峰值约 `20GB`
- 训练过程中成功生成：
  - `training_spec.json`
  - tensorboard 事件文件
  - `checkpoint-57`

从 `trainer_state.json` 可见：

- 第一个 epoch 结束时：
  - `global_step = 57`
  - `eval_loss ≈ 0.6744`

最终训练结束后：

- `final_metrics.json` 中：
  - `eval_loss = 0.5062161684036255`
  - `epoch = 2.0`

这说明：

- `0.6B` 路线在当前数据集和脚本下可以稳定训练
- loss 在两轮训练中继续下降，没有出现明显发散

### 3. smoke eval

使用脚本：

- [run_smoke_eval_qwen3_0p6b_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_gpu0.sh)

结果文件：

- `ai_engine/finetune_qwen3/logs/smoke_eval_qwen3_0p6b_round1_gpu0.json`

关键输出：

- `recent_hit`：
  - `最近拍到的是3月19日的触控笔桌面采集 01。`
- `no_hit`：
  - `目前没有找到与自行车相关的记录。`
- `partial_hit`：
  - `目前只找到触控笔相关记录，没有冰箱相关记录。`

结论：

- hit / no-hit / partial coverage 三类基本行为都已经正常

### 4. 固定 benchmark

新增脚本：

- [run_benchmark_qwen3_0p6b_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_gpu0.sh)

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_round1_gpu0.json`

核心指标：

- `false_no_answer_rate = 0.0`
- `partial_hallucination_rate = 0.0625`
- `natural_output_rate = 1.0`
- `natural_style_rate = 0.8`
- `evidence_utilization_rate = 0.9844`
- `partial_hit_precision = 0.9375`
- `partial_false_negative_rate = 0.0625`
- `partial_missing_negation_rate = 0.1875`
- `must_answer_focus_rate = 0.6875`

按 group 看：

- `no_hit`：
  - `natural_style_rate = 1.0`
- `partial_coverage`：
  - `partial_false_negative_rate = 0.0625`
  - `partial_missing_negation_rate = 0.1875`
- `stability`：
  - `natural_style_rate = 0.625`

### 5. 导出 0.6B release adapter

执行：

```bash
bash ai_engine/finetune_qwen3/scripts/export_release_adapter.sh \
  ai_engine/finetune_qwen3/outputs/qwen3_0p6b_lora_sft_round1 \
  qwen3_0p6b_braindance_round1 \
  Qwen/Qwen3-0.6B
```

导出目录：

- [qwen3_0p6b_braindance_round1](/ltx-data/BrainDance/ai_engine/finetune_qwen3/releases/qwen3_0p6b_braindance_round1)

---

## 三、结论

Part 21 的结论很直接：

1. `Qwen3-0.6B` 首轮 LoRA 训练已经真实跑通
2. `0.6B` 在当前 BrainDance 证据驱动 benchmark 上已经具备可用水平
3. 小模型路线不是纸面方案，而是已经有：
   - 训练产物
   - smoke eval 结果
   - benchmark 结果
   - release adapter 包

但同样需要明确：

- 当前 `must_answer_focus_rate = 0.6875`
- `partial_missing_negation_rate = 0.1875`

这说明 `0.6B` 已可用，但在“聚焦回答”和“部分命中时的否定表达”上，距离当前 `1.7B + LoRA` 还有差距。

---

## 四、下一步

下一轮应把重点转向：

1. 继续对比 `0.6B` 与 `1.7B` 的同题误差
2. 为 `0.6B` 定位“可接受的端侧版本”
3. 把 `1.7B` 的 merge 与量化真正落成部署产物
