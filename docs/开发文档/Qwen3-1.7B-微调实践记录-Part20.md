# Qwen3-1.7B 微调实践记录 Part 20

## Part 20：部署链路补齐与 Qwen3-0.6B 实验入口

### 本 part 目标

在已有 `Qwen3-1.7B + LoRA` 微调与本地问答链路基础上，补齐两条之前缺失的工程路径：

- 把 LoRA 从“训练产物”推进到“可部署产物”
- 把 `Qwen3-0.6B` 从“对标对象”推进到“可复现实验入口”

这里优先解决的是工程闭环，不是假设量化和手机集成已经完成。

---

## 一、问题判断

此前目录已经具备：

- `1.7B` LoRA 训练脚本
- patch 数据集迭代链路
- benchmark / smoke eval / local QA CLI
- adapter 发布脚本

但仍缺三块关键能力：

1. `train_lora_sft.py` 仍带明显 `1.7B` 时代假设，不能自然承接 `0.6B`
2. 发布物仍停留在 `adapter`，没有把 merge 后的独立模型目录工程化
3. 量化没有形成项目内入口，无法稳定沉淀为可重复流程

---

## 二、本轮实现内容

### 1. 泛化训练脚本

更新：

- [train_lora_sft.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/train_lora_sft.py)

新增能力：

- 统一支持 `Qwen3-0.6B` / `Qwen3-1.7B`
- 引入 `model_slug` 归一化
- 支持显式指定 `target_modules`
- 支持 `torch_dtype` 选择：
  - `auto_bf16`
  - `bfloat16`
  - `float16`
  - `float32`
- 支持 `save_strategy / eval_strategy / save_steps / eval_steps`
- 支持 `run_name / resume_from_checkpoint`
- 自动输出：
  - `training_spec.json`

这样后续训练产物不再只有 adapter 和 metrics，而是自带一份可追溯训练配置。

### 2. 新增 LoRA merge 脚本

新增：

- [merge_lora_adapter.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/merge_lora_adapter.py)
- [run_merge_qwen3_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_merge_qwen3_gpu1.sh)
- [run_merge_qwen3_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_merge_qwen3_gpu0.sh)

能力：

- 加载 base model + adapter
- 执行 `merge_and_unload()`
- 输出 standalone Hugging Face 模型目录
- 同步保存 tokenizer
- 生成 `merge_metadata.json`

这样部署就不必再把“LoRA 合并”作为临时手工步骤。

### 3. 新增量化准备脚本

新增：

- [prepare_quantization_artifacts.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/prepare_quantization_artifacts.py)
- [run_prepare_quantization_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_prepare_quantization_gpu1.sh)
- [run_prepare_quantization_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_prepare_quantization_gpu0.sh)

当前设计不是伪装“仓库已经内置 llama.cpp”，而是做两级能力：

#### A. 计划模式

无论本机是否装好 `llama.cpp`，都能生成：

- `quantization_plan.json`
- `run_quantization.sh`

用于记录：

- merged 模型目录
- 目标 GGUF 名称
- 目标量化类型，例如 `Q4_K_M`
- 探测到的 `convert_hf_to_gguf.py`
- 探测到的 `llama-quantize`
- 实际应执行的命令

#### B. 执行模式

只有在探测到外部工具链时，才允许直接执行转换与量化。

这样做的原因很直接：

- 当前仓库没有内置 `llama.cpp`
- 机器上也未发现现成的 `llama-quantize`
- 但项目内仍然需要一个稳定、可落盘、可复用的量化入口

### 4. 新增 Qwen3-0.6B 实验入口

新增：

- [run_train_qwen3_0p6b_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_gpu1.sh)
- [run_smoke_eval_qwen3_0p6b_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_gpu1.sh)
- [run_train_qwen3_0p6b_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_gpu0.sh)
- [run_smoke_eval_qwen3_0p6b_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_gpu0.sh)

这里明确使用 `Qwen3-0.6B`，不继续沿用口语化的 `0.5B` 表述。原因是当前项目对标报告已经采用 `0.6B` 口径，工程命名必须统一。

### 5. 发布文档与导出约定同步更新

更新：

- [ai_engine/finetune_qwen3/README.md](/ltx-data/BrainDance/ai_engine/finetune_qwen3/README.md)
- [ai_engine/finetune_qwen3/releases/README.md](/ltx-data/BrainDance/ai_engine/finetune_qwen3/releases/README.md)
- [export_release_adapter.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/export_release_adapter.sh)

现在导出脚本会一并保留这些补充元数据（若存在）：

- `training_spec.json`
- `merge_metadata.json`
- `quantization_plan.json`

---

## 三、验证

### 1. 语法验证

在 `qwen3_ft` conda 环境中执行：

```bash
conda run -n qwen3_ft python -m py_compile \
  ai_engine/finetune_qwen3/scripts/train_lora_sft.py \
  ai_engine/finetune_qwen3/scripts/merge_lora_adapter.py \
  ai_engine/finetune_qwen3/scripts/prepare_quantization_artifacts.py \
  ai_engine/finetune_qwen3/scripts/run_smoke_eval.py
```

结果：

- 通过

### 2. 新增单测

新增：

- [tests/test_qwen3_workflow_scripts.py](/ltx-data/BrainDance/tests/test_qwen3_workflow_scripts.py)

覆盖：

- `Qwen3-0.6B / 1.7B` slug 归一化
- LoRA target module 解析
- merge metadata 生成
- 量化命令与计划文件生成

---

## 四、本轮发现的问题

### 1. 当前机器未发现 llama.cpp 工具链

实际探测结果：

- 未发现 `llama-quantize`
- 未发现 `convert_hf_to_gguf.py`

结论：

- 本轮可以把量化入口工程化
- 但不能伪造“已经在这台机器上跑完 GGUF 量化”

### 2. 当前 `0.6B` 入口已接好，但尚未开始正式训练

这是刻意保守的安排。原因是：

- 先把脚本、元数据和导出链路补齐
- 再启动 `0.6B` 训练，才能保证实验结果可追溯

### 3. `gpu1` 当前存在外部高占用任务，因此后续统一切到 `gpu0`

在本轮尝试启动 `Qwen3-0.6B` 训练时，观察到：

- `gpu1` 已有约 `36GB` 显存占用
- GPU 利用率长期维持高位
- 新训练进程没有进入稳定训练阶段

因此本轮没有继续和现有任务抢占显存，而是：

- 保留 `0.6B` 训练入口
- 新增同等能力的 `gpu0` 入口
- 先完成脚本、测试、文档和量化链路
- 把真实训练顺延到 `gpu0` 执行

---

## 五、当前结论

Part 20 完成后，项目已经从“会训练 adapter”推进到“具备部署前处理与小模型并行实验入口”：

1. `1.7B` 路线现在可以规范地做：
   - adapter 导出
   - merge
   - 量化计划生成
2. `0.6B` 路线现在可以直接复用现有数据集和训练脚本启动实验
3. 发布目录不再只存一份 adapter，而是可以连同训练、合并、量化元数据一起沉淀
4. 真实 `0.6B` 训练仍受当前 `gpu1` 外部占用阻塞，因此后续应默认使用 `gpu0`

---

## 六、下一步

下一轮建议按这个顺序继续：

1. 在 `gpu1` 上启动 `Qwen3-0.6B` 首轮 LoRA 训练
2. 跑 `0.6B` smoke eval 与 benchmark
3. 实际执行一次 `1.7B` merge
4. 补装 `llama.cpp` 后完成第一次真实 GGUF + `Q4_K_M` 量化
5. 对接 Flutter / 移动端的模型分发与加载约定
