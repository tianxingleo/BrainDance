# BrainDance Qwen3 端侧大模型知识蒸馏与量化部署流水线

> 目录：`ai_engine/finetune_qwen3`  
> 状态：核心工程流水线，已成功推进至移动端量化部署阶段，并与 Flutter 端侧大模型运行架构 (`llamadart`) 打通闭环联动。

---

## 1. 核心目标与工程价值

本模块承载了 BrainDance 面向“隐私安全与移动端硬件瓶颈”的端侧 AI 大脑构建任务。通过自主设计的 `Qwen3-1.7B / 0.6B` 端侧化工程管线，实现以下核心目标：

- **定向知识蒸馏与“极低幻觉”控制**：依托 GPT-5.4 级预处理数据作为 Teacher 打下极高质量标准边界，进而通过 SFT/LoRA 技术引导端侧小网络学会“无充分证据精准拒答，有相关证据针对提取”，在移动环境显著降低大模型高频出现的意图错位和随机幻觉现象。
- **内存受限环境保真压测 (imatrix)**：不满足于粗放的权重量化手段，开创性引入包含大量真实数据的矩阵特征校准重要性矩阵（Importance Matrix）和 GGUF `Q4_K_M`/`Q5_K_M` 融合计算。在成功压低 60% 体重（达 1.2G）使其挤入 OPPO 等移动环境内存底座的同时，保持微调知识不磨损退化。
- **完整的抗泄漏工程流水线闭环**：沉淀从数据合成流、格式化控制阀（Formatters 兜底）到强约束反作弊双盲测联合测试套件全覆盖及 CI 发包规范。

---

## 2. 阶段结论（Part 16-30）

- **Part 16**：可观测性补齐（route 级统计）
- **Part 17**：`object_lookup` 检索专项优化
- **Part 18**：formatter 路由与体验层打磨
- **Part 19**：最小可用本地问答入口 `local_qa_cli.py`
- **Part 20**：LoRA merge / 量化准备 / `Qwen3-0.6B` 实验链路
- **Part 21**：`Qwen3-0.6B` 首轮 LoRA 训练与 benchmark
- **Part 22**：`Qwen3-1.7B` merge / GGUF / `Q4_K_M` 量化
- **Part 27**：`Q5_K_M` 补测、退化定位与图表补齐
- **Part 28**：importance matrix 量化复测，确认 `Q5_K_M + imatrix` 修复 strict 集回退
- **Part 29**：`0.6B LoRA`、`1.7B merged`、`1.7B Q5_K_M + imatrix` 部署候选小样本验证
- **Part 30**：仓库入口文档补齐与部署口径统一

当前统一结论：

- 当前部署主线：`1.7B Q5_K_M + imatrix GGUF`
- 备用方案：`0.6B LoRA`
- 质量基线：`1.7B merged`
- Flutter Recall 本地 AI 当前推荐对接方式：`GGUF + llamadart + local RAG`

对应记录见：

- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part16.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part17.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part18.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part19.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part20.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part21.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part22.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part27.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part28.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part29.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录-Part30.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-LoRA-对标评测报告-2026-03-22.md`
- `../../docs/04-本地问答与微调/Qwen3-1.7B-LoRA-严格无泄漏对标评测报告-2026-03-22.md`

---

## 3. 目录结构

- `configs/`：配置文件
- `data/`：训练/验证/评测题集
- `docs/`：路由策略说明
- `logs/`：评测与调试输出
- `outputs/`：训练产物（工作目录）
- `releases/`：可发布 adapter 包
- `scripts/`：训练、评测、调试脚本

---

## 4. 常用入口

### 4.1 本地问答（推荐）

```bash
# 单轮
python ai_engine/finetune_qwen3/scripts/local_qa_cli.py --question "我最近拍了什么？"

# 交互
python ai_engine/finetune_qwen3/scripts/local_qa_cli.py

# 查看路由与证据
python ai_engine/finetune_qwen3/scripts/local_qa_cli.py \
  --question "最近拍到过什么地球仪相关画面？" \
  --show_trace --show_evidence
```

### 4.2 调试链路

```bash
# 真实链路调试
python ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py

# 交互调试（含反馈标注）
python ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py --show_evidence
```

### 4.3 训练与评测

```bash
# LoRA SFT
python ai_engine/finetune_qwen3/scripts/train_lora_sft.py

# Part 17 评测
python ai_engine/finetune_qwen3/scripts/evaluate_object_lookup_part17.py

# Part 18 评测
python ai_engine/finetune_qwen3/scripts/evaluate_experience_part18.py
```

### 4.4 部署与小模型实验

```bash
# 统一使用 conda 环境
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_gpu1.sh
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_train_qwen3_0p6b_full_gpu1.sh

# Qwen3-0.6B smoke eval
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_gpu1.sh
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_full_gpu1.sh

# 合并 1.7B LoRA 到独立 HF 模型目录
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_merge_qwen3_gpu1.sh

# 生成 GGUF / 量化计划
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_prepare_quantization_gpu1.sh

# Qwen3-0.6B benchmark
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_gpu1.sh
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_full_gpu1.sh
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_benchmark_strict_qwen3_0p6b_full_gpu1.sh

# Part 29 部署候选小样本验证
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_deployment_eval_part29_gpu1.sh
```

---

## 5. 回归命令

```bash
pytest -q tests/test_part17_object_lookup.py tests/test_part18_formatters.py tests/test_local_qa_cli.py
pytest -q tests/test_qwen3_workflow_scripts.py

python -m py_compile \
  ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py \
  ai_engine/finetune_qwen3/scripts/evaluate_experience_part18.py \
  ai_engine/finetune_qwen3/scripts/local_qa_cli.py \
  ai_engine/finetune_qwen3/scripts/train_lora_sft.py \
  ai_engine/finetune_qwen3/scripts/train_full_sft.py \
  ai_engine/finetune_qwen3/scripts/merge_lora_adapter.py \
  ai_engine/finetune_qwen3/scripts/prepare_quantization_artifacts.py
```

---

## 6. 发布约定

发布 LoRA adapter 前，请先阅读：

- `ai_engine/finetune_qwen3/releases/README.md`

重点约定：

- 不提交 `outputs/` 训练工作目录
- 发布包放在 `releases/`
- 使用 `export_release_adapter.sh` 导出清洁产物
