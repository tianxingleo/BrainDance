# BrainDance Qwen3 本地问答微调与部署实验

> 目录：`ai_engine/finetune_qwen3`  
> 状态：独立实验目录，已推进到部署候选验证阶段，并与 Flutter Recall 本地 AI 入口形成可复用接入链路。

---

## 1. 目标

本目录用于 BrainDance 本地问答方向的 `Qwen3-1.7B / 0.6B` 实践，核心目标是：

- 让模型学会“基于检索证据回答”，而不是记忆用户事实
- 提升问答稳定性（有命中必答、无命中拒答）
- 沉淀可回归的脚本、数据集、评测日志和部署候选结论

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

# Qwen3-0.6B smoke eval
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_smoke_eval_qwen3_0p6b_gpu1.sh

# 合并 1.7B LoRA 到独立 HF 模型目录
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_merge_qwen3_gpu1.sh

# 生成 GGUF / 量化计划
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_prepare_quantization_gpu1.sh

# Qwen3-0.6B benchmark
conda run -n qwen3_ft bash ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_gpu1.sh

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
