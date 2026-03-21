# Qwen3-1.7B 微调实践记录 Part 19

## Part 19：最小可用本地问答接口

### 本 part 目标

在不接正式产品链路的前提下，把当前 debug 能力收成一个稳定入口：

- 输入：用户问题
- 中间：`query classify -> retrieval -> answer route`
- 输出：最终短答
- 可选：证据预览与链路摘要

目标不是再做一个调试台，而是给当前能力一个最小可用、可直接命令行使用的本地问答接口。

---

## 一、现状判断

仓库里原本已经有 [interactive_debug_chat.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py)：

- 可以命令行模拟用户问答
- 走的也是当前真实 retrieval 链路
- 已支持 `--show_evidence`

但它仍然偏 debug：

- 默认强调 session log
- 默认保留反馈采集和 issue bucket 语义
- 更适合“人工探针”和“问题归因”，不是最小稳定入口

因此本 part 的方向不是推倒重写，而是在其旁边单独收一个更轻的入口。

---

## 二、实现内容

### 1. 新增最小 CLI 入口

新增脚本：

- [local_qa_cli.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/local_qa_cli.py)

它直接复用：

- `retrieve_real_chain_case()`
- `generate_answer()`
- 当前 Part 18 已稳定的 formatter 路由

默认行为：

- 只显示最终短答
- 不弹反馈问题
- 不强制写 session summary
- 不暴露调试细节

### 2. 支持两种使用形态

#### 单轮模式

```bash
python ai_engine/finetune_qwen3/scripts/local_qa_cli.py --question "最近拍到过什么地球仪相关画面？"
```

适合：

- 命令行快速验证
- 后续 shell / 服务封装

#### 交互模式

```bash
python ai_engine/finetune_qwen3/scripts/local_qa_cli.py
```

支持最小命令：

- `/help`
- `/quit`
- `/last`

### 3. 可选证据与链路开关

保留两个显式开关：

- `--show_trace`
- `--show_evidence`

其中：

- `--show_trace` 输出 `query_class / retrieval_route / answer_route / hit_count`
- `--show_evidence` 额外输出最多 `3` 条 evidence 预览

这样默认用户态仍然是短答，但排查问题时不需要切回重型 debug 入口。

### 4. 新增 GPU 启动脚本

新增：

- [run_local_qa_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_local_qa_gpu0.sh)
- [run_local_qa_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_local_qa_gpu1.sh)

作用是直接复用现有 conda / CUDA / HF mirror 约定，降低启动成本。

使用方式：

```bash
bash ai_engine/finetune_qwen3/scripts/run_local_qa_gpu0.sh
```

或：

```bash
bash ai_engine/finetune_qwen3/scripts/run_local_qa_gpu0.sh --question "我最近拍了什么？"
```

---

## 三、验证

### 1. 新增测试

新增：

- [test_local_qa_cli.py](/ltx-data/BrainDance/tests/test_local_qa_cli.py)

覆盖：

- 链路摘要格式化输出
- `special_answer` 优先返回

### 2. 本轮测试命令

```bash
pytest -q tests/test_part17_object_lookup.py tests/test_part18_formatters.py tests/test_local_qa_cli.py
python -m py_compile \
  ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py \
  ai_engine/finetune_qwen3/scripts/evaluate_experience_part18.py \
  ai_engine/finetune_qwen3/scripts/local_qa_cli.py
```

本 part 完成时应满足：

- 最小 CLI 能独立运行
- 继续复用 Part 18 formatter 能力
- 不破坏已有 Part 17 / Part 18 回归

---

## 四、结论

Part 19 的价值不在于新能力本身，而在于把当前能力从“需要进 debug 台才能用”推进到“已经有稳定入口可直接问”。

当前建议使用顺序变为：

1. 日常本地问答：`local_qa_cli.py`
2. 需要看证据和路由：`local_qa_cli.py --show_trace --show_evidence`
3. 需要带反馈标注和 session 分析：`interactive_debug_chat.py`

这使得下一步如果要接正式产品链路，不需要再从零拼接问答主流程，只需要在这个最小入口基础上继续收口即可。
