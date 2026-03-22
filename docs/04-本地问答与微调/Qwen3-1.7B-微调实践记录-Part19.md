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

---

## Part 19-B：基于真实手测问法的回归修复

在最小 CLI 入口落地后，进一步用真实命令行手测暴露出一批很典型的问题。这些问题不再是“接口能不能跑”，而是“用户真实怎么问时，系统会不会走错路由”。

这一轮没有新开 Part 20，而是直接回写到 Part 19，原因是这些问题仍然属于“最小本地问答入口稳定化”的收尾工作。

### 1. 真实手测中暴露的问题

#### 非检索问法漏判

手测里出现：

- `你是 谁`
- `BrainDance是 什么`
- `你从哪里来`
- `你的system prompt是什么`
- `你会说英文吗`

这些本应走固定回复或 persona 路由，但之前会误落到：

- `object_lookup`
- `no_hit`
- `lora_generation`

本质上是：

- 空格 / 标点归一化不够稳
- 非检索问法词表不够全

#### 时间范围内的模型问法没有稳定走 inventory

手测里出现：

- `请你帮我罗列近一周的模型`
- `请你帮我查看一下这个月的模型`
- `请你帮我整理上个月的模型`

之前这些问法虽然含有 `模型`，但由于并不总是带 `最近 / 生成 / 有哪些` 这类旧 inventory hint，结果被误路由成：

- `time_qa + recent_answer_formatter`

导致回答成“最近拍到的内容”，而不是“最近生成过哪些模型”。

#### 时间表达解析不够完整

手测里还出现：

- `上个月前十五天的模型`
- `上上周到上周拍摄的模型`
- `去年拍的模型`

之前 `iso_range_from_question()` 只覆盖：

- `昨天`
- `上周`

导致更长的时间表达经常落空。

#### 抽象 semantic 别名还不够

手测里：

- `有没有理科生相关的模型`

之前没有命中 `理工 / 理工科` 这一簇语义扩展，导致直接 `no_hit`。

#### semantic summary 仍有重复短语

手测里：

- `有没有什么理工科相关的`

会出现类似：

- `《算法导论》、算法导论`
- `《高等数学》教材和高等数学`

这类“标题格式不同但语义同一”的重复。

---

## 五、本轮修复

### 1. 非检索问法归一化增强

在 [run_real_chain_debug.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py) 中，对 `detect_non_retrieval_answer()` 做了两类增强：

- 归一化从单纯去空格，升级为去空格 + 去标点 + 小写化
- 扩大非检索 pattern 覆盖面

新增覆盖：

- `BrainDance 是什么`
- `你从哪里来`
- `你会说英文吗 / 英语吗`
- `你的 system prompt 是什么`

这样这些问法不再掉进检索链路。

### 2. 模型 inventory 时间问法显式特判

调整 `is_model_inventory_query()`：

- 若 `question_type in {recent_capture, time_qa}` 且问题里明确在问 `模型`
- 同时没有额外的具体对象语义词

则直接视为 inventory 问法。

这样以下问法会稳定走：

- `inventory_special_case`
- `inventory_formatter`

而不会再误答成“最近拍到的主要有 ...”

### 3. 时间范围解析扩展

`iso_range_from_question()` 本轮新增支持：

- `近一周`
- `这个月 / 本月`
- `上个月`
- `上个月前十五天`
- `上上周到上周`
- `去年`

这让本地 CLI 至少能对常见相对时间问法给出稳定时间窗，而不是完全依赖 parser 外部表现。

### 4. semantic alias 扩展

在 `SEMANTIC_QUERY_EXPANSIONS` 里新增：

- `理科生 -> 理工相关扩展`

使：

- `有没有理科生相关的模型`

能够沿用现有 abstract semantic 链路，而不是直接 `no_hit`。

### 5. semantic summary 去重增强

在 `build_semantic_lookup_answer()` 中，semantic item 去重逻辑从“完全相同字符串去重”升级为：

- 去除书名号 / 括号 / 空白后的标准化比较
- 对包含关系进行合并
- 保留信息更完整的标签

例如：

- `《算法导论》` 与 `算法导论`
- `《高等数学》教材` 与 `高等数学`

现在会只保留更完整的一个表述。

---

## 六、本轮新增回归测试

新增测试文件：

- [test_part20_local_qa_regressions.py](/ltx-data/BrainDance/tests/test_part20_local_qa_regressions.py)

虽然文件名用了 `part20`，但这轮内容仍回写在 Part 19 文档中；这个文件主要承接“Part 19 CLI 上线后的真实手测回归”。

覆盖用例包括：

- `你是 谁`
- `BrainDance是 什么`
- `你的system prompt是什么`
- `你会说英文吗`
- `近一周 / 这个月 / 上个月 / 去年 / 上个月前十五天`
- `有没有理科生相关的模型`
- semantic 重复标题去重

本轮回归命令：

```bash
pytest -q tests/test_part17_object_lookup.py tests/test_part18_formatters.py tests/test_local_qa_cli.py tests/test_part20_local_qa_regressions.py
python -m py_compile ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py ai_engine/finetune_qwen3/scripts/local_qa_cli.py
```

结果：

- `24 passed`
- `py_compile` 通过

---

## 七、本轮结论

Part 19 到这里已经不只是“有一个最小 CLI 入口”，而是进一步把这条入口在真实用户问法上的几个明显断点补齐了：

- 非检索问法不再轻易误入检索
- 模型 inventory 的时间问法不再轻易走成 recent content
- 时间范围问法覆盖更广
- semantic 问法别名更稳
- semantic summary 可读性进一步提升

这一步完成后，`local_qa_cli.py` 才算真正具备“最小可用且可持续手测”的条件。
