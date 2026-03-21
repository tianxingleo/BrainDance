# Qwen3-1.7B 微调实践记录 Part 17

## Part 17-A：object_lookup 检索专项优化落地

### 时间

- 2026-03-21

### 本 part 目标

- 不开启新一轮 LoRA
- 只针对 `object_lookup` 做窄范围 retrieval 优化
- 优先处理：
  - query normalization
  - 多目标拆分
  - 向量候选与 lexical 候选的双阶段组织
  - object_lookup 专项统计口径补齐

### 本次判断

Part 16-C 已经把主问题收敛到 `object_lookup` 检索链路，且证据更偏向：

- `rpc_empty`
- `retrieval_miss`
- `lexical_fallback` 依赖

因此 Part 17 不再扩训练数据，也不碰当前 LoRA adapter，而是直接在 `run_real_chain_debug.py` 内把 `object_lookup` 检索路径做实。

### 本次改动

#### 1. query normalization 增强

修改文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增能力：

- 增加 `canonicalize_lookup_term()`
- 增加 `split_target_objects()`
- 增加 `is_generic_lookup_token()`

覆盖点：

- 泛后缀清洗
  - `模型`
  - `场景`
  - `内容`
  - `记录`
  - `画面`
  - `相关`
- 前缀噪声清洗
  - `最近`
  - `最近有没有`
  - `最近拍过`
  - `最近记录里`
  - `关于`
  - `有关`
- 多目标拆分
  - `显示器和钢琴`
  - `笔记本电脑、地球仪和钢琴`

当前规范化后，像下面这类问法都会被压到更稳定的实体词：

- `洛天依模型 -> 洛天依`
- `显示器画面 -> 显示器`
- `笔记本电脑相关内容 -> 笔记本电脑`

#### 2. object_lookup 改成多 query 向量候选构建

新增：

- `select_object_lookup_queries()`

当前策略：

- 不再只拿原始 `search_text` 打一次向量 RPC
- 会优先挑选规范化后的目标词 / 核心词作为向量查询候选
- 最多保留 `3` 个 query，避免检索扩散失控

这样做的目的不是“大改检索”，而是把 `object_lookup` 的向量入口从单点 query 改成更稳定的实体 query。

#### 3. lexical supplement 保留，但触发条件收窄

当前 `object_lookup` 的 lexical 路径保留，但只在以下情况触发：

- `candidate_rows` 为空
- `rpc_empty`
- `post_filter_empty`
- 已有候选但仍未覆盖明确 target

这一步刻意避免了“只因为候选数偏少就强行补 lexical”，否则会把正常的向量命中误计成 fallback 依赖。

#### 4. object_lookup 专项报表补 `post_filter_empty`

修改文件：

- `ai_engine/finetune_qwen3/scripts/evaluate_object_lookup_part17.py`
- `ai_engine/finetune_qwen3/scripts/summarize_interactive_debug_routes.py`

新增指标：

- `object_lookup_post_filter_empty_count`

这样 Part 17 后可以把：

- `rpc_empty`
- `post_filter_empty`

明确拆开看，而不是继续混在单一 fallback 统计里。

### 本 part 小结

Part 17-A 已经把 `object_lookup` 的 retrieval 优化主线落地到代码里：

- query normalization 更强
- 多目标拆分补齐
- 向量 query 从单点变成规范化多 query
- lexical supplement 仍保留，但不再泛化触发
- 专项统计口径补了 `post_filter_empty`

这一阶段的重点不是追求“固定集立刻大涨”，而是先把 Part 17 需要的检索动作和指标框架做正确。

## Part 17-B：检索逻辑回归测试与环境解耦

### 时间

- 2026-03-21

### 本 part 目标

- 给 Part 17 的检索逻辑补可重复的单测
- 让 retrieval-only 脚本不再强依赖生成环境

### 本次改动

#### 1. 新增检索专项单测

新增文件：

- `tests/test_part17_object_lookup.py`

覆盖点：

- 泛后缀清洗
- 多目标拆分
- query 选择优先级
- 候选合并去重与重排

本次单测聚焦的是“Part 17 新加的 retrieval 逻辑”，不去耦合真实 Supabase / DashScope 网络依赖。

#### 2. 生成依赖改为懒加载

修改文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

处理内容：

- 顶层不再强制 import `peft`
- 顶层不再强制要求 `torch` 必须可用
- retrieval-only 路径可以在普通 Python 环境运行
- 只有真正加载生成模型时才检查 `torch / peft / transformers`

这一步的直接收益是：

- `evaluate_object_lookup_part17.py` 可以单独跑 retrieval 评测
- route summary / fixed eval 不会再被 CUDA / bitsandbytes 环境问题卡死

### 验证情况

执行：

```bash
pytest -q tests/test_part17_object_lookup.py
python -m py_compile ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py
```

结果：

- `4 passed`
- `py_compile` 通过

### 本 part 小结

Part 17-B 把这轮改动从“能跑”推进到“可回归验证”：

- 检索逻辑有了单测
- retrieval-only 工具链不再被生成依赖绑死

## Part 17-C：固定评测复跑与结论收敛

### 时间

- 2026-03-21

### 本 part 目标

- 用固定 `12` 条 `object_lookup` 题集复跑 before/after
- 检查 Part 17 是否引入回归
- 判断是否继续在当前窄集上深调

### 执行命令

在 `qwen3_ft` 环境中执行：

```bash
source /home/jiangbeihu/miniconda3/etc/profile.d/conda.sh
conda activate qwen3_ft
python ai_engine/finetune_qwen3/scripts/evaluate_object_lookup_part17.py \
  --baseline_summary ai_engine/finetune_qwen3/logs/object_lookup_before_summary.json
```

### before / after

固定评测结果：

- `object_lookup_count = 12 -> 12`
- `object_lookup_hit_rate = 1.0 -> 1.0`
- `object_lookup_bad_rate = 0.0 -> 0.0`
- `object_lookup_lexical_fallback_rate = 0.4167 -> 0.4167`
- `object_lookup_retrieval_miss_bad_count = 0 -> 0`
- `object_lookup_retrieval_low_relevance_bad_count = 0 -> 0`
- `object_lookup_rpc_empty_count = 4 -> 4`
- `object_lookup_post_filter_empty_count = 0 -> 1`

after 的 route 分布：

- `vector_plus_filter = 7`
- `lexical_fallback = 5`

### 本次判断

这轮 Part 17 在固定窄集上的结论很明确：

#### 1. 没有引入回归

- hit rate 没掉
- bad rate 没升
- retrieval miss 没变差

说明 Part 17 当前代码形态可以保留。

#### 2. 固定窄集上没有拿到显著的数值提升

这并不意外，因为当前 `12` 条 fixed eval 在 Part 16 末已经很强：

- `hit_rate = 1.0`
- `bad_rate = 0.0`

在这种基线上，Part 17 更容易体现为：

- 代码鲁棒性增强
- normalization 更完整
- query 表达覆盖更稳
- 统计口径更细

而不是直接体现成 fixed set 指标暴涨。

#### 3. 当前最值得继续盯的是更广样本上的 `rpc_empty`

固定集里仍然有：

- `object_lookup_rpc_empty_count = 4`

说明向量空召回仍然存在，但这组 `12` 条题已经不足以继续放大差异。

### 结论

Part 17 当前阶段可以先收口，结论如下：

- `object_lookup` 检索专项优化已经完成第一轮落地
- 当前改动没有引入固定集回归
- fixed eval 上暂无显著增益，但代码鲁棒性和可观测性明显增强
- 下一步不该继续在这 `12` 条 fixed eval 上硬调
- 下一步应回到更广的 interactive / batch probe 样本上验证：
  - `rpc_empty`
  - `post_filter_empty`
  - `retrieval_miss`
  - `lexical_fallback` 的真实占比变化

### 本 part 小结

Part 17-C 的核心不是“数字大涨”，而是确认这轮 retrieval 优化：

- 没有把已有行为打坏
- 把 Part 17 需要的 normalization / candidate merge / 专项指标体系补齐了
- 说明后续应把验证重点切回更大样本，而不是继续围着固定窄集做局部调参

## 本文一段总结

Part 17 这一轮没有继续训练，也没有重写全链路，而是把 `object_lookup` 的 retrieval 主线做成了更稳的工程版本：

- query normalization 更强
- 多目标拆分落地
- 向量 query 选择更贴近稳定实体
- lexical supplement 触发条件更收敛
- object_lookup 专项统计补了 `post_filter_empty`
- retrieval-only 评测链路与生成环境解耦

最终结果是：固定 `12` 条题集上无回归、无显著新增收益，说明 Part 17 第一轮优化已经到达一个合理停点。接下来应该把验证重心切回更广的真实交互样本，而不是继续在当前 fixed eval 上反复微调。
