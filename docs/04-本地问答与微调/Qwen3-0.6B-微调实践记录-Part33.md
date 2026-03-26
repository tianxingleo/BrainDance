# Qwen3-0.6B 微调实践记录 Part 33

## Part 33：Qwen3-0.6B full SFT 的 `partial_coverage` 定向补样实验

### 时间

- 2026-03-23

### 执行结论

本轮实验确认：

> 在当前 `0.6B full SFT` 设置下，固定 `lr = 1e-5` 后追加一小批 `partial_coverage` 定向补样，能够修复个别 raw case，但没有把整体指标拉回优于 `full round1` 或 `0.6B LoRA` 的位置；`0.6B full` 仍应定位为风格探索支线。

---

## 一、本 part 目标

Part 32 已经把“继续降 learning rate”这条路试死了，本轮不再做纯 lr 搜索，只回答一个更窄的问题：

> 如果保持 `0.6B full` 的当前最佳点 `lr = 1e-5`、`epoch = 1` 不变，只围绕 `partial_coverage` 做一小批高针对性的补样，能不能把 unsupported leakage 压下去，同时保住风格收益？

本轮边界如下：

- 模型保持 `Qwen/Qwen3-0.6B`
- 训练方式保持 full SFT
- `lr = 1e-5`
- `epoch = 1`
- benchmark / strict benchmark 不改
- 不混大 patch 池
- 只补 `partial_coverage` 小样本

---

## 二、本轮改动

### 1. 新增 `partial_coverage` 补样构建脚本

新增文件：

- [build_full_partial_patch_dataset.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/build_full_partial_patch_dataset.py)

用途：

- 从 raw benchmark 与 strict benchmark 中抽取一小批 `partial_coverage` case
- 自动构造更明确的 supported / unsupported 对照问答
- 生成独立 patch 数据集
- 再与现有主训练集拼接成 round4 的 merged train / val 文件

脚本固定了两类模板：

- 问题模板：明确询问“`supported` 有，`unsupported` 也有吗”
- 答案模板：明确回答“`supported` 有记录，`unsupported` 暂无相关记录”

这样做的目的很直接：

- 不改训练框架
- 不引入新 loss 变量
- 只从数据层加强 `partial_coverage` 的纪律

### 2. 补样文件与规模

生成文件：

- [qwen3_0p6b_full_partial_patch_v1_train.jsonl](/ltx-data/BrainDance/ai_engine/finetune_qwen3/data/qwen3_0p6b_full_partial_patch_v1_train.jsonl)
- [qwen3_0p6b_full_partial_patch_v1_val.jsonl](/ltx-data/BrainDance/ai_engine/finetune_qwen3/data/qwen3_0p6b_full_partial_patch_v1_val.jsonl)

对应 merged 文件：

- `ai_engine/finetune_qwen3/outputs/datasets/qwen3_0p6b_full_partial_patch_v1_train.jsonl`
- `ai_engine/finetune_qwen3/outputs/datasets/qwen3_0p6b_full_partial_patch_v1_val.jsonl`

样本规模：

| 数据集 | 条数 |
|---|---:|
| patch train | 36 |
| patch val | 8 |
| merged train | 936 |
| merged val | 108 |

补样覆盖重点：

- raw `partial_coverage_006 / 008`
- strict `partial_coverage_006_rw`
- 同型 supported / unsupported 对照样本

---

## 三、训练与 smoke eval

### 1. 训练执行

使用环境：

- `conda run -n qwen3_ft`
- `CUDA_VISIBLE_DEVICES=1`

训练产物：

- [qwen3_0p6b_full_sft_round4_partial_patch_v1](/ltx-data/BrainDance/ai_engine/finetune_qwen3/outputs/qwen3_0p6b_full_sft_round4_partial_patch_v1)

训练日志：

- `ai_engine/finetune_qwen3/logs/train_qwen3_0p6b_full_round4_partial_patch_v1_gpu1.log`

关键结果：

- `trainable_params = 596,049,920`
- `train_examples = 936`
- `val_examples = 108`
- `eval_loss = 0.5221726894378662`
- `train_loss = 0.8378421330856065`
- `train_runtime ≈ 80.83s`

与前几轮 full 对照：

| 轮次 | 设置 | eval_loss |
|---|---|---:|
| Round 1 | base train, `lr=1e-5` | 0.4969 |
| Round 2 | base train, `lr=8e-6` | 0.7085 |
| Round 3 | base train, `lr=5e-6` | 1.0966 |
| Round 4 | partial patch v1, `lr=1e-5` | 0.5222 |

这说明：

- round4 没有像 `8e-6 / 5e-6` 那样明显失稳
- 但也没有在训练层面出现“明显优于 round1”的信号

### 2. smoke eval

输出文件：

- `ai_engine/finetune_qwen3/logs/smoke_eval_qwen3_0p6b_full_round4_partial_patch_v1_gpu1.json`

代表输出：

- `recent_hit`：`最近拍到的是触控笔桌面采集 01，3月19日。`
- `no_hit`：`目前没有找到与自行车相关的记录。`
- `partial_hit`：`目前只找到触控笔相关记录，没有冰箱相关记录。`

smoke 结果说明：

- 基本格式没有炸
- partial hit 的口吻符合本轮补样意图
- 但 smoke 只能证明“方向像是对的”，不能代替 benchmark

---

## 四、原始 80 题 benchmark 对比

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_full_round4_partial_patch_v1_gpu1.json`

与 `0.6B LoRA`、`full round1` 对比如下：

| 指标 | 0.6B LoRA | 0.6B full round1 | 0.6B full round4 patch |
|---|---:|---:|---:|
| false_no_answer_rate | **0.0000** | **0.0000** | **0.0000** |
| partial_hallucination_rate | **0.0625** | 0.1250 | 0.1250 |
| evidence_utilization_rate | 0.9844 | **1.0000** | 0.9844 |
| partial_hit_precision | **0.9375** | 0.8824 | 0.8824 |
| partial_false_negative_rate | 0.0625 | 0.0625 | 0.0625 |
| partial_missing_negation_rate | 0.1875 | **0.0625** | 0.1875 |
| must_answer_focus_rate | 0.6875 | **0.8125** | 0.6875 |
| natural_style_rate | 0.8000 | **0.9250** | **0.9250** |

### 1. raw 集结论

round4 的 raw 集表现可以概括成三句话：

1. `natural_style_rate` 保住了 round1 的高位：`0.9250`
2. 但 `partial_hallucination_rate` 没降，仍是 `0.1250`
3. 更关键的是，`partial_missing_negation / must_answer_focus / evidence_utilization` 都回退到了接近 `0.6B LoRA` 甚至不如 round1 的位置

也就是说：

> 这轮补样没有把 round1 的“风格更好 + 纪律不明显更差”继续往前推进，反而在 raw 集上丢掉了 round1 最有价值的那部分纪律收益。

### 2. raw 集分组观察

按组看，最明显的变化在三处：

- `partial_coverage.evidence_utilization_rate`：`1.0000 -> 0.9375`
- `partial_coverage.partial_missing_negation_rate`：`0.0625 -> 0.1875`
- `must_answer.must_answer_focus_rate`：`0.8125 -> 0.6875`

反过来看，唯一明确的正向变化只有：

- `stability.natural_style_rate`：`0.8750 -> 0.9375`

这再次说明：

- 补样带来的主要变化仍偏向表达风格
- 但它没有稳定转化成更好的 partial coverage 纪律

---

## 五、strict v3 / 64 题 benchmark 对比

输出文件：

- `ai_engine/finetune_qwen3/logs/benchmark_strict_v3_qwen3_0p6b_full_round4_partial_patch_v1_gpu1.json`

与 `0.6B LoRA`、`full round1` 对比如下：

| 指标 | 0.6B LoRA | 0.6B full round1 | 0.6B full round4 patch |
|---|---:|---:|---:|
| false_no_answer_rate | **0.0000** | **0.0000** | 0.0182 |
| partial_hallucination_rate | **0.0556** | **0.0556** | 0.1111 |
| evidence_utilization_rate | **1.0000** | **1.0000** | 0.9818 |
| partial_hit_precision | **0.9474** | **0.9474** | 0.8947 |
| partial_false_negative_rate | **0.0000** | **0.0000** | 0.0556 |
| partial_missing_negation_rate | **0.0556** | **0.0556** | 0.1111 |
| must_answer_focus_rate | **1.0000** | **1.0000** | **1.0000** |
| natural_style_rate | 0.8281 | **0.9375** | **0.9375** |

### 1. strict 集结论

strict 集的结论比 raw 集更明确：

1. `natural_style_rate` 保住了 `0.9375`
2. `must_answer_focus_rate` 仍是 `1.0000`
3. 但关键纪律指标全部回退：
   - `false_no_answer_rate`: `0.0000 -> 0.0182`
   - `partial_hallucination_rate`: `0.0556 -> 0.1111`
   - `partial_hit_precision`: `0.9474 -> 0.8947`
   - `partial_false_negative_rate`: `0.0000 -> 0.0556`
   - `partial_missing_negation_rate`: `0.0556 -> 0.1111`

因此 strict 集已经足够说明：

> `partial_coverage` 的定向补样并没有让 `0.6B full` 变成优于 round1 的版本，反而把 strict 集上原本打平 LoRA 的核心纪律打坏了一部分。

### 2. strict 集分组观察

strict 分组最关键的变化在：

- `partial_coverage.evidence_utilization_rate`：`1.0000 -> 0.9444`
- `partial_coverage.partial_false_negative_rate`：`0.0000 -> 0.0556`
- `partial_coverage.partial_missing_negation_rate`：`0.0556 -> 0.1111`
- `recent_hit.natural_style_rate`：`0.8889 -> 0.7778`

而 `stability.natural_style_rate` 则从 `0.8421` 升到 `0.8947`。

这说明补样并不是“纯负收益”，但收益仍集中在部分表达层，无法覆盖 strict 集上的纪律损失。

---

## 六、case 级分析

### 1. raw `partial_coverage_006`：本轮最成功的修复

`full round1`：

- `目前只找到了写字台，没有看到椅子。`

`full round4 patch`：

- `目前只找到椅子相关内容，未见写字台。`

这是本轮最明确的正向样本：

- 修掉了 supported / unsupported 对调
- 同时修掉了该 case 上的 `partial_hallucination`
- 也修掉了 `partial_false_negative`

这说明数据层补样并不是完全无效，它至少能对原始集中的同型 case 起作用。

### 2. raw `partial_coverage_008`：局部目标未稳住

`full round1`：

- `最近拍到海边的风景画，暂无相关记录。`

`full round4 patch`：

- `最近拍到过海边，但目前没有找到相关记录。`

问题在于：

- round1 是 unsupported leakage + 漏否定
- round4 变成了 unsupported leakage + false negative + evidence 未利用

也就是说：

> 它不是把错误消掉了，而是把错误形态换了一种。

### 3. strict `partial_coverage_006_rw`：最关键 strict case 仍没修掉

`0.6B LoRA`：

- `目前只找到椅子相关内容，未见写字台。`

`full round1`：

- `有椅子和写字台。`

`full round4 patch`：

- `有椅子和写字台。`

这是本轮最关键的反证：

- raw `partial_coverage_006` 被修了
- 但 strict 的同型改写 `partial_coverage_006_rw` 仍然完全没修

因此不能把 raw 里的局部修复误判成泛化收益。

### 4. strict `partial_coverage_008_rw`：引入新的 false no-answer

`full round1`：

- `目前只找到过风景画，海边暂无相关记录。`

`full round4 patch`：

- `海边没有相关记录。`

round4 的问题更严重：

- 丢失了 supported object `风景画`
- 变成 `false_no_answer`
- 同时被判定为 `partial_false_negative`

这类 case 说明补样有时会把模型往“只保留否定句”方向推得过头。

### 5. raw `partial_coverage_005 / 007 / 015`：出现新的漏否定与幻觉

`partial_coverage_005`

- round1：末句明确说 `咖啡机暂无相关记录`
- round4：直接省掉否定句

`partial_coverage_007`

- round1：`花瓶暂无相关记录`
- round4：只描述电脑桌场景，不再否定花瓶

`partial_coverage_015`

- round1：`那床暂时没有相关记录`
- round4：`那床可能还在等待你`

这里暴露出一个更危险的信号：

- 补样虽然强化了“supported / unsupported 二元表达”
- 但在其他 partial case 上，它反而放松了对 unsupported 项的明确否定

### 6. must-answer 焦点也被带偏

raw `must_answer_006 / 015` 在 round4 都出现了类似问题：

- 回答更长
- 带出整段场景描述
- 虽然没有事实错误
- 但不再像 round1 那样紧扣用户真正问的对象

这正对应了 raw 集里 `must_answer_focus_rate` 从 `0.8125` 回落到 `0.6875`。

### 7. case 级结论

case 级分析最终指向同一个判断：

1. 小规模定向补样可以修个别 raw case
2. 但这种修复没有稳定泛化到 strict 改写 case
3. 同时它又在其它 partial case 和 must-answer case 上引入了新的副作用

因此这轮结果还不足以支持继续沿“补一小批 partial patch”这条线加码。

---

## 七、本轮结论

### 1. 这轮实验的价值

Part 33 的价值不在于把 `0.6B full` 继续推强，而在于更明确地缩小了问题范围：

- 问题已经不是单纯 learning rate
- 也不是只要补一小批 `partial_coverage` 样本就能修好
- 当前 `0.6B full` 的 unsupported leakage 更像是一个泛化层面的稳定性问题

### 2. 对 `0.6B full` 支线的最新定位

现在可以把 `0.6B full` 的组织定位写得更清楚：

> `0.6B full` 仍定位为风格探索支线，不替代 `0.6B LoRA` 主线，也不影响现有 `1.7B` 部署主线。

进一步细化就是：

- `full round1 / lr=1e-5`：仍是当前 `0.6B full` 的暂时最佳点
- `round4 partial_patch_v1`：证明了数据层小补样能修局部 case，但不足以带来整体胜出

### 3. 下一步建议

本轮之后，不建议继续做下面这些事：

- 不再继续做小规模 pure lr 搜索
- 不再继续做同思路的小补样盲试
- 不把 `0.6B full` 推进到部署候选

如果后续还要继续探索，优先级应是：

1. 先停在 `full round1 / lr=1e-5` 作为 `0.6B full` 封板版本
2. 若必须继续，只做更窄、更可归因的训练策略实验
3. 但在当前项目主线下，这条支线已经足够回答“值不值得继续深挖”的问题

一句话总结：

> Part 33 证明了 `partial_coverage` 定向补样只能修局部、不能稳住全局；对当前 `0.6B full` 来说，最佳实验结论仍然是“到此为止，保留为风格探索支线，不替代 LoRA 主线”。
