# Qwen3 新增 Benchmark 云端与本地对标评测报告（2026-03-24）

## 1. 目标

这份报告专门补齐新增 benchmark 的最后一块：

不仅比较本地模型之间的差异，还把云端模型一起拉进同一套 `frozen retrieval` 口径里，回答两个更直接的问题：

1. 在 **训练后新增、模型没见过的真实数据** 上，本地最强版本到底能不能对到云端。
2. 当问题升级到 `左 / 右 / 远近 / 三者之间` 这种空间 hardcases 时，云端大模型是否已经明显强于本地版本。

## 2. 评测集与口径

### 2.1 新增未见数据 OOD

- 数据集：`ai_engine/finetune_qwen3/data/braindance_qwen3_unseen_ood_benchmark_20260324.json`
- frozen retrieval snapshot：`ai_engine/finetune_qwen3/data/braindance_qwen3_unseen_ood_retrieval_snapshot_20260324.json`
- 本地全量汇总：`ai_engine/finetune_qwen3/logs/unseen_ood_benchmark_20260324_frozen_all_local_summary.json`
- 云端汇总：`ai_engine/finetune_qwen3/logs/cloud_unseen_ood_benchmark_20260324_frozen_summary.json`

frozen retrieval 基线：

- `scoreable_case_count = 18`
- `retrieval_ok_rate = 0.6667`
- `blocked_case_count = 6`
- `must_answer retrieval_ok_rate = 0.5000`
- `no_hit retrieval_ok_rate = 0.5000`
- `partial_coverage retrieval_ok_rate = 1.0000`

### 2.2 空间关系高难 hardcases

- 数据集：`ai_engine/finetune_qwen3/data/braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json`
- frozen retrieval snapshot：`ai_engine/finetune_qwen3/data/braindance_qwen3_unseen_ood_spatial_hardcases_retrieval_snapshot_20260324.json`
- 本地全量汇总：`ai_engine/finetune_qwen3/logs/spatial_hardcases_candidates_20260324_frozen_all_local_summary.json`
- 云端汇总：`ai_engine/finetune_qwen3/logs/cloud_spatial_hardcases_20260324_frozen_summary.json`

### 2.3 评测脚本

- 本地 OOD：`ai_engine/finetune_qwen3/scripts/evaluate_unseen_ood_benchmark.py`
- 本地 spatial：`ai_engine/finetune_qwen3/scripts/evaluate_spatial_hardcases_candidates.py`
- 云端 OOD：`ai_engine/finetune_qwen3/scripts/evaluate_cloud_unseen_ood_benchmark.py`
- 云端 spatial：`ai_engine/finetune_qwen3/scripts/evaluate_cloud_spatial_hardcases.py`
- 静态图渲染：`ai_engine/finetune_qwen3/scripts/render_new_benchmark_comparison_figures.py`

本次云端模型沿用 strict 报告使用过的同一批候选：

- `qwen2.5-32b-instruct`
- `qwen3-32b`
- `qwen3-8b`
- `qwen-turbo`

图表输出目录：

- `ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/`
- 图表清单：`ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/manifest_20260324.json`

绘图依赖：

- `matplotlib`
- `seaborn`
- `adjustText`

## 3. 新增未见数据 OOD：云端与本地总表

先放统一总表。这里把 `16` 个本地版本和 `4` 个云端版本按同一标准排序，表中展示代表性第一梯队与关键对照项：

| 模型 | 类型 | 端到端通过率 | retrieval 正确时回答通过率 | partial_coverage 通过率 | 平均总时长(ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.7B Q5_K_M + imatrix | 本地 GGUF | **0.6667** | **1.0000** | **1.0000** | **19094.22** |
| qwen3-8b | 云端 | **0.6667** | **1.0000** | **1.0000** | 19903.24 |
| qwen2.5-32b-instruct | 云端 | **0.6667** | **1.0000** | **1.0000** | 20030.25 |
| 1.7B merged | 本地 HF | 0.6111 | 0.9167 | 0.8333 | **18489.98** |
| qwen3-32b | 云端 | 0.6111 | 0.9167 | 0.8333 | 20172.66 |
| 1.7B LoRA | 本地 HF | 0.5556 | 0.8333 | 0.6667 | **18683.20** |
| 0.6B full SFT round2 lr8e-6 | 本地 HF | 0.5556 | 0.8333 | 0.6667 | **18448.27** |
| qwen-turbo | 云端 | 0.4444 | 0.6667 | 0.3333 | 19787.20 |

补充说明：

- 全量本地 `16` 版本的完整排序，见 [Qwen3-新增Benchmark全量本地模型横评补全-2026-03-24.md](/ltx-data/BrainDance/docs/04-本地问答与微调/Qwen3-新增Benchmark全量本地模型横评补全-2026-03-24.md)
- 云端完整日志：
  - `ai_engine/finetune_qwen3/logs/cloud_unseen_ood_benchmark_20260324_frozen_results.json`
  - `ai_engine/finetune_qwen3/logs/cloud_unseen_ood_benchmark_20260324_frozen_summary.json`

### 3.1 结论不是“云端全面碾压”，而是“本地最佳已经对到了云端第一档”

新增未见数据 OOD 上，这次最重要的发现非常明确：

- `1.7B Q5_K_M + imatrix` 和云端最强两档 `qwen3-8b / qwen2.5-32b-instruct` 打成了同分
- 三者都达到：
  - `end_to_end_pass_rate = 0.6667`
  - `answer_pass_rate_when_retrieval_ok = 1.0000`
  - `partial_coverage_pass_rate = 1.0000`

这意味着：

在 retrieval 已经命中的前提下，当前最佳本地量化版在这套新增真实 OOD 上，已经不是“接近云端”，而是实际跑到了云端第一梯队。

### 3.2 云端 32B 并没有在这套新 benchmark 上压过本地主线

`qwen3-32b` 这次没有跑出比本地最强更高的结果，它对应的是：

- `0.6111 / 0.9167 / 0.8333`

这个成绩恰好和 `1.7B merged` 对齐，而且还更慢。

所以当前更准确的结论不是“云端 32B 一定更强”，而是：

- `qwen3-32b` 在这套新 benchmark 上只对齐到了 `1.7B merged`
- 真正拉到第一档的是 `qwen3-8b` 与 `qwen2.5-32b-instruct`

### 3.3 1.7B LoRA 主线依然稳，但已经不是新增 OOD 的唯一中心

`1.7B LoRA` 仍然保持在稳健梯队：

- `0.5556 / 0.8333 / 0.6667`

但这轮结果说明，新增真实 OOD 上更值得优先关注的对照关系已经变成：

- 本地部署首选：`1.7B Q5_K_M + imatrix`
- 本地高保真备选：`1.7B merged`
- 云端第一档：`qwen3-8b`、`qwen2.5-32b-instruct`

也就是说，新的 benchmark 已经把“LoRA 是唯一锚点”的叙事改成了“Q5 + imatrix / merged / LoRA 三线分工”。

### 3.4 qwen-turbo 明显不是这套 benchmark 的优选云端基线

`qwen-turbo` 只打到：

- `end_to_end_pass_rate = 0.4444`
- `answer_pass_rate_when_retrieval_ok = 0.6667`
- `partial_coverage_pass_rate = 0.3333`

它不仅落后于云端另外三档，也落后于多条本地主线。

因此如果后续还要保留云端对照，`qwen-turbo` 更适合作为“低成本云端基线”，不适合作为“高质量上限参考”。

### 3.5 图表：新增未见数据 OOD 第一梯队

下面这 3 张图分别对应：

1. 全量统一榜单
2. 速度/准确率/partial 覆盖三维关系
3. 第一梯队模型画像

![新增未见数据 OOD 统一榜单](../../ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/new_benchmark_unseen_leaderboard_20260324.png)

![新增未见数据 OOD 三维散点图](../../ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/new_benchmark_unseen_scatter_20260324.png)

![新增未见数据 OOD 第一梯队雷达图](../../ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/new_benchmark_unseen_radar_20260324.png)

## 4. 空间关系 hardcases：云端与本地总表

这组结果更“残酷”，因为它几乎把所有模型压成了同一类输出。

| 模型 | 类型 | spatial_direct_rate | generic_scene_summary_rate | refusal_rate | 平均总时长(ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6B full SFT round3 lr5e-6 | 本地 HF | **0.1000** | 0.9000 | **0.0000** | **13144.39** |
| 1.7B LoRA | 本地 HF | 0.0000 | 0.9000 | 0.1000 | **13163.92** |
| 1.7B merged | 本地 HF | 0.0000 | 0.9000 | 0.1000 | **13133.72** |
| 1.7B Q5_K_M + imatrix | 本地 GGUF | 0.0000 | 0.9000 | 0.1000 | **13270.78** |
| qwen3-32b | 云端 | 0.0000 | 0.9000 | 0.1000 | 13438.28 |
| qwen2.5-32b-instruct | 云端 | 0.0000 | 0.9000 | 0.1000 | 13501.30 |
| qwen3-8b | 云端 | 0.0000 | 0.9000 | 0.1000 | 13528.51 |
| qwen-turbo | 云端 | 0.0000 | 0.9000 | 0.1000 | 13545.44 |

补充说明：

- 本地完整 `16` 版本排序见 [Qwen3-新增Benchmark全量本地模型横评补全-2026-03-24.md](/ltx-data/BrainDance/docs/04-本地问答与微调/Qwen3-新增Benchmark全量本地模型横评补全-2026-03-24.md)
- 云端完整日志：
  - `ai_engine/finetune_qwen3/logs/cloud_spatial_hardcases_20260324_frozen_results.json`
  - `ai_engine/finetune_qwen3/logs/cloud_spatial_hardcases_20260324_frozen_summary.json`

### 4.1 云端和本地都没有真正解出空间题

这轮最重要的结论不是“谁更高 0.01”，而是：

- `4` 个云端模型全部 `spatial_direct_rate = 0.0000`
- `16` 个本地模型里只有 `1` 个版本碰到 `0.1000`
- 其余 `19` 个模型全部被压成：
  - `generic_scene_summary_rate = 0.9000`
  - `refusal_rate = 0.1000`

也就是说，云端没有因为参数量更大就自动跨过这个瓶颈。

### 4.2 这进一步证明空间题主瓶颈仍然在链路，不在模型

云端 32B、云端 8B、本地 1.7B、本地 0.6B、量化版、merged 版都收敛到同样的输出分布，说明：

- 不是某个单独模型“不会做空间推理”
- 而是现有 `retrieval_route + answer_route + formatter` 仍在把空间题扁平化成“列举场景里有什么”

因此空间 hardcases 的下一步不应该优先继续卷模型，而应该优先改：

1. retrieval 结果结构
2. spatial 专用 answer formatter
3. 对位置关系、前后关系、远近关系的结构化解析与约束模板

### 4.3 图表：空间题云端与本地一起失效

![空间 hardcases 直接回答率全景](../../ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/new_benchmark_spatial_bar_20260324.png)

![跨 benchmark 关键指标热力图](../../ai_engine/finetune_qwen3/audit/new_benchmark_figures_20260324/new_benchmark_cross_heatmap_20260324.png)

## 5. 最终结论

如果按严格报告的口径，把新增 benchmark 的云端与本地也一起看，这次最关键的结论有 6 条：

1. `新增未见数据 OOD` 上，当前最佳本地版本 `1.7B Q5_K_M + imatrix` 已经实际对到云端第一档，不再只是“勉强接近”。
2. 云端里真正强的是 `qwen3-8b` 和 `qwen2.5-32b-instruct`，`qwen3-32b` 这次只对齐到 `1.7B merged`，`qwen-turbo` 明显偏弱。
3. `1.7B merged` 在新增真实 OOD 上继续证明自己不是备份版本，而是稳定第一梯队。
4. `1.7B LoRA` 依然稳，但在新增 benchmark 里的角色已经从“唯一核心”变成“主线稳健基线之一”。
5. `空间关系 hardcases` 上，云端和本地一起失效，说明当前主瓶颈仍是链路和 formatter，而不是模型参数量。
6. 因而当前部署与研发的更优先方向是：
   - OOD 部署优先保留 `1.7B Q5_K_M + imatrix`
   - 质量对照保留 `qwen3-8b / qwen2.5-32b-instruct`
   - 空间题优先改链路，不优先继续卷更大的云端模型
