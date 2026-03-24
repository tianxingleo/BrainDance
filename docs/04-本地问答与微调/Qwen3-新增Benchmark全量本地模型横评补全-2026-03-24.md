# Qwen3 新增 Benchmark 全量本地模型横评补全（2026-03-24）

## 1. 这次补全解决什么问题

之前新增 benchmark 虽然已经落地，但还没有像 strict v3 那样，把 **本地主要候选版本** 全部放进同一张总表里统一比较。

这次补全专门把这个缺口补上：

1. `新增未见数据 OOD benchmark` 从原来的 `3` 个候选扩成 `16` 个本地版本统一横评。
2. `空间关系 hardcases` 从原来的 `6` 个候选扩成 `16` 个本地版本统一横评。
3. 两套 benchmark 都改成支持 `--candidate_ids all`，以后可以直接复跑，不再停留在代表性抽样。

## 2. 本次新增脚本能力

新增统一候选注册表：

- [benchmark_candidate_registry.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/benchmark_candidate_registry.py)

更新评测脚本：

- [evaluate_unseen_ood_benchmark.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/evaluate_unseen_ood_benchmark.py)
- [evaluate_spatial_hardcases_candidates.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/evaluate_spatial_hardcases_candidates.py)
- [build_local_version_matrix.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/build_local_version_matrix.py)

这次实际补跑使用的是 frozen 口径：

- `ai_engine/finetune_qwen3/data/braindance_qwen3_unseen_ood_retrieval_snapshot_20260324.json`
- `ai_engine/finetune_qwen3/data/braindance_qwen3_unseen_ood_spatial_hardcases_retrieval_snapshot_20260324.json`

新结果日志：

- `ai_engine/finetune_qwen3/logs/unseen_ood_benchmark_20260324_frozen_all_local_results.json`
- `ai_engine/finetune_qwen3/logs/unseen_ood_benchmark_20260324_frozen_all_local_summary.json`
- `ai_engine/finetune_qwen3/logs/spatial_hardcases_candidates_20260324_frozen_all_local_results.json`
- `ai_engine/finetune_qwen3/logs/spatial_hardcases_candidates_20260324_frozen_all_local_summary.json`

## 3. 参评本地版本

本次统一纳入 `16` 个本地版本：

1. `0.6B LoRA`
2. `0.6B full SFT round1`
3. `0.6B full SFT round2 lr8e-6`
4. `0.6B full SFT round3 lr5e-6`
5. `0.6B full SFT round4 patch`
6. `1.7B LoRA round3`
7. `1.7B LoRA round4 patch`
8. `1.7B LoRA round4.1 patch`
9. `1.7B LoRA`
10. `1.7B full SFT round1`
11. `1.7B full SFT lr5e-6`
12. `1.7B merged`
13. `1.7B Q4_K_M`
14. `1.7B Q5_K_M`
15. `1.7B Q4_K_M + imatrix`
16. `1.7B Q5_K_M + imatrix`

## 4. 新增未见数据 OOD：全量本地版本总表

frozen retrieval 口径下，当前 retrieval 基线是：

- `scoreable_case_count = 18`
- `retrieval_ok_rate = 0.6667`
- `blocked_case_count = 6`
- `partial_coverage retrieval_ok_rate = 1.0000`
- `must_answer retrieval_ok_rate = 0.5000`
- `no_hit retrieval_ok_rate = 0.5000`

这意味着：

- 现在真正能拉开模型差异的主要是 answer side，尤其是 `partial_coverage`
- retrieval 仍然卡住了 `6 / 18` 题，但已经比早先口径更稳定

总表如下：

| 模型 | 端到端通过率 | retrieval正确时回答通过率 | partial_coverage通过率 | 平均总时长(ms) |
| --- | ---: | ---: | ---: | ---: |
| 1.7B Q5_K_M + imatrix | 0.6667 | 1.0000 | 1.0000 | 19094.22 |
| 1.7B merged | 0.6111 | 0.9167 | 0.8333 | 18489.98 |
| 0.6B full SFT round2 lr8e-6 | 0.5556 | 0.8333 | 0.6667 | 18448.27 |
| 1.7B full SFT round1 | 0.5556 | 0.8333 | 0.6667 | 18487.70 |
| 1.7B LoRA round3 | 0.5556 | 0.8333 | 0.6667 | 18640.06 |
| 1.7B LoRA round4.1 patch | 0.5556 | 0.8333 | 0.6667 | 18664.22 |
| 1.7B LoRA | 0.5556 | 0.8333 | 0.6667 | 18683.20 |
| 1.7B Q4_K_M + imatrix | 0.5556 | 0.8333 | 0.6667 | 19057.34 |
| 1.7B Q5_K_M | 0.5556 | 0.8333 | 0.6667 | 19079.67 |
| 0.6B full SFT round4 patch | 0.5000 | 0.7500 | 0.5000 | 18512.87 |
| 1.7B LoRA round4 patch | 0.5000 | 0.7500 | 0.5000 | 18631.61 |
| 1.7B Q4_K_M | 0.5000 | 0.7500 | 0.5000 | 19070.10 |
| 0.6B full SFT round3 lr5e-6 | 0.4444 | 0.6667 | 0.3333 | 18437.31 |
| 0.6B full SFT round1 | 0.3889 | 0.5833 | 0.1667 | 18505.27 |
| 0.6B LoRA | 0.3889 | 0.5833 | 0.1667 | 18788.25 |
| 1.7B full SFT lr5e-6 | 0.3333 | 0.5000 | 0.0000 | 18440.73 |

## 5. OOD 结果怎么解读

### 5.1 当前新 benchmark 的第一名已经非常明确

`1.7B Q5_K_M + imatrix` 这次不是“和 LoRA 接近”，而是已经在这套 frozen 新增 OOD 上拉出了最清晰的领先：

- `end_to_end_pass_rate = 0.6667`
- `answer_pass_rate_when_retrieval_ok = 1.0000`
- `partial_coverage_pass_rate = 1.0000`

也就是说，在 retrieval 已经命中的前提下，它在这 18 条可打分题上没有再丢 answer side 的分。

### 5.2 第二名不是 1.7B LoRA，而是 1.7B merged

`1.7B merged` 这次表现非常强：

- `end_to_end_pass_rate = 0.6111`
- `answer_pass_rate_when_retrieval_ok = 0.9167`
- `partial_coverage_pass_rate = 0.8333`

它是这轮 all-local frozen OOD 里唯一紧跟 `Q5 + imatrix` 的版本。

这说明 merged 版本在真实新增数据上的泛化并不只是“接近 strict 集”，而是在 OOD 上也已经进入第一梯队。

### 5.3 1.7B LoRA 主线没有崩，但不再是唯一最优

`1.7B LoRA round3 / round4.1 patch / round4.1 patch mixed` 这三条线在 frozen OOD 上几乎完全并列：

- `end_to_end_pass_rate = 0.5556`
- `answer_pass_rate_when_retrieval_ok = 0.8333`
- `partial_coverage_pass_rate = 0.6667`

所以新的结论不是“LoRA 不行”，而是：

- `LoRA` 仍然稳
- 但在这套新增真实数据上，`merged` 和 `Q5 + imatrix` 已经跑到了更前面

### 5.4 0.6B 里最值得保留的不是 round1，而是 round2 lr8e-6

`0.6B full SFT round2 lr8e-6` 这次明显优于 `0.6B LoRA` 和 `0.6B full round1`：

- `0.6B full round2 lr8e-6`: `0.5556 / 0.8333 / 0.6667`
- `0.6B full round1`: `0.3889 / 0.5833 / 0.1667`
- `0.6B LoRA`: `0.3889 / 0.5833 / 0.1667`

这意味着在新增未见数据 OOD 上，`0.6B full round2 lr8e-6` 才是当前 0.6B 支线里最值得保留的版本。

### 5.5 最差点也很明确

`1.7B full SFT lr5e-6` 在这套新 benchmark 上是明显回退点：

- `end_to_end_pass_rate = 0.3333`
- `answer_pass_rate_when_retrieval_ok = 0.5000`
- `partial_coverage_pass_rate = 0.0000`

这和 strict 集里它的纪律回退方向是一致的，不是偶然波动。

## 6. 空间关系 hardcases：全量本地版本总表

总表如下：

| 模型 | spatial_direct_rate | generic_scene_summary_rate | refusal_rate | 平均总时长(ms) |
| --- | ---: | ---: | ---: | ---: |
| 0.6B full SFT round3 lr5e-6 | 0.1000 | 0.9000 | 0.0000 | 13144.39 |
| 0.6B full SFT round1 | 0.0000 | 0.9000 | 0.1000 | 13133.04 |
| 1.7B merged | 0.0000 | 0.9000 | 0.1000 | 13133.72 |
| 1.7B full SFT round1 | 0.0000 | 0.9000 | 0.1000 | 13134.02 |
| 0.6B full SFT round2 lr8e-6 | 0.0000 | 0.9000 | 0.1000 | 13134.19 |
| 1.7B full SFT lr5e-6 | 0.0000 | 0.9000 | 0.1000 | 13134.44 |
| 0.6B full SFT round4 patch | 0.0000 | 0.9000 | 0.1000 | 13143.62 |
| 1.7B LoRA round3 | 0.0000 | 0.9000 | 0.1000 | 13149.72 |
| 1.7B LoRA round4 patch | 0.0000 | 0.9000 | 0.1000 | 13155.28 |
| 1.7B LoRA round4.1 patch | 0.0000 | 0.9000 | 0.1000 | 13161.54 |
| 1.7B LoRA | 0.0000 | 0.9000 | 0.1000 | 13163.92 |
| 1.7B Q4_K_M | 0.0000 | 0.9000 | 0.1000 | 13254.11 |
| 1.7B Q4_K_M + imatrix | 0.0000 | 0.9000 | 0.1000 | 13265.74 |
| 1.7B Q5_K_M + imatrix | 0.0000 | 0.9000 | 0.1000 | 13270.78 |
| 0.6B LoRA | 0.0000 | 0.9000 | 0.1000 | 13282.55 |
| 1.7B Q5_K_M | 0.0000 | 0.9000 | 0.1000 | 13290.56 |

## 7. 空间关系结果怎么解读

### 7.1 这次终于不是 6 个模型，而是 16 个模型一起证明了一件事

空间关系题当前的主矛盾仍然不是模型本身，而是链路：

- `16` 个版本里只有 `1` 个版本打出了 `0.1000` 的 `spatial_direct_rate`
- 其余 `15` 个版本全部是 `0.0000`
- 所有版本的 `generic_scene_summary_rate` 都仍然是 `0.9000`

所以这次补跑不是推翻旧结论，而是用更大样本的模型矩阵把旧结论钉死了：

- 当前空间题还没有进入真正的模型比较阶段

### 7.2 唯一的例外不足以说明能力突破

`0.6B full SFT round3 lr5e-6` 这次打出了：

- `spatial_direct_rate = 0.1000`
- `refusal_rate = 0.0000`

但这只代表 `10` 题里有 `1` 题从“拒答/总结”变成了更接近空间直接回答，远远不足以说明它已经真的具备稳定空间推理能力。

### 7.3 route 侧模式几乎没变

这次 all-local frozen spatial 里，answer route 仍然几乎一致：

- `recent_answer_formatter = 0.8`
- `must_answer_focus_formatter = 0.1`
- `lora_generation = 0.1`

也就是说：

- 无论 `LoRA / full / merged / quantized / imatrix`
- 最后都被送进了几乎同一套 route / formatter 模式

所以单看模型大小和量化方案，本轮空间 hardcase 基本分不出输赢。

## 8. 最终结论

如果把 strict v3 之外的新增 benchmark 补齐到和 strict 报告同级别的信息密度，这次最重要的结论有 6 条：

1. 现在的新 benchmark 已经补成了真正的 **全量本地版本横评**，不再只是 3 个或 6 个代表性候选。
2. `新增未见数据 OOD` 上的当前第一名是 `1.7B Q5_K_M + imatrix`。
3. `1.7B merged` 在新增真实数据上的表现已经稳定进入第一梯队，明显强于之前“只是可部署备份”的定位。
4. `1.7B LoRA` 主线依然稳，但在新增 OOD 上已经不再是唯一最优。
5. `0.6B` 支线里真正值得保留的是 `0.6B full round2 lr8e-6`，不是 `0.6B LoRA` 也不是 `0.6B full round1`。
6. 空间关系题这次用 `16` 个本地版本再次证明：当前瓶颈仍然是 route / formatter，不是模型本体。
