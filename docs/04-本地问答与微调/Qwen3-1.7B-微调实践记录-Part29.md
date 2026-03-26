# Qwen3-1.7B 微调实践记录 Part 29

## Part 29：部署候选小样本验证与主线选择

### 本 part 目标

这一 part 不继续训练，不继续扩大 benchmark，只做 3 个部署候选的真实调用链验证：

1. `0.6B LoRA`
2. `1.7B merged`
3. `1.7B Q5_K_M + imatrix`

目标不是再做一轮“学术评测”，而是回答一个更工程化的问题：

- **下一阶段真正要接到 Recall 本地 AI 入口里的部署主线，到底选谁**

---

## 一、先处理 Flutter 真实调用链

本 part 开始前，先把 `tianxingleo` 分支合进了当前分支：

- merge commit: `65320b4`

这一步的作用不是泛泛同步，而是明确恢复 Recall 的本地 AI 入口与其依赖链路。合并后已经确认以下代码恢复到当前分支：

- `app/lib/pages/recall.dart`
- `app/lib/pages/recall/local_ai_panel.dart`
- `app/lib/services/local_rag_index.dart`
- `app/lib/services/local_text_embedder.dart`
- `app/pubspec.yaml`

关键事实：

1. 当前 Flutter 本地 AI 入口依赖 `llamadart`
2. 当前页面加载逻辑明确要求输入 **GGUF 模型路径**
3. 当前页面默认文件名就是 `qwen3-1.7b.gguf`
4. 当前页面下载、加载、问答的实现都围绕 `GGUF + llamadart` 路线

也就是说，**从现有 Flutter 代码现实出发，最容易接入的不是 HF adapter，也不是 merged HF 目录，而是 GGUF 模型**。

这点非常关键，因为它会直接影响部署主线选择。

额外说明：

- 当前机器上没有 `flutter` / `dart` CLI，因此这一轮不能做 `flutter analyze` 或真机打包验证
- 但代码链路已恢复，文件级别已确认到位

---

## 二、小样本真实调用集

数据文件：

- `ai_engine/finetune_qwen3/data/braindance_qwen3_deployment_eval_part29.json`

共 `18` 条，覆盖 6 类：

- `recent_hit`
- `no_hit`
- `partial_coverage`
- `must_answer`
- `inventory`
- `abstract_semantic`

这里故意不再做 64/80 题 benchmark，而是做 deployment-oriented 小样本：

1. 看能否在真实 retrieval 链路下稳定跑通
2. 看 formatter 路由是否仍正常触发
3. 看首字时间、总时长和资源占用
4. 看回答风格在真实调用链里是否还能接受

---

## 三、统一评测脚本

新增脚本：

- `ai_engine/finetune_qwen3/scripts/evaluate_deployment_candidates.py`
- `ai_engine/finetune_qwen3/scripts/run_deployment_eval_part29_gpu1.sh`

统一口径：

- `conda run -n qwen3_ft`
- `CUDA_VISIBLE_DEVICES=1`
- 真实 retrieval 链路：`retrieve_real_chain_case`
- 统一记录：
  - `retrieval_latency_ms`
  - `generation_latency_ms`
  - `time_to_first_token_ms`
  - `time_to_final_answer_ms`
  - `peak_memory_mb`
  - `peak_vram_mb`
  - `output_chars`
  - `retrieval_route`
  - `answer_route`

日志产物：

- `ai_engine/finetune_qwen3/logs/deployment_eval_part29_results.json`
- `ai_engine/finetune_qwen3/logs/deployment_eval_part29_summary.json`

---

## 四、总指标对比

### 1. 平均指标

| 候选 | 平均 retrieval(ms) | 平均 generation(ms) | 平均 TTFT(ms) | 平均总时长(ms) | 平均峰值内存(MB) | 平均峰值显存(MB) | 平均输出长度(字符) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B LoRA | 8216.88 | 319.58 | 67.96 | 8536.46 | 1536.11 | 1637.00 | 27.44 |
| 1.7B merged | 8229.07 | 209.79 | 27.51 | 8438.86 | 1738.32 | 3866.00 | 28.22 |
| 1.7B Q5+imatrix | 7461.37 | 1030.44 | 828.72 | 8491.82 | 668.95 | 793.56 | 25.33 |

### 2. 第一层判断

从这张表可以先得到 4 个明确结论：

1. **总时长三者非常接近**
   - 都在 `8.4s ~ 8.5s`
   - 当前真实链路下，总时长主要被 retrieval 吃掉了，不是纯生成问题
2. **1.7B merged 的生成最快、TTFT 最小**
   - 但它的显存占用最高
3. **1.7B Q5+imatrix 的生成最慢、TTFT 最慢**
   - 但它的内存和显存占用最低
4. **0.6B LoRA 在资源和速度上处于中间态**
   - 没有明显赢过 `Q5+imatrix` 的端侧资源优势
   - 也没有达到 `1.7B merged` 的生成速度

---

## 五、调用链稳定性与回答风格

### 1. formatter 路由是否还正常

三条路线都能正常跑通以下路由：

- `recent_answer_formatter`
- `inventory_formatter`
- `semantic_summary_formatter`
- `must_answer_focus_formatter`
- `lora_generation`

路由分布上没有出现整条链路失效的问题。也就是说：

- retrieval 仍能工作
- formatter 仍能命中
- 端侧候选之间的差异，主要发生在“生成回答质量和风格”，而不是路由断裂

### 2. case 级观察

#### `0.6B LoRA`

优点：

- `no_hit` 回答比较稳
- `inventory / semantic` formatter 路由可用
- 首字时间比 GGUF 明显更快

问题：

- `partial_coverage` 易给出过强判断
  - 例如：`冰箱和书架都有记录。`
- `inventory_003` 这类问法容易退成泛化否定
  - `目前没有找到与模型资产相关的记录。`

结论：

- 可用
- 但当前还不像“直接上线的主线版本”

#### `1.7B merged`

优点：

- 输出整体更稳
- 生成速度和 TTFT 最好
- 仍然是当前最合适的**质量基线**

问题：

- 资源最重
- 工程接入复杂度高，因为当前 Flutter 入口不是 HF 路线
- 某些 `partial_coverage` 回答会出现不够干净的双句拼接
  - `最近拍到过地毯，但没拍到打印机；也拍到过打印机，但没拍到地毯。`

结论：

- 质量基线成立
- 但不适合直接作为当前移动端主线

#### `1.7B Q5_K_M + imatrix`

优点：

- 资源最省
- 当前 Flutter Recall 本地 AI 入口就是 `GGUF + llamadart` 路线，工程接入最顺
- 近期通过 Part 28 已确认 strict 集核心任务指标已修回
- 在小样本真实调用里，`inventory / semantic / recent` 三类表现稳定

问题：

- TTFT 最慢
- 某些 `partial_coverage` 回答仍偏短促
  - `电视柜见，咖啡机见。`

结论：

- 当前最像“能直接接入产品链路继续往前推”的版本

---

## 六、部署决策表

| 候选 | 正确性 | 风格 | 首字时间 | 总时长 | 资源占用 | 工程复杂度 | 建议角色 |
|---|---|---|---|---|---|---|---|
| 0.6B LoRA | 中 | 中 | 较快 | 中 | 中 | 高 | 备用候选 |
| 1.7B merged | 高 | 较稳 | 最快 | 中 | 高 | 高 | 质量基线 |
| 1.7B Q5+imatrix | 较高 | 可接受但略短 | 最慢 | 中 | 最低 | 低 | 当前部署主线 |

### 这里的“工程复杂度”为什么这样判

不是抽象判断，而是由当前代码现状决定：

1. Recall 本地 AI 入口已经是 `llamadart + GGUF`
2. `Q5+imatrix` 可以直接沿当前链路继续接
3. `0.6B LoRA` 和 `1.7B merged` 目前都不是这条 Flutter 本地链路的原生输入

所以：

- **从“今天就继续往产品里接”的角度，`Q5+imatrix` 最顺**
- **从“保留一个质量对照上界”的角度，`1.7B merged` 最合适**

---

## 七、最终选择

### 1. 当前部署主线

- **`1.7B Q5_K_M + imatrix GGUF`**

原因：

1. 当前 Flutter 本地 AI 入口已恢复，且就是 `GGUF + llamadart`
2. Part 28 已经确认它修复了原始 `Q5` 的 strict 回退
3. Part 29 小样本里它的总时长并没有输很多，但资源占用最低
4. 它是现在最接近“直接可落地”的版本

### 2. 备用方案

- **`0.6B LoRA`**

原因：

1. 体量更小，后续仍有继续压缩端侧的价值
2. 小样本能跑通，回答不至于不可用
3. 如果后面需要进一步追求更小包体，它仍然是最值得继续打磨的轻量路线

但当前不把它定为主线，因为：

- 现有 Flutter 本地入口不是 adapter/HF 路线
- 它还没有形成足够明确的“速度/资源/质量同时更优”

### 3. 质量基线候选

- **`1.7B merged`**

原因：

1. 生成速度和 TTFT 最好
2. 输出稳定性仍然最适合做基线
3. 之后所有端侧版本都可以拿它继续做对照

---

## 八、这一轮不建议继续做的事

1. 不继续开新的训练轮次
2. 不继续扩大 benchmark
3. 不继续深挖 `Q4/Q5` 理论差异

当前更值得做的是：

1. 继续把 Recall 本地 AI 入口和 `Q5+imatrix GGUF` 真正串起来
2. 做更接近 App 体验的小样本验证
3. 如果端侧包体继续吃紧，再回到 `0.6B` 路线做下一轮压缩

---

## 九、本 part 产出

- merge commit：
  - `65320b4` `merge(app): 合并 tianxingleo 分支以恢复 Recall 本地 AI 入口与相关 Flutter 链路`
- 新增数据：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_deployment_eval_part29.json`
- 新增脚本：
  - `ai_engine/finetune_qwen3/scripts/evaluate_deployment_candidates.py`
  - `ai_engine/finetune_qwen3/scripts/run_deployment_eval_part29_gpu1.sh`
- 新增日志：
  - `ai_engine/finetune_qwen3/logs/deployment_eval_part29_results.json`
  - `ai_engine/finetune_qwen3/logs/deployment_eval_part29_summary.json`

一句话总结：

- **当前最该推进的不是继续训练，而是沿着已恢复的 Flutter GGUF 调用链，把 `1.7B Q5_K_M + imatrix` 作为部署主线往前接。**
