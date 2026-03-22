# Qwen3-1.7B 微调实践记录 Part 23

## Part 23：0.6B、1.7B merged、1.7B Q4 GGUF 的 benchmark 补测与 GPU 修复

### 本 part 目标

补齐下面三类产物的 benchmark 记录，并明确它们的运行后端：

- `Qwen3-0.6B` LoRA 微调版本
- `Qwen3-1.7B` LoRA merge 后的 Hugging Face 独立模型
- `Qwen3-1.7B` 的 `Q4_K_M` GGUF 量化版本

同时解决一个实际阻塞问题：

- 为什么 `Q4 GGUF` 一开始没有真正跑上 GPU
- 如何修复 `llama.cpp` 的 CUDA 构建链路，让它最终使用 `gpu1`

---

## 一、结论先行

截至 **2026-03-22**，这三类产物的 benchmark 状态如下：

| 模型/产物 | benchmark 状态 | 运行后端 | 结论 |
| --- | --- | --- | --- |
| `Qwen3-0.6B + LoRA` | 已跑 | Transformers + `gpu1` | 指标稳定，可继续作为端侧候选 |
| `Qwen3-1.7B merged` | 本轮补跑完成 | Transformers + `gpu1` | 质量最稳，适合继续作为标准 HF 部署基线 |
| `Qwen3-1.7B Q4_K_M GGUF` | 本轮补跑完成 | `llama.cpp` CUDA + `gpu1` | 可以跑 GPU，但质量相比 merged 明显回撤，当前更像“可部署候选”，不是“质量等价替代” |

本轮最重要的事实不是“Q4 终于能跑 GPU”本身，而是：

1. `Q4 GGUF` 已经能真实挂载 `gpu1` 推理
2. 但在当前 80 题固定 benchmark 上，`Q4` 质量明显弱于 `1.7B merged`
3. 这意味着当前不能把 `merge -> GGUF -> Q4` 视作“无损部署”

---

## 二、现有 benchmark 覆盖情况

### 1. `0.6B LoRA`

这条线此前已经跑过 benchmark，本轮又在 `gpu1` 上复跑了一次，结果与此前 `gpu0` 记录一致。

使用：

- [run_benchmark_qwen3_0p6b_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_gpu1.sh)

输出：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_round1_gpu1.json`
- 历史记录：`ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_round1_gpu0.json`

### 2. `1.7B merged`

此前只完成了：

- merge
- GGUF 转换
- 量化
- 最小加载验证

但没有针对 merge 后的独立 HF 模型目录跑标准 benchmark。

本轮补跑完成。

使用：

- [run_benchmark_qwen3_1p7b_merged_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_1p7b_merged_gpu1.sh)

输入模型：

- [qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0](/ltx-data/BrainDance/ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0)

输出：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_merged_round4_1_patch_mixed_gpu1.json`

### 3. `1.7B Q4 GGUF`

此前只做了：

- GGUF 产物生成
- `Q4_K_M` 量化
- 最小加载验证

但没有真正跑完整 benchmark。

本轮补跑完成。

使用：

- [evaluate_gguf_benchmark.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/evaluate_gguf_benchmark.py)
- [run_benchmark_qwen3_1p7b_q4_gguf_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_1p7b_q4_gguf_gpu1.sh)

输入模型：

- `ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0/model-f16-q4_k_m.gguf`

输出：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_q4_gguf_round4_1_patch_mixed_gpu1.json`

---

## 三、为什么 Q4 一开始没有跑上 GPU

根因不是模型问题，而是 `llama.cpp` 的本地构建方式有问题。

### 1. 初始状态是 CPU-only 构建

当前项目原先编译出来的 `llama.cpp` 位于：

- `ai_engine/finetune_qwen3/tools/llama.cpp/build/bin/llama-cli`

检查结果：

- [CMakeCache.txt](/ltx-data/BrainDance/ai_engine/finetune_qwen3/tools/llama.cpp/build/CMakeCache.txt:401) 中 `GGML_CUDA:BOOL=OFF`
- `llama-cli --list-devices` 返回空

这说明：

- 即使设置了 `CUDA_VISIBLE_DEVICES=1`
- 旧版 `llama-cli` 也不会实际使用 GPU
- 所以最开始那次 `GGUF` benchmark 只能算 CPU 验证，不算 GPU benchmark

### 2. 第一次重编失败：架构自动探测失败

第一次尝试 `GGML_CUDA=ON` 时，`cmake` 使用了 `CUDA_ARCHITECTURES=native`。

失败原因：

- 配置阶段没有正确探测到本机 GPU 架构
- `native` 无法展开

### 3. 第二次重编失败：系统默认 `nvcc` 太旧

显式指定 `89` 架构后，新的问题是：

- 系统默认拿的是 `/usr/bin/nvcc`
- 版本是 `11.5.119`
- 不支持 `compute_89`

这一步的报错本质上是：

- 不是 `llama.cpp` 不能编
- 而是默认 CUDA 编译器太旧

### 4. 最终修复方式

最终采用下面这组参数，编出了可用的 CUDA 版 `llama.cpp`：

```bash
cmake -S ai_engine/finetune_qwen3/tools/llama.cpp \
  -B ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda -j 8 \
  --target llama-cli llama-quantize
```

编译结果：

- `ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda/bin/llama-cli`
- `ai_engine/finetune_qwen3/tools/llama.cpp/build-cuda/bin/llama-quantize`

设备检测结果：

```text
ggml_cuda_init: found 1 CUDA devices
Available devices:
  CUDA0: NVIDIA L20
```

这里 `CUDA0` 对应的是：

- 由于脚本设置了 `CUDA_VISIBLE_DEVICES=1`
- 所以对 `llama.cpp` 来说，外部的 `gpu1` 会映射为进程内的 `CUDA0`

这也是为什么正式 benchmark 传入的是 `CUDA0`，但实际占用的是物理 `gpu1`。

---

## 四、本轮 benchmark 结果

### 1. 结果总表

| 指标 | `0.6B LoRA` | `1.7B merged` | `1.7B Q4 GGUF` |
| --- | ---: | ---: | ---: |
| `false_no_answer_rate` | `0.0000` | `0.0000` | `0.0000` |
| `partial_hallucination_rate` | `0.0625` | `0.0625` | `0.2500` |
| `natural_output_rate` | `1.0000` | `1.0000` | `1.0000` |
| `natural_style_rate` | `0.8000` | `0.8250` | `0.8375` |
| `evidence_utilization_rate` | `0.9844` | `1.0000` | `1.0000` |
| `partial_hit_precision` | `0.9375` | `0.9375` | `0.8000` |
| `partial_false_negative_rate` | `0.0625` | `0.0625` | `0.0000` |
| `partial_missing_negation_rate` | `0.1875` | `0.0625` | `0.2500` |
| `must_answer_specific_rate` | `1.0000` | `1.0000` | `1.0000` |
| `must_answer_focus_rate` | `0.6875` | `0.6875` | `0.5000` |

### 2. `0.6B LoRA`

日志：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_0p6b_round1_gpu1.json`

解读：

- 结果与本轮之前 `gpu0` 跑出的 `0.6B` benchmark 一致
- 当前 `0.6B` 仍然是一个有效端侧候选
- 但在部分命中否定表达与风格稳定性上，仍弱于 `1.7B merged`

### 3. `1.7B merged`

日志：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_merged_round4_1_patch_mixed_gpu1.json`

解读：

- `partial_hallucination_rate = 0.0625`，没有比 `0.6B` 变差
- `partial_missing_negation_rate = 0.0625`，明显优于 `0.6B` 的 `0.1875`
- `evidence_utilization_rate = 1.0`

当前可以把它看成：

- 本地 HF 推理口径下，当前最稳的 `1.7B` 标准部署版本

### 4. `1.7B Q4 GGUF`

日志：

- `ai_engine/finetune_qwen3/logs/benchmark_qwen3_1p7b_q4_gguf_round4_1_patch_mixed_gpu1.json`

运行后端：

- `runtime_backend = llama.cpp_gpu`
- `backend_note = detected devices: ['CUDA0: NVIDIA L20 ...']`

延迟摘要：

- `mean_seconds = 1.7212`
- `median_seconds = 1.7051`
- `p95_seconds = 2.0695`
- `max_seconds = 2.3932`

解读：

- `partial_hallucination_rate = 0.25`，比 merged 的 `0.0625` 差很多
- `partial_hit_precision = 0.8`，比 merged 的 `0.9375` 明显回退
- `must_answer_focus_rate = 0.5`，比 merged 的 `0.6875` 更弱
- `partial_missing_negation_rate = 0.25`，也弱于 merged

但它也有两个确认无误的结论：

1. 当前 `Q4 GGUF` 已经能真实在 `gpu1` 上跑
2. 当前 `Q4` 并不是损坏模型，依然能稳定输出自然语言，且 `false_no_answer_rate = 0`

所以现阶段更准确的结论是：

- `Q4` 已经具备“能部署、能跑 GPU、速度可接受”的工程可行性
- 但它还不具备“和 merged 质量基本等价”的任务质量水平

---

## 五、本轮新增脚本

本轮新增：

- [evaluate_gguf_benchmark.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/evaluate_gguf_benchmark.py)
- [run_benchmark_qwen3_0p6b_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_0p6b_gpu1.sh)
- [run_benchmark_qwen3_1p7b_merged_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_1p7b_merged_gpu1.sh)
- [run_benchmark_qwen3_1p7b_q4_gguf_gpu1.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_benchmark_qwen3_1p7b_q4_gguf_gpu1.sh)

它们分别解决：

- `0.6B` 在 `gpu1` 上复测
- `1.7B merged` 的标准 HF benchmark
- `1.7B Q4 GGUF` 的 `llama.cpp` 自动评测
- `llama.cpp` GPU/CPU 后端状态写入结果 JSON

---

## 六、本轮结论

Part 23 做完后，之前“只有产物、没有完整 benchmark”的问题已经补齐。

当前最清晰的判断如下：

1. `0.6B LoRA`：已经跑过 benchmark，且在 `gpu1` 复测一致
2. `1.7B merged`：本轮 benchmark 已补齐，而且是三者里质量最稳的版本
3. `1.7B Q4 GGUF`：本轮 benchmark 已补齐，且已真实修通 `gpu1 + llama.cpp CUDA`
4. 但当前 `Q4 GGUF` 的质量回退不能忽略，尤其是：
   - `partial_hallucination`
   - `partial_hit_precision`
   - `must_answer_focus`

这意味着：

- 如果当前目标是“离线端侧先跑起来”，`Q4 GGUF` 可以继续推进
- 如果当前目标是“尽量保住 BrainDance 任务质量”，`1.7B merged` 仍然是更可靠的基线

---

## 七、下一步

接下来更值得做的是：

1. 对 `Q4` 再测更高一档量化，例如 `Q5_K_M`
2. 对比 `Q4` 与 `Q5` 的任务质量回撤是否明显收敛
3. 补一个更贴近 Flutter 端实际调用方式的单轮/流式推理测试
4. 决定移动端首发版本到底选：
   - `0.6B LoRA`
   - `1.7B Q4 GGUF`
   - 或远程 `1.7B merged`
