# Qwen3-1.7B 微调实践记录 Part 22

## Part 22：1.7B LoRA 合并、GGUF 转换与 Q4_K_M 量化落地

### 本 part 目标

把 `Qwen3-1.7B + LoRA` 从“训练完成”继续推进到“端侧部署前可直接使用的模型产物”，完成：

- merge 到独立 Hugging Face 模型目录
- 安装并编译 `llama.cpp`
- 真实执行 GGUF 转换
- 真实执行 `Q4_K_M` 量化

---

## 一、本轮执行内容

### 1. 真实 merge 1.7B LoRA

使用：

- [run_merge_qwen3_gpu0.sh](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_merge_qwen3_gpu0.sh)

输入：

- base model：`Qwen/Qwen3-1.7B`
- adapter：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`

输出目录：

- [qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0](/ltx-data/BrainDance/ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0)

merge metadata：

- `torch_dtype = bfloat16`
- `device_map = cpu`
- `safe_serialization = true`

实际结果：

- 生成独立 `model.safetensors`
- 目录总体积约 `3.3G`

这一步证明：

- 当前 LoRA 产物可以顺利和底模融合
- 部署时不必强依赖运行时动态挂 adapter

### 2. 引入 llama.cpp 工具链

克隆到：

- `ai_engine/finetune_qwen3/tools/llama.cpp`

当前提交：

- `3306dbaef`

然后本地编译：

```bash
cmake -S ai_engine/finetune_qwen3/tools/llama.cpp \
  -B ai_engine/finetune_qwen3/tools/llama.cpp/build \
  -DGGML_CUDA=OFF

cmake --build ai_engine/finetune_qwen3/tools/llama.cpp/build -j 8 \
  --target llama-quantize llama-cli
```

编译结果：

- `llama-quantize`
- `llama-cli`

都已生成。

### 3. 真实执行 GGUF 转换与量化

执行：

```bash
conda run -n qwen3_ft python ai_engine/finetune_qwen3/scripts/prepare_quantization_artifacts.py \
  --merged_model_dir ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0 \
  --output_dir ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0 \
  --llama_cpp_dir ai_engine/finetune_qwen3/tools/llama.cpp \
  --quant_type Q4_K_M \
  --execute
```

输出目录：

- [qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0](/ltx-data/BrainDance/ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0)

生成产物：

- `model-f16.gguf`
- `model-f16-q4_k_m.gguf`
- `quantization_plan.json`
- `run_quantization.sh`

### 4. 量化结果

从量化日志可见：

- 原始 GGUF 大小：
  - `3281.97 MiB`
- `Q4_K_M` 后大小：
  - `1050.43 MiB`

即：

- 约从 `3.3G` 压到 `1.1G`
- 压缩后约 `5.12 BPW`

这已经进入移动端/端侧部署更现实的体积区间。

### 5. 最小加载验证

使用：

- `llama-cli`

对量化后 GGUF 做了最小加载测试，结果是：

- 模型可以被正常加载
- 能够开始生成回答

说明：

- 转换和量化产物不是损坏文件
- 已具备进一步接入 `llama.cpp` / Flutter / 移动端的前提

需要注意的是：

- `llama-cli` 默认会进入对话模式
- 如果直接拿它做子进程验证，输出会很多
- 后续接正式产品时应优先使用更严格的单轮调用方式，或直接用 SDK / 原生绑定

---

## 二、本轮结论

Part 22 完成后，`1.7B` 路线已经真正补齐到了部署前产物层：

1. LoRA 已成功 merge 为独立 HF 模型目录
2. `llama.cpp` 工具链已在项目内落地
3. GGUF 转换已真实完成
4. `Q4_K_M` 量化已真实完成
5. 最终可部署量化模型体积约 `1.1G`

这意味着当前项目不再只是“能训练 adapter”，而是已经拿到了：

- adapter 版本
- merged HF 版本
- GGUF fp16 版本
- GGUF `Q4_K_M` 版本

---

## 三、下一步

接下来更值得做的不是继续手工操作，而是收口成部署方案：

1. 明确 Flutter / `llama.cpp` 接入方式
2. 设计模型下载与校验方案
3. 决定上线版本优先用：
   - `0.6B LoRA`
   - 或 `1.7B Q4_K_M`
4. 做端侧延迟、内存、发热实测
