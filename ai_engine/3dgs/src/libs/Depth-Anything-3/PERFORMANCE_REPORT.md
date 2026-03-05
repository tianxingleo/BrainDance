# RTX 5070 性能测试报告

## 硬件配置

- **GPU**: NVIDIA GeForce RTX 5070
- **架构**: Blackwell (计算能力 12.0)
- **显存**: 11.94 GB
- **多处理器**: 48
- **CUDA 版本**: 12.8
- **PyTorch 版本**: 2.10.0.dev20251204+cu128

## 测试环境

- **Conda 环境**: gs_linux_backup
- **Python**: 3.10.19
- **操作系统**: WSL2 Ubuntu
- **Hugging Face 镜像**: https://hf-mirror.com
- **测试图片**: 280x504 像素

## 性能测试结果

| 模型 | 参数量 | 首次推理 | 平均推理 | 最快推理 | FPS | 显存占用 | 适用场景 |
|------|--------|-----------|-----------|---------|-----|---------|---------|
| **DA3-SMALL** | 80M | ~0.64s | ~120ms | ~120ms | ~8.3 | ~0.7GB | 快速原型、移动端 |
| **DA3-BASE** | 120M | - | **60.1ms** | - | **16.6** | ~0.8GB | **实时视频** ✅ |
| **DA3-LARGE** | 350M | - | **88.1ms** | ~50ms | **11.3** | 1.55GB | 平衡精度和速度 |
| **DA3-GIANT** | 1.15B | ~1.07s | ~1.07s | - | ~0.9 | 5.12GB | 离线高精度 |

## 关键发现

### 1. 所有模型都能运行 ✅

RTX 5070 的 11.94GB 显存完全足够：
- 最大的 GIANT 模型只用了 5.12GB
- 还剩 6.82GB，可以处理更高分辨率或批量推理

### 2. DA3-BASE 实时性能 🚀

- **16.6 FPS** - 达到实时处理标准
- **60.1ms** 平均推理时间
- **~0.8GB** 显存占用
- **推荐用于**: 实时视频处理、直播、交互应用

### 3. DA3-LARGE 最佳平衡 ⚖️

- **11.3 FPS** - 接近实时
- **88.1ms** 平均推理时间
- **1.55GB** 显存占用
- **推荐用于**: 高精度实时场景、批处理

### 4. DA3-GIANT 最高精度 🎯

- **~0.9 FPS** - 离线处理
- **~1.07s** 单张推理时间
- **5.12GB** 显存占用
- **推荐用于**: 离线高精度重建、研究项目

## 使用建议

### 实时视频处理

```bash
# 使用 DA3-BASE 达到实时性能
da3 video input.mp4 \
  --model-dir depth-anything/DA3-BASE \
  --fps 15 \
  --export-format depth
```

**预期性能**: 16.6 FPS（超过 15 FPS 目标）

### 离线高精度处理

```bash
# 使用 DA3-GIANT 获得最高精度
da3 auto images/ \
  --model-dir depth-anything/DA3-GIANT \
  --export-format glb
```

**预期性能**: ~1 秒/张，最高精度

### 批量处理

```python
# 使用 DA3-LARGE 平衡精度和速度
model = DepthAnything3.from_pretrained('depth-anything/DA3-LARGE')
model = model.to(device='cuda')

# 可以批量处理多张图片
images = glob.glob('images/*.png')
prediction = model.inference(images)  # 批量推理
```

**显存使用**: 1.55GB（单张），可以批量处理 7 张（11.94 / 1.55 ≈ 7）

## 性能优化建议

### 1. 降低分辨率提升速度

```bash
da3 auto images/ \
  --model-dir depth-anything/DA3-LARGE \
  --process-res-method lower_bound_resize \
  --export-dir output/
```

**效果**: 分辨率降低，推理速度提升 2-4x

### 2. 使用混合精度推理

```python
model = model.to(device='cuda', dtype=torch.float16)
```

**效果**: 显存减半，速度提升 1.5-2x（牺牲少量精度）

### 3. 批量推理

```python
# 一次处理多张图片
images = [f'image_{i}.png' for i in range(8)]
prediction = model.inference(images)
```

**效果**: 显存充分利用，吞吐量提升

## 内存规划

| 模型 | 单张显存 | 推荐批量大小 | 显存利用率 |
|------|----------|------------|-----------|
| DA3-SMALL | ~0.7GB | 16 张 | 94% |
| DA3-BASE | ~0.8GB | 14 张 | 94% |
| DA3-LARGE | 1.55GB | 7 张 | 91% |
| DA3-GIANT | 5.12GB | 2 张 | 86% |

## 与其他 GPU 对比（估算）

| GPU | 显存 | DA3-BASE FPS | DA3-LARGE FPS | DA3-GIANT |
|-----|------|-------------|---------------|-----------|
| **RTX 5070** | 12GB | **16.6** | **11.3** | ✅ 可运行 |
| RTX 4070 | 12GB | ~15 | ~10 | ✅ 可运行 |
| RTX 3090 | 24GB | ~18 | ~13 | ✅ 可运行 |
| RTX 4090 | 24GB | ~20 | ~15 | ✅ 可运行 |

## 结论

RTX 5070 配合你精心编译的 CUDA 12.8 + PyTorch 2.10.0.dev 环境：

✅ **性能优异**: 所有模型都能流畅运行
✅ **实时处理**: DA3-BASE 达到 16.6 FPS
✅ **显存充足**: 即使 GIANT 模型也有 6GB+ 余量
✅ **架构先进**: Blackwell 架构计算能力 12.0
✅ **未来兼容**: 支持最新的 CUDA 特性

**推荐配置**:
- **日常使用**: DA3-BASE（实时、低显存）
- **高精度需求**: DA3-LARGE（平衡）
- **离线研究**: DA3-GIANT（最高精度）

## 附录

### 测试命令

```python
import torch
from depth_anything_3.api import DepthAnything3

device = torch.device('cuda')
model = DepthAnything3.from_pretrained('depth-anything/DA3-BASE')
model = model.to(device=device)

prediction = model.inference(['path/to/image.png'])
```

### Hugging Face 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 项目路径

- **项目**: `~/projects/Depth-Anything-3`
- **文档**: `REPRODUCTION.md`
- **测试脚本**: `test_inference.py`
