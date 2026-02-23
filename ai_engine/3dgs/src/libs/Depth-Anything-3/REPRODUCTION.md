# Depth Anything 3 复现指南

## 环境信息

- **操作系统**: WSL2 (Ubuntu)
- **Python 版本**: 3.10.19
- **CUDA 版本**: 12.4 / 12.8 (PyTorch 使用 12.8)
- **PyTorch 版本**: 2.10.0.dev20251204+cu128
- **Conda 环境**: gs_linux_backup

## 安装步骤

### 1. 克隆项目

```bash
cd ~/projects
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
```

### 2. 激活 conda 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup
```

### 3. 安装依赖

由于 moviepy==1.0.3 与当前环境有冲突，需要修改 pyproject.toml：

```toml
# 将 moviepy==1.0.3 改为 moviepy（不指定版本）
"moviepy",
```

然后安装：

```bash
pip install -e .
```

如果遇到 moviepy 安装失败，可以单独安装新版本：

```bash
pip install moviepy
```

### 4. 修复 moviepy 导入问题

修改 `src/depth_anything_3/utils/export/gs.py`：

```python
# 将
import moviepy.editor as mpy

# 改为
import moviepy as mpy
```

### 5. 安装 gsplat（可选，用于 3D Gaussian Splatting）

```bash
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### 6. 安装 Gradio（可选，用于 Web UI）

```bash
pip install -e ".[app]"
```

## 使用方法

### Python API

```python
import torch
from depth_anything_3.api import DepthAnything3

# 加载模型
device = torch.device('cuda')
model = DepthAnything3.from_pretrained('depth-anything/DA3-SMALL')
model = model.to(device=device)

# 推理
prediction = model.inference(['path/to/image.png'])

# 访问结果
print(prediction.depth.shape)        # [N, H, W]
print(prediction.conf.shape)         # [N, H, W]
print(prediction.extrinsics.shape)  # [N, 3, 4]
print(prediction.intrinsics.shape)   # [N, 3, 3]
```

### 命令行接口 (CLI)

```bash
# 自动模式
da3 auto assets/examples/SOH --export-format glb

# 处理单张图片
da3 image path/to/image.png --export-dir output/

# 处理图片目录
da3 images assets/examples/SOH --export-dir output/

# 处理视频
da3 video assets/examples/robot_unitree.mp4 --fps 15

# 启动后端服务
da3 backend --model-dir depth-anything/DA3-SMALL --gallery-dir workspace/gallery

# 启动 Gradio Web UI
da3 gradio
```

## 重要提示

### 使用 Hugging Face 镜像

由于网络问题，建议使用 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或在 Python 代码中：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

或在命令行中：

```bash
HF_ENDPOINT=https://hf-mirror.com python your_script.py
```

### 可用模型

根据硬件选择合适的模型：

- **DA3-SMALL**: 80M 参数，最快速度
- **DA3-BASE**: 120M 参数，平衡性能
- **DA3-LARGE**: 350M 参数，更好的精度
- **DA3-GIANT**: 1.15B 参数，最高精度
- **DA3NESTED-GIANT-LARGE-1.1**: 1.40B 参数，带公制深度

完整模型列表见：[Model Cards](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards)

## 测试

运行测试脚本：

```bash
cd ~/projects/Depth-Anything-3
HF_ENDPOINT=https://hf-mirror.com python test_inference.py
```

测试脚本会：
1. 加载 DA3-SMALL 模型
2. 处理 `assets/examples/SOH/` 中的所有图片
3. 保存深度图、置信度图和处理后的图片到 `output/` 目录

## 常见问题

### 1. ModuleNotFoundError: No module named 'moviepy.editor'

**原因**: moviepy 2.x 版本不再有 editor 模块

**解决**: 修改导入语句为 `import moviepy as mpy`

### 2. Network error when downloading models

**原因**: 无法访问 huggingface.co

**解决**: 使用镜像 `export HF_ENDPOINT=https://hf-mirror.com`

### 3. CUDA out of memory

**原因**: 模型过大或图片分辨率过高

**解决**:
- 使用更小的模型（DA3-SMALL 或 DA3-BASE）
- 降低图片分辨率
- 使用 `--process-res-method lower_bound_resize` 参数

## 性能参考

在测试图片 (280x504) 上的推理时间（DA3-SMALL）：

- 第一次推理: ~0.64 秒（包含模型加载）
- 后续推理: ~0.12 秒

## 输出格式说明

### Prediction 对象

- `processed_images`: 处理后的 RGB 图像 [N, H, W, 3]
- `depth`: 深度图 [N, H, W]
- `conf`: 置信度图 [N, H, W]
- `extrinsics`: 相机外参 [N, 3, 4]
- `intrinsics`: 相机内参 [N, 3, 3]

### 深度值

- **相对深度**: DA3-SMALL, DA3-BASE, DA3-LARGE, DA3-GIANT
- **公制深度（米）**: DA3METRIC-LARGE, DA3NESTED-GIANT-LARGE-1.1

对于公制深度模型，可以使用以下公式转换：

```python
metric_depth = focal * net_output / 300.
```

其中 `focal` 是焦距（像素单位，通常是内参矩阵 K 中 fx 和 fy 的平均值）。

## 参考资料

- [GitHub 仓库](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [模型卡片](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards)
- [CLI 文档](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/CLI.md)
- [API 文档](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)
