# 🧠 BrainDance 3DGS Engine - 环境配置指南

> 本文档详细记录 `gs_linux_backup` conda 环境的完整配置，包括混合安装方式（conda + pip + 源码编译）和 RTX 50 系列显卡的特殊处理。

---

## 1. 环境概览

| 项目 | 配置 |
|------|------|
| **环境名称** | `gs_linux_backup` |
| **Python 版本** | `3.10.19` |
| **Conda 位置** | `/home/ltx/miniconda3/envs/gs_linux_backup` |
| **PyTorch 版本** | `2.10.0.dev20251204+cu128` (每日构建版) |
| **CUDA 版本** | `12.8` |
| **CUDA 可用** | ✅ True |
| **安装方式** | 混合安装 (conda + pip + 源码编译) |

⚠️ **注意**: 由于 RTX 50 系列显卡支持不完善，部分组件需要特殊处理或降级。

---

## 2. 硬件要求与兼容性

### 测试环境
| 组件 | 配置 | 状态 |
|------|------|------|
| **显卡** | NVIDIA GeForce RTX 5070 | ⚠️ 部分兼容 |
| **显存** | 12GB | ✅ 够用 |
| **CUDA** | 12.8 | ✅ 兼容 |
| **内存** | 32GB | 推荐 |

### RTX 50 系列已知问题

| 问题 | 解决方案 |
|------|----------|
| PyTorch 每日构建版可能不稳定 | 使用 `--torch.compile` 时加 `torch.backends.cudnn.benchmark = False` |
| 部分 CUDA 扩展编译失败 | 使用预编译 wheel 或降级 CUDA |
| `xformers` 与 RTX 50 兼容问题 | 禁用 xformers 或使用 CPU 模式 |
| `pytorch3d` 编译失败 | 使用预编译版本或 Docker 镜像 |

---

## 3. 安装方式说明

本环境采用 **混合安装** 方式：

```
┌─────────────────────────────────────────────────────────┐
│  gs_linux_backup 环境                                    │
├─────────────────────────────────────────────────────────┤
│  📦 conda 安装                                           │
│     ├── Python 3.10                                     │
│     ├── CUDA toolkit 相关依赖                           │
│     ├── 编译类依赖 (ninja, cmake, llvm)                 │
│     └── 系统级库 (ffmpeg, colmap 等)                    │
├─────────────────────────────────────────────────────────┤
│  📦 pip 安装 (PyPI)                                     │
│     ├── 纯 Python 包 (supabase, dashscope 等)           │
│     └── 预编译 wheel (opencv-python, numpy 等)          │
├─────────────────────────────────────────────────────────┤
│  📦 pip 安装 (Git)                                      │
│     ├── nerfstudio (Git 源码安装)                      │
│     ├── pytorch3d (Git 源码安装)                       │
│     └── 其他需要最新特性的包                            │
├─────────────────────────────────────────────────────────┤
│  📦 pip -e (本地开发)                                   │
│     ├── sam3d_objects (/home/ltx/workspace/ai/sam-3d-objects) │
│     └── sharp (/home/ltx/projects/BrainDance/ai_engine/3dgs/src/libs/ml-sharp) │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 详细安装步骤

### 4.1 系统级依赖 (Ubuntu/Debian)

```bash
# 基础编译工具
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libomp-dev \
    llvm-dev \
    clangd

# 3D 相关系统库
sudo apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev

# COLMAP/GLOMAP 依赖
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    ceres-solver \
    libgflags-dev \
    libgoogle-glog-dev \
    libboost-all-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libglew-dev \
    libsuitesparse-dev
```

### 4.2 创建 Conda 环境

```bash
# 创建环境
conda create -n gs_linux_backup python=3.10 -y
conda activate gs_linux_backup

# 添加 conda-forge 源 (获取一些系统库)
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### 4.3 安装 PyTorch (CUDA 12.x)

```bash
# 安装 PyTorch 每日构建版 (支持最新 CUDA)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4.4 安装 nerfstudio (Git 源码)

```bash
# 从 GitHub 源码安装 (获取最新功能)
pip install git+https://github.com/nerfstudio-project/nerfstudio.git@50e0e3c70c775e89333256213363badbf074f29d

# nerfstudio 会自动安装大部分依赖，但可能版本冲突
# 如需固定版本:
# pip install nerfstudio==1.1.5
```

### 4.5 安装 pytorch3d (Git 源码)

```bash
# pytorch3d 需要从源码编译以获得最佳性能
pip install git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba

# 如果编译失败，可以使用预编译版本 (可能较旧)
# pip install pytorch3d==0.7.4 -f https://dl.fbaipublicfiles.com/pytorch3d/packages.html
```

### 4.6 安装 gsplat

```bash
# gsplat 高斯光栅化库
pip install gsplat==1.5.3

# 如果需要最新功能:
# pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### 4.7 安装 spconv (稀疏卷积)

```bash
# spconv 需要与 CUDA 版本匹配
pip install spconv-cu121==2.3.8
pip install cumm-cu121==0.7.11

# 如果安装失败，需要从源码编译:
# git clone https://github.com/traveller59/spconv.git
# cd spconv
# python setup.py build
# pip install .
```

### 4.8 安装 open3d 和 pymeshlab

```bash
# open3d
pip install open3d==0.19.0

# pymeshlab (需要系统安装 MeshLab)
# 1. 先下载 MeshLab: https://www.meshlab.net/#download
# 2. 安装 MeshLab
# 3. 然后安装 pymeshlab
pip install pymeshlab==2025.7
```

### 4.9 安装其他核心依赖

```bash
# 图像处理
pip install opencv-python==4.12.0.88
pip install opencv-python-headless==4.10.0.84
pip install pillow==12.0.0
pip install imageio==2.37.2
pip install imageio-ffmpeg==0.6.0
pip install av==16.0.1
pip install decord==0.6.0

# AI 模型
pip install ultralytics==8.3.240
pip install transformers==4.57.3
pip install sentence-transformers==5.1.2
pip install accelerate==1.12.0
pip install einops==0.8.1
pip install timm==1.0.22

# 云端依赖
pip install supabase==2.27.0
pip install python-dotenv==1.2.1
pip install dashscope==1.25.2
pip install openai==2.15.0

# 工具库
pip install loguru==0.7.3
pip install colorlog==6.10.1
pip install tqdm==4.67.1
pip install rich==13.9.4
pip install pyyaml==6.0.3
pip install numpy==2.1.3
pip install scipy==1.15.3
pip install plyfile==1.1.3
```

### 4.10 安装本地开发包 (Editable)

```bash
# SAM3D 单图引擎
# 路径: /home/ltx/workspace/ai/sam-3d-objects
pip install -e /home/ltx/workspace/ai/sam-3d-objects

# Apple SHARP 模型封装
# 路径: /home/ltx/projects/BrainDance/ai_engine/3dgs/src/libs/ml-sharp
pip install -e /home/ltx/projects/BrainDance/ai_engine/3dgs/src/libs/ml-sharp
```

---

## 5. RTX 50 系列特殊处理

### 5.1 禁用问题组件

如果遇到兼容性问题，可以在代码中添加：

```python
import torch

# 禁用可能导致问题的组件
torch.backends.cudnn.benchmark = False  # 禁用 cuDNN benchmark
torch.backends.cuda.matmul.allow_tf32 = False  # 禁用 TF32
torch.backends.cudnn.allow_tf32 = False

# 如果 xformers 有问题
import os
os.environ["DISABLE_XFORMERS"] = "1"
```

### 5.2 内存管理 (针对 12GB 显存)

```python
# 在训练前添加
torch.cuda.empty_cache()

# 限制 PyTorch 内存增长
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

### 5.3 降级方案

如果 RTX 50 完全不兼容，考虑：

```bash
# 方案 A: 使用 CPU 模式 (慢但稳定)
export CUDA_VISIBLE_DEVICES=""

# 方案 B: 使用 Docker 镜像 (已预配置)
# nvidia-docker run -it --rm \
#   -e CUDA_VISIBLE_DEVICES=0 \
#   pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# 方案 C: 降级到 CUDA 11.8
# pip uninstall torch torchvision torchaudio
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 6. 验证安装

### 6.1 基础检查

```bash
conda activate gs_linux_backup

# Python 版本
python --version
# Python 3.10.19

# PyTorch + CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 6.2 核心依赖检查

```bash
python -c "
import nerfstudio
import gsplat
import supabase
import open3d
import trimesh
import ultralytics
import dashscope
print('✓ All core dependencies loaded')
"
```

### 6.3 训练命令测试

```bash
# 测试 ns-train 是否可用
ns-train --help | head -20

# 测试 gsplat
python -c "import gsplat; print(f'gsplat version: {gsplat.__version__}')"
```

---

## 7. 常见问题解决

### Q1: 编译 spconv/cumm 失败

```bash
# 确保安装了完整的 CUDA toolkit
conda install -c conda-forge cuda-toolkit

# 或者使用 conda 安装
conda install -c nvidia cuda-nvcc

# 清理后重试
pip uninstall spconv cumm
pip install spconv-cu121==2.3.8
```

### Q2: pytorch3d 编译超时

```bash
# 使用预编译版本 (但版本较旧)
pip uninstall pytorch3d
pip install pytorch3d==0.7.4 -f https://dl.fbaipublicfiles.com/pytorch3d/packages.html
```

### Q3: nerfstudio 安装后导入失败

```bash
# 重新安装
pip uninstall nerfstudio -y
pip install git+https://github.com/nerfstudio-project/nerfstudio.git

# 检查依赖
pip check
```

### Q4: RTX 50 显存爆炸

```python
# 在代码中添加显存限制
import torch
torch.cuda.set_per_process_memory_fraction(0.8)  # 使用不超过 80% 显存
```

### Q5: CUDA out of memory

```python
# 降低批大小
# 启用梯度检查点
# 使用混合精度训练
```

---

## 8. 环境变量配置

在项目根目录创建 `.env` 文件：

```bash
# Supabase 配置
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=your_service_role_key_here
SUPABASE_BUCKET=braindance-assets
SUPABASE_TABLE=processing_tasks

# AI API 配置
DASHSCOPE_API_KEY=your_dashscope_key_here
OPENAI_API_KEY=your_openai_key_here

# 训练参数
MAX_IMAGES=500
TRAINING_ITERATIONS=15000
MIN_QUALITY_SCORE=40

# RTX 50 兼容设置
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
DISABLE_XFORMERS=1
```

---

## 9. 完整一键安装脚本

```bash
#!/bin/bash
# install_braindance_env.sh - BrainDance 3DGS 环境一键安装脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "BrainDance 3DGS 环境安装脚本"
echo "=========================================="

# 1. 检查系统依赖
echo "[1/6] 检查系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake ninja-build pkg-config \
    libomp-dev llvm-dev clangd \
    ffmpeg libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libegl1-mesa-dev

# 2. 创建 conda 环境
echo "[2/6] 创建 conda 环境..."
conda create -n gs_linux_backup python=3.10 -y
source /home/ltx/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup

# 3. 安装 PyTorch
echo "[3/6] 安装 PyTorch (CUDA 12.8)..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 4. 安装 3D 核心库
echo "[4/6] 安装 3D 核心库..."
pip install gsplat==1.5.3
pip install open3d==0.19.0
pip install pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba || \
    pip install pytorch3d==0.7.4 -f https://dl.fbaipublicfiles.com/pytorch3d/packages.html

# 5. 安装 nerfstudio
echo "[5/6] 安装 nerfstudio..."
pip install git+https://github.com/nerfstudio-project/nerfstudio.git@50e0e3c70c775e89333256213363badbf074f29d || \
    pip install nerfstudio==1.1.5

# 6. 安装其他依赖
echo "[6/6] 安装其他依赖..."
pip install \
    nerfstudio==1.1.5 \
    supabase==2.27.0 \
    python-dotenv==1.2.1 \
    dashscope==1.25.2 \
    ultralytics==8.3.240 \
    opencv-python==4.12.0.88 \
    imageio==2.37.2 \
    loguru==0.7.3 \
    plyfile==1.1.3

# 安装本地包
if [ -d "/home/ltx/workspace/ai/sam-3d-objects" ]; then
    pip install -e /home/ltx/workspace/ai/sam-3d-objects
fi

if [ -d "/home/ltx/projects/BrainDance/ai_engine/3dgs/src/libs/ml-sharp" ]; then
    pip install -e /home/ltx/projects/BrainDance/ai_engine/3dgs/src/libs/ml-sharp
fi

echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo "请运行: conda activate gs_linux_backup"
echo "然后验证: python -c 'import torch; print(torch.__version__)'"
```

---

## 10. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-01-20 | 初始文档，混合安装方式 (conda + pip + 源码) |
| 2026-01-20 | 添加 RTX 50 系列特殊处理章节 |
| 2026-01-20 | 添加完整一键安装脚本 |

---

## 11. 相关文档

- [项目主文档](../../README.md)
- [AI Engine 文档](../README.md)
- [3DGS 快速开始](./README.md)
- [Supabase 配置](../supabase/README.md)

---

<div align="center">

**BrainDance Team** - *物理世界注定走向无序，而我们在比特世界重建永恒。*

</div>
