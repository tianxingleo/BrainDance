<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">Depth Anything 3: Recovering the Visual Space from Any Views</h1>
<!-- <h2 style="border-top: none; margin-top: 3px;">Recovering the Visual Space from Any Views</h2> -->


[**Haotong Lin**](https://haotongl.github.io/)<sup>&ast;</sup> · [**Sili Chen**](https://github.com/SiliChen321)<sup>&ast;</sup> · [**Jun Hao Liew**](https://liewjunhao.github.io/)<sup>&ast;</sup> · [**Donny Y. Chen**](https://donydchen.github.io)<sup>&ast;</sup> · [**Zhenyu Li**](https://zhyever.github.io/) · [**Guang Shi**](https://scholar.google.com/citations?user=MjXxWbUAAAAJ&hl=en) · [**Jiashi Feng**](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)
<br>
[**Bingyi Kang**](https://bingykang.github.io/)<sup>&ast;&dagger;</sup>

&dagger;project lead&emsp;&ast;Equal Contribution

<a href="https://arxiv.org/abs/2511.10647"><img src='https://img.shields.io/badge/arXiv-Depth Anything 3-red' alt='Paper PDF'></a>
<a href='https://depth-anything-3.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything 3-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<!-- <a href='https://huggingface.co/datasets/depth-anything/VGB'><img src='https://img.shields.io/badge/Benchmark-VisGeo-yellow' alt='Benchmark'></a> -->
<!-- <a href='https://huggingface.co/datasets/depth-anything/data'><img src='https://img.shields.io/badge/Benchmark-xxx-yellow' alt='Data'></a> -->

</div>

This work presents **Depth Anything 3 (DA3)**, a model that predicts spatially consistent geometry from
arbitrary visual inputs, with or without known camera poses.
In pursuit of minimal modeling, DA3 yields two key insights:
- 💎 A **single plain transformer** (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization,
- ✨ A singular **depth-ray representation** obviates the need for complex multi-task learning.

🏆 DA3 significantly outperforms
[DA2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation,
and [VGGT](https://github.com/facebookresearch/vggt) for multi-view depth estimation and pose estimation.
All models are trained exclusively on **public academic datasets**.

<!-- <p align="center">
  <img src="assets/images/da3_teaser.png" alt="Depth Anything 3" width="100%">
</p> -->
<p align="center">
  <img src="assets/images/demo320-2.gif" alt="Depth Anything 3 - Left" width="70%">
</p>
<p align="center">
  <img src="assets/images/da3_radar.png" alt="Depth Anything 3" width="100%">
</p>


## 📰 News
- **18-02-2026:** 🎯 **智能对齐脚本 V7 发布**！采用 XY 紧凑度判定算法，完美解决桌面物体 vs 桌底伪影的区分问题。详见 [对齐 Pipeline 指南](docs/ALIGNMENT_PIPELINE_GUIDE.md)。
- **18-02-2026:** 🚀 **DA3 → 3DGS 双重对齐 Pipeline** 上线！结合 COLMAP 和 Open3D 的优势，实现训练前粗对齐 + 训练后精细校正。
- **11-12-2025:** 🚀 New models and [**DA3-Streaming**](da3_streaming/README.md) released! Handle ultra-long video sequence inference with less than 12GB GPU memory via sliding-window streaming inference. Special thanks to [Kai Deng](https://github.com/DengKaiCQ) for his contribution to DA3-Streaming!
- **08-12-2025:** 📊 [Benchmark evaluation pipeline](docs/BENCHMARK.md) released! Evaluate pose estimation & 3D reconstruction on 5 datasets.
- **30-11-2025:** Add [`use_ray_pose`](#use-ray-pose) and [`ref_view_strategy`](docs/funcs/ref_view_strategy.md) (reference view selection for multi-view inputs).
- **25-11-2025:** Add [Awesome DA3 Projects](#-awesome-da3-projects), a community-driven section featuring DA3-based applications.
- **14-11-2025:** Paper, project page, code and models are all released.

## ✨ Highlights

### 🏆 Model Zoo
We release three series of models, each tailored for specific use cases in visual geometry.

- 🌟 **DA3 Main Series** (`DA3-Giant`, `DA3-Large`, `DA3-Base`, `DA3-Small`) These are our flagship foundation models, trained with a unified depth-ray representation. By varying the input configuration, a single model can perform a wide range of tasks:
  + 🌊 **Monocular Depth Estimation**: Predicts a depth map from a single RGB image.
  + 🌊 **Multi-View Depth Estimation**: Generates consistent depth maps from multiple images for high-quality fusion.
  + 🎯 **Pose-Conditioned Depth Estimation**: Achieves superior depth consistency when camera poses are provided as input.
  + 📷 **Camera Pose Estimation**:  Estimates camera extrinsics and intrinsics from one or more images.
  + 🟡 **3D Gaussian Estimation**: Directly predicts 3D Gaussians, enabling high-fidelity novel view synthesis.

- 📐 **DA3 Metric Series** (`DA3Metric-Large`) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

- 🔍 **DA3 Monocular Series** (`DA3Mono-Large`). A dedicated model for high-quality relative monocular depth estimation. Unlike disparity-based models (e.g.,  [Depth Anything 2](https://github.com/DepthAnything/Depth-Anything-V2)), it directly predicts depth, resulting in superior geometric accuracy.

🔗 Leveraging these available models, we developed a **nested series** (`DA3Nested-Giant-Large`). This series combines a any-view giant model with a metric model to reconstruct visual geometry at a real-world metric scale.

### 🛠️ Codebase Features
Our repository is designed to be a powerful and user-friendly toolkit for both practical application and future research.
- 🎨 **Interactive Web UI & Gallery**: Visualize model outputs and compare results with an easy-to-use Gradio-based web interface.
- ⚡ **Flexible Command-Line Interface (CLI)**: Powerful and scriptable CLI for batch processing and integration into custom workflows.
- 💾 **Multiple Export Formats**: Save your results in various formats, including `glb`, `npz`, depth images, `ply`, 3DGS videos, etc, to seamlessly connect with other tools.
- 🔧 **Extensible and Modular Design**: The codebase is structured to facilitate future research and the integration of new models or functionalities.


<!-- ### 🎯 Visual Geometry Benchmark
We introduce a new benchmark to rigorously evaluate geometry prediction models on three key tasks: pose estimation, 3D reconstruction, and visual rendering (novel view synthesis) quality.

- 🔄 **Broad Model Compatibility**: Our benchmark is designed to be versatile, supporting the evaluation of various models, including both monocular and multi-view depth estimation approaches.
- 🔬 **Robust Evaluation Pipeline**: We provide a standardized pipeline featuring RANSAC-based pose alignment, TSDF fusion for dense reconstruction, and a principled view selection strategy for novel view synthesis.
- 📊 **Standardized Metrics**: Performance is measured using established metrics: AUC for pose accuracy, F1-score and Chamfer Distance for reconstruction, and PSNR/SSIM/LPIPS for rendering quality.
- 🌍 **Diverse and Challenging Datasets**: The benchmark spans a wide range of scenes from datasets like HiRoom, ETH3D, DTU, 7Scenes, ScanNet++, DL3DV, Tanks and Temples, and MegaDepth. -->


## 🚀 Quick Start

### 📦 Installation

```bash
pip install xformers torch\>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
```

For detailed model information, please refer to the [Model Cards](#-model-cards) section below.

### 💻 Basic Usage

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)
```

```bash

export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend

# CLI video processing with feature visualization
da3 video assets/examples/robot_unitree.mp4 \
    --fps 15 \
    --use-backend \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/robo \
    --export-format glb-feat_vis \
    --feat-vis-fps 15 \
    --process-res-method lower_bound_resize \
    --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_CLI/SOH \
    --model-dir ${MODEL_DIR}

```

The model architecture is defined in [`DepthAnything3Net`](src/depth_anything_3/model/da3.py), and specified with a Yaml config file located at [`src/depth_anything_3/configs`](src/depth_anything_3/configs). The input and output processing are handled by [`DepthAnything3`](src/depth_anything_3/api.py). To customize the model architecture, simply create a new config file (*e.g.*, `path/to/new/config`) as:

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params

  name: vitb
  out_layers: [5, 7, 9, 11]
  alt_start: 4
  qknorm_start: 4
  rope_start: 4
  cat_token: True

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params

  dim_in: &head_dim_in 1536
  output_dim: 2
  features: &head_features 128
  out_channels: &head_out_channels [96, 192, 384, 768]
```

Then, the model can be created with the following code snippet.
```python
from depth_anything_3.cfg import create_object, load_config

Model = create_object(load_config("path/to/new/config"))
```



## 🛠️ Community Enhancements

这个 fork 在原仓库基础上，新增了完整的 3D 重建生态系统，让用户可以从视频直接生成高质量的 3D 模型。

### 🌟 What's New (vs Original Repository)

**完整 3D 重建生态系统**：

1. **🎯 智能点云对齐系统**（新增，2026-02-18）
   - **V7 智能对齐脚本**：XY 紧凑度判定（推荐用于桌面物体场景）
     - 选择"分布更聚焦"的一侧作为正面
     - 自动计算场景尺度，动态调整参数
     - 支持毫米/米/任意比例单位
   - **V4 智能对齐脚本**：DBSCAN 聚类判定
     - 自适应尺度 + 聚类连通性分析
     - 适用于复杂伪影场景
   - **DA3 → 3DGS 双重对齐 Pipeline**：
     - 训练前 COLMAP 对齐（曼哈顿世界假设）
     - 训练后 Open3D RANSAC 精细校正
     - 智能跳过机制，避免过度旋转
   - 完整文档：[ALIGNMENT_PIPELINE_GUIDE.md](docs/ALIGNMENT_PIPELINE_GUIDE.md)

2. **🎯 DA3 → SuGaR Pipeline**（新增）
   - 一键将 DA3 输出转换为 SuGaR 可用的 COLMAP 格式
   - 自动完成 4 个步骤：格式转换 → 二进制转换 → 数据整理 → SuGaR 训练
   - 支持快速预览（15-30分钟）、标准质量（1小时）、高质量（2小时）
   - 输出可直接在 Blender 中编辑或在线查看
   - 完整文档：[DA3_TO_SUGAR_QUICKSTART.md](DA3_TO_SUGAR_QUICKSTART.md)

3. **🚀 DA3 → DN-Splatter Pipeline**（新增）
   - 端到端 pipeline：DA3 → DN-Splatter → 3DGS PLY
   - 支持深度约束和法线约束，有效消除白墙漂浮物
   - 自动训练 30000 步并导出标准 PLY 格式
   - 内存优化：支持 RTX 5070，<12GB VRAM

4. **📹 Video Processing & Streaming**（增强）
   - 批量视频深度估计工具（支持 720p/1080p）
   - DA3-Streaming 支持超长视频（<12GB VRAM）
   - 滑动窗口推理 + 循环闭包检测
   - 生成高质量点云（PLY 格式）

5. **🔧 COLMAP Integration**（增强）
   - 改进的 DA3 到 COLMAP 格式转换工具
   - 支持文本格式和二进制格式
   - 完整的参数验证和错误提示
   - 兼容 SuGaR、DN-Splatter 等下游工具

6. **⚡ Performance Benchmarking**（新增）
   - 全面的性能测试工具（测试所有 DA3 模型）
   - RTX 5070 优化结果

---

### 📦 Complete Toolset

#### 🎯 3D Gaussian Splatting Pipelines

**智能点云对齐系统**（新增，2026-02-18）
- `align_target_object_plyv7.py` - **V7 智能对齐脚本**（推荐）
  - XY 紧凑度判定，选择"分布更聚焦"的一侧
  - 适用于桌面物体 vs 桌底伪影的区分
  - 自适应尺度计算，支持任意单位模型
- `align_target_object_ply.py` - **V4 智能对齐脚本**
  - DBSCAN 聚类分析，智能判定正反方向
  - 适用于复杂伪影场景
- `run_da3_to_3dgs_aligned.py` - **双重对齐 Pipeline**
  - COLMAP + Open3D 双重对齐
  - 训练前粗对齐 + 训练后精细校正
- `batch_align_existing_ply.py` - 批量扶正已有 PLY
- `auto_align_ply.py` - 独立扶正工具
- 文档：[ALIGNMENT_PIPELINE_GUIDE.md](docs/ALIGNMENT_PIPELINE_GUIDE.md)

**DA3 × SuGaR**（新增，推荐用于高质量重建）
- `da3_to_sugar_pipeline.sh` - 一键完整 pipeline
- `convert_da3_to_colmap.py` - DA3 输出转 COLMAP 文本格式（已改进）
- `colmap_text_to_binary.py` - COLMAP 文本转二进制格式
- 文档：
  - [DA3_TO_SUGAR_QUICKSTART.md](DA3_TO_SUGAR_QUICKSTART.md) - 快速开始
  - [DA3_TO_SUGAR_PIPELINE.md](DA3_TO_SUGAR_PIPELINE.md) - 完整指南
  - [DA3_TO_SUGAR_IMPLEMENTATION.md](DA3_TO_SUGAR_IMPLEMENTATION.md) - 实现细节

**DA3 × DN-Splatter**（新增，推荐用于快速重建）
- `run_da3_to_dn_splatter_pipeline.py` - 端到端 pipeline
- `run_da3_to_dn_splatter.py` - 独立转换工具
- `run_direct_dn_splatter.py` - 直接 DN-Splatter 训练
- `batch_export_ply.py` - 批量导出 PLY
- 文档：[DN_SPLATTER_PIPELINE_GUIDE.md](docs/DN_SPLATTER_PIPELINE_GUIDE.md)

**Classic 3DGS**（新增，直接生成）
- `generate_3dgs.py` - 直接从 DA3 输出生成 3DGS
- `run_da3_3dgs.sh` - 自动化 pipeline
- `run_gradio_direct.sh` - Gradio UI
- 文档：[DA3_3DGS_GUIDE.md](DA3_3DGS_GUIDE.md)

#### 📹 Video Processing & Streaming

**Video Depth Estimation**
- `process_video.py` - 批量视频深度估计
  - 支持可配置 FPS 提取
  - 支持 720p、1080p 分辨率
  - 导出深度图、置信度图、处理后的帧

**Long Video Streaming**
- `run_sugar_streaming.sh` - DA3-Streaming 处理超长视频
  - 分块处理 + 重叠（内存高效）
  - 循环闭包检测
  - 生成 PLY 点云
- 文档：[DA3_STREAMING_GUIDE.md](DA3_STREAMING_GUIDE.md)

#### 🔧 Format Conversion & Integration

**COLMAP Integration**（增强）
- `convert_da3_to_colmap.py` - DA3 输出转 COLMAP 格式（已改进）
  - ✅ 完整的参数验证和错误提示
  - ✅ 支持中文注释和输出
  - ✅ 自动符号链接或复制图像
  - ✅ 支持文本格式和二进制格式转换
- `colmap_text_to_binary.py` - COLMAP 文本转二进制

**其他集成**
- `run_da3_glomap_pipeline.py` - GLOMAP 集成 pipeline
- `run_da3_to_3dgs_direct.py` - 直接 DA3 到 3DGS 转换

#### 📊 Performance & Testing

**Benchmarking**
- `benchmark.py` - 全面的性能测试工具
  - 测试所有 DA3 模型大小（SMALL, BASE, LARGE, GIANT）
  - 测量推理时间、FPS、VRAM 使用
- 文档：[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)

**Testing & Utilities**
- `test_inference.py` - 快速推理测试
- `inspect_npz.py` - NPZ 文件检查工具
- `cleaned_help.txt`, `ns_help.txt` - 额外文档

#### 📖 Installation & Documentation

**Setup Guides**
- [REPRODUCTION.md](REPRODUCTION.md) - 完整安装指南
  - WSL2 + CUDA 12.8 环境设置
  - Bug 修复（moviepy 导入、HF 镜像）
  - 模型下载说明
- [REPRODUCTION_SUMMARY.md](REPRODUCTION_SUMMARY.md) - 快速参考

**Model Weights**
- `weights/model.safetensors` - DA3 模型检查点
- `weights/config.json` - 模型配置
- `weights/dino_salad.ckpt` - SALAD 权重

---

### 📁 输出目录结构

#### DA3 主输出目录

```
output/
├── sugar_streaming/              # SuGaR streaming 输出（默认）
│   ├── camera_poses.ply         # 相机位姿点云
│   ├── camera_poses.txt         # 相机位姿文本
│   ├── intrinsic.txt            # 相机内参
│   ├── loop_closures.txt        # 闭环检测结果
│   ├── colmap_data/            # COLMAP 数据（二进制）
│   ├── colmap_text/            # COLMAP 数据（文本格式）
│   ├── glomap_ws/              # GLOMAP 工作空间
│   ├── pcd/                   # 点云文件
│   ├── extracted/              # 提取的图像
│   ├── results_output/         # 结果输出
│   ├── da3_3dgs_pipeline/     # 3DGS 训练输出
│   ├── da3_3dgs_colmap_aligned_pipeline/   # COLMAP 对齐后的3DGS
│   └── da3_3dgs_aligned_pipeline/         # 融合对齐后的3DGS
├── video_depth/                # 视频深度估计输出
├── video_test/                # 测试视频输出
├── sugar_video/               # SuGaR 视频处理输出
├── da3_3dgs/                # DA3 3DGS 输出
├── quick_mesh/               # 快速mesh输出
├── da3_dn_splatter_dataset/    # DA3→DN-Splatter 数据集
└── da3_dn_splatter_output/    # DA3→DN-Splatter 输出
```

#### 跨项目输出

- **SuGaR**: `~/projects/SuGaR/output/3dgs/` - DA3→3DGS PLY
- **DN-Splatter**: `~/projects/dn-splatter/output/` - DN-Splatter 训练输出

### 🔗 数据流向与集成

#### DA3 → 3DGS Pipeline

```
视频 → DA3 (深度估计+位姿估计) → colmap_text/ → 3DGS 训练 → PLY 文件
```

#### 跨项目集成

```
DA3 output/
├── colmap_text/              # 文本格式COLMAP数据
│   ├───► SuGaR/output/3dgs/     (DA3→3DGS PLY)
│   └───► dn-splatter/            # 深度先验
└── camera_poses.txt
```

#### 输出数据层级

```
基础层 (DA3)
  ├── colmap_text/              # 文本格式COLMAP数据
  ├── colmap_data/              # 二进制格式COLMAP数据
  ├── camera_poses.txt         # 相机位姿文本
  ├── pcd/                     # 点云文件
  └── depth图像               # 深度估计结果

中级层 (3DGS训练)
  ├── da3_3dgs_pipeline/       # 纯3DGS训练输出
  ├── da3_2dgs_pipeline/       # 2DGS训练输出
  └── da3_dn_splatter_dataset/   # DN-Splatter数据集

高级层 (高质量输出)
  ├── SuGaR/output/             # SuGaR训练输出（跨项目）
  │   ├── vanilla_gs/           # Vanilla 3DGS
  │   ├── coarse/               # Coarse SuGaR
  │   ├── refined/              # Refined SuGaR
  │   └── refined_mesh/         # Mesh输出
  └── dn-splatter/output/       # DN-Splatter输出（跨项目）

后处理层 (对齐和优化)
  ├── da3_3dgs_aligned_pipeline/        # COLMAP对齐
  ├── da3_3dgs_colmap_aligned_pipeline/ # Open3D对齐
  └── da3_3dgs_aligned_pipeline/       # 双重对齐
```

### 🎯 快速选择指南

#### 根据需求选择脚本

| 需求 | 推荐脚本 | 预计时间 | 说明 |
|------|---------|---------|------|
| 最快速度获取3DGS | `da3_to_3dgs.sh` | 15-30分钟 | 纯3DGS训练 |
| 高质量几何 | `da3_to_2dgs.sh` | 15-30分钟 | 2DGS，几何质量更好 |
| 最高质量+Mesh | `da3_to_sugar_pipeline.sh` | 2-3小时 | SuGaR完整流程 |
| 推荐方案（双重对齐） | `run_da3_to_3dgs_aligned.py` | 15-30分钟 | COLMAP+Open3D |
| 批量扶正已有PLY | `batch_align_existing_ply.py` | - | 批量对齐 |

#### 对齐脚本选择

| 场景 | 推荐脚本 | 原因 |
|------|---------|------|
| 桌面物体场景 | `align_target_object_plyv7.py` | XY紧凑度判定，区分物体vs桌底 |
| 复杂伪影场景 | `align_target_object_ply.py` | DBSCAN聚类，处理伪影 |
| 训练前粗对齐 | `da3_to_3dgs_aligned_colmap.sh` | COLMAP model_aligner |
| 训练后精细校正 | `da3_to_3dgs_aligned_open3d.sh` | Open3D RANSAC |
| 双重对齐（推荐） | `run_da3_to_3dgs_aligned.py` | COLMAP（训练前）+ Open3D（训练后） |

---

### 🚀 Getting Started

#### 快速开始 - 推荐流程

**1. 基础推理测试**
```bash
python test_inference.py
```

**2. SuGaR Pipeline（推荐用于高质量 3D 重建）**
```bash
cd /home/ltx/projects/Depth-Anything-3

# 快速预览（约30分钟）
./da3_to_sugar_pipeline.sh output/sugar_streaming my_scene dn_consistency short false true

# 标准质量（约1小时）
./da3_to_sugar_pipeline.sh output/sugar_streaming my_scene dn_consistency short true false
```

**3. DN-Splatter Pipeline（推荐用于快速重建）**
```bash
python run_da3_to_dn_splatter_pipeline.py
```

**4. 视频深度估计**
```bash
# 编辑 process_video.py 中的 VIDEO_PATH
python process_video.py
```

**5. 长视频流处理**
```bash
# 编辑 run_sugar_streaming.sh 中的 VIDEO_PATH
bash run_sugar_streaming.sh
```

**6. 性能测试**
```bash
python benchmark.py
```

---

### ✨ Features & Benefits

✅ **完整生态** - 从视频到 3D 模型的完整 pipeline
✅ **内存高效** - DA3-Streaming 支持超长视频（<12GB VRAM）
✅ **循环闭包** - SIM3 优化防止漂移
✅ **批量处理** - 自动化视频帧提取和处理
✅ **COLMAP 就绪** - 直接格式转换用于下游工具
✅ **RTX 5070 优化** - 针对最新 GPU 架构优化
✅ **中文支持** - 工具和文档包含中文注释
✅ **详细文档** - 每个功能都有完整的使用指南

---

**Fork by**: [@tianxingleo](https://github.com/tianxingleo)
**Last Updated**: 2026-02-18
**Tested on**: WSL2 Ubuntu, CUDA 12.8, RTX 5070
**License**: Inherits from original repository

## 📚 Useful Documentation

### Pipeline & Usage
- 🎯 [点云自动扶正 Pipeline 指南](docs/ALIGNMENT_PIPELINE_GUIDE.md) - **智能对齐系统完整文档**
- 🎯 [DA3 → SuGaR Pipeline Guide](DA3_TO_SUGAR_PIPELINE.md) - 完整pipeline文档
- ⚡ [DA3 → SuGaR Quick Start](DA3_TO_SUGAR_QUICKSTART.md) - 快速开始指南
- 🎯 [SuGaR Modes Technical Details](SUGAR_MODES_TECHNICAL_DETAILS.md) - 模式技术对比
- 🧠 [SDF Regularization Guide](SDF_REGULARIZATION_GUIDE.md) - **SDF约束详细使用指南**
- 📋 [SuGaR Official Default Iterations](SUGAR_OFFICIAL_DEFAULT_ITERATIONS.md) - 官方默认迭代次数

### Core Documentation
- 🖥️ [Command Line Interface](docs/CLI.md)
- 📑 [Python API](docs/API.md)
- 📊 [Benchmark Evaluation](docs/BENCHMARK.md)

## 🗂️ Model Cards

Generally, you should observe that DA3-LARGE achieves comparable results to VGGT.

The Nested series uses an Any-view model to estimate pose and depth, and a monocular metric depth estimator for scaling. 

⚠️ Models with the `-1.1` suffix are retrained after fixing a training bug; prefer these refreshed checkpoints. The original `DA3NESTED-GIANT-LARGE`, `DA3-GIANT`, and `DA3-LARGE` remain available but are deprecated. You could expect much better performance for street scenes with the `-1.1` models.

| 🗃️ Model Name                  | 📏 Params | 📊 Rel. Depth | 📷 Pose Est. | 🧭 Pose Cond. | 🎨 GS | 📐 Met. Depth | ☁️ Sky Seg | 📄 License     |
|-------------------------------|-----------|---------------|--------------|---------------|-------|---------------|-----------|----------------|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE)                     | 0.12B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL)                     | 0.08B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)              | 0.35B     | ✅             |              |               |       | ✅             | ✅         | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE)                | 0.35B     | ✅             |              |               |               |       | ✅         | Apache 2.0     |


## ❓ FAQ

- **Monocular Metric Depth**: To obtain metric depth in meters from `DA3METRIC-LARGE`, use `metric_depth = focal * net_output / 300.`, where `focal` is the focal length in pixels (typically the average of fx and fy from the camera intrinsic matrix K). Note that the output from `DA3NESTED-GIANT-LARGE` is already in meters.

- <a id="use-ray-pose"></a>**Ray Head (`use_ray_pose`)**:  Our API and CLI support `use_ray_pose` arg, which means that the model will derive camera pose from ray head, which is generally slightly slower, but more accurate. Note that the default is `False` for faster inference speed. 
  <details>
  <summary>AUC3 Results for DA3NESTED-GIANT-LARGE</summary>
  
  | Model | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ | 
  |-------|------|-------|-----|---------|-----------|
  | `ray_head` | 84.4 | 52.6 | 93.9 | 29.5 | 89.4 |
  | `cam_head` | 80.3 | 48.4 | 94.1 | 28.5 | 85.0 |

  </details>




- **Older GPUs without XFormers support**: See [Issue #11](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11). Thanks to [@S-Mahoney](https://github.com/S-Mahoney) for the solution!


## 🏢 Awesome DA3 Projects

A community-curated list of Depth Anything 3 integrations across 3D tools, creative pipelines, robotics, and web/VR viewers, including but not limited to these. You are welcome to submit your DA3-based project via PR, and we will review and feature it if applicable.

- [DA3-blender](https://github.com/xy-gao/DA3-blender): Blender addon for DA3-based 3D reconstruction from a set of images. 

- [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3): ComfyUI nodes for Depth Anything 3, supporting single/multi-view and video-consistent depth with optional point‑cloud export.

- [DA3-ROS2-Wrapper](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper): Real-time DA3 depth in ROS2 with multi-camera support. 

- [DA3-ROS2-CPP-TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt): DA3 ROS2 C++ TensorRT Inference Node: a ROS2 node for DA3 depth estimation using TensorRT for real-time inference.

- [VideoDepthViewer3D](https://github.com/amariichi/VideoDepthViewer3D): Streaming videos with DA3 metric depth to a Three.js/WebXR 3D viewer for VR/stereo playback.


## 🧑‍💻 Official Codebase Core Contributors and Maintainers

<table>
  <tr>
    <td align="center">
      <a href="https://bingykang.github.io/">
        <img src="https://images.weserv.nl/?url=https://bingykang.github.io/images/bykang_homepage.jpeg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub><b>Bingyi Kang</b></sub>
    </td>
    <td align="center">
      <a href="https://haotongl.github.io/">
        <img src="https://images.weserv.nl/?url=https://haotongl.github.io/assets/img/prof_pic.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Haotong Lin</sub>
    </td>
    <td align="center">
      <a href="https://github.com/SiliChen321">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/195901058?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Sili Chen</sub>
    </td>
    <td align="center">
      <a href="https://liewjunhao.github.io/">
        <img src="https://images.weserv.nl/?url=https://liewjunhao.github.io/images/liewjunhao.png?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
       </a>
        <br />
        <sub>Jun Hao Liew</sub>
    </td>
    <td align="center">
      <a href="https://donydchen.github.io/">
        <img src="https://images.weserv.nl/?url=https://donydchen.github.io/assets/img/profile.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Donny Y. Chen</sub>
    </td>
    <td align="center">
      <a href="https://github.com/DengKaiCQ">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/59907452?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Kai Deng</sub>
    </td>
  </tr>
</table>

## 📝 Citations
If you find Depth Anything 3 useful in your research or projects, please cite our work:

```
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
