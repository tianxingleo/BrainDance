# DA3 3DGS 生成完整指南

## 📌 重要说明

**DA3-Streaming** 和 **DA3 3DGS** 是两个不同的功能：

| 功能 | DA3-Streaming | DA3 3DGS |
|------|--------------|----------|
| 输入 | 视频帧 | 图像/视频 |
| 输出 | 点云 + 相机位姿 | **3D Gaussians** + 渲染视频 |
| 模型 | DA3-GIANT | DA3NESTED-GIANT-LARGE |
| 用途 | SLAM/重建 | Novel View Synthesis |

## 🎯 方案对比

### 方案 1：DA3 原生 3DGS（推荐）

**优势**：
- ✅ 直接预测 3D Gaussians（无需训练）
- ✅ 支持视频渲染
- ✅ 质量最高

**劣势**：
- ❌ 需要重新用 DA3 处理原始图像
- ❌ 需要 Gradio UI（暂时）

### 方案 2：DA3-Streaming + 3DGS 训练

**优势**：
- ✅ 已有点云和相机位姿
- ✅ 可用标准 3DGS 工具

**劣势**：
- ❌ 需要额外训练步骤
- ❌ 质量取决于训练过程

## 🚀 推荐方案：使用 DA3 Gradio UI

### 步骤 1：启动 Gradio 应用

```bash
cd ~/projects/Depth-Anything-3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup

# 启动 Gradio
da3 gradio \
    --model-dir ./weights \
    --workspace-dir ./workspace \
    --gallery-dir ./gallery \
    --host 0.0.0.0 \
    --port 7860 \
    --share
```

### 步骤 2：在浏览器中使用

1. **打开浏览器**：访问显示的 URL（通常是 `http://127.0.0.1:7860`）

2. **上传输入**：
   - **图像文件夹**：选择包含图像的文件夹
   - **或视频**：直接上传视频文件

3. **配置选项**：
   - ✅ 勾选 **"Export 3DGS Video"**
   - 选择视频质量：`low` / `medium` / `high`
   - 选择轨迹模式：`original` / `smooth` / `interpolate` 等

4. **点击 Process**：
   - 等待处理完成
   - 下载生成的 3DGS 视频

### 步骤 3：查看输出

输出文件结构：
```
workspace/
└── [scene_name]/
    ├── scene.glb              # 3D 场景（可查看）
    ├── scene.jpg              # 预览图
    ├── gs_video/
    │   └── [scene]_extend.mp4  # ⭐ 3DGS 渲染视频
    └── depth_vis/             # 深度可视化（可选）
```

## 🔧 方案 2：使用 DA3-Streaming 输出

如果你已经有 DA3-Streaming 的输出（点云 + 相机位姿），可以将其转换为 COLMAP 格式并直接使用 3DGS 工具训练。

### 步骤 1：转换数据

运行提供的转换脚本：

```bash
cd ~/projects/Depth-Anything-3
# 运行脚本将位姿、内参、点云转换为 COLMAP 格式
python convert_da3_to_colmap.py
```

转换完成后，数据保存在：`output/sugar_streaming/colmap_data/`

### 步骤 2：使用 3DGS 训练

如果你想使用 **SuGaR** 内部的 Gaussian Splatting 工具进行训练：

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gs_linux_backup

# 进入 SuGaR 的 gaussian_splatting 目录
cd /home/ltx/projects/SuGaR/gaussian_splatting

# 开始训练
python train.py \
    -s /home/ltx/projects/Depth-Anything-3/output/sugar_streaming/colmap_data \
    -m /home/ltx/projects/Depth-Anything-3/output/sugar_streaming/gs_trained_result \
    --iteration 15000
```

## 📊 输出对比

| 方案 | 输出文件 | 质量 | 速度 |
|------|---------|------|------|
| **DA3 3DGS (原生)** | `.mp4` 渲染视频 | ⭐⭐⭐⭐⭐ | 快 |
| **DA3-Streaming + 训练** | `.ply` 点云 | ⭐⭐⭐ | 慢 |

## 💡 常见问题

### Q1: DA3-Streaming 的点云能直接用吗？

**A**: 不能直接用于 3DGS 渲染，需要：
1. 转换为 COLMAP 格式
2. 使用 3DGS 工具训练
3. 渲染新视角

### Q2: DA3 3DGS 需要训练吗？

**A**: **不需要！** DA3 直接预测 3D Gaussians，立即可渲染。

### Q3: 能不能用命令行直接生成 3DGS？

**A**: 目前 **CLI 不支持**，只能通过 Gradio UI。未来版本可能会添加。

## 🎬 快速测试

如果你想快速体验 3DGS，可以：

```bash
# 1. 准备测试图像（从视频中提取）
mkdir -p test_images
ffmpeg -i /home/ltx/projects/SuGaR/video.mp4 \
    -vf "fps=0.5,scale=720:-1" \
    test_images/frame_%04d.png

# 2. 启动 Gradio
da3 gradio \
    --model-dir ./weights \
    --workspace-dir ./workspace \
    --gallery-dir ./gallery \
    --host 0.0.0.0 \
    --port 7860

# 3. 在浏览器中上传 test_images 文件夹
```

## 📚 参考

- [DA3 README](README.md)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [GSplat](https://github.com/nerfstudio-project/gsplat)
