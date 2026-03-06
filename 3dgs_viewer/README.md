# WebGL Demo 使用指南

## 概述

`3dgs_viewer` 是 BrainDance 项目的 Web 3D 高斯点云查看器 Demo，用于在浏览器中实时查看和交互 3DGS（3D Gaussian Splatting）重建的场景。

## 目录结构

```
3dgs_viewer/
├── .env                      # 环境配置文件
├── run_glomap.py            # 主流水线脚本（视频 → 3DGS → WebGL）
├── calc_transform.py        # 坐标转换计算工具
├── export_poses.py          # 位姿导出工具
├── fix_poses.py             # 位姿修复工具
├── sync_images.py           # 图片同步工具
├── tag_poses.py             # 镜头语义打标工具
├── evaluate_poses.py        # 位姿评估工具
└── my-3dgs-viewer/          # Vue 3 + Three.js 前端查看器
    ├── public/
    │   └── models/          # 输出模型和相机位姿
    ├── src/
    │   ├── App.vue          # 主应用组件
    │   └── components/
    │       └── GaussianViewer.vue  # 3DGS 查看器组件
    └── package.json
```

## 快速开始

### 1. 环境准备

#### Python 依赖（AI 引擎）
```bash
# 确保已安装必要依赖
pip install nerfstudio gsplat opencv-python numpy plyfile
```

#### 前端依赖
```bash
cd 3dgs_viewer/my-3dgs-viewer
npm install
```

### 2. 配置环境变量

编辑 [.env](3dgs_viewer/.env) 文件：

```bash
# Supabase 配置（可选）
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=your_key_here
SUPABASE_BUCKET=braindance-assets

# 训练参数
TRAINING_ITERATIONS=15000    # 训练迭代次数（预览 7000，高质量 30000）
MAX_IMAGES=500               # 最大处理图片数
MIN_QUALITY_SCORE=40         # AI 质检及格线（0-100）

# 多模态 LLM API（用于语义打标）
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 完整流水线（推荐）

使用 [run_glomap.py](3dgs_viewer/run_glomap.py:1) 从视频生成 3DGS 模型并自动部署到 WebGL 查看器：

```bash
cd 3dgs_viewer

# 使用默认视频 (test.mp4)
python run_glomap.py

# 使用自定义视频
python run_glomap.py /path/to/your/video.mp4
```

**流水线步骤：**
1. **视频抽帧**：使用 FFmpeg 从视频提取帧
2. **AI 质检**：智能过滤模糊和低质量图片
3. **位姿估计**：使用 GLOMAP/COLMAP 进行相机位姿估计
4. **3DGS 训练**：使用 Nerfstudio 训练高斯点云模型
5. **后处理**：球形切割去除背景点云
6. **自动部署**：将模型和位姿数据同步到 WebGL 目录

**输出位置：**
- 模型文件：`my-3dgs-viewer/public/models/scene_auto_sync.ply`
- 相机位姿：`my-3dgs-viewer/public/models/webgl_poses.json`
- 参考图片：`my-3dgs-viewer/public/models/images/`

### 4. 启动查看器

```bash
cd my-3dgs-viewer

# 开发模式（热重载）
npm run dev

# 生产构建
npm run build

# 预览生产构建
npm run preview
```

访问 `http://localhost:5173` 查看效果。

## 功能详解

### 主流水线 ([run_glomap.py](3dgs_viewer/run_glomap.py:1))

**核心参数：**
- `MAX_IMAGES`：限制最大图片数量（防止显存溢出）
- `TRAINING_ITERATIONS`：训练迭代次数
- `KEEP_PERCENTILE`：保留点云百分比（0.9 = 保留 90% 最近点云）
- `FORCE_SPHERICAL_CULLING`：强制开启球形切割

**智能清洗策略：**
1. **质量清洗**：剔除最差的 15% 废片
2. **均匀采样**：在时间轴上均匀采样，保证视角覆盖

**质量检测：**
- COLMAP 匹配率检测（< 35% 自动终止）
- "No convergence" 错误检测

### 辅助工具

#### 图片同步 ([sync_images.py](3dgs_viewer/sync_images.py:1))

从现有训练数据同步图片到查看器：

```bash
python sync_images.py
```

#### 坐标转换 ([calc_transform.py](3dgs_viewer/calc_transform.py:1))

计算 OpenCV 到 WebGL 的坐标转换矩阵。

#### 位姿导出 ([export_poses.py](3dgs_viewer/export_poses.py:1))

导出对齐后的相机位姿到 WebGL 格式。

#### 语义打标 ([tag_poses.py](3dgs_viewer/tag_poses.py:1))

为相机镜头添加语义标签（支持中文自然语言搜索）：

```bash
python tag_poses.py
```

**功能：**
- 使用多模态 LLM（Qwen-VL）分析每个视角的场景内容
- 生成中文语义标签（如"全景视角"、"特写镜头"等）
- 支持基于标签的智能搜索和导航

### WebGL 查看器功能

查看器位于 [GaussianViewer.vue](3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue:1)，支持以下功能：

#### 基础控制
- **鼠标拖拽**：旋转视角
- **滚轮**：缩放
- **右键拖拽**：平移

#### 高级功能
- **镜头飞跃**：点击相机位姿图标，平滑飞越到对应视角
- **语义搜索**：输入中文描述（如"全景"），自动找到匹配镜头
- **自动旋转**：开启后场景自动旋转展示
- **VR 模式**：支持 WebXR VR 设备

#### 调试工具
- **手动微调**：精确调整相机位置和角度
- **复制数据**：导出当前相机矩阵用于调试
- **轨迹预览**：显示相机运动轨迹的可视化

## 常见问题

### Q: 训练失败或点云质量差？
A: 检查以下几点：
1. 视频质量：确保视频清晰、稳定
2. 图片数量：调整 `MAX_IMAGES` 参数（推荐 200-500）
3. 质检阈值：降低 `MIN_QUALITY_SCORE`（如设为 30）
4. 光照条件：避免弱光或剧烈变化的光照

### Q: 网页无法加载模型？
A: 检查：
1. PLY 文件是否存在于 `public/models/` 目录
2. 浏览器控制台是否有 CORS 错误
3. 文件大小是否过大（建议 < 100MB）

### Q: 相机位姿不准确？
A: 使用以下工具调试：
1. 运行 [evaluate_poses.py](3dgs_viewer/evaluate_poses.py:1) 评估位姿质量
2. 使用 [fix_poses.py](3dgs_viewer/fix_poses.py:1) 修复偏差
3. 检查 `webgl_poses.json` 中的矩阵数据

### Q: 如何调整切割参数？
A: 编辑 [run_glomap.py](3dgs_viewer/run_glomap.py:137) 中的：
- `KEEP_PERCENTILE`：保留点云比例（0.5-0.95）
- `FORCE_SPHERICAL_CULLING`：是否强制切割

### Q: 如何使用语义搜索？
A:
1. 确保已配置 `DASHSCOPE_API_KEY`
2. 运行 `python tag_poses.py` 生成标签
3. 在查看器搜索框输入中文描述
4. 点击"飞跃"按钮自动导航

## 技术栈

### 后端（AI 引擎）
- **Nerfstudio**：3DGS 训练框架
- **GLOMAP**：位姿估计引擎
- **OpenCV**：图像处理
- **FFmpeg**：视频抽帧

### 前端
- **Vue 3**：响应式框架
- **Three.js**：3D 渲染引擎
- **@mkkellogg/gaussian-splats-3d**：3DGS 渲染库
- **GSAP**：动画引擎

## 相关文件

- [环境配置](3dgs_viewer/.env)
- [Vue 配置](3dgs_viewer/my-3dgs-viewer/package.json)
- [主应用](3dgs_viewer/my-3dgs-viewer/src/App.vue)
- [查看器组件](3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue)
