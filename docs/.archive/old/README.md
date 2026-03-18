<div align="center">

# 🧠 BrainDance | 超梦

**“并不是我们在回忆过去，而是过去依然驻留在那里，等待我们重新步入。”**

[English](https://www.google.com/search?q=README_EN.md) | [简体中文](README.md)

</div>

## 📖 序言：从“记录画面”到“收藏空间”

当我们在翻看旧照片时，我们在看什么？ 我们看到的是被压扁的三维世界，是丢失了深度、光影和空气感的切片。传统的相册只能告诉我们“那里有什么”，却无法让我们再次“置身其中”。

**BrainDance (超梦)** 项目的灵感源于《赛博朋克 2077》，但我们不追求反乌托邦的狂想，而是致力于探索**人类记忆存储的终极形态**。

这不仅仅是一个 3D 扫描工具，它是一个**面向未来的三维数字记忆库**，是你的**第二大脑在物理空间的可视化延伸**。我们希望捕捉那个下午阳光洒在书桌上的微尘，捕捉那一刻的空气与氛围——这才是记忆的本体。

## ✨ 核心理念与创新

### 1. 重新定义记忆：由 2D 迈向 3DGS

传统的 3D 建模（Mesh）像是“捏泥人”，由三角形拼接而成，虽有形状却无灵魂。 **BrainDance** 采用 **3D Gaussian Splatting (高斯泼溅)** 技术，它更像是“全息投影”。它由无数个携带颜色、透明度和光影信息的“光点”组成。

- **光影折射**：完美还原玻璃、水面、金属的光泽。
- **半透明质感**：捕捉发丝、烟雾与薄纱的朦胧。
- **临场感**：不仅仅是看照片，而是能再次感受到那天的“氛围”。

### 2. 物理世界的搜索引擎 ("Ctrl+F" for Reality)

BrainDance 不止于展示，更在于**回溯**。结合多模态大模型（Multimodal AI），我们将实现对物理世界的语义检索。

- *“我的钥匙落在哪里了？”*
- *“去年生日那天的蛋糕是什么颜色的？”*
- 你不再是在海量图片中翻找，而是直接在你的三维记忆库中进行**语义搜索**。

### 3. 端云协同的算力架构

鉴于 3DGS 高昂的训练成本与移动端的功耗限制，我们设计了高效的 **Mobile-Cloud-Mobile** 闭环：

- **采集 (Mobile)**：轻量级 App 进行倾斜摄影与位姿记录（ARCore/ARKit）。
- **重构 (Cloud)**：云端集群运行 SfM (COLMAP) 与 3DGS 训练 (Nerfstudio/Gsplat)。
- **体验 (Mobile)**：流式加载压缩后的 PLY 模型，实现低延迟的 60fps 漫游。

## 🏗️ 系统架构与技术栈

本项目包含以下核心模块：

### 计算后端 (AI Engine)

基于 `nerfstudio` 和 `colmap` 的自动化处理流水线。

- **数据预处理**：`process_3dgs.py`
  - 视频抽帧 (`ffmpeg`)
  - 特征点提取与匹配 (`COLMAP`)
  - 相机位姿解算与稀疏点云生成
- **模型训练**：
  - 基于 Gaussian Splatting 的场景优化
  - 自动裁剪与密度控制
- **输出转换**：
  - 生成 Web 友好的压缩格式
  - 渲染全景视频流

### 业务架构

- **消息队列**：Redis/RabbitMQ 处理异步训练任务
- **存储**：对象存储 (OSS/S3) 管理海量图像与模型数据

## 🚀 快速开始 (后端引擎)

### 环境要求

- OS: Linux (Ubuntu 20.04+ 推荐) / Windows (需配置 CUDA 环境)
- GPU: NVIDIA GPU (8GB+ VRAM)
- CUDA: 11.8+

### 安装依赖

```
# 克隆项目
git clone [https://github.com/YourUsername/BrainDance.git](https://github.com/YourUsername/BrainDance.git)
cd BrainDance

# 建立虚拟环境
conda create -n braindance python=3.10
conda activate braindance

# 安装核心依赖 (以 Nerfstudio 为例)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install nerfstudio
```

### 运行 Demo

将拍摄好的视频放入 `data/input` 目录，运行自动化脚本：

```
# 自动处理视频 -> COLMAP -> 3DGS 训练
python ai_engine/demo/process_3dgs_clean.py --video_path ./data/input/my_room.mp4 --output_dir ./data/output/
```

脚本将自动执行以下步骤：

1. **视频转序列帧**：提取关键帧。
2. **COLMAP 稀疏重建**：计算相机位姿 (`transforms.json`)。
3. **Nerfstudio 训练**：开始高斯泼溅迭代。
4. **导出**：生成可查看的 Viewer 链接或 `.ply` 文件。

## 🗓️ 演进路线

- [x] **Phase 1: 原型验证**
  - [x] 完成视频到 3DGS 的自动化 Python 脚本 (`process_3dgs.py`)。
  - [x] 验证 COLMAP 位姿解算与 Nerfstudio 的对接。
  - [x] 本地 Viewer 预览。
- [x] **Phase 2: 云端化与移动端适配**
- [x] 部署后端 API 接口，接收移动端上传的数据包。
- [x] 优化移动端 WebGL 查看器 (基于 LumaAI 或 SuperSplat 内核)。
- [x] 实现模型压缩，适配 4G/5G 网络传输。
- [x] **Phase 3: 记忆深搜 (Memory RAG)**
- [x] 接入多模态大模型 (Gemini/GPT-4V)。
- [x] 实现 3D 场景内的物体识别与标签化。
- [x] 开发自然语言检索接口 ("Show me the books on the shelf")。

## 🤝 贡献与社区

BrainDance 是一个探索性质的项目，如果你对 **计算机视觉、Web图形学、后端架构** 或 **赛博朋克哲学** 感兴趣，欢迎加入我们。

- 提交 Issue 分享你的想法
- Fork 项目并提交 Pull Request

## 📝 版权与致谢

本项目深受以下开源项目启发：

- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Colmap](https://colmap.github.io/)

License: MIT

<div align="center"> <sub>Made with ❤️ by the BrainDance Team. 



 "To live is to express, and to express is to create."</sub> </div>
