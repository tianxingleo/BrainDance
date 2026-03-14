<div align="center">

<img src="./app/assets/icon_square_transparent.png" alt="BrainDance Logo" width="120" />


# 🕯️ BrainDance | 流光 · 记

**“物理世界注定走向无序，而我们在比特世界重建永恒。”**

“——这是物理世界的搜索引擎，第二大脑的空间可视化。”

[English](./README.en.md) | [简体中文](README.md)

### 🏆 面向空间计算时代的三维语义记忆引擎

#### An Anti-Entropy Engine for Human Memory

![Flutter](https://img.shields.io/badge/Flutter-Mobile%20Client-02569B?logo=flutter&logoColor=white) ![Supabase](https://img.shields.io/badge/Supabase-BaaS-3ECF8E?logo=supabase&logoColor=white) ![Python](https://img.shields.io/badge/Python-AI%20Worker-3776AB?logo=python&logoColor=white) ![Vue](https://img.shields.io/badge/Vue-Dashboard-4FC08D?logo=vuedotjs&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-Vector%20Search-6C47FF) ![Three.js](https://img.shields.io/badge/Three.js-3D%20Viewer-000000) ![Deno](https://img.shields.io/badge/Deno-Edge%20Function-000000?logo=deno&logoColor=white) ![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg)

</div>

## 📖 项目概述 (Overview)

**BrainDance (流光 · 记)** 是一个**面向移动端的“可检索三维记忆库”**。

不同于传统的相册只能记录二维的“画面”，BrainDance 利用 **3D Gaussian Splatting (高斯泼溅)** 等计算机图形学前沿技术，将现实世界的物理空间以 1:1 的高保真度转化为数字资产。

更进一步，我们结合了 **Multimodal AI (多模态大模型)** 与 **RAG (检索增强生成)** 技术，让这些三维场景具备了“语义”。你可以像在搜索引擎里一样**搜索现实世界**，让空间记忆不再只是“能看”，而是“能找、能回溯、能定位”。

### 核心特性

- **📷 移动端低成本扫描**：利用手机采集视频与图片素材，通过云端计算和 AI 质检提升低质量素材下的重建成功率。
- **🔍 空间语义检索 (Spatial RAG)**：结合多模态大模型、向量检索与空间锚点，实现“Ctrl+F”搜索物理世界。
- **⏳ 时光剥离 (Time Peeling)**：围绕同一空间在不同时间的扫描结果，探索面向时间维度的空间记忆表达。
- **☁️ 端云协同渲染**：Mobile 采集 -> Cloud 高性能计算 -> Mobile/Web 轻量化查看。

### 技术栈速览

**前端与交互**

![Flutter](https://img.shields.io/badge/Flutter-Mobile-02569B?logo=flutter&logoColor=white) ![Riverpod](https://img.shields.io/badge/Riverpod-State%20Management-0EA5E9) ![WebView](https://img.shields.io/badge/WebView-Model%20Viewer-4B5563) ![Vue 3](https://img.shields.io/badge/Vue%203-Dashboard-4FC08D?logo=vuedotjs&logoColor=white) ![Vite](https://img.shields.io/badge/Vite-Build%20Tool-646CFF?logo=vite&logoColor=white) ![TypeScript](https://img.shields.io/badge/TypeScript-Web-3178C6?logo=typescript&logoColor=white) ![Three.js](https://img.shields.io/badge/Three.js-3D%20Rendering-000000) ![ECharts](https://img.shields.io/badge/ECharts-Visualization-AA344D)

**云服务与数据**

![Supabase](https://img.shields.io/badge/Supabase-Platform-3ECF8E?logo=supabase&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Relational%20DB-4169E1?logo=postgresql&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-Embedding%20Search-6C47FF) ![Deno](https://img.shields.io/badge/Deno-Edge%20Functions-000000?logo=deno&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-Local%20Infra-2496ED?logo=docker&logoColor=white)

**AI 与三维重建**

![Python](https://img.shields.io/badge/Python-Worker-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?logo=pytorch&logoColor=white) ![FFmpeg](https://img.shields.io/badge/FFmpeg-Frame%20Extraction-007808?logo=ffmpeg&logoColor=white) ![COLMAP](https://img.shields.io/badge/COLMAP-SfM-1F6FEB) ![GLOMAP](https://img.shields.io/badge/GLOMAP-Global%20Pose-0F9D58) ![Nerfstudio](https://img.shields.io/badge/Nerfstudio-3DGS%20Pipeline-FF6F00) ![Qwen-VL](https://img.shields.io/badge/Qwen--VL-Multimodal-FF6A00)

## 📜 序言：对抗熵增的战争

**物理学**告诉我们，宇宙的终极命运是**熵增**。房屋会变旧，物品会破碎，秩序会变成混乱。

在**生物学**层面，熵增表现为**遗忘**。海马体的衰退让我们忘记了回家的路，忘记了爱人的脸。在**社会学**层面，熵增表现为**消亡**。城市更新的推土机下，老街、胡同与那些烟火气终将化为尘土。

现有的技术，2D 照片与视频，只是对现实苍白的“截屏”。它们丢失了深度，丢失了光影，更丢失了空间感。它们无法对抗遗忘，因为它们本身就是扁平的。

**BrainDance (流光 · 记)** 不仅仅是一个 App，它是人类对抗时间熵增的通用工具。利用 **3D Gaussian Splatting (高斯泼溅)** 与 **Multimodal AI (多模态大模型)** 等前沿技术，我们试图捕获光的场域，**在数字世界里建立负熵**，为每个人、每座城，留下一份可以穿越时间的**空间档案**。

## 🌌 价值坐标系：微观、宏观与纵深

BrainDance 的价值架构跨越了三个维度，构建了一个从个体到文明的完整记忆生态：

### 1. 微观尺度 (The Micro Scale)

> **"为即将消逝的记忆，建立数字海马体。"**

- **个人见证 (Spatial Journal)**：
  - 当你毕业离开住了 4 年的宿舍，或搬离充满回忆的出租屋时，一次扫描即可把整个物理空间折叠成可回访的数字记忆。
- **医疗辅助 (The Cure)**：
  - 对于阿尔茨海默症等记忆障碍场景，沉浸式空间回访比平面照片更接近真实的熟悉感。

### 2. 宏观尺度 (The Macro Scale)

> **"一座城市的数字方舟，对抗文明的断层。"**

- **众包档案馆 (Crowd-Sourced Archive)**：
  - 汇聚用户扫描数据，为即将消失的街区、店铺和建筑留下一份可进入、可浏览的三维档案。
- **集体记忆 (Collective Memory)**：
  - 让后人不是阅读历史，而是亲自“走进”历史。

### 3. 时间尺度 (The Temporal Scale)

> **"空间计算时代的‘数字胶片’。"**

- **数字底片 (The Digital Negative)**：
  - 2D 视频的分辨率会过时，空间资产却可以在未来设备上继续被重新渲染。
- **面向未来 (Future-Proof)**：
  - 我们今天保存的不是单纯影像，而是面向下一代 XR 终端的原生空间资产。

## ⚡ 核心功能与技术哲学

### 空间 RAG：像搜索文字一样搜索现实

我们不仅重建了“形”，更赋予了“意”。通过集成 **Multimodal LLM (多模态大语言模型)** 的视觉理解能力，BrainDance 将非结构化的 3D 场景转化为**可检索的语义数据库**。

- **User Query**: "爷爷留下的那块怀表在哪？"
- **System Action**: 语义理解 -> 空间索引匹配 -> 摄像机自动飞越 -> **显示最相关的空间位置与视角**

### 时光剥离：在同一空间里回看不同时间

BrainDance 的目标不只是记录一个静态场景，而是逐步形成同一空间在不同时间下的多层记忆切片。对于房间布置变化、成长记录、装修过程、城市更新等场景，这种“在同一坐标系下看变化”的方式，比传统按时间排序的照片更接近真实记忆。

## 🛠️ 技术架构与实现 (Technical Architecture)

本项目采用 **Supabase BaaS 架构**，实现了从移动端采集到云端重建的端云协同流程。

系统当前主要由四部分组成，通过 **Supabase** 解耦：

1. **Client (Flutter)**  
   负责素材采集、上传、任务创建、状态查看与模型浏览。客户端直接连接 Supabase Storage / DB，并通过 Realtime 获取任务进度。

2. **Backend as a Service (Supabase)**  
   提供 PostgreSQL、Storage、Auth、Realtime 与 Edge Functions：
   - **PostgreSQL + pgvector**：存储任务、资产与语义向量。
   - **Storage**：统一管理原始素材、缩略图、模型文件和相关输出。
   - **Realtime**：为移动端与 Dashboard 提供状态同步。
   - **RLS**：基于数据库策略控制用户资产访问权限。

3. **Edge Functions (Deno)**  
   当前仓库已包含 `supabase/functions/search-models`，用于承载语义搜索接口，负责 Embedding 调用、时间解析与向量检索。

4. **AI Worker (Python)**  
   部署在 Linux / WSL GPU 节点，监听 `processing_tasks`，根据 `task_type` 执行不同流水线，上传结果并回写日志、评分、标签和资产信息。

### 当前已接入的主要任务类型

- `video_3dgs`
- `multi_image`
- `single_image_sam3d`
- `single_image_sharp`
- `da3_feed_forward_3dgs`
- `da3_sugar` / `da3+sugar`
- `da3_2dgs` / `da3+2dgs`
- `sparse2dgs`

## 📂 目录结构 (Directory Structure)

本项目遵循 **Monorepo** 策略，所有服务托管于同一仓库，按模块拆分：

```text
BrainDance/
├── app/                  # [Flutter] 移动端客户端
│   ├── lib/              #   - 页面、配置、服务
│   ├── assets/           #   - 图标、字体、内置 WebGL 资源
│   └── pubspec.yaml      #   - Flutter 依赖定义
│
├── ai_engine/            # [Python] 核心算法引擎
│   ├── 3dgs/             #   - GPU Worker 与 3D/2D 重建流水线
│   │   ├── src/core/     #       - Worker、工厂、主流程
│   │   ├── src/pipelines/#       - 各类 Pipeline 实现
│   │   ├── src/libs/     #       - nerfstudio / SuGaR / SHARP 等子模块
│   │   ├── tests/        #       - 测试脚本
│   │   ├── requirements.txt
│   │   └── main.py
│   └── demo/             #   - 演示脚本与实验代码
│
├── supabase/             # [BaaS] 本地后端基础设施
│   ├── migrations/       #   - SQL 迁移
│   ├── functions/        #   - Edge Functions
│   │   └── search-models/#       - 语义搜索函数
│   ├── config.toml
│   └── README.md
│
├── dashboard/            # [Vue 3 + Vite] 系统状态看板
│   ├── src/              #   - Dashboard 前端代码
│   ├── .env.example      #   - 环境变量模板
│   └── package.json
│
├── 3dgs_viewer/          # [Tools] 3DGS 脚本与辅助工具
├── docs/                 # [Doc] 项目文档与技术报告
└── README.md
```

> **说明**
> - `app/`、`dashboard/`、`supabase/functions/` 均已纳入本仓库
> - 根 README 只保留项目总览与最短启动路径，模块细节请看各子目录 README

## 🚀 快速开始 (Quick Start)

### 环境要求 (Prerequisites)

- **AI Engine**: NVIDIA GPU, Python 3.10+, CUDA 11.8+/12.x
- **Infrastructure**: Docker, Supabase CLI
- **Client**: Flutter SDK, Android Studio / Xcode
- **Dashboard**: Node.js 18+

### 测试环境 (Testing Environment)

#### 移动端测试设备

- **OPPO Find X8**
- **OPPO Reno 14**

#### 服务器配置

- **当前 AI Engine 测试/推荐服务器（本机，2026-03-09 实测）**
- **CPU**: Intel Xeon Platinum 8260 × 2（双路，96 线程）
- **内存**: 503GiB（约 512GB）
- **显卡**: NVIDIA L20 46GB × 2（双卡）
- **操作系统**: Ubuntu 22.04.5 LTS（Kernel 6.8.0-100-generic）

**AI Engine 最低配置（兼容基线）**
- **CPU**: Intel Core i5-14600KF
- **内存**: 64GB RAM
- **显卡**: NVIDIA RTX 5070 12GB

### 部署步骤 (Deployment)

#### 0. 获取完整代码（子模块 + LFS）

```bash
git lfs install
git clone --recurse-submodules https://github.com/tianxingleo/BrainDance.git
cd BrainDance
git lfs pull
git submodule foreach --recursive 'git lfs pull || true'
```

#### 1. 启动基础设施 (Supabase Local)

```bash
cd supabase
supabase start
```

启动后记下：

- `API URL`
- `anon key`
- `service_role key`

#### 2. 启动计算引擎 (AI Worker)

```bash
conda create -n braindance python=3.10
conda activate braindance
cd ai_engine/3dgs
pip install -r requirements.txt
pip uninstall -y nerfstudio
pip install -e src/libs/nerfstudio
python main.py
```

如需本地调试单个视频：

```bash
cd ai_engine/3dgs
python main.py /path/to/video.mp4
```

#### 3. 启动 Dashboard

先在 `dashboard/.env` 中配置：

```env
VITE_SUPABASE_URL=http://127.0.0.1:54321
VITE_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
VITE_STORAGE_BUCKETS=braindance-assets
VITE_SUPABASE_EDGE_FUNCTIONS=search-models,test-timeout
```

然后运行：

```bash
cd dashboard
npm install
npm run dev
```

#### 4. 启动移动端 (App)

在 `app/.env` 中填入客户端所需的 Supabase 配置后运行：

```bash
cd app
flutter pub get
flutter run
```

当前移动端主链路已经包括：

- 登录与会话管理
- 视频上传与任务创建
- 任务状态页
- Recall 资产页
- 基于 WebView 的移动端 WebGL 模型查看

## 🗂️ 数据流与存储约定

Storage 默认以 `braindance-assets` bucket 为中心，当前常见路径为：

```text
{user_id}/{scene_id}/raw/video.mp4
{user_id}/{scene_id}/raw/image.png
{user_id}/{scene_id}/raw/images.zip
{user_id}/{scene_id}/raw/thumbnail.jpg

{user_id}/{scene_id}/output/point_cloud.ply
{user_id}/{scene_id}/output/point_cloud.splat
{user_id}/{scene_id}/output/point_cloud.ksplat
{user_id}/{scene_id}/output/transforms.json
```

数据库中的关键表包括：

- `processing_tasks`：任务状态、日志、质量分数、任务类型、参数
- `model_assets`：模型路径、描述、标签、对象与 Embedding
- `memory_poses`：帧级空间锚点与向量

## 🔎 语义搜索 (Semantic Search)

当前仓库已包含 Edge Function：

- `supabase/functions/search-models`

其职责是：

1. 解析自然语言中的检索目标与时间条件。
2. 调用 Embedding 接口生成向量。
3. 通过 `pgvector` 与数据库函数检索相关场景和空间锚点。

本地运行示例：

```bash
cd supabase/functions/search-models
supabase functions serve search-models --no-verify-jwt --env-file .env.local
```

## 📚 文档导航 (Documentation)

- [`docs/01-入门指南/快速开始.md`](./docs/01-入门指南/快速开始.md)
- [`docs/01-入门指南/本地部署.md`](./docs/01-入门指南/本地部署.md)
- [`supabase/README.md`](./supabase/README.md)
- [`ai_engine/3dgs/README.md`](./ai_engine/3dgs/README.md)
- [`docs/开发文档/设计及创新性分析报告.md`](./docs/开发文档/设计及创新性分析报告.md)

## 📜 版权与开源协议 (License & Copyright)

**BrainDance (流光·记)** 的所有核心工程架构与业务代码的知识产权均归属原作者及项目贡献者所有。

本项目采用 **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)** 协议对外公开源码。

- ✅ **允许**：个人学习、学术研究、非营利组织使用以及二次修改
- ❌ **严禁任何商业用途**：包括但不限于企业内部生产使用、收费服务、闭源商业集成等

> **第三方依赖特别声明**
> 本项目 AI 引擎部分集成了多个上游研究项目与子模块，使用时需同时遵守其各自许可证与非商业研究协议。
