<div align="center">


# 🕯️ BrainDance | 流光 · 记

**“物理世界注定走向无序，而我们在比特世界重建永恒。”**

“——这是物理世界的搜索引擎，第二大脑的空间可视化。”

[English](./README.en.md) | [简体中文](README.md)

### 🏆 面向空间计算时代的三维语义记忆引擎

#### An Anti-Entropy Engine for Human Memory

![Supabase](https://img.shields.io/badge/Supabase-Enabled-3ECF8E?logo=supabase&logoColor=white) ![Flutter](https://img.shields.io/badge/Flutter-Client-02569B?logo=flutter&logoColor=white) ![Python](https://img.shields.io/badge/Python-Worker-3776AB?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-blue)

</div>

## 📖 项目概述 (Overview)

**BrainDance (流光 · 记)** 是一个**面向移动端的“可检索三维记忆库”**。

不同于传统的相册只能记录二维的“画面”，BrainDance 利用 **3D Gaussian Splatting (高斯泼溅)** 等计算机图形学前沿技术，将现实世界的物理空间（如你的房间、即将拆迁的老街、珍藏的手办）以 1:1 的高保真度转化为数字资产。

更进一步，我们结合了 **Multimodal AI (多模态大模型)** 与 **RAG (检索增强生成)** 技术，让这些三维场景具备了“语义”。你可以像在搜索引擎里一样**搜索现实世界**——问它“我的钥匙落在哪里了？”，镜头便会自动飞跃并聚焦到那一刻的时空。

### 核心特性

- **📷 移动端低成本扫描**：利用手机采集视频流与位姿数据，通过大量优化与AI介入，极大提高了低拍摄质量下生成的3DGS模型质量，大大降低 3DGS 建模门槛。
- **🔍 空间语义检索 (Spatial RAG)**：结合多模态大模型理解场景内容，实现“Ctrl+F”搜索物理世界。
- **⏳ 时光剥离 (Time Peeling)**：在同一坐标系下叠加多维时间切片，实现从“现在”回溯到“过去”的视觉体验。
- **☁️ 端云协同渲染**：Mobile 采集 -> Cloud 高性能计算 -> Mobile/XR 轻量化查看。

## 📜 序言：对抗熵增的战争

**物理学**告诉我们，宇宙的终极命运是**熵增**。 房屋会变旧，物品会破碎，秩序会变成混乱。

在**生物学**层面，熵增表现为**遗忘**。海马体的衰退让我们忘记了回家的路，忘记了爱人的脸。 在**社会学**层面，熵增表现为**消亡**。城市更新的推土机下，老街、胡同与那些烟火气终将化为尘土。

现有的技术——2D 照片与视频，只是对现实苍白的“截屏”。它们丢失了深度，丢失了光影，更丢失了空间感。它们无法对抗遗忘，因为它们本身就是扁平的。

**BrainDance (流光 · 记)** 不仅仅是一个 App，它是人类对抗时间熵增的通用工具。 利用 **3D Gaussian Splatting (高斯泼溅)** 与 **Multimodal AI (多模态大模型)**等前沿技术，我们试图捕获光的场域，**在数字世界里建立负熵**，为每个人、每座城，留下一份可以穿越时间的**空间档案**。

## 🌌 价值坐标系：微观、宏观与纵深

BrainDance 的价值架构跨越了三个维度，构建了一个从个体到文明的完整记忆生态：

### 1. 微观尺度 (The Micro Scale)：

> **"为即将消逝的记忆，建立数字海马体。"**

- **个人见证 (Spatial Journal)**：
  - **空间折叠**：当你毕业离开住了4年的宿舍，或搬离充满回忆的出租屋时，一键扫描，即可将整个物理空间打包带走。带不走房子，但能带走“家”。
- **医疗辅助 (The Cure)**：
  - **怀旧疗法**：对于**阿尔茨海默症**患者，通过 VR/MR 设备，让老人重返几十年前的“老家”。墙上的照片会说话，桌上的饭菜冒着热气。这种沉浸式的熟悉感，是药物无法替代的慰藉。

### 2. 宏观尺度 (The Macro Scale)：

> **"一座城市的数字方舟，对抗文明的断层。"**

- **众包档案馆 (Crowd-Sourced Archive)**：
  - 城市在生长，也在消失。BrainDance 汇聚成千上万用户的扫描数据，将即将拆迁的弄堂、老字号店铺、百年古树拼凑成一张**城市的 3DGS 地图**。
- **集体记忆 (Collective Memory)**：
  - 即使物理实体被推倒重建，但在 BrainDance 的平行宇宙里，那条充满烟火气的街道依然存在。后人不再通过冷冰冰的文字，而是亲自“走进”历史，触摸文明的肌理。

### 3. 时间尺度 (The Temporal Scale)：

> **"空间计算时代的‘数字胶片’。"**

- **数字底片 (The Digital Negative)**：
  - 2D 视频的分辨率是锁死的，平面设备必将过时。但 BrainDance 记录的是**光场 (Radiance Field)**。
- **面向未来 (Future-Proof)**：
  - 就像胶片电影可以重制成 4K 蓝光一样，今天扫描的数据，在未来 Apple Vision Pro 或 Meta Quest 普及的时代，将释放出比现在强百倍的临场感。我们是在为下一个计算时代储备**原生空间资产**。

## ⚡ 核心功能与技术哲学

### 空间 RAG：像搜索文字一样搜索现实

我们不仅重建了“形”，更赋予了“意”。 通过集成 **Multimodal LLM (多模态大语言模型)** 的视觉理解能力，BrainDance 将非结构化的 3D 场景转化为**可检索的语义数据库**。

- **User Query**: "爷爷留下的那块怀表在哪？"
- **System Action**: 语义理解 -> 空间索引匹配 -> 摄像机自动飞越 -> **显示抽屉深处的怀表**。



### 🛠️ 技术架构与实现 (Technical Architecture)

本项目采用 **Supabase BaaS 架构**，实现了从移动端采集到云端重建的全自动化流程。

系统由四部分组成，通过 **Supabase** 进行解耦：

1. **Client (Flutter)**:
   - 负责视频采集与上传，支持断点续传。
   - 直接连接 Supabase Storage/DB，无中间件。
   - 通过 Realtime 监听任务进度（支持灵动岛/Live Activities）。
2. **Backend as a Service (Supabase)** 本项目彻底移除了传统中间件（Redis/MinIO/Go），完全基于 Supabase 构建 Serverless 架构：
   - **PostgreSQL (The "Everything" Store)**:
     - **Vector DB**: 启用 `pgvector` 扩展，存储 3D 场景的语义向量 (Embedding)，实现多模态 RAG 检索。
     - **Job Queue**: 基于 `processing_tasks` 表与 `FOR UPDATE SKIP LOCKED` 机制，实现高性能、原子性的任务分发队列（替代 Redis）。
   - **Storage**:
     - **Bucket**: 统一管理原始视频流 (Raw) 与训练完成的 3D 模型 (PLY/Splat)，通过 Policy 控制公开/私有访问权限。
   - **Auth & Security**:
     - **RLS (Row Level Security)**: 数据库原生的行级安全策略，确保用户只能访问属于自己的 3D 资产，零后端代码实现严密的鉴权。
   - **Realtime**:
     - **WebSocket**: 客户端订阅数据库变更流 (CDC)，实现毫秒级的任务进度推送与多端状态同步（替代前端轮询）。
3. **Edge Functions (Deno)**:
   - **Serverless API**: 承载轻量级业务逻辑。
   - **Semantic Search**: 负责 RAG 语义检索接口，调用 LLM Embedding API 并进行向量匹配，保护 API Key 不泄露。
4. **AI Worker (Python)**:
    - 部署在 WSL/Linux 显卡服务器上的纯计算节点。
    - **Consumer**: 监听数据库的任务队列，自动拉取视频或图片。
    - **Training**: 运行 3DGS/Nerfstudio 训练管线，生成 PLY 模型。
    - **Single Image**: 支持基于 SAM3D 的单图 3DGS 生成，无需视频。
    - **Understanding**: 调用多模态大模型 (Qwen-VL) 进行场景理解与自动打标 (Auto-Tagging)。



------

### 📂 目录结构 (Directory Structure)

本项目遵循 **Monorepo** 策略，所有服务托管于同一仓库，严格执行模块隔离。

Plaintext

```
BrainDance/
├── ai_engine/            # [Python] 核心算法引擎 (Worker)
│   ├── 3dgs/             #   - 3DGS 核心引擎
│   │   ├── src/          #   - 源代码
│   │   │   ├── core/         #       - Pipeline 基类、工厂、Worker
│   │   │   ├── pipelines/    #       - Pipeline 实现
│   │   │   │   ├── video_3dgs.py      #       - 视频 3DGS Pipeline
│   │   │   │   └── single_image_sam3d.py  #       - 单图 SAM3D Pipeline
│   │   │   ├── modules/      #       - 功能模块
│   │   │   │   ├── sam3d_engine/      #           - SAM3D 单图引擎
│   │   │   │   ├── nerf_engine.py     #           - 3DGS 训练引擎
│   │   │   │   ├── glomap_runner.py   #           - 位姿解算
│   │   │   │   └── knowledge_base.py  #           - RAG 知识库
│   │   │   ├── libs/         #       - 内嵌依赖库
│   │   │   │   └── sam-3d-objects/    #           - SAM3D 推理库
│   │   │   └── utils/        #       - 工具函数
│   │   ├── tests/            #   - 测试脚本
│   │   ├── requirements.txt  #   - Python 依赖
│   │   └── main.py           #   - 程序入口
│   ├── demo/              #   - 演示脚本与测试数据
│   ├── models/            #   - AI 模型缓存目录
│   ├── rag/               #   - RAG 数据处理
│   └── log/               #   - 日志文件
│
├── supabase/              # [BaaS] 云端基础设施 (Serverless)
│   ├── migrations/        #   - [SQL] 数据库结构变更历史
│   ├── seed.sql           #   - 初始化测试数据
│   ├── config.toml        #   - Supabase 本地开发配置
│   └── README.md          #   - ☁️ 后端部署指南
│
├── docs/                  # [Doc] 项目文档
│   ├── API_DOC.md         #   - API 接口文档
│   ├── BrainDance 项目协作规范与开发协议 (v1.0).md  #   - 开发规范
│   ├── 代办/               #   - 待办事项
│   └── 技术报告/           #   - 技术报告
│
└── README.md              #   - 本文件
```

> **说明**: 
> - `app/` (Flutter 移动端) 正在开发中，尚未纳入本仓库
> - `supabase/functions/` (搜索 Edge Functions) 正在开发中

## 🚀 快速开始 (Quick Start)

### 环境要求 (Prerequisites)

- **AI Engine**: NVIDIA GPU (CUDA 11.8+), Python 3.10+ (Conda 推荐)
- **Infrastructure**: [Docker Desktop](https://www.docker.com/), [Supabase CLI](https://supabase.com/docs/guides/cli)
- **Client**: Flutter SDK (3.10+), Android Studio / Xcode

### 部署步骤 (Deployment)

本项目支持 **全本地化部署**，无需云端账号即可完整运行。

#### 1. 启动基础设施 (Supabase Local)

基于 Docker 一键拉起数据库、存储桶、鉴权服务与 Edge Functions。



```bash
# 1. 进入项目根目录
cd BrainDance

# 2. 启动 Supabase 本地服务
supabase start

# 3. 🎉 记录输出的 API URL 和 Keys (Anon / Service Role)
#    - API URL: http://127.0.0.1:54321
#    - DB URL: postgresql://postgres:postgres@127.0.0.1:54322/postgres
#    - Studio: http://127.0.0.1:54323 (Web 控制台)
```

#### 2. 启动计算引擎 (AI Worker)

Worker 负责监听本地 Supabase 的任务队列并调用 GPU 进行训练。

```bash
cd ai_engine

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量 (复制示例并填入上一步的 URL/Key)
cp .env.example .env
# ⚠️ 注意：Worker 必须使用 SERVICE_ROLE_KEY 以绕过 RLS 权限

# 3. 启动 Worker
python src/worker.py
# 输出: 🚀 [Worker] Connected to Supabase Local. Listening for tasks...
```

#### 3. 启动移动端 (App)

客户端直接连接本地 Supabase 网关。

```bash
cd app

# 1. 配置连接信息
#    编辑 lib/config.dart (或 .env)，填入 API URL 和 ANON_KEY

# 2. 运行 App
flutter run
```



## 时序图：

```mermaid
sequenceDiagram
    autonumber
    participant User as 👤 用户 (User)
    participant App as 📱 移动端 (Flutter)
    participant Edge as ⚡ 边缘函数 (Edge Function)
    participant Supabase as 🐘 Supabase (DB/Storage/Realtime)
    participant LLM as ☁️ 大模型 API (DashScope)
    participant Worker as ⚙️ 计算节点 (Python Worker)

    Note over User, Supabase: === 阶段一：直接上传与任务创建 (Direct Upload) ===
    User->>App: 1. 点击“新建记忆”，拍摄视频
    App->>Supabase: 2. 直传视频文件 (Storage Bucket: raw)
    Supabase-->>App: 返回文件路径 (Path)
    App->>Supabase: 3. 插入任务记录 (Insert processing_tasks) <br/> {status: "pending", user_id: "..."}
    Supabase-->>App: 写入成功
    
    par 实时反馈
        Supabase-->>App: 4. Realtime 推送: 状态=Pending
        App-->>User: 显示“排队中...” (灰色状态)
    end

    Note over Supabase, Worker: === 阶段二：异步消费与生产 (Async Worker) ===
    loop 智能轮询 (Smart Polling)
        Worker->>Supabase: 5. 抢占任务 (RPC: pop_next_task) <br/> 使用 SKIP LOCKED 锁
    end
    Supabase-->>Worker: 返回任务 (Task A)
    
    Worker->>Supabase: 6. 下载视频 (Storage)
    
    rect rgb(240, 248, 255)
        Note right of Worker: GPU 密集计算
        Worker->>Worker: 7a. 3DGS 训练 (Nerfstudio)
        Worker->>Worker: 7b. 场景理解 & 自动打标 (Qwen-VL)
    end
    
    Worker->>Supabase: 8. 上传 .ply 模型 (Storage Bucket: output)
    Worker->>Supabase: 9. 资产入库 (Insert model_assets) <br/> 包含 Embedding 向量 (pgvector)
    Worker->>Supabase: 10. 更新任务状态 (Update status="completed")
    
    Supabase-->>App: 11. Realtime 推送: 状态=Completed
    App-->>User: 灵动岛/通知栏弹出 "记忆构建完成"

    Note over User, Worker: === 阶段三：浏览与 RAG 搜索 (Visualize & Search) ===
    User->>App: 12. 查看“我的模型”
    App->>Supabase: 查询资产表 (Select model_assets) <br/> RLS 自动过滤
    Supabase-->>App: 返回列表 (含 .ply 下载链接)
    App->>App: 13. 加载 3D 视图 (Flutter 渲染)
    
    User->>App: 14. 搜索 "我的钥匙在哪"
    App->>Edge: 15. 调用搜索接口 (POST /functions/v1/search)
    
    rect rgb(255, 250, 240)
        Note right of Edge: Serverless 语义检索
        Edge->>LLM: 16. 获取文本向量 (Embedding)
        LLM-->>Edge: 返回 Vector
        Edge->>Supabase: 17. 向量匹配 (RPC: match_model_assets) <br/> pgvector 相似度计算
        Supabase-->>Edge: 返回 Top-K 结果
    end
    
    Edge-->>App: 返回 JSON 结果
    App-->>User: 展示搜索结果并定位视角
```

------



### 📚 参考与致谢 (References & Acknowledgements)

本项目是站在巨人肩膀上的探索。核心算法与渲染能力大量借鉴并集成了以下优秀的开源项目，特此致谢：

#### Core Algorithms (算法核心)

- **[nerfstudio](https://github.com/nerfstudio-project/nerfstudio)**: 提供了模块化最强的 NeRF/3DGS 训练框架，本项目的训练管线基于 `splatfacto` 模型修改。
- **[gsplat](https://github.com/nerfstudio-project/gsplat)**: 极速 CUDA 光栅化后端，为云端训练提供了性能保障。
- **[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)**: Inria 的原始论文实现，奠定了理论基础。
- **[SAM3D](https://github.com/ ScreenVerse/sam-3d-objects)**: 单图 3DGS 生成框架，支持从单张照片重建高质量 3D 模型。
- **[SHARP](https://github.com/apple/ml-sharp)**: Apple 的高质量单图 3DGS 生成模型，通过神经网络直接预测高斯泼溅参数。

#### Infrastructure & AI (基础设施与人工智能)

- **[Supabase](https://github.com/supabase/supabase)**: 本项目的灵魂。提供了开箱即用的 Auth、Storage 和 Realtime 能力，让我们能专注于 3D 业务逻辑。
- **[pgvector](https://github.com/pgvector/pgvector)**: PostgreSQL 的向量扩展，替代了 ChromaDB，为本项目提供了高性能的 RAG 检索能力。
- **[Qwen-VL](https://github.com/QwenLM/Qwen-VL)**: 强大的多模态大模型，赋予了 3D 场景”被理解”的能力（自动打标与描述）。
- **[Ultralytics YOLO](https://www.ultralytics.com/zh/yolo)**: 使用了最新的 **YOLO12** (2025) 和 **YOLO26** 目标检测模型，为场景提供实时物体识别与开放词汇检测能力。YOLO12 引入以注意力为中心的架构，YOLO26 专为边缘和低功耗设备设计。
- **[Qwen3-Embedding](https://help.aliyun.com/zh/model-studio/general-text-embedding/)**: 阿里巴巴 2025 年最新发布的文本嵌入模型系列，在多项 benchmarks 上达到 SOTA 水平，为本项目的 RAG 检索提供高精度语义向量支持。

#### Rendering & Viewer (渲染与查看器)

- **[GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)**: 基于 Three.js 的 Web 端查看器，是我们移动端 WebView 渲染的灵感来源。
- **[antimatter15/splat](https://github.com/antimatter15/splat)**: 另一个优秀的 WebGL 实现，提供了早期的思路参考。

#### Mobile Client Ecosystem (移动端与视觉设计)

- **[TDesign Flutter](https://tdesign.tencent.com/flutter)**: 腾讯开源的企业级移动端组件库。
  - **Design System**: 采用了 TDesign 的设计规范与深色模式 (`theme.json` 来源：[TDesign Dark Mode](https://tdesign.tencent.com/flutter/dark-mode))。
  - **Iconography**: 使用了 TDesign Icons (如 Video-camera Icon) 保持视觉统一 ([来源](https://tdesign.tencent.com/icons))。
- **HarmonyOS Sans**: 采用华为鸿蒙字体作为应用主字体，优化多屏阅读体验。
- **Dependencies (关键依赖)**:
  - `Dart SDK`: 3.10.7
  - `tdesign_flutter`: ^0.2.7
  - `camera`: ^0.11.3
  - `photo_manager`: ^3.8.3
  - `video_thumbnail_pro`: ^0.7.3
  - `image_picker`: ^1.1.2
  - `path_provider`: ^2.1.5
  - `path`: ^1.9.1
  - `cupertino_icons`: ^1.0.8




我们相信，科技不应只是冰冷的参数竞赛。 **最好的科技，是让此在（Dasein）不再孤独，让瞬间成为永恒。**

<div align="center">


<sub>Made with ❤️ by the BrainDance Team: 烫锟斤拷烫. Dedicated to everyone fighting against entropy.</sub>

</div>
