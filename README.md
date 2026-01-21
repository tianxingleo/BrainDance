<div align="center">


# 🕯️ BrainDance | 流光 · 记

**"物理世界注定走向无序，而我们在比特世界重建永恒。"**

"——这是物理世界的搜索引擎，第二大脑的空间可视化。"

[English](./README.en.md) | [简体中文](README.md)

### 🏆 面向空间计算时代的三维语义记忆引擎

#### An Anti-Entropy Engine for Human Memory

![Supabase](https://img.shields.io/badge/Supabase-Enabled-3ECF8E?logo=supabase&logoColor=white) ![Flutter](https://img.shields.io/badge/Flutter-Client-02569B?logo=flutter&logoColor=white) ![Python](https://img.shields.io/badge/Python-Worker-3776AB?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-blue)

</div>

## 📖 项目概述 (Overview)

**BrainDance (流光 · 记)** 是一个**面向移动端的"可检索三维记忆库"**。

不同于传统的相册只能记录二维的"画面"，BrainDance 利用 **3D Gaussian Splatting (高斯泼溅)** 等计算机图形学前沿技术，将现实世界的物理空间（如你的房间、即将拆迁的老街、珍藏的手办）以 1:1 的高保真度转化为数字资产。

更进一步，我们结合了 **Multimodal AI (多模态大模型)** 与 **RAG (检索增强生成)** 技术，让这些三维场景具备了"语义"。你可以像在搜索引擎里一样**搜索现实世界**——问它"我的钥匙落在哪里了？"，镜头便会自动飞跃并聚焦到那一刻的时空。

### 核心特性

- **📷 移动端低成本扫描**：利用手机采集视频流与位姿数据，通过大量优化与AI介入，极大提高了低拍摄质量下生成的3DGS模型质量，大大降低 3DGS 建模门槛。
- **🔍 空间语义检索 (Spatial RAG)**：结合多模态大模型理解场景内容，实现"Ctrl+F"搜索物理世界。
- **⏳ 时光剥离 (Time Peeling)**：在同一坐标系下叠加多维时间切片，实现从"现在"回溯到"过去"的视觉体验。
- **☁️ 端云协同渲染**：Mobile 采集 -> Cloud 高性能计算 -> Mobile/XR 轻量化查看。

## 📜 序言：对抗熵增的战争

**物理学**告诉我们，宇宙的终极命运是**熵增**。 房屋会变旧，物品会破碎，秩序会变成混乱。

在**生物学**层面，熵增表现为**遗忘**。海马体的衰退让我们忘记了回家的路，忘记了爱人的脸。 在**社会学**层面，熵增表现为**消亡**。城市更新的推土机下，老街、胡同与那些烟火气终将化为尘土。

现有的技术——2D 照片与视频，只是对现实苍白的"截屏"。它们丢失了深度，丢失了光影，更丢失了空间感。它们无法对抗遗忘，因为它们本身就是扁平的。

**BrainDance (流光 · 记)** 不仅仅是一个 App，它是人类对抗时间熵增的通用工具。 利用 **3D Gaussian Splatting (高斯泼溅)** 与 **Multimodal AI (多模态大模型)**等前沿技术，我们试图捕获光的场域，**在数字世界里建立负熵**，为每个人、每座城，留下一份可以穿越时间的**空间档案**。

## 🌌 价值坐标系：微观、宏观与纵深

BrainDance 的价值架构跨越了三个维度，构建了一个从个体到文明的完整记忆生态：

### 1. 微观尺度 (The Micro Scale)：个体记忆的数字化永生

每个人都是自己记忆的主角。通过 BrainDance，你可以：

- 用手机扫描自己的房间、收藏的手办、童年的玩具。
- 那些终将破损、遗失、遗忘的实物，将以数字形式永久保存。
- 结合 AI 语义标注，你可以像搜索文本一样搜索这些三维记忆——"我那套灌篮高手的手办在哪里？""去年在云南买的那条披肩是什么颜色？"
- **价值**：对抗个体遗忘，构建第二大脑的数字记忆层。

### 2. 中观尺度 (The Meso Scale)：城市文化遗产的抢救性数字化

推土机不会因为你没来得及记录就停止运转。 BrainDance 可以：

- 快速扫描整条老街、胡同、历史建筑群。
- 以 1:1 的高保真度留存城市记忆。
- 结合 RAG 技术，这些三维模型可以承载历史叙事——每块砖、每扇窗背后都有故事。
- **价值**：为城市留下可检索、可交互的"数字遗体"，对抗城市化进程中的文化熵增。

### 3. 宏观尺度 (The Macro Scale)：文明的负熵体

当城市、国家、文明的记忆碎片被 BrainDance 连接，将形成：

- **可检索的历史**：不再需要翻阅枯燥的档案或古籍，直接"走进"历史场景。
- **可交互的文明**：不再只是看历史纪录片，而是可以在三维空间中"行走"于历史场景中。
- **可传承的遗产**：不再担心战争、天灾、人祸导致的文明断代，数字化的记忆永远留存。
- **价值**：构建人类文明的负熵体，对抗熵增的终极使命。

## 🚀 快速开始

### 环境准备

1. 确保已安装 Python 3.10+ 和 Node.js 18+
2. 安装 [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started) 并启动本地实例
3. 克隆本仓库并安装依赖：

```bash
git clone https://github.com/yourname/BrainDance.git
cd BrainDance
# 安装 Python 依赖
pip install -r ai_engine/3dgs/requirements.txt
# 安装 Flutter 依赖
cd flutter_client && flutter pub get
```

### 启动服务

```bash
# 启动 Supabase（确保已登录）
supabase start

# 启动 AI 引擎服务
cd ai_engine/3dgs && python main.py

# 启动 Flutter 应用（开发模式）
cd flutter_client && flutter run
```

详细配置请参考 [快速开始指南](./docs/快速开始指南.md)

## 🏗️ 系统架构

BrainDance 采用**端云协同**架构，由三大核心模块构成：

```
┌─────────────────────────────────────────────────────────────┐
│                      Flutter Client                          │
│  - 视频采集与预处理                                           │
│  - 3DGS 模型实时预览                                         │
│  - 语义检索 UI                                               │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API / WebSocket
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     Supabase Backend                         │
│  - 用户认证与权限管理                                         │
│  - 原始视频、元数据存储                                       │
│  - RAG 向量数据库                                             │
│  - 消息队列（任务调度）                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ 任务队列触发
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Worker (Python)                        │
│  - 视频下载与预处理                                           │
│  - 3D Gaussian Splatting 训练                                 │
│  - SAM3D 单图/多图生成 3DGS                                    │
│  - SHARP 单图生成 3DGS                                        │
│  - 场景分析与语义标注                                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ 上传结果
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   对象存储 & CDN                              │
│  - 训练好的 3DGS 模型（.ply, .safetensors）                   │
│  - 语义分析结果                                               │
└─────────────────────────────────────────────────────────────┘
```

### 核心 AI 能力

1. **3D Gaussian Splatting (3DGS)**：基于高斯泼溅的新视角合成技术，支持从视频或单图生成三维场景
2. **SAM3D**：基于 SAM (Segment Anything Model) 的三维分割技术，实现精细化的三维场景编辑
3. **SHARP**：基于单图的高斯泼溅生成技术，极大降低三维重建门槛
4. **Spatial RAG**：空间感知的检索增强生成，支持自然语言查询三维场景内容

### 技术栈详情

| 模块 | 技术栈 | 说明 |
|------|--------|------|
| 移动端 | Flutter | 跨平台移动应用开发框架 |
| 后端服务 | Supabase | 开源 Firebase 替代方案，提供 Auth、DB、Storage、Edge Functions |
| 3DGS 训练 | Python + PyTorch | 3D 高斯泼溅训练流水线 |
| 多模态理解 | Python + LangChain + OpenAI/Claude/DashScope | RAG 流水线与 LLM 集成 |
| 向量数据库 | Supabase pgvector | 存储语义向量，支持相似性搜索 |
| 任务队列 | Supabase Realtime + Edge Functions | 异步任务调度 |

## 📂 项目结构

```
BrainDance/
├── ai_engine/                 # AI 推理服务
│   ├── 3dgs/                  # 3DGS 相关代码
│   │   ├── src/               # 核心源代码
│   │   │   ├── config.py      # 配置文件
│   │   │   ├── core/          # 核心模块（工厂模式、流水线、Worker）
│   │   │   ├── modules/       # 功能模块（SAM3D, SHARP, Glomap, 图像处理等）
│   │   │   ├── pipelines/     # 流水线定义（单图/视频 3DGS 生成）
│   │   │   └── utils/         # 工具函数
│   │   ├── scripts/           # 辅助脚本
│   │   ├── tests/             # 测试用例
│   │   └── requirements.txt   # Python 依赖
│   └── models/                # 模型文件（不提交到 git，由 .gitignore 忽略）
│
├── flutter_client/            # Flutter 移动端应用
│   ├── lib/                   # Dart 源代码
│   ├── assets/                # 静态资源
│   └── pubspec.yaml           # Flutter 依赖
│
├── docs/                      # 项目文档
│   ├── API文档/               # API 设计文档
│   ├── 架构说明/              # 系统架构设计文档
│   ├── 测试指南/              # 测试用例与验收标准
│   └── 贡献指南/              # 代码贡献规范
│
├── supabase/                  # Supabase 配置
│   ├── config/                # 数据库 schema 与配置
│   ├── migrations/            # 数据库迁移脚本
│   ├── functions/             # Edge Functions
│   └── seed.sql               # 初始数据
│
└── README.md                  # 项目说明文档
```

## 🧪 测试

项目包含以下测试类型：

1. **单元测试**：针对核心工具函数和模块
2. **集成测试**：针对完整流水线的端到端测试
3. **性能测试**：针对 3DGS 训练与推理的性能基准

运行测试：

```bash
# 运行 Python 测试
cd ai_engine/3dgs && pytest tests/

# 运行 Flutter 测试
cd flutter_client && flutter test
```

## 🤝 贡献

欢迎贡献代码！请先阅读 [贡献指南](./docs/5.贡献/贡献指南.md)，了解代码规范、Pull Request 流程等内容。

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](./LICENSE) 文件。

## 🙏 致谢

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - 原始 3DGS 实现
- [SAM (Segment Anything Model)](https://github.com/facebookresearch/segment-anything) - 三维分割模型
- [supabase](https://github.com/supabase/supabase) - 开源后端即服务解决方案
- [Flutter](https://flutter.dev/) - 跨平台应用框架
- 所有为这个项目贡献代码和反馈的开发者们！
