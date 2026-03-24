
---

# 🧠 BrainDance AI Engine - 3DGS Cloud Node

> **BrainDance 核心计算后端**：基于 Supabase 消息队列与 Nerfstudio 的云原生 3D 高斯泼溅（3DGS）生成引擎。

本项目是 BrainDance 的核心 AI 算力节点，负责监听云端任务、自动下载用户上传的视频、执行全自动 3D 重建流程（COLMAP/GLOMAP + Splatfacto），并将最终的 3D 模型（PLY）与日志实时回传至云端。

## ✨ 核心特性

* **☁️ 云原生架构**：通过 Supabase 实现完全解耦的“生产者-消费者”模式，前端只管发任务，后端自动排队处理。
* **🔄 全自动流水线**：一键完成 `视频抽帧` -> `位姿解算 (COLMAP)` -> `AI 场景分析` -> `3DGS 训练` -> `模型后处理`。
* **📡 实时状态同步**：训练进度与详细日志实时推送到 Supabase 数据库，支持前端远程监控。
* **🚀 混合运行模式**：支持 `本地调试模式`（直接处理文件）和 `云端监听模式`（无人值守工单处理）。

## 📂 项目结构

```text
BrainDance/
├── main.py                    # [入口] 程序启动入口 (模式选择器)
├── .env                       # [配置] 环境变量 (Supabase Key 等敏感信息)
├── src/
│   ├── config.py              # [配置] PipelineConfig 配置类定义
│   ├── core/                  # [核心逻辑]
│   │   ├── pipeline.py        # 3DGS 生成主流水线 (run_pipeline)
│   │   └── worker.py          # Supabase 轮询器与上传下载逻辑
│   ├── modules/               # [功能模块]
│   │   ├── ai_segmentor.py    # AI 语义分割与物体提取
│   │   ├── glomap_runner.py   # GLOMAP/COLMAP 位姿解算封装
│   │   ├── image_proc.py      # 图像预处理 (抽帧、锐化等)
│   │   └── nerf_engine.py     # Nerfstudio 训练引擎封装
│   └── utils/                 # [工具库]
│       ├── common.py          # 通用工具 (如时间格式化)
│       ├── cv_algorithms.py   # 计算机视觉算法 (Mask处理、包围盒计算)
│       ├── geometry.py        # 几何分析 (自动计算裁剪框)
│       └── ply_utils.py       # PLY 点云后处理工具

```

## 🛠️ 环境准备

### 1. 硬件要求

* **GPU**: NVIDIA RTX 30/40/50 系列 (显存 >= 8GB)
* **OS**: Linux (推荐 Ubuntu 22.04) 或 Windows WSL2
* **CUDA**: 11.8 或 12.x

### 2. 软件依赖

确保已安装 `nerfstudio` 和 `ffmpeg`。

```bash
# 1. 创建并激活 Conda 环境
conda create -n braindance python=3.10
conda activate braindance

# 2. 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖库
pip install nerfstudio supabase python-dotenv

```

### 3. 环境变量配置 (.env)

在项目根目录下新建 `.env` 文件，填入你的 Supabase 配置：

```ini
# Supabase 连接信息
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=your_service_role_key_here

# 存储桶与表名配置
SUPABASE_BUCKET=braindance-assets
SUPABASE_TABLE=processing_tasks

```

## 💾 数据库设计 (Supabase)

请确保 Supabase 中包含以下 Table 和 Storage Bucket。

### Table: `processing_tasks`

| 字段名 | 类型 | 描述 |
| --- | --- | --- |
| `id` | `uuid` | 主键，自动生成 |
| `user_id` | `text` | 用户 ID |
| `scene_id` | `text` | 场景/项目唯一标识 |
| `status` | `text` | 状态: `pending` (排队), `processing` (处理中), `completed` (完成), `failed` (失败) |
| `logs` | `jsonb` | 实时日志数组，结构: `[{"ts": 123, "msg": "..."}]` |
| `created_at` | `timestamp` | 创建时间 |

### Storage Bucket: `braindance-assets`

文件存储路径规范：

* **输入视频**: `{user_id}/{scene_id}/raw/video.mp4`
* **输出模型**: `{user_id}/{scene_id}/output/point_cloud.ply`

## 🚀 运行指南

### 模式 A：云端监听模式 (生产环境)

启动 Worker，持续监听 Supabase 的 `pending` 任务。

```bash
python main.py

```

*输出示例：*

> 🚀 [CloudWorker] 启动! 正在监听表: [processing_tasks]
> .....
> 📥 [接收任务] ID: ... | Scene: party_01

### 模式 B：本地调试模式 (开发测试)

不经过数据库，直接处理本地视频文件。

```bash
python main.py /path/to/your/video.mp4

```

*输出示例：*

> 💿 启动本地模式: video.mp4
> ... (开始直接运行 Pipeline)

## 🔄 工作流原理

1. **用户** 在前端上传视频，Supabase Storage 存入文件，Database 插入一条 `status='pending'` 的记录。
2. **Worker** (`worker.py`) 轮询检测到新记录，将状态改为 `processing`。
3. **下载**：自动从 Storage 下载 `video.mp4` 到本地临时目录。
4. **生成**：调用 `pipeline.py`，执行 COLMAP 解算与 Gaussian Splatting 训练。
5. **同步**：训练过程中，日志通过回调函数实时写入 Database 的 `logs` 字段。
6. **上传**：训练结束后，将生成的 PLY 模型上传回 Storage。
7. **完成**：更新 Database 记录状态为 `completed`。

---

*BrainDance Team © 2026*

这就是你现在的 **BrainDance 智能资产库 (RAG System)** 的完整设计文档。

这是一个**“语义+元数据”混合检索系统**，旨在把你的 3D 模型从“死文件”变成“活知识”。

---

### 1. 设计核心逻辑 (Design Philosophy)

目前的架构设计遵循以下 4 个核心原则：

1. **资产化 (Permanent Assets)**：
* 分离“流水线任务”(`processing_tasks`) 和 “数字资产”(`model_assets`)。
* 任务是暂时的（会失败、重试），资产是永久的（只存成功的、高质量的结果）。


2. **富文本上下文 (Rich Context Embedding)**：
* 向量化不仅仅是基于简单的描述。我们将 **“核心物体 + 详细描述 + 物品清单 + 环境标签”** 组合成一个“加权文本块”，提高搜索命中率。


3. **唯一性 (Idempotency)**：
* 以 `scene_id` 为唯一锚点。同一个场景无论跑多少次，数据库里永远只有最新的一条记录，避免“影分身”。


4. **混合检索 (Hybrid Search)**：
* **向量搜索 (Vector)**：负责模糊匹配（“找个像红杯子的”）。
* **标量过滤 (Scalar)**：负责精确限制（“只要上周生成的”、“分数大于80的”）。



---

### 2. 数据库结构 (Schema Structure)

这是目前 Supabase 中 `model_assets` 表的物理结构：

| 字段名 | 类型 | 作用 | 备注 |
| --- | --- | --- | --- |
| **`id`** | `uuid` | 主键 | 自动生成 |
| **`scene_id`** | `text` | **唯一标识符** | 对应文件名，设有 `UNIQUE` 约束，用于去重/更新 |
| **`user_id`** | `text` | 归属用户 | 用于多用户隔离（预留） |
| **`description`** | `text` | 场景描述 | AI 生成的完整自然语言描述 |
| **`objects`** | `text[]` | 物品清单 | 数组，如 `["红色杯子", "木桌"]` |
| **`tags`** | `text[]` | 环境标签 | 数组，如 `["室内", "弱光", "办公"]` |
| **`embedding`** | `vector(1536)` | **语义向量** | 核心数据，由加权文本生成 (OpenAI/Qwen v2 标准) |
| **`ply_path`** | `text` | 文件路径 | 也就是 Storage 里的 `Key`，用于下载 |
| **`preview_img_path`** | `text` | 预览图 | (预留) 未来可存缩略图 URL |
| **`meta_info`** | `jsonb` | 技术元数据 | 存分数、引擎版本等，如 `{"quality_score": 85}` |
| **`created_at`** | `timestamptz` | 创建时间 | 用于时间范围过滤 |

---

### 3. 内容生成策略 (Content Strategy)

当 `Worker` 调用 `knowledge_base.py` 入库时，它不仅仅是存数据，还在做**数据清洗和加权**。

#### A. 向量内容的构成

为了让搜索更精准，我们在生成 `embedding` 前，会构造一段**加权文本 (Weighted Text)**：

```text
核心物体: [物品A]。 [物品A]。     <-- 重复两次，人为增加 30-50% 的权重
详细描述: [AI生成的长难句...]。    <-- 提供上下文
包含物品: [物品A, 物品B...]。     <-- 提供关键词覆盖
环境标签: [标签1, 标签2...]。     <-- 提供背景信息

```

* **为什么要重复？** Embedding 模型通常对文本开头的内容更敏感。重复核心物体可以让向量在多维空间中更靠近该物体类别。

#### B. 元数据的构成

`meta_info` 字段是一个 JSONB，目前存储：

```json
{
  "quality_score": 85,          // AI 打分
  "quality_reason": "光照充足",  // AI 评价
  "engine_version": "nerfstudio-splatfacto"
}

```

---

### 4. 搜索与交互流程 (Search Workflow)

当用户发起搜索时，系统经历以下步骤：

1. **意图解析 (Python/LLM)**：
* 用户输入：“找一下上周做的红色杯子”
* LLM 提取：
* `Query`: "红色杯子" (去掉了时间词)
* `Filter`: Start="2025-XX-XX", End="2025-XX-XX"




2. **向量化 (Python/Embedding)**：
* 将 "红色杯子" 转换为 `[0.12, -0.98, ...]` (1536维)。


3. **数据库执行 RPC (Supabase/PostgreSQL)**：
* 调用 `match_model_assets` 函数。
* **第一层 (Vector)**：计算余弦相似度，找出所有像“红色杯子”的模型。
* **第二层 (Filter)**：执行 SQL `WHERE created_at BETWEEN ...`，过滤掉不在时间范围内的。
* **排序**：按相似度从高到低返回。


4. **结果返回**：
* 返回 `ply_path`，前端或脚本即可直接下载模型。



### 总结

你现在拥有的不仅仅是一个文件存储系统，而是一个**具备“理解能力”的 3D 资产管理平台**。它知道你每个模型长什么样、包含什么东西、是什么时候做的，并且能通过自然语言瞬间找到它。

