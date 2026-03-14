# BrainDance AI Engine - 3DGS Cloud Node

> BrainDance 当前主用的计算节点，负责任务监听、三维重建、结果上传和语义资产回写。

本项目是 BrainDance 的核心 AI 算力节点，负责监听云端任务、自动下载用户上传的视频、执行全自动 3D 重建流程（COLMAP/GLOMAP + Splatfacto），并将最终的 3D 模型（PLY）与日志实时回传至云端。

## 文档导航

本文档分为两个部分：

### 第一部分：Cloud Node 部署指南
- [环境准备](#环境准备) - 硬件与软件依赖
- [运行指南](#运行指南) - 云端监听模式与本地调试模式
- [论文 Pipeline 使用指南](#论文-pipeline-使用指南) - DA3+SuGaR / DA3+2DGS / Sparse2DGS
- [工作流原理](#工作流原理) - 任务从创建到完成的基本流程

适合读者：准备部署或调试 Worker 的开发者。

### 第二部分：RAG 系统设计文档
- [设计核心逻辑](#1-设计核心逻辑-design-philosophy) - 架构原则
- [数据库结构](#2-数据库结构-schema-structure) - Schema 设计
- [内容生成策略](#3-内容生成策略-content-strategy) - 向量化策略
- [搜索与交互流程](#4-搜索与交互流程-search-workflow) - 检索流程

适合读者：需要理解语义检索数据结构和搜索流程的开发者。

## 核心特性

- **任务驱动**：通过 Supabase 中的 `processing_tasks` 解耦前端和计算节点。
- **自动流水线**：覆盖 `抽帧 -> 位姿解算 -> 训练 -> 后处理 -> 上传` 的基本过程。
- **状态回写**：运行日志和任务状态会持续回写数据库，便于前端和 Dashboard 观察进度。
- **双模式运行**：既支持云端监听模式，也支持直接处理本地文件的调试模式。

## 第三方依赖与论文引用

本目录集成了多个上游研究项目与子模块，例如 `nerfstudio`、`2d-gaussian-splatting`、`Sparse2DGS`、`SuGaR`、`Depth-Anything-3`、`ml-sharp`、`sam-3d-objects`。

这些依赖的论文、Citation、致谢和许可证已单独整理在：

- [THIRD_PARTY_ATTRIBUTIONS.md](./THIRD_PARTY_ATTRIBUTIONS.md)

使用本目录中的代码、模型与子模块时，请同时遵守各项目自身许可证以及可能存在的模型许可证、非商业限制。

## 项目结构

```text
BrainDance/
├── main.py                    # [入口] 程序启动入口 (模式选择器)
├── .env                       # [配置] 环境变量 (Supabase Key 等敏感信息)
├── src/
│   ├── config.py              # [配置] PipelineConfig 配置类定义
│   ├── core/                  # [核心逻辑]
│   │   ├── pipeline.py        # 本地单次运行入口
│   │   ├── worker.py          # Supabase 轮询与任务执行
│   │   ├── factory.py         # task_type 到 pipeline 的映射
│   │   └── pipeline_base.py   # 各类 pipeline 的公共逻辑
│   ├── pipelines/             # [任务流水线]
│   │   ├── video_3dgs.py
│   │   ├── single_image_sam3d.py
│   │   ├── single_image_sharp.py
│   │   ├── da3_feed_forward_pipeline.py
│   │   ├── da3_sugar_pipeline.py
│   │   ├── da3_2dgs_pipeline.py
│   │   └── sparse2dgs.py
│   ├── modules/               # [功能模块]
│   │   ├── scene_analyzer.py
│   │   ├── knowledge_base.py
│   │   ├── rag_memory.py
│   │   ├── spatial_anchor.py
│   │   └── sam3d_engine/
│   ├── libs/                  # [上游仓库与子模块]
│   │   ├── nerfstudio/
│   │   ├── SuGaR/
│   │   ├── Sparse2DGS/
│   │   ├── 2d-gaussian-splatting/
│   │   ├── Depth-Anything-3/
│   │   ├── ml-sharp/
│   │   └── sam-3d-objects/
│   └── utils/                 # [工具库]
│       ├── common.py
│       ├── cv_algorithms.py
│       ├── geometry.py
│       └── ply_utils.py
├── tests/                     # 测试脚本
├── scripts/                   # 辅助脚本
└── ENVIRONMENT.md             # 环境依赖说明
```

## 环境准备

### 1. 硬件要求

**最低配置**
* **GPU**: NVIDIA RTX 5070 12GB 或更高
* **CPU**: Intel Core i5-14600KF 或同级处理器
* **RAM**: 64GB
* **OS**: Linux (推荐 Ubuntu 22.04) 或 Windows WSL2
* **CUDA**: 12.8

**推荐配置（生产环境）**
* **GPU**: NVIDIA L20 45GB × 2 (双卡配置)
* **CPU**: Intel Xeon Platinum 8260 × 2 (双路配置)
* **RAM**: 512GB
* **OS**: Linux (Ubuntu 22.04)
* **CUDA**: 12.8

### 2. 软件依赖

依赖以当前仓库实际环境为准，统一说明见：

- `ENVIRONMENT.md`（系统版本、COLMAP/GLOMAP/FFmpeg、全量 Python 依赖、编译/非编译分类、`nerfstudio` 子模块安装）

基础安装流程：

```bash
# 1. 创建并激活 Conda 环境
conda create -n Braindance python=3.10
conda activate Braindance

# 2. 拉取子模块（包含项目 fork 的 nerfstudio）
cd /path/to/BrainDance
git submodule sync --recursive
git submodule update --init --recursive
git lfs pull
git submodule foreach --recursive 'git lfs pull || true'

# 3. 安装最小依赖
cd ai_engine/3dgs
pip install -r requirements.txt

# 4. 强制使用仓库内 nerfstudio（已包含 weights_only=False 修复）
pip uninstall -y nerfstudio
pip install -e src/libs/nerfstudio

# 5. 验证导入路径
python -c "import nerfstudio, pathlib; print(pathlib.Path(nerfstudio.__file__).resolve())"
# 预期路径包含: ai_engine/3dgs/src/libs/nerfstudio
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

## 数据库设计 (Supabase)

这一部分只列出 Worker 当前依赖的最小对象，不替代表结构设计文档。

### Table: `processing_tasks`

| 字段名 | 类型 | 描述 |
| --- | --- | --- |
| `id` | `uuid` | 主键，自动生成 |
| `user_id` | `text` | 用户 ID |
| `scene_id` | `text` | 场景/项目唯一标识 |
| `status` | `text` | 状态: `pending` (排队), `processing` (处理中), `completed` (完成), `failed` (失败) |
| `logs` | `jsonb` | 实时日志数组，结构: `[{"ts": 123, "msg": "..."}]` |
| `created_at` | `timestamp` | 创建时间 |

### Table: `model_assets`

| 字段名 | 类型 | 描述 |
| --- | --- | --- |
| `scene_id` | `text` | 场景唯一标识，用于 upsert |
| `user_id` | `text` | 所有者 ID |
| `description` | `text` | 场景描述 |
| `objects` | `text[]` | 场景物体列表 |
| `tags` | `text[]` | 标签 |
| `embedding` | `vector(1536)` | 语义向量 |
| `ply_path` | `text` | 模型文件路径 |
| `preview_img_path` | `text` | 预览图路径 |
| `meta_info` | `jsonb` | 质量分、引擎版本等附加信息 |

### Table: `memory_poses`

该表用于保存帧级空间锚点与相关向量数据，供后续空间检索使用。

### Storage Bucket: `braindance-assets`

文件存储路径规范：

- **输入视频**: `{user_id}/{scene_id}/raw/video.mp4`
- **输入单图**: `{user_id}/{scene_id}/raw/image.png`
- **输入多图**: `{user_id}/{scene_id}/raw/images.zip`
- **输出模型**: `{user_id}/{scene_id}/output/point_cloud.ply`
- **输出位姿**: `{user_id}/{scene_id}/output/webgl_poses.json`

## 运行指南

### 模式 A：云端监听模式

启动 Worker，持续监听 Supabase 的 `pending` 任务。

```bash
python main.py

```

输出示例：

> 🚀 [CloudWorker] 启动! 正在监听表: [processing_tasks]
> .....
> 📥 [接收任务] ID: ... | Scene: party_01

### 模式 B：本地调试模式

不经过数据库，直接处理本地视频文件。

```bash
python main.py /path/to/your/video.mp4

```

输出示例：

> 💿 启动本地模式: video.mp4
> ... (开始直接运行 Pipeline)

## 论文 Pipeline 使用指南

下面三条是近期新增的“论文复现/组合型”流水线，统一通过 `processing_tasks.task_type` 触发。

### 1) `da3_sugar` / `da3+sugar`（DA3 + SuGaR）

- 用途：先用 DA3 做位姿与深度，再由 SuGaR 使用 mesh/SDF 约束 3DGS；通常质量更高，但速度更慢。
- 输入：`{user_id}/{scene_id}/raw/video.mp4`
- 典型场景：室内空间、需要后续 mesh/refinement 的资产化流程。

常用参数（写入 `task_params`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `regularization` | `dn_consistency` | SuGaR 正则类型：`dn_consistency` / `density` / `sdf` |
| `refinement_time` | `short` | 精炼时长：`short` / `medium` / `long` |
| `fast_mode` | `true` | 快速模式，通常更快产出可交付 PLY |
| `high_poly` | `false` | 是否启用更高面数相关流程 |
| `gpu_index` | 环境默认 | 映射到 `CUDA_VISIBLE_DEVICES` |
| `da3_repo_path` | 自动探测 | DA3 仓库路径（需含 `da3_to_sugar_pipeline.sh`） |
| `sugar_repo_path` | 自动探测 | SuGaR 仓库路径（需含 `train_fast.py`） |

### 2) `da3_2dgs` / `da3+2dgs`（DA3 + 2DGS）

- 用途：少量图片输入，先 DA3 解算，再用 2DGS 训练；可视为 Nerfstudio 3DGS 的替代方案，在一定程度上质量更高，输出 2DGS。
- 输入优先级：`raw/images.zip`（推荐） -> 下载失败时回退 `raw/image.png`
- 典型场景：移动端多张照片上传、希望用较少图像得到可用点云。

常用参数（写入 `task_params`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `iterations` | `7000` | 2DGS 训练步数（质量优先可提到 30k） |
| `max_images` | `60` | 最大参与训练图片数 |
| `keep_ratio` | `1.0` | 输入保留比例（0~1） |
| `gpu_index` | `1` | 训练显卡索引 |
| `enable_scene_analysis` | `false` | 是否先做 AI 质检 |
| `render_after_train` | `false` | 是否训练后调用 `render.py` |
| `dgs_repo_path` | 自动探测 | 2DGS 仓库路径（需含 `train.py`） |

### 3) `sparse2dgs`（Sparse2DGS）

- 用途：少量图片输入 + COLMAP 稀疏重建 + Sparse2DGS 训练，输出 2DGS。
- 输入：`{user_id}/{scene_id}/raw/images.zip`
- 最低要求：至少 3 张有效图片（少于 3 张会直接失败）。
- 典型场景：照片数量有限，但希望比常规少图流程有更强几何约束。

常用参数（写入 `task_params`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `iterations` | `7000` | Sparse2DGS 训练步数 |
| `resolution` | `2` | 对应 `train.py -r`，值越小分辨率越高、耗时更大 |
| `depth_ratio` | `1.0` | 深度损失权重比例 |
| `lambda_dist` | `1000` | 几何约束强度 |
| `conda_env` | `Braindance` | 用于 `conda run -n` 的环境名 |
| `sparse2dgs_repo_path` | `/ltx-data/Sparse2DGS` | Sparse2DGS 仓库路径 |
| `colmap_matcher` | `exhaustive_matcher` | COLMAP 匹配器（少图推荐 exhaustive） |
| `colmap_mapper` | `mapper` | COLMAP mapper（失败时会回退 `mapper`） |

### 选型建议

| 目标 | 推荐 task_type | 原因 |
| --- | --- | --- |
| 视频输入，追求更高质量且可接受更慢速度 | `da3_sugar` | SuGaR 的 mesh/SDF 约束能提升质量，但耗时更高 |
| 少图（2~60 张），希望替代 Nerfstudio 3DGS 并输出 2DGS | `da3_2dgs` | 2DGS 作为替代方案，在一定程度上质量更高 |
| 少量图片直接生成 2DGS | `sparse2dgs` | 针对少图输入的 Sparse2DGS 路线 |

### 前端接入最小步骤

1. 上传素材到标准路径（`video.mp4` / `images.zip` / `image.png`）。
2. 向 `processing_tasks` 插入任务：`status='pending'` + 正确 `task_type`。
3. 可选传入 `task_params` 调参（建议先默认参数跑通，再微调）。
4. 监听 `status` 和 `logs`，完成后从 `output/point_cloud.*` 获取交付文件。

## 工作流原理

1. **用户** 在前端上传视频，Supabase Storage 存入文件，Database 插入一条 `status='pending'` 的记录。
2. **Worker** (`worker.py`) 轮询检测到新记录，将状态改为 `processing`。
3. **下载**：自动从 Storage 下载 `video.mp4` 到本地临时目录。
4. **生成**：调用 `pipeline.py`，执行 COLMAP 解算与 Gaussian Splatting 训练。
5. **同步**：训练过程中，日志通过回调函数实时写入 Database 的 `logs` 字段。
6. **上传**：训练结束后，将生成的 PLY 模型上传回 Storage。
7. **完成**：更新 Database 记录状态为 `completed`。

---

# 第二部分：RAG 系统设计文档

> 这一部分说明 Worker 侧资产入库和语义搜索的数据组织方式。

本部分说明 BrainDance 当前的“语义 + 元数据”混合检索设计。目标不是单纯存一个模型文件，而是把模型整理成后续可以检索和定位的资产。

## 1. 设计核心逻辑 (Design Philosophy)

目前的设计主要遵循以下 4 个原则：

1. **资产化 (Permanent Assets)**
   分离“流水线任务”(`processing_tasks`) 和“数字资产”(`model_assets`)。
   任务是暂时的，资产是长期保留的成功结果。

2. **富文本上下文 (Rich Context Embedding)**
   向量化不只基于一句描述，而是把 **核心物体 + 详细描述 + 物品清单 + 环境标签** 组合成一个加权文本块，提高搜索命中率。

3. **唯一性 (Idempotency)**
   以 `scene_id` 为唯一锚点。同一个场景无论跑多少次，数据库里只保留最新的一条资产记录。

4. **混合检索 (Hybrid Search)**
   向量搜索负责模糊匹配，标量过滤负责时间范围、质量分等精确限制。

## 2. 数据库结构 (Schema Structure)

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

## 3. 内容生成策略 (Content Strategy)

当 `Worker` 调用 `knowledge_base.py` 入库时，除了写入数据，还会做一次面向搜索的文本整理。

#### A. 向量内容的构成

为了让搜索更精准，我们在生成 `embedding` 前，会构造一段**加权文本 (Weighted Text)**：

```text
核心物体: [物品A]。 [物品A]。     <-- 重复两次，人为增加 30-50% 的权重
详细描述: [AI生成的长难句...]。    <-- 提供上下文
包含物品: [物品A, 物品B...]。     <-- 提供关键词覆盖
环境标签: [标签1, 标签2...]。     <-- 提供背景信息

```

- **为什么要重复？** Embedding 模型通常对文本开头更敏感。重复核心物体，是为了让向量更稳定地靠近该物体类别。

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

## 4. 搜索与交互流程 (Search Workflow)

当用户发起搜索时，系统经历以下步骤：

1. **意图解析 (Python/LLM)**
   用户输入“找一下上周做的红色杯子”后，LLM 会提取：
   `Query`: `"红色杯子"`
   `Filter`: `Start="2025-XX-XX", End="2025-XX-XX"`

2. **向量化 (Python/Embedding)**
   将 `"红色杯子"` 转换为 1536 维向量。

3. **数据库执行 RPC (Supabase/PostgreSQL)**
   调用 `match_model_assets` 函数。
   第一层做向量相似度匹配，第二层做时间范围等标量过滤，最后按相似度排序。

4. **结果返回**
   返回 `ply_path`，前端或脚本即可继续下载和展示模型。


### 小结

这套设计的重点，不是把模型文件单独存起来，而是让模型同时具备描述、物体清单、标签和时间信息，后续才能稳定支撑自然语言检索。

---

## 相关文档

- [上层文档](../README.md) - BrainDance AI Engine 总览
- [快速开始指南](../../docs/01-入门指南/快速开始.md) - 项目级启动路径
- [API 接口文档](../../docs/API_DOC.md) - 完整接口说明
- [返回顶部](#braindance-ai-engine---3dgs-cloud-node)
