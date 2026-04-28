# BrainDance 计算后端 Pipeline 全链路梳理

> 文档版本：2026-04-22
> 涉及代码目录：`ai_engine/3dgs/`

---

## 1. 系统总览

BrainDance（流光 · 记）是一个空间记忆系统，核心能力是将用户拍摄的视频/图片重建为 3D 高斯泼溅（3DGS）模型，并为模型注入语义标签、向量嵌入，使其可被自然语言检索。

计算后端位于 `ai_engine/3dgs/`，采用 **Supabase BaaS + Python Worker** 架构：

```
Flutter 客户端
    │  上传视频/图片到 Supabase Storage
    │  在 processing_tasks 表创建任务（status=pending）
    ▼
Supabase（PostgreSQL + Storage + Edge Functions）
    │  任务队列 + 结果存储 + 语义搜索
    ▼
AI Worker（Python）        ← 本文档重点
    │  轮询任务 → 下载资源 → 执行 Pipeline → 上传结果
    ▼
Supabase Storage / model_assets 表
```

---

## 2. 代码目录结构

```
ai_engine/3dgs/
├── main.py                          # 入口：本地模式 / 云端 Supervisor 模式
├── config/
│   ├── default.toml                 # 默认配置
│   └── local.toml                   # 本地覆盖配置（不入库）
├── .env                             # 环境变量
└── src/
    ├── config.py                    # PipelineConfig 配置类
    ├── core/
    │   ├── pipeline_base.py         # BasePipeline 抽象基类
    │   ├── factory.py               # PipelineFactory 工厂
    │   ├── supervisor.py            # WorkerSupervisor 进程管理器
    │   ├── worker.py                # CloudWorker 云端工作者
    │   └── local_runner.py          # 本地调试运行器
    ├── pipelines/
    │   ├── video_3dgs.py            # 标准 视频→3DGS 流水线
    │   ├── single_image_sam3d.py    # 单图→SAM3D 流水线
    │   ├── single_image_sharp.py    # 单图→SHARP 流水线
    │   ├── da3_feed_forward_pipeline.py  # DA3 前馈直接生成
    │   ├── da3_sugar_pipeline.py    # DA3 + SuGaR 流水线
    │   ├── da3_2dgs_pipeline.py     # DA3 + 2DGS 流水线
    │   ├── sparse2dgs.py            # 少量图片→Sparse2DGS
    │   └── image_to_3d.py           # 多图→3DGS（骨架）
    ├── modules/
    │   ├── image_proc.py            # 图片质量筛选（去模糊）
    │   ├── scene_analyzer.py        # AI 场景质检 & 语义分析
    │   ├── glomap_runner.py         # GLOMAP/COLMAP 位姿解算
    │   ├── da3_runner.py            # Depth Anything 3 位姿解算
    │   ├── ai_segmentor.py          # AI 语义分割（YOLO + SAM）
    │   ├── nerf_engine.py           # Nerfstudio 3DGS 训练
    │   ├── sharp_engine.py          # SHARP 单图重建引擎
    │   ├── spatial_anchor.py        # 空间语义锚点提取
    │   ├── rag_memory.py            # 向量记忆存储
    │   └── knowledge_base.py        # 知识库管理
    └── utils/
        ├── common.py                # 通用工具函数
        └── ply_utils.py             # PLY 模型压缩与变换
```

---

## 3. 启动流程

### 3.1 入口 `main.py`

三种运行模式，由命令行参数决定：

| 模式 | 触发条件 | 说明 |
|------|----------|------|
| **云端模式** | 无位置参数 | 启动 `WorkerSupervisor`，轮询数据库任务 |
| **子 Worker 模式** | `--child-worker` | 由 Supervisor 拉起的子进程，执行 `CloudWorker` |
| **本地模式** | 传入视频文件路径 | 直接在本地执行 Pipeline，用于调试 |

```
python main.py                        # → 云端 Supervisor 模式
python main.py --child-worker         # → 子 Worker（Supervisor 调用）
python main.py video.mp4              # → 本地模式（默认 video_dual_chain）
python main.py video.mp4 --task-type video_3dgs --project-name my_scene
```

### 3.2 配置加载 `src/config.py`

`PipelineConfig` 是贯穿整个 Pipeline 的配置中心，采用三层优先级：

```
环境变量（.env） > local.toml > default.toml
```

关键配置分组：

| 分组 | 代表字段 | 用途 |
|------|----------|------|
| API | `dashscope_api_key`, `dashscope_vl_model` | AI 视觉语言模型调用 |
| Supabase | `supabase_url`, `supabase_bucket`, `supabase_table` | 云端存储与任务表 |
| 训练 | `gpu_index`, `iterations`, `max_images`, `mapper_type` | GPU、训练轮次、图片上限、位姿引擎 |
| 仓库路径 | `sam3d_repo_path`, `da3_repo_path`, `sharp_repo_path` 等 | 各 AI 模型仓库本地路径 |
| 交付 | `model_delivery_format`, `compression_opacity_threshold` | 模型压缩输出格式 |

---

## 4. 任务调度层

### 4.1 WorkerSupervisor（`src/core/supervisor.py`）

Supervisor 是云端模式下的**进程管理器**，负责拉起并监控子 Worker 进程：

```
WorkerSupervisor
    │  轮询 worker_nodes 表的 desired_state 字段
    │  ┌──────────┬──────────┬──────────────┐
    │  │ run      │ pause    │ interrupt    │
    │  │ 确保子进程 │ 不重启   │ 发 SIGINT    │
    │  │ 在线      │ 子进程   │ 中断当前任务 │
    │  └──────────┴──────────┴──────────────┘
    ▼
  子进程: python main.py --child-worker
```

- 轮询间隔：`WORKER_SUPERVISOR_POLL_INTERVAL`（默认 3s）
- 支持优雅中断：先 SIGINT → 等待宽限期（20s）→ terminate → kill
- 心跳：将 worker 状态定期写入 `worker_nodes` 表

### 4.2 CloudWorker（`src/core/worker.py`）

CloudWorker 是**实际执行任务的主体**，由 Supervisor 拉起或手动运行：

```
CloudWorker.start()
    │  启动心跳线程
    │  进入主循环 _tick()
    ▼
_tick() 每次循环：
    1. 检查 desired_state（run/pause/interrupt）
    2. 查询 processing_tasks 表 WHERE status='pending' LIMIT 1
    3. 发现任务 → 调用 _process_task()
    4. 无任务 → 休眠 3s
```

### 4.3 任务全生命周期 `_process_task()`

一个任务从 `pending` 到终态的完整流程：

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 A: 锁定任务                                                │
│   UPDATE processing_tasks SET status='processing', logs=[]      │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 B: 下载资源                                                │
│   根据 task_type 决定下载策略：                                   │
│   - video_3dgs / video_dual_chain → 下载 video.mp4              │
│   - single_image_sam3d / single_image_sharp → 多路径探测下载图片  │
│   - sparse2dgs → 优先 images.zip，回退 video.mp4                │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 C: 执行 Pipeline                                          │
│   PipelineFactory.get_pipeline(task_type, context)              │
│   pipeline.run(input_path, params)                              │
│   ├─ 单链: _run_pipeline_once()                                 │
│   └─ 双链: _run_video_dual_chain() → 快链 + 慢链               │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 D: 模型压缩与上传                                          │
│   _compress_if_needed()  → PLY → splat/ksplat 格式              │
│   _upload_and_upsert_assets() → 上传模型 + transforms + 预览图  │
│   → 写入 model_assets 表（含语义标签、描述）                     │
│   → 写入知识库（rag_docs）                                       │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 E: 完结                                                    │
│   UPDATE processing_tasks SET status='completed'                │
├─────────────────────────────────────────────────────────────────┤
│ 异常: status → 'failed'                                         │
│ 清理: 删除临时视频、工作区目录                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 日志同步机制

Worker 采用**内存缓冲 + 覆盖写入**策略：

```python
current_task_logs = []          # 内存日志列表
_record_cloud_log(task_id, msg) # 仅含 emoji 的日志才入缓冲
_sync_log(task_id)              # UPDATE tasks SET logs=current_task_logs
```

只把带 emoji 的关键进度日志同步到云端，避免大量进度刷屏。

### 4.5 Worker 心跳与远程控制

Worker 通过 `worker_nodes` 表实现远程控制：

| 字段 | 说明 |
|------|------|
| `worker_id` | Worker 唯一标识 |
| `status` | 当前状态：idle / busy / stopping / offline |
| `current_task_id` | 正在执行的任务 |
| `desired_state` | 控制指令：run / pause / interrupt |
| `last_heartbeat` | 最后心跳时间 |

---

## 5. Pipeline 抽象层

### 5.1 BasePipeline（`src/core/pipeline_base.py`）

所有 Pipeline 的抽象基类，采用**策略模式 + 模板方法**：

```python
class BasePipeline(ABC):
    def __init__(self, context: dict)
        # context 包含: task_id, scene_id, work_root, log_callback, supabase

    @abstractmethod
    def run(self, input_path: str, params: dict) -> Tuple[str, dict]:
        # 子类必须实现
        # 返回: (ply_path, metadata)

    def log(self, message, level="INFO")     # 日志（通过回调回传）
    def cleanup(self)                         # 清理临时资源
    def run_rag_analysis(self, input_path)    # AI 语义分析
    def upload_and_record(self, ply_path, metadata, params)  # 上传并入库
```

### 5.2 PipelineFactory（`src/core/factory.py`）

工厂模式，根据 `task_type` 字符串创建对应的 Pipeline 实例：

| task_type | Pipeline 类 | 输入类型 | 说明 |
|-----------|-------------|----------|------|
| `video_3dgs` | `Video3DGSPipeline` | 视频 | 标准 4 步流水线 |
| `video_dual_chain` | `Video3DGSPipeline` | 视频 | 快慢双链（Worker 编排） |
| `multi_image` | `MultiImagePipeline` | ZIP | 多图重建（骨架） |
| `single_image_sam3d` | `SingleImageSAM3DPipeline` | 单图 | SAM3D 引擎 |
| `single_image_sharp` | `SingleImageSharpPipeline` | 单图 | SHARP 引擎 |
| `da3_feed_forward_3dgs` | `DA3FeedForwardPipeline` | 视频 | DA3 前馈直接生成 |
| `da3_sugar` / `da3+sugar` | `DA3SuGaRPipeline` | 视频 | DA3 + SuGaR 组合 |
| `da3_2dgs` / `da3+2dgs` | `DA3TwoDGSPipeline` | 视频 | DA3 + 2DGS 组合 |
| `sparse2dgs` | `Sparse2DGSPipeline` | ZIP/视频/单图 | Sparse2DGS |

---

## 6. 各 Pipeline 详细流程

### 6.1 Video3DGSPipeline（标准视频→3DGS）

最核心、最完整的流水线，4 步处理：

```
输入: video.mp4
  │
  ├─ Step 1: 数据准备
  │   ├─ FFmpeg 抽帧 (fps=5, 最长边 1920px, Lanczos)
  │   ├─ ImageProcessor.smart_filter_blurry_images()  去模糊
  │   └─ 均匀采样限制到 max_images 张（默认 300）
  │
  ├─ Step 1.5: AI 质检（可选）
  │   ├─ SceneAnalyzer.run() → 调用 Qwen-VL 打分
  │   ├─ 返回: score, tags, description, objects, reason
  │   └─ score < min_quality_score → 抛出异常，任务失败
  │
  ├─ Step 2: 位姿解算
  │   ├─ mapper_type=glomap → GlomapRunner
  │   │   ├─ COLMAP feature_extractor（GPU 特征提取）
  │   │   ├─ COLMAP sequential_matcher（特征匹配）
  │   │   ├─ GLOMAP global_mapper（全局 SfM）
  │   │   └─ 生成 transforms.json
  │   └─ mapper_type=da3 → DA3Runner
  │       └─ Depth Anything 3 深度估计 + 位姿解算
  │
  ├─ Step 3: AI 语义分割（可选）
  │   └─ AISegmentor.run()
  │       ├─ Qwen-VL 检测中心物体
  │       ├─ YOLO 目标检测
  │       ├─ SAM 精确分割
  │       └─ 生成透明 PNG mask
  │
  ├─ Step 4: 3DGS 训练与导出
  │   ├─ NerfstudioEngine.train()  → Splatfacto 模型训练
  │   ├─ NerfstudioEngine.export() → 导出 PLY
  │   ├─ SpatialAnchorExtractor    → 空间语义锚点提取
  │   └─ SceneAnalyzer.select_best_preview() → 最佳封面帧选择
  │
  └─ Step 5: 上传入库
      ├─ upload_and_record() → Supabase Storage + model_assets 表
      └─ 返回 (ply_path, metadata)
```

### 6.2 快慢双链模式（video_dual_chain）

**仅 Worker 云端模式**下生效的高级编排策略，目标是让用户尽快看到预览模型：

```
输入: video.mp4
  │
  ├─ 提取候选帧（fps=5, 采样 N 帧）
  ├─ SceneAnalyzer.select_best_image() → 选最佳帧
  ├─ SceneAnalyzer.classify_scene_or_object() → 场景/物体判定
  │
  ├─── ⚡ 快链（秒级~分钟级）
  │    ├─ 判定=object 且 VRAM≥25GB → SAM3D
  │    ├─ 判定=object 且 VRAM<25GB → SHARP（降级）
  │    └─ 判定=scene → SHARP
  │    → 快速生成预览模型，上传
  │
  └─── 🐢 慢链（分钟级~小时级）
       ├─ 默认 video_3dgs（完整 4 步流水线）
       └─ 可选 da3_feed_forward_3dgs
       → 高质量最终模型，覆盖上传

  任一链路成功即视为成功；双链均失败则任务失败。
```

### 6.3 SingleImageSAM3DPipeline（单图→SAM3D）

```
输入: single_image.png
  │
  ├─ SAM3DEngine.run()
  │   └─ Meta SAM-3D-Objects 模型生成 3DGS
  ├─ 后处理: 朝向修正（绕 X 轴旋转 -90°）
  ├─ RAG 语义分析 → tags, description, objects
  └─ upload_and_record()
```

### 6.4 SingleImageSharpPipeline（单图→SHARP）

```
输入: single_image.png
  │
  ├─ SharpEngine.run()
  │   └─ SHARP 模型生成 3DGS
  ├─ 后处理: 绕 X 轴旋转 180° 修正朝向
  ├─ RAG 语义分析
  └─ upload_and_record()
```

### 6.5 DA3FeedForwardPipeline（视频→DA3 前馈生成）

**无需 Nerfstudio 训练**，由 DA3 直接反投影构建 3DGS：

```
输入: video.mp4
  │
  ├─ Step 1: 视频抽帧 + 去模糊 + 采样
  ├─ Step 1.5: AI 质检（可选）
  ├─ Step 2: DA3Runner.run() → 位姿与深度解算
  ├─ Step 3: feed_forward_3dgs_from_streaming.py
  │   └─ DA3 直接反投影 → gs_ply/0000_perfect_merged.ply
  └─ upload_and_record()
```

### 6.6 DA3SuGaRPipeline（视频→DA3→SuGaR）

在 DA3 解算基础上叠加 SuGaR 精炼：

```
输入: video.mp4
  │
  ├─ Step 1: 视频抽帧 (fps=1) + 去模糊
  ├─ Step 1.5: AI 质检
  ├─ Step 2: DA3Runner.run() → 位姿与深度
  ├─ Step 3: da3_to_sugar_pipeline.sh
  │   ├─ SuGaR 训练（支持 regularization, refinement_time 等参数）
  │   └─ 输出 refined PLY / coarse PLY
  └─ upload_and_record()
```

### 6.7 DA3TwoDGSPipeline（视频→DA3→2DGS）

用 2D Gaussian Splatting 替代 Nerfstudio 3DGS 训练：

```
输入: video.mp4（仅支持视频，需≥24帧）
  │
  ├─ Step 1: 视频抽帧 + 去模糊
  ├─ Step 2: AI 质检（可选，默认关闭）
  ├─ Step 3: DA3Runner.run() → 位姿与深度
  ├─ Step 4: 2DGS 训练
  │   ├─ 组装 COLMAP sparse 数据集
  │   ├─ python train.py（2d-gaussian-splatting 仓库）
  │   └─ 输出 point_cloud.ply
  └─ 返回 (ply_path, metadata)
```

### 6.8 Sparse2DGSPipeline（少量图片→Sparse2DGS）

支持 ZIP 多图、视频随机抽帧、甚至单图（至少 3 张）：

```
输入: images.zip / video.mp4 / single_image.png
  │
  ├─ Step 1: 准备图片
  │   ├─ ZIP → 解压
  │   ├─ 视频 → 随机抽帧（可配 seed、间隔、数量）
  │   └─ 展平目录、去空目录
  │
  ├─ Step 2: COLMAP 位姿解算
  │   ├─ feature_extractor
  │   ├─ exhaustive_matcher / sequential_matcher
  │   └─ mapper → sparse/0
  │
  ├─ Step 3: 组装 Sparse2DGS 输入
  │   ├─ 复制 images + sparse 到场景目录
  │   ├─ 生成 DTU 格式 cam_*.txt（内参+外参+深度范围）
  │   └─ 创建符号链接到 Sparse2DGS/dtu_sparse/
  │
  ├─ Step 4: Sparse2DGS 训练
  │   ├─ 构建 diff-surfel-rasterization 扩展（首次）
  │   └─ conda run train.py → point_cloud.ply
  │
  └─ 返回 (ply_path, metadata)
```

---

## 7. 核心模块详解

### 7.1 ImageProcessor（`src/modules/image_proc.py`）

图片质量筛选模块：

- **网格化拉普拉斯方差**：将图片分成 3×3 网格，每个网格计算 Laplacian 方差
- **去模糊**：按方差从高到低排序，保留 top `keep_ratio`（默认 85%）
- **均匀采样**：超过 `max_images` 时等间距采样

### 7.2 SceneAnalyzer（`src/modules/scene_analyzer.py`）

AI 场景分析与质检，调用 Qwen-VL（通义千问视觉语言模型）：

| 方法 | 功能 |
|------|------|
| `run(images_dir)` | 批量图片质检：随机采样 → AI 打分 → 返回 (passed, score, reason, tags, description, objects) |
| `analyze_single_image(path)` | 单图语义分析：返回 description, tags, objects, score |
| `select_best_image(paths)` | 从多张图片中选最佳帧 |
| `classify_scene_or_object(path)` | 判定图片是场景还是物体 |
| `select_best_preview(frames, images_dir)` | 从训练结果中选择最佳封面帧 |

### 7.3 GlomapRunner（`src/modules/glomap_runner.py`）

基于 COLMAP + GLOMAP 的传统 SfM 位姿解算：

```
COLMAP feature_extractor  →  COLMAP sequential_matcher  →  GLOMAP global_mapper  →  transforms.json
```

- 支持 GPU/CPU 自动切换
- 环境隔离防止库冲突
- 重建率低于 20% 判定失败

### 7.4 DA3Runner（`src/modules/da3_runner.py`）

基于 Depth Anything 3 的位姿与深度解算：

- 支持多 HuggingFace 端点
- 输出 COLMAP 兼容格式
- 自动格式转换（DA3 → COLMAP text → binary）

### 7.5 AISegmentor（`src/modules/ai_segmentor.py`）

AI 语义分割，三级流水线：

```
Qwen-VL 中心物体检测  →  YOLO 目标检测  →  SAM 精确分割  →  透明 PNG mask
```

- 生成 mask 图片并更新 `transforms.json`
- 剔除未成功分割的图片

### 7.6 NerfstudioEngine（`src/modules/nerf_engine.py`）

3DGS 训练引擎，封装 Nerfstudio Splatfacto：

- 自动修复 CUDA 环境
- 场景自适应 collider 计算
- 支持 `fast_mode` 降低训练轮次
- 导出 PLY + WebGL 相机参数

### 7.7 SpatialAnchorExtractor（`src/modules/spatial_anchor.py`）

空间语义锚点提取：

- 从训练结果中提取相机位姿
- 生成向量嵌入
- 存入 `memory_poses` 表，用于空间搜索

### 7.8 RagMemory & KnowledgeBase

| 模块 | 功能 |
|------|------|
| `RagMemory` | 将场景描述生成向量嵌入，存入 `rag_docs` 表 |
| `KnowledgeBase` | 管理 `model_assets` 的语义元数据（描述、标签、对象） |

---

## 8. 数据库 Schema

### 8.1 processing_tasks（任务表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 主键 |
| `user_id` | text | 用户 ID |
| `scene_id` | text | 场景标识 |
| `status` | text | pending → processing → completed / failed |
| `task_type` | text | 任务类型（video_3dgs, single_image_sam3d 等） |
| `task_params` | JSONB | 任务参数（mapper_type, iterations, use_mask 等） |
| `logs` | JSONB | 实时日志数组 `[{ts, msg}, ...]` |
| `tags` | text[] | AI 标签 |
| `quality_score` | integer | AI 质量评分 |
| `quality_reason` | text | 质量评价原因 |
| `display_name` | text | 显示名称 |
| `created_at` | timestamptz | 创建时间 |

### 8.2 model_assets（模型资产表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 主键 |
| `scene_id` | text | 场景标识（唯一约束） |
| `user_id` | text | 用户 ID |
| `source_task_id` | UUID | 来源任务 |
| `description` | text | AI 生成的场景描述 |
| `objects` | text[] | 识别到的物体 |
| `tags` | text[] | AI 标签 |
| `embedding` | vector(1536) | 语义向量（用于相似性搜索） |
| `ply_path` | text | PLY 模型在 Storage 中的路径 |
| `preview_img_path` | text | 预览图路径 |
| `meta_info` | JSONB | 扩展元数据 |
| `display_name` | text | 显示名称 |
| `created_at` | timestamptz | 创建时间 |

### 8.3 worker_nodes（Worker 注册表）

| 字段 | 类型 | 说明 |
|------|------|------|
| `worker_id` | text | Worker 唯一标识（主键） |
| `hostname` | text | 主机名 |
| `pid` | integer | 进程 ID |
| `status` | text | idle / busy / stopping / offline |
| `current_task_id` | UUID | 当前执行的任务 |
| `desired_state` | text | 控制指令：run / pause / interrupt |
| `last_heartbeat` | timestamptz | 最后心跳时间 |

### 8.4 其他表

| 表名 | 用途 |
|------|------|
| `rag_docs` | 知识库文档（content + embedding） |
| `memory_poses` | 空间锚点（位姿 + embedding + tags） |
| `memory_links` | 记忆关联关系 |
| `collections` | 用户收藏/分组 |
| `community_posts` | 社区分享 |

---

## 9. 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Flutter 客户端                                │
│  上传视频 → Supabase Storage                                            │
│  创建任务 → processing_tasks (status=pending)                           │
│  轮询任务状态 ← processing_tasks (status=processing/completed/failed)    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│                        Supabase (BaaS)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐      │
│  │ PostgreSQL    │  │ Storage      │  │ Edge Functions (Deno)    │      │
│  │ - tasks      │  │ - videos     │  │ - agent-recall (搜索)    │      │
│  │ - assets     │  │ - models     │  │ - search-models (向量)   │      │
│  │ - poses      │  │ - previews   │  │ - spatial-search-agent   │      │
│  │ - rag_docs   │  │              │  │                          │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────┘      │
└─────────┼─────────────────┼─────────────────────────────────────────────┘
          │ 轮询/上传        │ 下载/上传
┌─────────▼─────────────────▼─────────────────────────────────────────────┐
│                     AI Worker (Python)                                  │
│                                                                         │
│  WorkerSupervisor                                                       │
│    └── CloudWorker                                                      │
│          │                                                              │
│          ├── 下载资源 (video.mp4 / image.png / images.zip)              │
│          │                                                              │
│          ├── PipelineFactory.get_pipeline(task_type)                    │
│          │     │                                                        │
│          │     ├── Video3DGSPipeline                                    │
│          │     │    ├─ FFmpeg 抽帧 → ImageProcessor 去模糊              │
│          │     │    ├─ SceneAnalyzer AI 质检                             │
│          │     │    ├─ GlomapRunner / DA3Runner 位姿解算                │
│          │     │    ├─ AISegmentor 语义分割                             │
│          │     │    └─ NerfstudioEngine 训练 → PLY                      │
│          │     │                                                        │
│          │     ├── SingleImageSAM3DPipeline                             │
│          │     │    └─ SAM3DEngine → PLY                                │
│          │     │                                                        │
│          │     ├── SingleImageSharpPipeline                             │
│          │     │    └─ SharpEngine → PLY                                │
│          │     │                                                        │
│          │     ├── DA3FeedForwardPipeline                               │
│          │     │    └─ DA3Runner → 反投影 → PLY                         │
│          │     │                                                        │
│          │     ├── DA3SuGaRPipeline                                     │
│          │     │    └─ DA3Runner → SuGaR 精炼 → PLY                    │
│          │     │                                                        │
│          │     ├── DA3TwoDGSPipeline                                    │
│          │     │    └─ DA3Runner → 2DGS 训练 → PLY                      │
│          │     │                                                        │
│          │     └── Sparse2DGSPipeline                                   │
│          │          └─ COLMAP → Sparse2DGS 训练 → PLY                   │
│          │                                                              │
│          ├── 模型压缩 (PLY → splat/ksplat)                              │
│          ├── 上传 PLY + transforms.json + 预览图 → Storage              │
│          ├── 写入 model_assets 表（语义元数据）                          │
│          ├── 生成 embedding → rag_docs / model_assets.embedding         │
│          └── UPDATE processing_tasks SET status='completed'             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键设计模式

| 模式 | 应用场景 |
|------|----------|
| **策略模式** | BasePipeline 抽象接口，不同 Pipeline 实现不同算法 |
| **工厂模式** | PipelineFactory 根据 task_type 动态创建 Pipeline 实例 |
| **模板方法** | BasePipeline 提供 `run_rag_analysis()`、`upload_and_record()` 等通用方法 |
| **观察者模式** | `log_callback` 回调实现实时日志回传 |
| **双链模式** | video_dual_chain 的快链+慢链并行策略 |

---

## 11. 容错与健壮性设计

1. **日志同步非阻塞**：日志推送失败不影响核心训练，只打印警告
2. **下载多路径探测**：单图任务尝试十几种可能的存储路径，自动兜底
3. **GPU/CPU 自动降级**：GlomapRunner GPU 失败自动回退 CPU
4. **VRAM 感知调度**：双链模式根据 GPU 显存选择 SAM3D 或 SHARP
5. **模型压缩容错**：压缩失败则上传原始 PLY
6. **RAG 分析容错**：语义分析失败不影响模型上传，仅跳过元数据
7. **Worker 远程控制**：通过 worker_nodes 表实现 pause/interrupt，避免僵尸进程
8. **任务隔离**：每个任务独立的 `task_output_dir`，结束后清理全部中间产物
