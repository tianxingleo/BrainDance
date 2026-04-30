
# BrainDance API 接入文档 (v2.0)

> **同步说明**：本文档与 `docs/03-API参考/API接口指南.md` 保持同步，内容来源为 `README.md` 技术架构章节与 `supabase/functions/` 代码目录。
>
> 如需查阅最新版本，请优先参考 [`docs/03-API参考/API接口指南.md`](../03-API参考/API接口指南.md)。

本文档描述了 BrainDance 3DGS 引擎的前端接入规范。本项目采用 **Supabase BaaS 架构**，使用 Edge Functions (Deno) 提供语义搜索与 Agent 能力。

## 1. 环境配置 (Environment)

### 1.1 服务地址

| 服务名称 | 本地开发 (Local) | 生产环境 (Prod) | 说明 |
| :--- | :--- | :--- | :--- |
| **Supabase URL** | `http://127.0.0.1:54321` | `https://<你的项目ID>.supabase.co` | 核心数据库、Auth、Storage |
| **Edge Functions** | `http://127.0.0.1:54321/functions/v1` | `https://<项目ID>.supabase.co/functions/v1` | 语义搜索、Agent Recall、API 保护 |

### 1.2 密钥配置 (Public Keys)

前端初始化 SDK 时需要使用 Supabase Anon Key。**严禁在前端使用 `service_role` key。**

#### 获取 API Key

- **本地开发**：运行 `supabase start` 后，终端会显示 Anon Key
- **生产环境**：在 Supabase Dashboard -> Settings -> API 获取

#### 环境变量配置

前端项目应配置以下环境变量：

```env
# .env.local (Vite)
VITE_SUPABASE_URL=http://127.0.0.1:54321  # 本地开发
# VITE_SUPABASE_URL=https://<项目ID>.supabase.co  # 生产环境

VITE_SUPABASE_ANON_KEY=<你的-Anon-Key>
```

**Flutter 配置示例**：
```dart
// lib/config/supabase_config.dart
class SupabaseConfig {
  static const String url = String.fromEnvironment('SUPABASE_URL',
    defaultValue: 'http://127.0.0.1:54321');
  static const String anonKey = String.fromEnvironment('SUPABASE_ANON_KEY');
}
```

> 安全警告：严禁在代码中硬编码 API Key，必须使用环境变量。

---

## 2. 用户鉴权 (Authentication)

本项目完全托管于 Supabase Auth。

- **SDK**: 直接使用 `supabase_flutter` 的 Auth 方法。
- **登录/注册**: 不需要后端写接口，直接调用 SDK：
  - `supabase.auth.signUp()`
  - `supabase.auth.signInWithPassword()`
- **Token 管理**: SDK 会自动维护 Session，后续对 Database 的操作会自动携带 JWT Token，无需手动处理。

---

## 3. 数据库交互 (Database as API)

前端通过 Supabase SDK 直接读写数据库表。以下是表结构契约。

### 3.1 任务表: `processing_tasks`
用于创建新的 3D 生成任务，并监听进度。

- **权限**: 开发态当前允许公共访问；Dashboard 额外依赖只读策略直接查询该表。
- **操作**: `Insert` (创建), `Select` (查询), `Realtime` (监听)。

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | uuid | -- | 主键，**插入时留空**，数据库会自动生成并返回 |
| `scene_id` | string | 是 | **场景唯一标识**，建议前端生成 `timestamp_random` |
| `display_name` | string | -- | 任务展示名称（用于列表显示），为空时可回退 `scene_id` |
| `user_id` | string | 是 | 当前登录用户的 ID (`auth.users.id` 字符串) |
| `task_type` | string | -- | 任务类型，默认 `video_3dgs` |
| `task_params` | jsonb | -- | 任务参数，JSON 对象 |
| `status` | string | 是 | 固定填 `pending` |
| `logs` | json | -- | (只读) 实时日志，格式 `[{"ts":..., "msg":...}]` |
| `quality_score`| int | -- | (只读) AI 评分 |
| `quality_reason`| string | -- | (只读) AI 评分原因 |
| `tags` | string[] | -- | (只读) AI 标签 |

> Dashboard 当前会优先显示 `display_name`，如果为空则回退到 `scene_id`。

**task_type 可选值:**

| 值 | 说明 | 输入文件 |
|---|------|---------|
| `video_3dgs` | 视频转3DGS（传统流程） | `video.mp4` |
| `da3_feed_forward_3dgs` | 视频转3DGS（前馈快速生成） | `video.mp4` |
| `da3_sugar` / `da3+sugar` | SuGaR 使用 mesh/SDF 约束 3DGS（质量更高、速度更慢） | `video.mp4` |
| `da3_2dgs` / `da3+2dgs` | Nerfstudio 3DGS 的替代路线（输出 2DGS） | `video.mp4` |
| `single_image_sam3d` | 单图转3DGS（SAM3D） | `image.png` |
| `single_image_sharp` | 单图转3DGS（SHARP） | `image.png` |
| `sparse2dgs` | 少量图片生成 2DGS（Sparse2DGS） | `images.zip` / `video.mp4` |

**task_params 字段说明 (sparse2dgs):**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `iterations` | int | 7000 | Sparse2DGS 训练迭代数 |
| `resolution` | int | 2 | 对应 `train.py -r`，值越小分辨率越高 |
| `depth_ratio` | float | 1.0 | 深度损失权重比例参数 |
| `lambda_dist` | float | 1000 | 几何约束损失权重 |
| `conda_env` | string | `Braindance` | 运行 Sparse2DGS 的 Conda 环境名 |
| `sparse2dgs_repo_path` | string | `/ltx-data/Sparse2DGS` | Sparse2DGS 仓库路径 |
| `colmap_matcher` | string | `exhaustive_matcher` | COLMAP 匹配器（少图推荐 exhaustive） |
| `colmap_mapper` | string | `mapper` | COLMAP 解算器（可选 `global_mapper`） |
| `video_sample_count` | int | 12 | 当输入为视频时，随机抽取的帧数 |
| `video_random_seed` | int | 42 | 视频随机抽帧种子 |
| `min_video_frame_gap` | int | 3 | 视频随机抽帧时的最小帧间隔 |
| `video_max_edge` | int | 0 | 抽帧后图片长边限制，0 表示不缩放 |

**task_params 字段说明 (single_image_sam3d):**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `mask_path` | string | 可选，自定义Mask图片路径 |

**task_params 字段说明 (da3_feed_forward_3dgs):**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `frame_interval` | int | 5 | 前馈生成时的帧间隔，值越小使用帧数越多（1=使用全部帧） |
| `conf_threshold` | float | 0.5 | 深度置信度阈值，值越高过滤越严格 |

**task_params 字段说明 (da3_sugar / da3+sugar):**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `regularization` | string | `dn_consistency` | SuGaR正则化：`dn_consistency` / `density` / `sdf` |
| `refinement_time` | string | `short` | 精炼时长：`short` / `medium` / `long` |
| `high_poly` | bool | `false` | 完整流程时是否高面数mesh |
| `fast_mode` | bool | `true` | 是否走fast模式（仅coarse，通常输出PLY更快） |
| `gpu_index` | int | 环境默认 | 绑定GPU索引（会转成 `CUDA_VISIBLE_DEVICES`） |
| `sugar_repo_path` | string | 自动探测 | SuGaR仓库路径 |
| `da3_repo_path` | string | 自动探测 | DA3仓库路径 |
| `sugar_scene_name` | string | `scene_id` | 覆盖SuGaR内部场景名 |

**task_params 字段说明 (da3_2dgs / da3+2dgs):**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `iterations` | int | 30000 | 2DGS 训练迭代数（高质量建议 30000） |
| `gpu_index` | int | 1 | 绑定 GPU 索引（默认第二张卡） |
| `extract_fps` | float | 2.0 | 抽帧帧率（每秒抽取图片数） |
| `max_edge` | int | 1920 | 抽帧后最长边限制 |
| `blur_keep_ratio` | float | 0.85 | 去模糊筛选后保留比例 |
| `max_images` | int | `MAX_IMAGES` | 参与重建的最大图片数（超出会均匀采样） |
| `min_images` | int | 24 | 最小有效帧数门槛（低于此值直接失败） |
| `enable_scene_analysis` | bool | false | 是否启用 AI 质检 |
| `render_after_train` | bool | false | 训练后是否执行 render.py |
| `dgs_repo_path` | string | 自动探测 | 2DGS 仓库路径（可覆盖） |

**创建视频任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260118_001',
  'display_name': '客厅模型-第一版',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'video_3dgs',
  'status': 'pending'
}).select();
```

**创建 DA3+SuGaR 视频任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260313_sugar_001',
  'display_name': '客厅漫游-DA3+SuGaR',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'da3_sugar',
  'task_params': {
    'regularization': 'dn_consistency',
    'refinement_time': 'short',
    'fast_mode': true,
    'high_poly': false
  },
  'status': 'pending'
}).select();
```

**创建 Sparse2DGS 多图任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260313_sparse2dgs_001',
  'display_name': '展柜小场景-Sparse2DGS',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'sparse2dgs',
  'task_params': {
    'iterations': 7000,
    'resolution': 2,
    'depth_ratio': 1.0,
    'lambda_dist': 1000
  },
  'status': 'pending'
}).select();
```

**创建单图任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260119_001',
  'display_name': '手办单图重建',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'single_image_sam3d',
  'task_params': {},
  'status': 'pending'
}).select();
```

### 3.2 资产表: `model_assets`
用于存储生成成功的模型资产。
**用途**：前端直接查询此表以展示"我的模型列表"或"模型总数"。

- **权限**: 开发态当前允许公共访问；Dashboard 依赖只读策略查询聚合数据。
- **操作**: `Select` (列表/详情), `Delete` (删除)。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | uuid | 资产唯一 ID |
| `scene_id` | string | 对应任务的场景 ID |
| `user_id` | string | 资产所属用户 ID |
| `description` | text | AI 生成的场景描述 (用于展示) |
| `objects` | string[] | 场景内关键物体列表 |
| `tags` | array | 标签列表，如 `["室内", "红色"]` |
| `embedding` | vector(1536) | 语义向量（1536维，用于 pgvector 搜索） |
| `ply_path` | text | **关键**：文件在 Storage 中的相对路径，需拼接下载链接 |
| `preview_img_path` | text | 预览图 URL 或相对路径 |
| `meta_info` | jsonb | 扩展元数据（如 `quality_score` / `quality_reason`） |
| `created_at` | timestamp | 创建时间 |

**获取我的模型列表 (Dart):**
```dart
final assets = await supabase.from('model_assets')
    .select('*')
    .order('created_at', ascending: false);
```

### 3.3 空间锚点表: `memory_poses`

存储视频流水线拆出的关键帧位姿、描述和向量，支撑 Recall 空间检索。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | uuid | 主键 |
| `model_id` | uuid | 关联 `model_assets.id`，级联删除 |
| `frame_index` | int | 帧序号 |
| `image_path` | text | 关键帧图片路径 |
| `pose_data` | jsonb | 帧级位姿数据 |
| `caption` | text | AI 生成的帧描述 |
| `tags` | string[] | 标签列表 |
| `embedding` | vector(1536) | 语义向量 |

### 3.4 社区贴文表: `community_posts`

用于 Community 页的公共贴文流和地图探索。

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | uuid | -- | 主键，自动生成 |
| `user_id` | string | -- | 发布者 ID |
| `model_asset_id` | uuid | -- | 关联 `model_assets.id`，删除资产后会置空 |
| `model_name` | string | -- | 展示用模型名称快照 |
| `title` | string | 是 | 贴文标题 |
| `caption` | string | 是 | 贴文文案，默认空字符串 |
| `place_name` | string | 是 | 地点名称 |
| `latitude` | double | 是 | 纬度 |
| `longitude` | double | 是 | 经度 |
| `cover_image_url` | string | -- | 封面图地址 |
| `metadata` | jsonb | -- | 扩展元数据 |
| `created_at` | timestamp | -- | 创建时间 |
| `updated_at` | timestamp | -- | 更新时间 |

### 3.5 Worker 节点表: `worker_nodes`

用于 AI Engine Worker 注册、心跳和 Dashboard 集群控制。

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `worker_id` | string | 是 | Worker 实例 ID，主键 |
| `hostname` | string | -- | 节点主机名 |
| `pid` | int | -- | 进程 ID |
| `status` | string | 是 | `starting / idle / busy / stopping / offline / error` |
| `current_task_id` | uuid | -- | 当前任务 ID |
| `current_scene_id` | string | -- | 当前场景 ID |
| `desired_state` | string | 是 | Dashboard 控制目标：`run / pause / interrupt` |
| `control_note` | string | -- | 控制备注 |
| `control_requested_at` | timestamp | -- | 控制请求时间 |
| `last_heartbeat` | timestamp | 是 | 最近心跳 |
| `started_at` | timestamp | 是 | 启动时间 |
| `stopped_at` | timestamp | -- | 停止时间 |
| `metadata` | jsonb | -- | 在线超时、停止原因等附加信息 |

---

## 4. 文件存储 (Storage)

### 4.1 存储桶配置
- **Bucket 1**: `braindance-assets`
- **用途**: 3D 生成任务的原始素材、中间结果和输出模型
- **权限**: Public (公开读取)
- **Bucket 2**: `braindance-models`
- **用途**: Flutter Recall 本地 AI 模型发布与下载
- **权限**: Public (公开读取)

### 4.2 目录结构规范
前端**必须**严格遵守以下路径格式，否则后端 Worker 无法读取文件。

```text
braindance-assets/ (Bucket)
{user_id}/{scene_id}/raw/video.mp4
{user_id}/{scene_id}/raw/image.png
{user_id}/{scene_id}/raw/images.zip
{user_id}/{scene_id}/raw/thumbnail.jpg

{user_id}/{scene_id}/output/point_cloud.ply
{user_id}/{scene_id}/output/point_cloud.splat
{user_id}/{scene_id}/output/point_cloud.ksplat
{user_id}/{scene_id}/output/transforms.json
```

`braindance-models` 当前目录约定：

```text
catalog/model_catalog.json

releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf
releases/qwen3-1.7b-braindance-q5-k-m.gguf
releases/qwen3-1.7b-braindance-q4-k-m.gguf
releases/qwen3-1.7b-braindance-merged/*
releases/qwen3-0.6b-braindance-round1/*
```

### 4.3 下载链接拼接
`{Supabase_URL}/storage/v1/object/public/braindance-assets/{user_id}/{scene_id}/output/point_cloud.ply`

Flutter Recall 本地 AI 默认模型下载链接：
`{Supabase_URL}/storage/v1/object/public/braindance-models/releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf`

---

## 5. 语义搜索与 Agent 接口

本项目使用 **Supabase Edge Function (Deno)** 实现语义搜索与 Agent 能力。当前已形成分层架构：

- `search-models`：基础语义搜索，负责 Embedding、时间解析与向量检索
- `agent-recall`：正式统一 Agent 入口，支持多模式路由、流式事件输出与前端会话状态
- `spatial-search-agent`：LangChain / Agent 实验入口，复用共享 Core
- `time-compare-agent`：面向双时间窗口的专用时间对比能力

### 5.1 search-models 接口

| 环境 | URL |
| :--- | :--- |
| **本地开发** | `http://127.0.0.1:54321/functions/v1/search-models` |
| **生产环境** | `https://<项目ID>.supabase.co/functions/v1/search-models` |

- **Method**: `POST`
- **Content-Type**: `application/json`
- **认证**: 需要携带 `Authorization` Header (使用 Anon Key)

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :---: | :--- |
| `query` | string | 是 | 搜索关键词，支持自然语言 |

**请求示例**:
```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <ANON_KEY>' \
  -d '{"query":"红色杯子"}'
```

### 5.2 agent-recall 接口

| 环境 | URL |
| :--- | :--- |
| **本地开发** | `http://127.0.0.1:54321/functions/v1/agent-recall` |
| **生产环境** | `https://<项目ID>.supabase.co/functions/v1/agent-recall` |

- 支持 `spatial_search`、`asset_metadata`、`time_compare`、`creative`、`memory_graph` 五类模式
- 支持 `SSE / NDJSON` 流式协议
- 输出结构：`answer + evidence + actions + top_candidates + tool_trace`
- 动作协议：`open_scene`、`fly_to_pose`

### 5.3 Flutter 调用示例

```dart
Future<List<SearchResult>> searchModels(String query) async {
  try {
    final response = await supabase.functions.invoke(
      'search-models',
      body: {'query': query},
    );
    if (response.data['success'] == true) {
      final results = response.data['results'] as List;
      return results.map((r) => SearchResult.fromJson(r)).toList();
    } else {
      throw Exception(response.data['error'] ?? '搜索失败');
    }
  } catch (e) {
    debugPrint('搜索出错: $e');
    return [];
  }
}
```

---

## 6. 核心业务流程 (Workflows)

### 流程一：新建任务 (Create Task)
1.  **生成 ID**: 前端生成一个 `scene_id`。
2.  **上传视频**: 将文件上传至 Storage: `{user_id}/{scene_id}/raw/video.mp4`。
3.  **写入数据库**: 向 `processing_tasks` 插入一条记录，状态为 `pending`。
4.  **监听状态**: 使用 Supabase Realtime 订阅该条记录的 `UPDATE` 事件。

### 流程二：Agent 检索 (Agent Recall)
1.  用户输入自然语言查询。
2.  调用 Edge Function: `supabase.functions.invoke('agent-recall', body: {...})`。
3.  系统自动路由到合适的模式（空间搜索、资产元数据、时间对比等）。
4.  通过 SSE/NDJSON 流式返回状态、工具调用和最终结果。

### 流程三：查看我的模型 (My Models)
1.  调用 Supabase SDK: `.from('model_assets').select('*')`。
2.  获取 `ply_path` 字段并拼接下载链接进行渲染。

### 流程四：Worker 集群控制 (Worker Cluster Control)
1.  Worker 启动时向 `worker_nodes` 执行 `upsert`，注册心跳。
2.  Dashboard 读取 `worker_nodes` 渲染实例列表。
3.  Dashboard 需要暂停实例时，更新 `desired_state='pause'`。

---

*本文档版本: v2.0 | 同步日期: 2026-04-30 | 内容来源: README.md、supabase/functions/、docs/03-API参考/*
