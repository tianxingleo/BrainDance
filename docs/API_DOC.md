

# BrainDance API 接入文档 (v1.2)

本文档描述了 BrainDance 3DGS 引擎的前端接入规范。本项目采用 **Supabase (BaaS) + Python (微服务)** 的混合架构。

## 1. 环境配置 (Environment)

### 1.1 服务地址

| 服务名称 | 本地开发 (Local) | 生产环境 (Prod) | 说明 |
| :--- | :--- | :--- | :--- |
| **Supabase URL (项目ID在网页Reference ID)** | `http://127.0.0.1:54321` | `https://<你的项目ID>.supabase.co` | 核心数据库、Auth、Storage |
| **Python API（还没做完）** | `http://127.0.0.1:8000` | `https://api.braindance.com` | 仅用于语义搜索 |

### 1.2 密钥 (Public Keys)

前端初始化 SDK 时请使用以下 Key。**严禁在前端使用 `service_role` key。**

- **Supabase Anon Key（具体之前发了好像不记得了）**: `xxxx`

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
| `id` | uuid | ❌ | 主键，**插入时留空**，数据库会自动生成并返回 |
| `scene_id` | string | ✅ | **场景唯一标识**，建议前端生成 `timestamp_random` |
| `display_name` | string | ❌ | 任务展示名称（用于列表显示），为空时可回退 `scene_id` |
| `user_id` | string | ✅ | 当前登录用户的 ID (`auth.users.id` 字符串) |
| `task_type` | string | ❌ | 任务类型，默认 `video_3dgs` |
| `task_params` | jsonb | ❌ | 任务参数，JSON 对象 |
| `status` | string | ✅ | 固定填 `pending` |
| `logs` | json | ❌ | (只读) 实时日志，格式 `[{"ts":..., "msg":...}]` |
| `quality_score`| int | ❌ | (只读) AI 评分 |
| `quality_reason`| string | ❌ | (只读) AI 评分原因 |
| `tags` | string[] | ❌ | (只读) AI 标签 |

> Dashboard 当前会优先显示 `display_name`，如果为空则回退到 `scene_id`。

**task_type 可选值:**

| 值 | 说明 | 输入文件 |
|---|------|---------|
| `video_dual_chain` | 视频快慢双链（快链先交付，慢链自动接续） | `video.mp4` |
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

**task_params 字段说明 (video_dual_chain):**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `slow_pipeline` | string | `video_3dgs` | 慢链类型：`video_3dgs` / `da3_feed_forward_3dgs`（每次仅跑 1 条慢链） |
| `sam3d_vram_threshold_gb` | int | 25 | SAM3D 显存门槛（GB），低于阈值会降级为 SHARP |
| `best_frame_sample_count` | int | 8 | 从视频抽样候选帧数量，用于快链最佳帧挑选 |

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

**task_params 通用参数（所有会产出 3DGS 模型的任务都可用）:**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `delivery_format` | string | `splat` | 模型交付格式：`splat` / `ksplat` / `ply` |
| `compression_opacity_threshold` | float | `0.05` | `.ply -> .splat` 时透明度剔除阈值 |
| `ksplat_alpha_threshold` | int | `1` | `.ply -> .ksplat` 时透明度剔除阈值（传给 Node 工具） |

> `delivery_format=ksplat` 时，Worker 所在环境必须安装 Node.js，并配置 `KSPLAT_SCRIPT_PATH` 指向 `GaussianSplats3D/util/create-ksplat.js`。  
> 若 `.ksplat` 压缩失败，当前实现会自动回退并上传原始 `.ply`（任务不会失败）。

**论文 Pipeline 选型建议（重点）:**

| task_type | 适合什么场景 | 输入要求 | 推荐起步参数 |
|---|---|---|---|
| `da3_sugar` / `da3+sugar` | 质量优先且可接受更慢速度（mesh/SDF 约束 3DGS） | `raw/video.mp4` | `regularization=dn_consistency`, `refinement_time=short`, `fast_mode=true` |
| `da3_2dgs` / `da3+2dgs` | 希望替代 Nerfstudio 3DGS 并输出 2DGS | `raw/video.mp4`（建议连续走拍视频） | `iterations=30000`, `extract_fps=2.0`, `min_images=24` |
| `sparse2dgs` | 少量图片直接生成 2DGS | `raw/images.zip`（至少 3 张） | `iterations=7000`, `resolution=2`, `depth_ratio=1.0` |

**上传约定（非常关键）:**

1. `da3_2dgs` / `da3+2dgs` 必须上传视频到 `{user_id}/{scene_id}/raw/video.mp4`。
2. `sparse2dgs` 使用 `images.zip` 到 `{user_id}/{scene_id}/raw/images.zip`。
3. `da3_2dgs` 不支持单图或少量图片回退。

**创建视频任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260118_001',
  'display_name': '客厅模型-第一版',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'video_dual_chain',
  'task_params': {
    'slow_pipeline': 'video_3dgs',
    'sam3d_vram_threshold_gb': 25,
    'best_frame_sample_count': 8
  },
  'status': 'pending'
}).select();
```

**创建 DA3 前馈式3DGS任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260118_002',
  'display_name': '办公室扫描-快速版',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'da3_feed_forward_3dgs',
  'task_params': {
    'frame_interval': 2,  // 使用更多帧以提高质量
    'conf_threshold': 0.5  // 深度置信度阈值
  },
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

**创建 DA3+2DGS 视频任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260313_2dgs_001',
  'display_name': '室内走拍-DA3+2DGS',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'da3_2dgs',
  'task_params': {
    'iterations': 30000,
    'gpu_index': 1,
    'extract_fps': 2.0,
    'min_images': 24
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
  'task_params': {},  // 可选自定义参数
  'status': 'pending'
}).select();
```

**创建 SHARP 单图任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260120_001',
  'display_name': '桌面静物重建',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'single_image_sharp',
  'task_params': {},  // SHARP 无额外参数
  'status': 'pending'
}).select();
```

### 3.2 资产表: `model_assets`
用于存储生成成功的模型资产。
**用途**：前端直接查询此表以展示“我的模型列表”或“模型总数”。

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
| `ply_path` | text | **关键**：文件在 Storage 中的相对路径，需拼接下载链接。当前可能是 `point_cloud.ply` / `point_cloud.splat` / `point_cloud.ksplat` |
| `preview_img_path` | text | 预览图 URL 或相对路径 |
| `meta_info` | jsonb | 扩展元数据（如 `quality_score` / `quality_reason`） |
| `created_at` | timestamp | 创建时间 |

**获取我的模型列表 (Dart):**
```dart
// 获取当前用户的所有模型，按时间倒序
final assets = await supabase.from('model_assets')
    .select('*')
    .order('created_at', ascending: false);
```

### 3.3 社区贴文表: `community_posts`

用于 Community 页的公共贴文流和地图探索。

- **权限**: 开发态当前允许公共读写。
- **操作**: `Select` (社区流), `Insert` (发布贴文)。

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | uuid | ❌ | 主键，自动生成 |
| `user_id` | string | ❌ | 发布者 ID |
| `model_asset_id` | uuid | ❌ | 关联 `model_assets.id`，删除资产后会置空 |
| `model_name` | string | ❌ | 展示用模型名称快照 |
| `title` | string | ✅ | 贴文标题 |
| `caption` | string | ✅ | 贴文文案，默认空字符串 |
| `place_name` | string | ✅ | 地点名称 |
| `latitude` | double | ✅ | 纬度 |
| `longitude` | double | ✅ | 经度 |
| `cover_image_url` | string | ❌ | 封面图地址 |
| `metadata` | jsonb | ❌ | 扩展元数据 |
| `created_at` | timestamp | ❌ | 创建时间 |
| `updated_at` | timestamp | ❌ | 更新时间 |

**读取社区流 (Dart):**
```dart
final posts = await supabase.from('community_posts').select('''
  id,
  title,
  caption,
  place_name,
  latitude,
  longitude,
  user_id,
  created_at,
  model_name,
  cover_image_url,
  model_assets (
    scene_id,
    description,
    ply_path,
    preview_img_path
  )
''').order('created_at', ascending: false).limit(24);
```

**发布社区贴文 (Dart):**
```dart
await supabase.from('community_posts').insert({
  'user_id': supabase.auth.currentUser?.id,
  'model_asset_id': selectedModelId,
  'model_name': sceneId,
  'title': '清晨刚亮时的断桥',
  'caption': '薄雾和湖面的反光被一起留在模型里。',
  'place_name': '杭州西湖',
  'latitude': 30.258,
  'longitude': 120.140,
  'cover_image_url': previewUrl,
});
```

### 3.4 Worker 节点表: `worker_nodes`

用于 AI Engine Worker 注册、心跳和 Dashboard 集群控制。

- **权限**: 当前迁移为开发态全开放策略；Dashboard 直接读写该表。
- **操作**: `Upsert` (Worker 注册/心跳), `Select` (Dashboard 监控), `Update` (下发控制目标)。

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `worker_id` | string | ✅ | Worker 实例 ID，主键 |
| `hostname` | string | ❌ | 节点主机名 |
| `pid` | int | ❌ | 进程 ID |
| `status` | string | ✅ | `starting / idle / busy / stopping / offline / error` |
| `current_task_id` | uuid | ❌ | 当前任务 ID |
| `current_scene_id` | string | ❌ | 当前场景 ID |
| `desired_state` | string | ✅ | Dashboard 控制目标，当前约定 `run / pause / interrupt` |
| `control_note` | string | ❌ | 控制备注 |
| `control_requested_at` | timestamp | ❌ | 控制请求时间 |
| `last_heartbeat` | timestamp | ✅ | 最近心跳 |
| `started_at` | timestamp | ✅ | 启动时间 |
| `stopped_at` | timestamp | ❌ | 停止时间 |
| `metadata` | jsonb | ❌ | 在线超时、停止原因等附加信息 |

**Dashboard 暂停 Worker (TypeScript):**
```ts
await supabase
  .from('worker_nodes')
  .update({
    desired_state: 'pause',
    control_note: 'pause requested from dashboard',
    control_requested_at: new Date().toISOString(),
  })
  .eq('worker_id', workerId)
```

---

## 4. 文件存储 (Storage)

### 4.1 存储桶配置
- **Bucket 1**: `braindance-assets`
- **用途**: 3D 任务素材、中间结果、输出模型
- **权限**: Public (公开读取)
- **Bucket 2**: `braindance-models`
- **用途**: Flutter Recall 本地 AI 下载用的端侧模型发布仓
- **权限**: Public (公开读取)

### 4.2 目录结构规范
前端**必须**严格遵守以下路径格式，否则后端 Worker 无法读取文件。

```text
braindance-assets/ (Bucket)
└── {user_id}/                   <-- 第一级：用户隔离
    └── {scene_id}/              <-- 第二级：项目/场景隔离
        ├── raw/                 <-- 原始素材
        │   ├── video.mp4        # 视频任务 (task_type: video_3dgs / da3_2dgs / da3+2dgs)
        │   ├── images.zip       # 多图任务 (task_type: sparse2dgs)
        │   └── image.png        # 单图任务 (task_type: single_image_sam3d)
        ├── processed/           <-- 抽帧图片
        │   ├── frame_001.jpg
        │   └── frame_002.jpg
        └── output/              <-- 训练结果
            ├── point_cloud.splat   # 默认输出（推荐）
            ├── point_cloud.ksplat  # 可选输出（需 Node + 脚本）
            ├── point_cloud.ply     # 回退或显式指定 delivery_format=ply
            ├── transforms.json
            ├── webgl_poses.json
            ├── preview.jpg
            └── images/
                ├── frame_00001.jpg
                └── ...
```

> 注：`webgl_poses.json` 与 `output/images/*` 主要由视频流水线（含空间锚点提取）生成，单图任务可能只产出 `point_cloud.*`（后缀取决于 `delivery_format`）。

`braindance-models` 当前目录约定：

```text
braindance-models/ (Bucket)
├── catalog/
│   └── model_catalog.json
└── releases/
    ├── qwen3-1.7b-braindance-q5-k-m-imatrix.gguf   # 当前 Flutter 默认模型
    ├── qwen3-1.7b-braindance-q5-k-m.gguf
    ├── qwen3-1.7b-braindance-q4-k-m.gguf
    ├── qwen3-1.7b-braindance-merged/
    └── qwen3-0.6b-braindance-round1/
```

### 4.3 下载链接拼接
`{Supabase_URL}/storage/v1/object/public/braindance-assets/{user_id}/{scene_id}/output/point_cloud.{ply|splat|ksplat}`

Flutter Recall 本地 AI 当前默认模型下载链接：
`{Supabase_URL}/storage/v1/object/public/braindance-models/releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf`

### 4.4 Worker 相关环境变量（压缩/交付）

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `MODEL_DELIVERY_FORMAT` | `splat` | 全局默认输出格式：`splat` / `ksplat` / `ply` |
| `COMPRESSION_OPACITY_THRESHOLD` | `0.05` | `.splat` 压缩透明度阈值 |
| `KSPLAT_ALPHA_THRESHOLD` | `1` | `.ksplat` 工具透明度阈值 |
| `KSPLAT_SCRIPT_PATH` | 空 | `create-ksplat.js` 绝对路径（仅 `ksplat` 需要） |

---

## 5. 语义搜索接口 (Search API)

本项目使用 **Supabase Edge Function (Deno)** 实现语义搜索功能，替代原有的 Python HTTP 接口。

### 5.1 接口地址

| 环境 | URL | 说明 |
| :--- | :--- | :--- |
| **本地开发** | `http://127.0.0.1:54321/functions/v1/search-models` | 需要启动本地 Supabase |
| **生产环境** | `https://<项目ID>.supabase.co/functions/v1/search-models` | 云端 Edge Function |

### 5.2 请求说明

- **Method**: `POST`
- **Content-Type**: `application/json`
- **认证**: 需要携带 `Authorization` Header (使用 Anon Key)

### 5.3 请求参数

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :---: | :--- |
| `query` | string | ✅ | 搜索关键词，支持自然语言，如 "红色杯子"、"上周拍的照片" |

**请求示例**:
```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <ANON_KEY>' \
  -d '{"query":"红色杯子"}'
```

### 5.4 响应格式

**成功响应**:
```json
{
  "success": true,
  "intent": {
    "original_query": "红色杯子",
    "parsed_search_text": "红色杯子",
    "filter_start": null,
    "filter_end": null
  },
  "results": [
    {
      "id": "uuid",
      "scene_id": "scene_20260118_001",
      "description": "桌子上的红色马克杯...",
      "ply_path": "user_123/scene_001/output/point_cloud.splat",
      "created_at": "2026-01-18T10:00:00Z",
      "similarity": 0.89
    }
  ]
}
```

**错误响应**:
```json
{
  "success": false,
  "error": "错误描述信息"
}
```

### 5.5 智能时间过滤

系统会自动解析自然语言中的时间词：

| 用户输入 | 解析结果 |
| :--- | :--- |
| "红色杯子" | `search_text: "红色杯子"`, `time: 无过滤` |
| "上周拍的红色杯子" | `search_text: "红色杯子"`, `time: 上周的时间范围` |
| "去年生日的照片" | `search_text: "生日照片"`, `time: 去年的范围` |

### 5.6 Flutter 调用示例

```dart
/// 语义搜索模型
///
/// 使用 Edge Function 进行自然语言搜索
/// 支持自动时间过滤和语义匹配
///
/// @param query 搜索关键词，支持自然语言
/// @returns 搜索结果列表
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

/// 搜索结果模型
class SearchResult {
  final String id;
  final String sceneId;
  final String description;
  final String plyPath;
  final DateTime createdAt;
  final double similarity;

  SearchResult({
    required this.id,
    required this.sceneId,
    required this.description,
    required this.plyPath,
    required this.createdAt,
    required this.similarity,
  });

  /// 获取模型下载链接
  String getModelUrl(String supabaseUrl) {
    return '$supabaseUrl/storage/v1/object/public/braindance-assets/$plyPath';
  }

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    return SearchResult(
      id: json['id'] ?? '',
      sceneId: json['scene_id'] ?? '',
      description: json['description'] ?? '',
      plyPath: json['ply_path'] ?? '',
      createdAt: DateTime.tryParse(json['created_at'] ?? '') ?? DateTime.now(),
      similarity: (json['similarity'] ?? 0).toDouble(),
    );
  }
}
```

### 5.7 错误码说明

| HTTP 状态码 | 错误信息 | 说明 |
| :--- | :--- | :--- |
| 400 | `缺少或无效的搜索关键词 'query'` | 未提供 query 参数 |
| 400 | `搜索关键词不能为空` | query 为空字符串 |
| 400 | `搜索关键词过长（最大 500 字符）` | query 超过 500 字符 |
| 500 | `未配置 DASHSCOPE_API_KEY` | 服务器配置错误 |
| 500 | `向量生成失败` | AI 服务调用失败 |
| 500 | `数据库查询失败` | 数据库错误 |

---

## 6. 核心业务流程 (Workflows)

前端请按以下顺序实现业务逻辑：

### 流程一：新建任务 (Create Task)
1.  **生成 ID**: 前端生成一个 `scene_id`。
2.  **上传视频**: 将文件上传至 Storage: `{user_id}/{scene_id}/raw/video.mp4`。
3.  **写入数据库**: 向 `processing_tasks` 插入一条记录，状态为 `pending`。
4.  **监听状态**: 使用 Supabase Realtime 订阅该条记录的 `UPDATE` 事件。
    *   当 `status` 变为 `processing` -> 显示进度条。
    *   当 `logs` 数组更新 -> 显示实时日志。
    *   当 `status` 变为 `completed` -> 拼接 URL 下载并展示模型。

### 流程二：搜索模型 (Search)
1.  用户输入自然语言搜索词，如"红色杯子"或"上周拍的照片"。
2.  调用 Edge Function: `supabase.functions.invoke('search-models', body: {'query': query})`。
3.  系统自动：
    - 解析搜索意图（提取搜索词和时间范围）。
    - 调用 AI 生成语义向量。
    - 在向量数据库中搜索相似模型。
4.  获取返回列表，使用 `results[].ply_path` 拼接下载链接进行渲染。

### 流程三：查看我的模型 (My Models)
1.  调用 Supabase SDK: `.from('model_assets').select('*')`。
2.  获取 `ply_path` 字段 (例如 `user_123/scene_001/output/point_cloud.splat`)。
3.  **前端拼接下载链接**:
    `https://<ProjectID>.supabase.co/storage/v1/object/public/braindance-assets/` + `ply_path`
4.  将完整链接喂给 3D 渲染组件进行展示。

### 流程四：单图3DGS任务 (Create Single Image Task)
1.  **生成 ID**: 前端生成一个 `scene_id`。
2.  **上传图片**: 将文件上传至 Storage: `{user_id}/{scene_id}/raw/image.png`。
3.  **写入数据库**: 向 `processing_tasks` 插入一条记录，设置 `task_type` 为 `single_image_sam3d`。
4.  **监听状态**: 使用 Supabase Realtime 订阅该条记录的 `UPDATE` 事件。
    *   当 `status` 变为 `processing` -> 显示进度条。
    *   当 `logs` 数组更新 -> 显示实时日志。
    *   当 `status` 变为 `completed` -> 拼接 URL 下载并展示模型。

### 流程五：发布社区贴文 (Share To Community)
1.  前端先从 `model_assets` 读取可分享模型。
2.  用户填写 `title / caption / place_name / latitude / longitude`。
3.  向 `community_posts` 插入记录，并带上 `model_asset_id`。
4.  Community 页通过关联查询 `model_assets` 取回 `ply_path`、描述和预览图。

### 流程六：Worker 集群控制 (Worker Cluster Control)
1.  Worker 启动时向 `worker_nodes` 执行 `upsert`，注册 `worker_id`、`hostname`、`pid` 和 `started_at`。
2.  Worker 运行期间持续更新 `status`、`current_task_id`、`current_scene_id` 和 `last_heartbeat`。
3.  Dashboard 读取 `worker_nodes` 渲染实例列表。
4.  Dashboard 需要暂停实例时，更新 `desired_state='pause'`。
5.  Worker 观察到状态变更后停止接新任务，并优雅退出。
