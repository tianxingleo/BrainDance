

# BrainDance API 接入文档 (v1.0)

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

- **权限**: 用户仅可读写自己的数据 (RLS 开启)。
- **操作**: `Insert` (创建), `Select` (查询), `Realtime` (监听)。

| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | uuid | ❌ | 主键，**插入时留空**，数据库会自动生成并返回 |
| `scene_id` | string | ✅ | **场景唯一标识**，建议前端生成 `timestamp_random` |
| `user_id` | uuid | ✅ | 当前登录用户的 ID |
| `task_type` | string | ❌ | 任务类型，默认 `video_3dgs` |
| `task_params` | json | ❌ | 任务参数，JSON格式 |
| `status` | string | ✅ | 固定填 `pending` |
| `logs` | json | ❌ | (只读) 实时日志，格式 `[{"ts":..., "msg":...}]` |
| `quality_score`| int | ❌ | (只读) AI 评分 |

**task_type 可选值:**

| 值 | 说明 | 输入文件 |
|---|------|---------|
| `video_3dgs` | 视频转3DGS（传统流程） | `video.mp4` |
| `da3_feed_forward_3dgs` | 视频转3DGS（前馈快速生成） | `video.mp4` |
| `single_image_sam3d` | 单图转3DGS（SAM3D） | `image.png` |
| `single_image_sharp` | 单图转3DGS（SHARP） | `image.png` |

**task_params 字段说明 (single_image_sam3d):**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `mask_path` | string | 可选，自定义Mask图片路径 |

**task_params 字段说明 (da3_feed_forward_3dgs):**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `frame_interval` | int | 5 | 前馈生成时的帧间隔，值越小使用帧数越多（1=使用全部帧） |
| `conf_threshold` | float | 0.5 | 深度置信度阈值，值越高过滤越严格 |

**创建视频任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260118_001',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'video_3dgs',
  'status': 'pending'
}).select();
```

**创建 DA3 前馈式3DGS任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260118_002',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'da3_feed_forward_3dgs',
  'task_params': {
    'frame_interval': 2,  // 使用更多帧以提高质量
    'conf_threshold': 0.5  // 深度置信度阈值
  },
  'status': 'pending'
}).select();
```

**创建单图任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260119_001',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'single_image_sam3d',
  'task_params': '{}',  // 可选自定义参数
  'status': 'pending'
}).select();
```

**创建 SHARP 单图任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260120_001',
  'user_id': supabase.auth.currentUser!.id,
  'task_type': 'single_image_sharp',
  'task_params': '{}',  // SHARP 无额外参数
  'status': 'pending'
}).select();
```

### 3.2 资产表: `model_assets`
用于存储生成成功的模型资产。
**用途**：前端直接查询此表以展示“我的模型列表”或“模型总数”。

- **权限**: 读写 (RLS 开启，用户只能查询和删除**属于自己**的数据)。
- **操作**: `Select` (列表/详情), `Delete` (删除)。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | uuid | 资产唯一 ID |
| `scene_id` | string | 对应任务的场景 ID |
| `description` | text | AI 生成的场景描述 (用于展示) |
| `tags` | array | 标签列表，如 `["室内", "红色"]` |
| `quality_score`| int | 质量评分 (0-100) |
| `ply_path` | text | **关键**：文件在 Storage 中的相对路径，需拼接下载链接 |
| `created_at` | timestamp | 创建时间 |

**获取我的模型列表 (Dart):**
```dart
// 获取当前用户的所有模型，按时间倒序
final assets = await supabase.from('model_assets')
    .select('*')
    .order('created_at', ascending: false);
```

---

## 4. 文件存储 (Storage)

### 4.1 存储桶配置
- **Bucket**: `braindance-assets`
- **权限**: Public (公开读取)

### 4.2 目录结构规范
前端**必须**严格遵守以下路径格式，否则后端 Worker 无法读取文件。

```text
braindance-assets/ (Bucket)
└── {user_id}/                   <-- 第一级：用户隔离
    └── {scene_id}/              <-- 第二级：项目/场景隔离
        ├── raw/                 <-- 原始素材
        │   ├── video.mp4        # 视频任务 (task_type: video_3dgs)
        │   └── image.png        # 单图任务 (task_type: single_image_sam3d)
        ├── processed/           <-- 抽帧图片
        │   ├── frame_001.jpg
        │   └── frame_002.jpg
        └── output/              <-- 训练结果
            ├── point_cloud.ply
            └── gaussian_splat.splat
```

### 4.3 下载链接拼接
`{Supabase_URL}/storage/v1/object/public/braindance-assets/{user_id}/{scene_id}/output/point_cloud.ply`

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
      "ply_path": "user_123/scene_001/output/point_cloud.ply",
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
2.  获取 `ply_path` 字段 (例如 `user_123/scene_001/output/point_cloud.ply`)。
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