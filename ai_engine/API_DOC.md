

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
| `status` | string | ✅ | 固定填 `pending` |
| `logs` | json | ❌ | (只读) 实时日志，格式 `[{"ts":..., "msg":...}]` |
| `quality_score`| int | ❌ | (只读) AI 评分 |

**创建任务示例 (Dart):**
```dart
final res = await supabase.from('processing_tasks').insert({
  'scene_id': 'scene_20260118_001',
  'user_id': supabase.auth.currentUser!.id,
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
{user_id}/                   <-- 第一级：用户 UUID
  └── {scene_id}/            <-- 第二级：场景 ID (与数据库一致)
      ├── raw/
      │   └── video.mp4      <-- [上传] 原始视频，固定文件名
      └── output/
          ├── point_cloud.ply  <-- [下载] 3D 模型
          └── transforms.json  <-- [下载] 预览参数
```

### 4.3 下载链接拼接
`{Supabase_URL}/storage/v1/object/public/braindance-assets/{user_id}/{scene_id}/output/point_cloud.ply`

---

## 5. 自定义接口 (Python API)（这一部分还没做好，只是本地跑了，api没有做，先留个坑位）

部分复杂逻辑无法通过 Supabase 直接完成，需调用以下 HTTP 接口。

### 5.1 语义搜索模型
通过自然语言搜索历史模型库。

- **URL**: `/search`
- **Method**: `GET`

**请求参数 (Query Params):**

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | string | 是 | 搜索词，如 "红色的杯子" |
| `start` | date | 否 | 开始时间 `2025-01-01` |
| `end` | date | 否 | 结束时间 `2025-12-31` |

**响应示例 (JSON):**

```json
{
  "code": 200,
  "data": [
    {
      "scene_id": "scene_20260118_001",
      "description": "桌子上的红色马克杯...",
      "score": 0.89,
      "model_url": "https://.../point_cloud.ply",
      "created_at": "2026-01-18T10:00:00Z"
    }
  ]
}
```

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
1.  用户输入文字。
2.  调用 Python API `/search?q=...`。
3.  获取返回列表，直接使用列表中的 `model_url` 进行渲染展示。

### 流程三：查看我的模型 (My Models)
1.  调用 Supabase SDK: `.from('model_assets').select('*')`。
2.  获取 `ply_path` 字段 (例如 `user_123/scene_001/output/point_cloud.ply`)。
3.  **前端拼接下载链接**:
    `https://<ProjectID>.supabase.co/storage/v1/object/public/braindance-assets/` + `ply_path`
4.  将完整链接喂给 3D 渲染组件进行展示。