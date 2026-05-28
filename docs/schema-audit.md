# Supabase Schema 审查报告

> 生成时间：2026-04-30 | 基于 12 个 SQL 迁移文件 + 全代码搜索

---

## 一、Schema 总览

### 数据库表清单

| 表名 | 迁移文件 | RLS | 安全级别 |
|------|---------|-----|---------|
| `model_assets` | init_schema.sql + 2个ALTER | **未启用** | 严重 |
| `processing_tasks` | init_schema.sql + 3个ALTER | 已启用 | 低（策略全开放） |
| `rag_docs` | init_schema.sql | **未启用** | 高 |
| `tasks` | init_schema.sql | 已启用 | 低（策略全开放） |
| `memory_poses` | create_memory_poses_table.sql | 已启用 | 良好（用户级策略） |
| `community_posts` | create_community_posts.sql | 已启用 | 低（策略全开放） |
| `worker_nodes` | create_worker_nodes.sql | 已启用 | 低（策略全开放） |
| `related_model_links` | create_memory_links_and_collections.sql | **未显式启用** | 中 |
| `memory_collections` | create_memory_links_and_collections.sql | **未显式启用** | 中 |
| `memory_collection_items` | create_memory_links_and_collections.sql | **未显式启用** | 中 |

### Storage Bucket

| Bucket | 策略 |
|--------|------|
| `braindance-assets` | 4个策略：3个用户级文件夹策略 + 1个 `Enable all storage access`（全开放，覆盖前3个） |

---

## 二、表字段详细对比

### 2.1 `model_assets`

**Schema 字段**：
- `id` uuid PK
- `scene_id` text UNIQUE NOT NULL
- `user_id` text
- `source_task_id` uuid
- `description` text
- `objects` text[]
- `tags` text[]
- `embedding` vector(1536)
- `ply_path` text
- `preview_img_path` text
- `meta_info` jsonb default '{}'
- `created_at` timestamptz
- `display_name` text (迁移 20260325 添加)
- `place_id` uuid (迁移 20260326 添加)
- `memory_thread_id` uuid (迁移 20260326 添加)
- `version_label` text (迁移 20260326 添加)
- `summary_title` text (迁移 20260326 添加)
- `event_label` text (迁移 20260326 添加)
- `agent_meta` jsonb (迁移 20260326 添加)

**代码实际使用**：
| 位置 | 使用的字段 | 状态 |
|------|-----------|------|
| app/search-models/shared.ts | id, scene_id, description, ply_path, similarity | 匹配 |
| app/community/repository.dart | id, scene_id, ply_path, display_name | 匹配 |
| app/recall/*.dart | scene_id, ply_path, tags, display_name, preview_img_path | 匹配 |
| supabase/assetTools.ts | id, scene_id, description, objects, tags, ply_path, meta_info, display_name | 匹配 |
| supabase/memoryTools.ts | id, scene_id, display_name, summary_title, event_label, agent_meta, place_id | 匹配 |
| dashboard/App.vue | count(*) only | 匹配 |

**发现**：
- `source_task_id` 在代码中几乎未被直接查询（仅在 init 插入时设置）
- `place_id`、`memory_thread_id`、`version_label` 等新增字段仅在 memoryTools.ts 中使用，Flutter 端尚未集成
- **RLS 未启用** — model_assets 是最核心的表，但完全没有行级安全策略

### 2.2 `processing_tasks`

**Schema 字段**：
- `id` uuid PK
- `user_id` text NOT NULL
- `scene_id` text NOT NULL
- `status` text default 'pending'
- `created_at` timestamptz
- `updated_at` timestamptz
- `logs` jsonb default '[]'
- `tags` text[] default '{}'
- `quality_score` integer default 0
- `quality_reason` text
- `subject` text
- `category` text
- `description` text
- `keywords` text[] default '{}'
- `task_type` text default 'video_3dgs' (迁移 20260121 添加)
- `task_params` jsonb default '{}' (迁移 20260121 添加)
- `display_name` text (迁移 20260307 添加)

**代码实际使用**：
| 位置 | 使用的字段 | 状态 |
|------|-----------|------|
| app/generate_submission.dart | user_id, scene_id, status, task_type, task_params | 匹配 |
| app/task_list.dart | select('*') | 匹配 |
| app/recall_data_sync.dart | status, scene_id | 匹配 |
| app/viewer_navigation.dart | scene_id, status | 匹配 |
| supabase/confirm-text-image/index.ts | status, task_params | 匹配 |
| supabase/time-compare-agent/agent.ts | status, scene_id | 匹配 |
| supabase/memoryTools.ts | status | 匹配 |
| dashboard/App.vue | status, created_at, task_type | 匹配 |

**发现**：
- `quality_score`、`quality_reason`、`subject`、`category`、`keywords` 字段定义了但在代码中几乎未被使用（可能在 ai_engine 直接写入）
- `logs` 字段仅在 dashboard 可能查看，代码中未直接查询
- RLS 策略 `USING (true)` — 相当于无保护

### 2.3 `rag_docs`

**Schema 字段**：id, content, metadata, embedding(1536)

**代码使用**：
- Dashboard 引用了 `rag_docs` 做行数统计（`.from('rag_docs').select('*', { count: 'exact', head: true })`）
- 无其他代码直接使用此表

**发现**：
- **RLS 未启用**
- 此表似乎是 LangChain 的 RAG 存储，但当前代码中几乎未被实际使用
- 可能是遗留表或预留给 `#57 langchain` issue 使用

### 2.4 `tasks`

**Schema 字段**：id, user_id, source_path, status, worker_id, result_data, created_at, updated_at

**代码使用**：
- Dashboard 引用了 `tasks` 做行数统计
- 无其他代码直接使用此表

**发现**：
- **疑似废弃表** — 功能已被 `processing_tasks` 完全覆盖
- `processing_tasks` 有更丰富的字段（task_type, task_params, display_name 等）
- 建议评估是否可以删除或合并

### 2.5 `memory_poses`

**Schema 字段**：id, model_id(FK), image_name, transform_matrix, tag, embedding(1536), created_at

**代码使用**：
- supabase/assetTools.ts — 查询 memory_poses 获取空间锚点
- supabase/memoryTools.ts — 查询和管理 poses
- supabase/time-compare-agent/agent.ts — 时间对比时查询 poses
- supabase/spatialAgent.ts — 空间搜索时查询 poses

**发现**：
- **RLS 策略设计良好** — 基于用户身份的 SELECT/INSERT/UPDATE/DELETE 策略
- 是所有表中安全策略设计最完善的

### 2.6 `community_posts`

**Schema 字段**：id, user_id, model_asset_id(FK), model_name, title, caption, place_name, latitude, longitude, cover_image_url, metadata, created_at, updated_at

**代码使用**：
- app/community/repository.dart — 完整 CRUD

**发现**：
- RLS 策略 `USING (true) WITH CHECK (true)` — 完全开放，任何人可增删改
- 应至少限制 anon 只能 SELECT，INSERT/UPDATE/DELETE 需要认证

### 2.7 `worker_nodes`

**Schema 字段**：worker_id(PK), hostname, pid, status, current_task_id, current_scene_id, desired_state, control_note, control_requested_at, last_heartbeat, started_at, stopped_at, metadata

**代码使用**：
- dashboard/App.vue — 查询在线 worker 数量和状态

**发现**：
- RLS 策略 `USING (true) WITH CHECK (true)` — 完全开放
- Worker 状态表被任何人可以修改，存在安全隐患
- 建议限制为 service_role 才能写入，anon/authenticated 只能 SELECT

### 2.8 `related_model_links`、`memory_collections`、`memory_collection_items`

**代码使用**：
- 仅在 supabase/memoryTools.ts 中使用
- Flutter 端尚未集成这些功能

**发现**：
- **RLS 未显式启用**
- 这些是新表（2026-03-26 创建），可能还在开发中

---

## 三、安全问题汇总

### 严重

1. **`model_assets` 表未启用 RLS** — 这是核心数据表，包含所有 3D 模型信息
2. **Storage 策略矛盾** — `Enable all storage access` 策略（using true）覆盖了用户级文件夹策略
3. **`rag_docs` 表未启用 RLS** — embedding 数据完全暴露

### 高

4. **`community_posts` 全开放策略** — 任何人可增删改社区帖子
5. **`worker_nodes` 全开放策略** — 任何人可修改 Worker 状态
6. **`processing_tasks` 全开放策略** — 任何人可修改任务状态
7. **`tasks` 全开放策略** — 同上

### 中

8. **`related_model_links` 等三个新表未启用 RLS** — 可能在开发中
9. **`tasks` 表疑似废弃** — 与 `processing_tasks` 功能重叠

---

## 四、命名一致性检查

### 表名一致性
- ✅ 所有代码中使用的表名与 schema 定义一致
- ✅ Storage bucket `braindance-assets` 在所有模块中一致使用

### Edge Function 入口点

| Function | 入口文件 | 状态 |
|----------|---------|------|
| agent-recall | index.ts | 正常 |
| spatial-search-agent | index.ts | 正常 |
| time-compare-agent | index.ts | 正常 |
| search-models | index.ts | 正常 |
| text-to-image | index.ts | 正常 |
| confirm-text-image | index.ts | 正常 |
| test-timeout | index.ts | 测试用 |

### 字段命名风格
- ✅ 风格一致：snake_case
- ✅ 时间字段统一使用 timestamptz
- ✅ JSON 字段统一使用 jsonb

---

## 五、建议修复优先级

### P0（演示前必须修复）
1. 为 `model_assets` 启用 RLS
2. 移除 Storage 的 `Enable all storage access` 策略

### P1（近期修复）
3. 收紧 `community_posts` 的 RLS — anon 只能 SELECT
4. 收紧 `worker_nodes` 的 RLS — 仅 service_role 可写
5. 为 `related_model_links` 等新表启用 RLS

### P2（后续优化）
6. 评估是否删除 `tasks` 表
7. 为 `rag_docs` 启用 RLS 或标注为 service_role only
8. 收紧 `processing_tasks` 的 RLS 策略
9. 清理 `model_assets` 中未使用的 `source_task_id` 字段
