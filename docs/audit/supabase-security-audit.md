# Supabase 安全与 Schema 审计报告

## 审计概要
- **审计时间**：2026-04-30
- **审计范围**：supabase/migrations/（12 个 SQL 文件）、supabase/functions/（7 个 Edge Function）、supabase/seed.sql
- **发现问题数**：21 个
- **P0: 4 个 | P1: 7 个 | P2: 6 个 | P3: 4 个**

---

## Schema 一致性

### 表结构汇总

| 表名 | 主键 | RLS 启用 | 关键字段 |
|------|------|---------|---------|
| `processing_tasks` | uuid | ✅ | status, task_type, task_params, logs, quality_score, user_id, scene_id, display_name |
| `model_assets` | uuid | ❌（未启用） | scene_id, user_id, ply_path, preview_img_path, embedding, objects, tags, display_name, place_id, agent_meta 等 |
| `rag_docs` | bigint | ❌（未启用） | content, metadata, embedding |
| `tasks` | uuid | ✅ | status, source_path, worker_id, result_data |
| `memory_poses` | uuid | ✅ | model_id, image_name, transform_matrix, tag, embedding |
| `community_posts` | uuid | ✅ | user_id, model_asset_id, title, caption, latitude, longitude |
| `worker_nodes` | text(worker_id) | ✅ | status, desired_state, current_task_id, last_heartbeat |
| `related_model_links` | uuid | ❌（未启用） | source_model_id, target_model_id, relation_type |
| `memory_collections` | uuid | ❌（未启用） | user_id, title, collection_type |
| `memory_collection_items` | uuid | ❌（未启用） | collection_id, model_id, sort_order |

### task_type 枚举定义

| 位置 | 值 |
|------|-----|
| Migration `add_task_type_and_params` 注释 | `video_3dgs`, `single_image_sam3d`, `single_image_sharp` |
| Worker `factory.py` | `video_3dgs`, `video_dual_chain`, `multi_image`, `single_image_sam3d`, `single_image_sharp`, `da3_feed_forward_3dgs`, `da3_sugar`, `da3+sugar`, `da3_2dgs`, `da3+2dgs`, `sparse2dgs` |
| Flutter `generate.dart` | `video_3dgs`, `video_dual_chain`, `single_image_sam3d`, `single_image_sharp`, `da3_feed_forward_3dgs`, `da3_sugar`, `da3_2dgs`, `sparse2dgs` |
| Migration CHECK 约束 | **被注释掉**，无强制约束 |

**不一致**：Migration 注释只列了 3 种，实际有 11 种。CHECK 约束被注释掉，任何字符串都能写入 task_type。

### status 枚举定义

| 位置 | processing_tasks status | worker_nodes status | worker_nodes desired_state |
|------|------------------------|---------------------|--------------------------|
| Migration 默认值 | `pending` | `starting` | `run` |
| Worker 实际写入 | `pending` → `processing` → `completed` / `failed` | `starting`, `idle`, `busy`, `stopping`, `offline`, `error` | `run`, `pause` |
| Dashboard 读取 | `pending`, `processing`, `completed`, `failed` | 同上 | 同上 |
| Migration 注释 | 无 | 有注释说明 | 有注释说明 |

**不一致**：processing_tasks 的 status 没有注释说明合法值，也没有 CHECK 约束。

---

## P0 级问题

### P0-1: model_assets 表未启用 RLS，任何人可读写全部模型数据

- **严重程度**：P0
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:8-21`
- **问题描述**：`model_assets` 是核心业务表，存储所有 3D 模型的路径、描述、标签和 embedding。init_schema 中未对其执行 `ALTER TABLE ... ENABLE ROW LEVEL SECURITY`。虽然 `processing_tasks` 和 `tasks` 启用了 RLS，但 model_assets 没有任何行级安全保护。
- **风险**：任何持有 anon key 的人（前端代码中公开）都可以读取、修改、删除所有用户的模型数据，包括 ply_path 和 preview_img_path。
- **复现方法**：使用 anon key 调用 `supabase.from('model_assets').select('*')` 即可读取全部数据。
- **建议修复**：添加 migration `ALTER TABLE model_assets ENABLE ROW LEVEL SECURITY`，并创建基于 user_id 的读写策略。
- **建议创建 Issue**：是

### P0-2: rag_docs 表未启用 RLS，语义索引数据完全暴露

- **严重程度**：P0
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:46-51`
- **问题描述**：`rag_docs` 存储了语义搜索的内容、metadata 和 embedding，但未启用 RLS。 anon 角色拥有全部 CRUD 权限。
- **风险**：攻击者可以读取全部 RAG 文档（可能包含场景分析详情），也可以注入恶意文档污染搜索结果。
- **建议修复**：启用 RLS 并限制 anon 只能通过 Edge Function 间接查询。
- **建议创建 Issue**：是

### P0-3: init_schema 中 "Enable all storage access" 策略允许 anon 完全控制 Storage

- **严重程度**：P0
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:390-396`
- **问题描述**：存在一个名为 `Enable all storage access` 的 Storage policy，允许 `public` 对所有 bucket 执行 ALL 操作（`using (true)`）。虽然前面有更精细的 user folder 策略，但这个全开策略优先级更高，等于覆盖了前面的所有保护。
- **风险**：任何人可以读取、写入、删除 Storage 中所有文件，包括其他用户的 3D 模型文件。
- **复现方法**：使用 anon key 调用 `supabase.storage.from('braindance-assets').list()` 即可列出全部文件。
- **建议修复**：删除此全开策略。如果需要 service_role 的全量访问，应通过 service_role key 在后端实现。
- **建议创建 Issue**：是

### P0-4: worker_nodes 表 "Allow all for dev" 策略允许 anon 控制 Worker

- **严重程度**：P0
- **涉及文件**：`supabase/migrations/20260320143000_create_worker_nodes.sql:61-67`
- **问题描述**：策略名明确标注 "for dev"，但已存在于正式 migration 中。anon 角色可以 UPDATE desired_state、control_note 等字段，直接控制 Worker 的暂停/中断/恢复。
- **风险**：任何人知道 URL + anon key 即可远程操控所有 Worker 节点。
- **建议修复**：将写入权限限制为 service_role 或通过带认证的 Edge Function 间接操作。
- **建议创建 Issue**：是

---

## P1 级问题

### P1-1: processing_tasks "Allow all for dev" 策略过宽

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:345-350`
- **问题描述**：processing_tasks 有一个 `using (true)` 的全开策略，允许 anon 角色执行 ALL 操作。虽然后续 migration 添加了 `dashboard_read_processing_tasks` 的 SELECT 策略，但原始的 ALL 策略仍然生效，anon 仍然可以 INSERT/UPDATE/DELETE 任务。
- **风险**：任何人可以创建伪造任务、修改任务状态、删除其他用户的任务。
- **建议修复**：收紧为用户只能操作自己的任务（基于 user_id = auth.uid()）。
- **建议创建 Issue**：是

### P1-2: community_posts "Allow all for dev" 策略允许匿名发布和篡改

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260317170000_create_community_posts.sql:29-34`
- **问题描述**：社区帖子表使用全开策略，任何人都可以发布、修改、删除帖子，包括其他人的帖子。
- **建议修复**：SELECT 公开，INSERT/UPDATE/DELETE 限制为帖子作者。
- **建议创建 Issue**：是

### P1-3: related_model_links / memory_collections / memory_collection_items 未启用 RLS

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260326122000_create_memory_links_and_collections.sql`
- **问题描述**：这三个新表（弱图谱关系、记忆专题、专题成员）都没有 RLS 策略，也没有授予 anon 权限的语句。默认情况下 PostgreSQL 表对 anon 不可见，但如果后续通过 grant 授权可能遗漏 RLS 保护。
- **建议修复**：主动启用 RLS 并添加基于 user_id 的策略。
- **建议创建 Issue**：是

### P1-4: task_type CHECK 约束被注释掉，无枚举保护

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260121000000_add_task_type_and_params.sql:40-42`
- **问题描述**：Migration 中预留了 `task_type_check` 约束但被注释掉。当前任何字符串都能写入 task_type，包括拼写错误（如 `video_3dgs` vs `video3dgs`）。
- **风险**：前端或 Worker 写入非法 task_type 不会被数据库拒绝，导致任务静默丢失或路由错误。
- **建议修复**：启用 CHECK 约束或使用 PostgreSQL ENUM 类型。
- **建议创建 Issue**：是

### P1-5: model_assets.user_id 类型为 text 而非 uuid，与 auth.uid() 不一致

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:11`
- **问题描述**：`model_assets.user_id` 定义为 `text`，而 `auth.uid()` 返回 `uuid`。在 memory_poses 的 RLS 策略中使用了 `model_assets.user_id = auth.uid()::text` 做了类型转换，但如果 user_id 存储的格式不一致（如包含/不包含连字符），可能导致 RLS 策略失效。
- **建议修复**：统一 user_id 为 `uuid` 类型，与 Supabase Auth 系统保持一致。processing_tasks.user_id 也有同样问题。
- **建议创建 Issue**：是

### P1-6: processing_tasks 缺少 updated_at 自动更新触发器

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:31`
- **问题描述**：`updated_at` 有默认值 `now()` 但没有自动更新触发器。Worker 更新任务状态时不会自动更新 updated_at，导致该字段可能停留在创建时间。
- **建议修复**：添加 `CREATE TRIGGER ... BEFORE UPDATE ON processing_tasks FOR EACH ROW EXECUTE FUNCTION moddatetime(updated_at)`。
- **建议创建 Issue**：否（小改动，可直接修复）

### P1-7: worker_nodes 缺少 user_id 或归属字段

- **严重程度**：P1
- **涉及文件**：`supabase/migrations/20260320143000_create_worker_nodes.sql:1-15`
- **问题描述**：worker_nodes 没有 user_id 或 owner 字段，无法区分哪个用户/系统注册了哪个 Worker。当前全开 RLS 策略下这不是问题，但收紧权限后无法实现"只能控制自己的 Worker"。
- **建议修复**：增加 owner_id 字段并在 RLS 策略中使用。
- **建议创建 Issue**：是

---

## P2 级问题

### P2-1: dashboard_read 系列策略对 anon 开放全表 SELECT

- **严重程度**：P2
- **涉及文件**：`supabase/migrations/20260320143000_add_dashboard_table_read_policies.sql`
- **问题描述**：5 张表的 dashboard_read 策略都对 anon 开放了 `USING (true)` 的 SELECT 权限。这意味着任何人都能读取全部 processing_tasks、model_assets、memory_poses、tasks、rag_docs 的所有行。
- **风险**：信息泄露——其他用户的任务数据、模型路径、场景描述对 anon 完全可见。
- **建议修复**：Dashboard 应使用 authenticated 角色而非 anon。或者创建专用的 dashboard_role。

### P2-2: tasks 表与 processing_tasks 表功能重叠

- **严重程度**：P2
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:55-64`
- **问题描述**：存在两个任务表：`tasks`（user_id 为 uuid）和 `processing_tasks`（user_id 为 text）。从代码来看，业务主链路使用 `processing_tasks`，`tasks` 表几乎未使用但仍然存在并占用 Dashboard 查询资源。
- **建议修复**：确认 tasks 表是否已废弃，如果是则添加注释标记。

### P2-3: community_posts.user_id 为 text 且无 NOT NULL 约束

- **严重程度**：P2
- **涉及文件**：`supabase/migrations/20260317170000_create_community_posts.sql:3`
- **问题描述**：社区帖子的 user_id 可以为 NULL，意味着可以发布匿名帖子。在比赛/演示场景可能需要，但也可能被滥用。
- **建议修复**：根据业务需求决定是否添加 NOT NULL。

### P2-4: model_assets.embedding 维度硬编码为 1536

- **严重程度**：P2
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:16`
- **问题描述**：embedding 固定为 1536 维（OpenAI text-embedding-ada-002 的维度）。如果切换到其他 embedding 模型（如 Cohere、本地模型），需要重建索引。
- **建议修复**：记录当前 embedding 模型选择，考虑后续迁移策略。

### P2-5: 缺少 model_assets 的 user_id 索引

- **严重程度**：P2
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql`
- **问题描述**：model_assets 有 scene_id 唯一索引和 embedding 索引，但没有 user_id 索引。Flutter 端按用户查询模型的场景很常见。
- **建议修复**：添加 `CREATE INDEX idx_model_assets_user_id ON model_assets(user_id)`。

### P2-6: seed.sql 可能包含开发用测试数据

- **严重程度**：P2
- **涉及文件**：`supabase/seed.sql`
- **问题描述**：seed.sql 在 `supabase start` 时自动执行。如果包含测试用户或测试数据，可能污染生产环境。
- **建议修复**：审查 seed.sql 内容，确保只包含必要的初始化数据。

---

## P3 级问题

### P3-1: match_model_assets 函数缺少用户过滤

- **严重程度**：P3
- **涉及文件**：`supabase/migrations/20260118144558_init_schema.sql:95-118`
- **问题描述**：向量搜索函数 `match_model_assets` 没有 user_id 参数，意味着搜索结果是全局的。在多用户场景下，用户可能搜索到其他人的模型。
- **建议修复**：增加 `filter_user_id` 参数。

### P3-2: Edge Function 缺少统一 CORS 配置

- **严重程度**：P3
- **涉及文件**：各 Edge Function 的 index.ts
- **问题描述**：CORS headers 在各 Function 中独立配置，可能不一致。
- **建议修复**：抽取公共 CORS 处理到 `_shared/` 中。

### P3-3: 缺少数据库备份/恢复策略

- **严重程度**：P3
- **问题描述**：没有看到 pg_dump 或 Supabase 备份的自动化配置。
- **建议修复**：添加定期备份脚本或依赖 Supabase 平台备份。

### P3-4: grant 权限语句过于冗长，建议用角色继承

- **严重程度**：P3
- **涉及文件**：init_schema.sql 中的大量 GRANT 语句
- **问题描述**：每张表都对 anon、authenticated、postgres、service_role 单独授权，非常冗长。
- **建议修复**：考虑使用 PostgreSQL 角色继承简化权限管理。

---

## Edge Function 安全评估

| Function | 输入验证 | 认证检查 | 错误处理 | 风险 |
|----------|---------|---------|---------|------|
| agent-recall | 部分验证 | 使用 service_role | 中等 | service_role 绕过 RLS |
| search-models | 基本验证 | anon 可调用 | 中等 | 无用户隔离 |
| spatial-search-agent | 部分验证 | 使用 service_role | 中等 | 同 agent-recall |
| time-compare-agent | 基本验证 | 使用 service_role | 中等 | 同上 |
| text-to-image | 未审计 | 未审计 | 未审计 | 需进一步审查 |
| confirm-text-image | 未审计 | 未审计 | 未审计 | 需进一步审查 |
| test-timeout | 测试用途 | 无 | 最小 | 不应部署到生产 |

**关键发现**：Agent 类 Edge Function 使用 `service_role` key 创建 Supabase client，绕过了所有 RLS 策略。这在功能上是必要的（Agent 需要跨用户搜索），但也意味着如果 Edge Function 有输入注入漏洞，影响面极大。

---

## 建议新建 Issue 清单

- [ ] [P0] model_assets 表启用 RLS 并添加 user_id 隔离策略
- [ ] [P0] rag_docs 表启用 RLS，限制为只通过 Edge Function 查询
- [ ] [P0] 删除 Storage 的 "Enable all storage access" 全开策略
- [ ] [P0] worker_nodes 表收紧 RLS 策略，从 "Allow all for dev" 改为 service_role only
- [ ] [P1] processing_tasks 收紧 RLS，删除 "Allow all for dev" 策略
- [ ] [P1] community_posts 收紧 RLS，INSERT/UPDATE/DELETE 限制为帖子作者
- [ ] [P1] 为 related_model_links、memory_collections、memory_collection_items 启用 RLS
- [ ] [P1] 启用 task_type CHECK 约束或使用 ENUM 类型
- [ ] [P1] 统一 user_id 字段类型为 uuid
- [ ] [P1] 为 processing_tasks 添加 updated_at 自动更新触发器
- [ ] [P2] Dashboard 读取策略从 anon 改为 authenticated
- [ ] [P2] 审查并清理 tasks 表（是否已废弃）
- [ ] [P2] 为 model_assets 添加 user_id 索引
- [ ] [P3] match_model_assets 函数增加 user_id 过滤参数
