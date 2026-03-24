# BrainDance Supabase

`supabase/` 提供 BrainDance 当前使用的本地后端基础设施，包括数据库迁移、Edge Functions 和本地开发配置。

## 目录内容

- `migrations/`：数据库结构与策略迁移
- `functions/search-models/`：自然语言搜索接口
- `functions/test-timeout/`：测试用函数
- `config.toml`：Supabase CLI 本地配置
- `deploy-functions.sh`：函数部署脚本

## 当前职责

这一层主要承担：

- `processing_tasks` 等业务表的数据存储
- `worker_nodes` 的注册、心跳与集群控制
- `community_posts` 的社区贴文存储
- `pgvector` 向量检索
- Storage 文件管理
- Realtime 状态同步
- Edge Functions 承载搜索接口

## 本地启动

先确保 Docker 和 Supabase CLI 可用：

```bash
cd supabase
supabase start
```

默认本地端口以 `config.toml` 为准，当前常用地址为：

- API: `http://127.0.0.1:54321`
- PostgreSQL: `postgresql://postgres:postgres@127.0.0.1:54322/postgres`
- Studio: `http://127.0.0.1:54323`

首次启动时会执行 `migrations/` 下的 SQL 迁移。

## 当前迁移内容

从现有迁移文件看，当前仓库至少覆盖了这些核心对象：

- `processing_tasks`
- `worker_nodes`
- `model_assets`
- `memory_poses`
- `rag_docs`
- `tasks`
- `match_memory_poses` 相关检索能力
- Dashboard 读取 `braindance-assets` 的存储策略

现有迁移文件包括：

- `20260118144558_init_schema.sql`
- `20260121000000_add_task_type_and_params.sql`
- `20260225020151_create_memory_poses_table.sql`
- `20260306123456_match_memory_poses.sql`
- `20260307090000_add_display_name_to_processing_tasks.sql`
- `20260309195500_add_storage_read_policies_for_dashboard.sql`
- `20260317170000_create_community_posts.sql`
- `20260320143000_add_dashboard_table_read_policies.sql`
- `20260320143000_create_worker_nodes.sql`

当前与最近几次表结构变更直接相关的对象可以概括为：

- `processing_tasks`：任务主表，新增了 `display_name` 用于前端列表展示
- `model_assets`：模型资产表，供 Recall、Community 和 Dashboard 查询
- `memory_poses`：空间锚点与向量检索表
- `community_posts`：社区贴文表，引用 `model_assets.id`
- `worker_nodes`：Worker 注册、心跳、当前任务和控制状态表
- `dashboard_read_*` 策略：为 Dashboard 直连读表补齐只读 RLS

## Storage 约定

当前项目默认围绕 `braindance-assets` bucket 工作，路径约定与根 README 保持一致：

```text
{user_id}/{scene_id}/raw/video.mp4
{user_id}/{scene_id}/raw/image.png
{user_id}/{scene_id}/raw/images.zip
{user_id}/{scene_id}/raw/thumbnail.jpg

{user_id}/{scene_id}/output/point_cloud.ply
{user_id}/{scene_id}/output/point_cloud.splat
{user_id}/{scene_id}/output/point_cloud.ksplat
{user_id}/{scene_id}/output/transforms.json
```

注意：

- 现有策略和代码都默认使用 `braindance-assets`
- 当前仓库中的 `seed.sql` 为空，不应假设 bucket 一定会被自动创建
- 如果本地环境里还没有这个 bucket，需要在 Studio 或脚本中手动创建

## Edge Functions

### `search-models`

这是当前仓库里的主要函数，用于承载自然语言搜索接口。它负责：

1. 解析查询中的检索目标和时间范围
2. 调用 Embedding 接口生成向量
3. 通过 `pgvector` 查询相关场景或空间锚点

本地运行：

```bash
cd supabase/functions/search-models
supabase functions serve search-models --no-verify-jwt --env-file .env.local
```

如果要测试接口，可以参考 [tests/README.md](/home/ltx/projects/BrainDance/tests/README.md)。

### `test-timeout`

这是一个辅助测试函数，主要用于本地联调，不承担核心业务能力。

## 与其他模块的关系

- `app/`：使用 Anon Key 直连数据库、Storage 和 Realtime
- `dashboard/`：读取任务、资产、空间锚点和 `worker_nodes` 状态，并可写入 `desired_state`
- `ai_engine/3dgs/`：监听 `processing_tasks`，回写结果到数据库和 Storage，同时持续更新 `worker_nodes`

## 说明

这份 README 只描述当前仓库里的 Supabase 层，不覆盖线上部署策略，也不替代具体的表结构设计文档。更完整的系统链路请参考项目根目录的 [README.md](/home/ltx/projects/BrainDance/README.md)。
