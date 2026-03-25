# BrainDance Supabase

`supabase/` 提供 BrainDance 当前使用的本地后端基础设施，包括数据库迁移、Edge
Functions 和本地开发配置。

## 目录内容

- `migrations/`：数据库结构与策略迁移
- `functions/search-models/`：自然语言搜索接口
- `functions/spatial-search-agent/`：基于 LangChain 的空间检索 Agent
- `functions/agent-recall/`：稳定响应协议的 Recall Agent 总入口
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

当前项目当前使用两个 Storage bucket：

- `braindance-assets`：3DGS 任务素材与输出
- `braindance-models`：端侧模型发布与分发

其中 `braindance-assets` 路径约定与根 README 保持一致：

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

`braindance-models` 当前发布目录约定如下：

```text
catalog/model_catalog.json

releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf
releases/qwen3-1.7b-braindance-q5-k-m.gguf
releases/qwen3-1.7b-braindance-q4-k-m.gguf
releases/qwen3-1.7b-braindance-merged/*
releases/qwen3-0.6b-braindance-round1/*
```

其中：

- `catalog/model_catalog.json` 用于记录默认推荐模型与候选用途
- Flutter Recall 本地 AI 默认下载对象为 `releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf`
- `qwen3-1.7b-braindance-merged/` 与 `qwen3-0.6b-braindance-round1/` 保留为完整发布目录，便于后续做 HF / LoRA 侧部署实验

注意：

- Worker 与 Dashboard 的现有资产路径默认仍围绕 `braindance-assets`
- Flutter Recall 本地 AI 默认模型下载入口已切到 `braindance-models`
- 当前仓库中的 `seed.sql` 为空，不应假设 bucket 一定会被自动创建
- 如果本地环境里还没有这些 bucket，需要在 Studio 或脚本中手动创建

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

如果使用 `supabase start` 启动本地自部署容器，请把 `DASHSCOPE_API_KEY`
写在仓库下的 `supabase/.env.local`，并确保
[`supabase/config.toml`](/home/ltx/projects/BrainDance/supabase/config.toml)
通过 `[edge_runtime.secrets]` 从宿主机环境读取该变量；修改后需要重启本地
Supabase，Edge Runtime 才会重新加载它。

如果要测试接口，可以参考
[tests/README.md](/home/ltx/projects/BrainDance/tests/README.md)。

### `agent-recall`

`agent-recall` 是当前按架构路线新增的稳定总入口，负责：

1. 识别用户查询意图
2. 复用 `search-models/shared.ts` 中的共享搜索逻辑组织召回结果
3. 组织 `answer + evidence + actions` 结构化响应
4. 把空间证据和前端动作一起返回给 Flutter / Web

当前动作协议已经稳定为：

- `open_scene`
- `fly_to_pose`
- `highlight_region`

本地运行：

```bash
cd supabase/functions/agent-recall
supabase functions serve agent-recall --no-verify-jwt
```

最小回归题集位于
[tests/agent_recall_cases.jsonl](/home/ltx/projects/BrainDance/tests/agent_recall_cases.jsonl)。详细路线见：[docs/02-架构设计/Agent规划与LangChain实践路线.md](/home/ltx/projects/BrainDance/docs/02-架构设计/Agent规划与LangChain实践路线.md)。

### `test-timeout`

这是一个辅助测试函数，主要用于本地联调，不承担核心业务能力。

### `spatial-search-agent`

这是新增的空间检索 Agent，用于把用户自然语言请求编排成“意图解析 → 类型判断 →
搜索工具调用 → scene/pose 选择 → 可视化动作生成”的闭环。

它当前具备：

1. 解析用户意图
2. 判断是在找物体、位置、时间还是场景
3. 通过 LangChain tool calling 调用多个检索工具
4. 选择最可信的 scene / pose
5. 返回 `open_model`、`fly_to_pose`、`highlight_hotspot`
   等可视化动作，以及可直接给 3D Viewer 使用的 `viewer_payload`
6. 处理模型资产元数据类请求，包括：
   - `list_model_assets`
   - `rename_model_asset`
   - `batch_patch_model_metadata`
   - `get_model_asset_bundle`
   - `compare_model_assets`
7. 对写工具默认走 `dry_run` 预览，只有请求显式传入
   `executionMode: "execute"` 时才正式写库
8. 可选接收前端多选传入的 `selectedModelIds`，把 Agent 操作范围限制在已选模型内

本地运行：

```bash
cd supabase/functions/spatial-search-agent
supabase functions serve spatial-search-agent --no-verify-jwt
```

请求体示例：

```json
{
  "query": "把这几个模型统一改成宿舍-{{created_date}}",
  "selectedModelIds": [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222"
  ],
  "executionMode": "preview"
}
```

## 与其他模块的关系

- `app/`：使用 Anon Key 直连数据库、Storage 和 Realtime
- `dashboard/`：读取任务、资产、空间锚点和 `worker_nodes` 状态，并可写入
  `desired_state`
- `ai_engine/3dgs/`：监听 `processing_tasks`，回写结果到数据库和
  Storage，同时持续更新 `worker_nodes`

## 说明

这份 README 只描述当前仓库里的 Supabase
层，不覆盖线上部署策略，也不替代具体的表结构设计文档。更完整的系统链路请参考项目根目录的
[README.md](/home/ltx/projects/BrainDance/README.md)。
