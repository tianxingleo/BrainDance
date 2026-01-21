# 🧠 BrainDance - Supabase Backend

**3DGS 生成引擎与语义搜索数据库**

这是一个基于 Supabase 的本地后端环境，服务于 BrainDance 项目。它不仅提供 PostgreSQL 数据库和对象存储，还集成了 `pgvector` 用于 RAG（检索增强生成）语义搜索功能。

> **📚 相关文档**
> - [项目主文档](../README.md) - 项目概述、架构说明
> - [docs 文档索引](../docs/README.md) - 完整文档导航
> - [API 接口文档](../docs/API_DOC.md) - 前端接入规范
> - [本地部署指南](../docs/LOCAL_DEPLOYMENT.md) - 完整部署流程
> - [开发环境配置](../docs/开发环境配置.md) - 详细配置指南

## 🏗️ 架构说明 (Architecture)

本项目作为 **数据中台**，连接前端与计算节点：

*   📱 **Frontend (Flutter)**: 直接读写数据库（创建任务、监听进度、下载模型）。
*   ⚙️ **AI Engine (Python)**: 监听任务队列，下载视频，生成 3D 模型，回填 Embedding 向量。
*   🗄️ **Supabase**: 负责身份认证 (Auth)、任务调度 (DB)、文件存储 (Storage) 和 向量检索 (Vector)。
*   ⚡ **Edge Functions (Deno)**: Serverless 函数层，负责语义搜索接口，保护 API Key 不暴露给前端。

---

## 🌐 服务端口 (Service Ports)

| 服务 | 端口 | 访问地址 | 说明 |
|------|------|----------|------|
| **Kong (API Gateway)** | 54321 | http://127.0.0.1:54321 | 主 API 入口（REST, GraphQL, Storage, Auth, Edge Functions） |
| **PostgreSQL** | 54322 | postgresql://postgres:postgres@127.0.0.1:54322/postgres | 数据库连接 |
| **Studio** | 54323 | http://127.0.0.1:54323 | Web 管理界面（推荐使用） |
| **Inbucket (Mailpit)** | 54324 | http://127.0.0.1:54324 | 邮件测试服务器 |
| **Analytics** | 54327 | - | 分析服务 |

**常用访问地址：**
- REST API: http://127.0.0.1:54321/rest/v1
- GraphQL: http://127.0.0.1:54321/graphql/v1
- Storage: http://127.0.0.1:54321/storage/v1
- Edge Functions: http://127.0.0.1:54321/functions/v1/{function-name}
- Studio: http://127.0.0.1:54323

---

## 📋 前置要求 (Prerequisites)

在开始之前，请确保你的机器上安装了以下工具：

1.  **Docker Desktop** (必须保持运行状态)
2.  **Supabase CLI**
    *   **MacOS:** `brew install supabase/tap/supabase`
    *   **Windows:** `scoop bucket add supabase https://github.com/supabase/scoop-bucket.git; scoop install supabase`

---

## 🚀 快速开始 (Quick Start)

### 1. 启动服务
确保 Docker 已启动，在当前目录下运行：

```bash
cd supabase
supabase start
```

首次启动会自动拉取镜像并执行 `supabase/migrations` 下的所有 SQL 脚本（包括开启 `vector` 插件、创建表结构）。

### 2. 获取连接信息
启动成功后，控制台会输出 API URL 和 Keys。**这是连接 Python 后端的关键信息。**

### 3. 配置 Python 环境
请将终端输出的 `API URL` 和 `service_role key` 复制到 `ai_engine/3dgs/.env` 文件中：

```env
# ../ai_engine/3dgs/.env
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=sb_secret_xxxx... (填 service_role key)
```

---

## 🗃️ 数据库结构 (Schema)

本项目核心包含以下业务表（定义位于 `migrations` 文件夹）：

### 1. 任务流水线 (`processing_tasks`)
*   **作用**: 管理 3D 生成任务的生命周期。
*   **字段**:
  - `id` (uuid): 任务唯一标识
  - `user_id` (text): 用户 ID
  - `scene_id` (text): 场景 ID
  - `status` (text): 状态 (pending/processing/completed/failed)
  - `logs` (jsonb): 实时执行日志
  - `quality_score` (integer): AI 质检评分
  - `description`, `tags`, `keywords`: 元数据
*   **流转**: 前端写入 `pending` -> Python 接单改 `processing` -> 完成改 `completed`。

### 2. 3D 资产知识库 (`model_assets`)
*   **作用**: 存储已完成的高质量模型及其 **向量嵌入 (Embeddings)**。
*   **字段**:
  - `id` (uuid): 资产唯一标识
  - `scene_id` (text): 场景 ID
  - `user_id` (text): 所有者 ID
  - `description` (text): AI 生成的场景描述
  - `objects` (text[]): 场景中的物体列表
  - `tags` (text[]): 环境标签
  - `embedding` (vector(1536)): 语义向量（用于 RAG 搜索）
  - `ply_path` (text): 3D 模型文件路径
  - `meta_info` (jsonb): 质量评分、引擎版本等
*   **RAG**: 支持通过 RPC 函数 `match_model_assets` 进行自然语言语义搜索。

### 3. RAG 文档库 (`rag_docs`)
*   **作用**: 存储用于 RAG 检索的文档内容。
*   **字段**:
  - `id` (bigint): 文档 ID
  - `content` (text): 文档内容
  - `metadata` (jsonb): 文档元数据
  - `embedding` (vector(1536)): 文档向量

### 4. 通用任务表 (`tasks`)
*   **作用**: 通用任务队列（备用）。
*   **字段**:
  - `id` (uuid): 任务 ID
  - `user_id` (uuid): 用户 ID
  - `source_path` (text): 源文件路径
  - `status` (text): 任务状态
  - `result_data` (jsonb): 结果数据

### 5. Auth Schema (系统表)
*   `auth.users`: 用户账户信息
*   `auth.sessions`: 会话数据
*   `auth.refresh_tokens`: 刷新令牌

### 6. Storage Schema (系统表)
*   `storage.buckets`: 存储桶配置
*   `storage.objects`: 文件元数据

---

## 📦 存储规范 (Storage)

系统已通过 Seed 自动创建存储桶：**`braindance-assets`** (Public)。

**⚠️ 严格的文件路径规范：**
Python Worker 和 Flutter 前端均依赖此路径结构，请勿随意修改：

```text
braindance-assets/
└── {user_id}/
    └── {scene_id}/
        ├── raw/
        │   └── video.mp4        (输入: 原始视频)
        └── output/
            ├── point_cloud.ply  (输出: 3D模型)
            └── transforms.json  (输出: 预览配置)
```

---

## 🔧 环境变量配置 (Environment Variables)

### Python Worker 必需变量

```bash
# 必填
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=sb_secret_xxxx...  # service_role key

# 可选
SUPABASE_BUCKET=braindance-assets  # 存储桶名称
SUPABASE_TABLE=processing_tasks     # 任务表名称
```

### 获取方式
1. 运行 `supabase start` 后，控制台会输出连接信息
2. 或查看 `supabase/.env.1` 文件（首次启动时生成）

---

## ⚡ Edge Functions (语义搜索)

本项目使用 **Supabase Edge Functions (Deno)** 实现语义搜索功能。

### 功能说明

- **自然语言搜索**: 用户输入"红色杯子"，系统自动理解意图并搜索相关 3D 模型
- **智能时间过滤**: 支持"上周拍的"、"上个月"等自然语言时间描述
- **语义向量匹配**: 使用 AI 生成语义向量，在 pgvector 中进行相似度搜索

### 文件位置

```
supabase/functions/search-models/
├── index.ts      # Edge Function 主程序 (Deno/TypeScript)
├── test.ts       # 自动化测试 (Deno Test)
└── .env.local    # 本地环境变量配置 (不提交 git)
```

### 快速开始

#### 1. 配置环境变量

编辑 `supabase/functions/search-models/.env.local`：

```bash
# DashScope API Key (必填)
DASHSCOPE_API_KEY=sk-your-api-key-here
```

获取 DashScope Key: https://dashscope.console.aliyun.com/

#### 2. 启动 Edge Function

```bash
cd supabase/functions/search-models
supabase functions serve search-models --no-verify-jwt --env-file .env.local
```

启动成功后显示：
```
Serving functions at:
- http://127.0.0.1:54321/functions/v1/search-models
```

#### 3. 测试搜索接口

```bash
# 简单搜索
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"红色杯子"}'

# 带时间过滤的搜索
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"上周拍的照片"}'
```

#### 4. 运行自动化测试

```bash
deno test --allow-all supabase/functions/search-models/test.ts
```

### API 文档

| 项目 | 说明 |
| :--- | :--- |
| **URL** | `/functions/v1/search-models` |
| **Method** | `POST` |
| **Content-Type** | `application/json` |

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
| :--- | :--- | :---: | :--- |
| `query` | string | ✅ | 搜索关键词，支持自然语言 |

**请求示例**:
```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"红色杯子"}'
```

**响应示例**:
```json
{
  "success": true,
  "intent": {
    "original_query": "红色杯子",
    "parsed_search_text": "红色杯子",
    "filter_start": null,
    "filter_end": null
  },
  "results": []
}
```

### 技术实现

| 组件 | 技术 |
| :--- | :--- |
| **运行时** | Deno 1.46+ |
| **语言** | TypeScript |
| **LLM** | DashScope qwen-plus (意图解析) |
| **Embedding** | DashScope text-embedding-v2 (1536 维) |
| **向量搜索** | pgvector (`match_model_assets` RPC) |
| **数据库** | PostgreSQL + Supabase JS |

### 开发说明

#### 添加新的 Edge Function

```bash
# 创建新的 Edge Function
supabase functions new my-new-function

# 编辑代码
supabase/functions/my-new-function/index.ts
```

#### 修改现有代码

1. 编辑 `supabase/functions/search-models/index.ts`
2. 重启 Edge Function:
   ```bash
   # Ctrl+C 停止当前服务
   supabase functions serve search-models --no-verify-jwt --env-file .env.local
   ```

#### 查看日志

```bash
supabase functions logs search-models
```

### 相关文档

- [API 接入文档](../docs/API_DOC.md) - 前端调用接口说明
- [本地部署指南](../docs/LOCAL_DEPLOYMENT.md) - 完整的本地开发指南
- [API 测试报告](../docs/API_TEST_REPORT.md) - 测试结果记录

---

## 💾 备份与恢复 (Backup & Restore)

### 备份数据库
```bash
cd supabase
docker exec supabase_db_BrainDance pg_dump -U postgres -d postgres \
  --format=custom --compress=9 -f /tmp/backup.dump
docker cp supabase_db_BrainDance:/tmp/backup.dump ./backups/
```

### 恢复数据库
```bash
cd supabase
docker cp ./backups/backup.dump supabase_db_BrainDance:/tmp/
docker exec supabase_db_BrainDance pg_restore -U postgres -d postgres \
  -c /tmp/backup.dump
```

### 备份存储文件
```bash
# 打包存储卷
docker run --rm \
  -v supabase_storage_BrainDance:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine:latest \
  sh -c "cd /source && tar czvf /backup/storage.tar.gz ."
```

---

## 💻 常用开发命令

### 1. 修改表结构 (Migrations)
如果你在 Studio (http://localhost:54323) 修改了表结构，必须同步到本地文件，否则队友拉取代码后会丢失结构。

```bash
# 自动生成迁移文件 (基于本地 DB 的变更)
supabase db diff -f update_schema_v1

# 这会在 supabase/migrations/ 下生成新的 sql 文件
```

### 2. 重置环境 (Reset)
如果数据库数据脏了，或者想测试"从零部署"的流程：

```bash
# 清空数据库 -> 重新应用 Migrations -> 重新应用 Seed
supabase db reset
```

### 3. 停止服务
```bash
supabase stop
```

### 4. 查看日志
```bash
# 查看所有容器日志
docker compose logs -f

# 查看特定服务日志
docker logs supabase_db_BrainDance -f
```

### 5. Edge Functions 命令

```bash
# 启动 Edge Function 开发服务器
cd supabase/functions/search-models
supabase functions serve search-models --no-verify-jwt --env-file .env.local

# 部署 Edge Function 到云端
supabase functions deploy search-models

# 查看 Edge Function 日志
supabase functions logs search-models

# 运行 Edge Function 测试
deno test --allow-all supabase/functions/search-models/test.ts
```

---

## 🧪 测试与验证

### 验证服务状态
```bash
# 检查容器运行状态
docker ps | grep supabase

# 测试数据库连接
psql "postgresql://postgres:postgres@localhost:54322/postgres" -c "\dt"
```

### 验证 API 可访问
```bash
# 测试 REST API
curl http://127.0.0.1:54321/rest/v1/processing_tasks

# 测试 Studio
# 浏览器访问 http://127.0.0.1:54323
```

---

## ⚠️ 常见问题

**Q: Python 脚本报错 `PGRST202` 找不到函数？**
A: 这是因为 SQL 函数缓存未刷新。请进入 Studio -> Settings -> API -> 点击 **"Reload schema cache"**。

**Q: 搜索功能报错 `vector` 类型不存在？**
A: 确保迁移脚本中包含了 `create extension if not exists vector;`，并且 Docker 容器已正确加载该插件。

**Q: 启动时 Docker 报错端口占用？**
A: 检查 `54322` (DB) 或 `54321` (API) 端口是否被占用，或尝试 `supabase stop` 后重试。

**Q: 存储文件上传失败？**
A: 检查 `braindance-assets` 存储桶是否存在，确认存储服务正常运行。

**Q: 如何查看数据库中的数据？**
A: 使用 Studio (http://localhost:54323)，使用 `postgres/postgres` 登录。

---

## 📚 相关资源

- [Supabase 官方文档](https://supabase.com/docs)
- [pgvector 使用指南](https://github.com/pgvector/pgvector)
- [项目架构说明](../README.md)
