# 🧠 BrainDance - Supabase Backend

**3DGS 生成引擎与语义搜索数据库**

这是一个基于 Supabase 的本地后端环境，服务于 BrainDance 项目。它不仅提供 PostgreSQL 数据库和对象存储，还集成了 `pgvector` 用于 RAG（检索增强生成）语义搜索功能。

## 🏗️ 架构说明 (Architecture)

本项目作为 **数据中台**，连接前端与计算节点：

*   📱 **Frontend (Flutter)**: 直接读写数据库（创建任务、监听进度、下载模型）。
*   ⚙️ **AI Engine (Python)**: 监听任务队列，下载视频，生成 3D 模型，回填 Embedding 向量。
*   🗄️ **Supabase**: 负责身份认证 (Auth)、任务调度 (DB)、文件存储 (Storage) 和 向量检索 (Vector)。

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

本项目核心包含两张业务表（定义位于 `migrations` 文件夹）：

### 1. 任务流水线 (`processing_tasks`)
*   **作用**: 管理 3D 生成任务的生命周期。
*   **流转**: 前端写入 `pending` -> Python 接单改 `processing` -> 完成改 `completed`。
*   **日志**: 包含 `logs` (JSON) 和 AI 质检报告 (`quality_score`, `tags`)。

### 2. 3D 资产知识库 (`model_assets`)
*   **作用**: 存储已完成的高质量模型及其 **向量嵌入 (Embeddings)**。
*   **RAG**: 支持通过 RPC 函数 `match_model_assets` 进行自然语言语义搜索。

---

## 📦 存储规范 (Storage)

系统已通过 Seed 自动创建存储桶：**`braindance-assets`** (Public)。

**⚠️ 严格的文件路径规范：**
Python Worker 和 Flutter 前端均依赖此路径结构，请勿随意修改：

```text
{user_id}/
  └── {scene_id}/
      ├── raw/
      │   └── video.mp4        (输入: 原始视频)
      └── output/
          ├── point_cloud.ply  (输出: 3D模型)
          └── transforms.json  (输出: 预览配置)
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
如果数据库数据脏了，或者想测试“从零部署”的流程：

```bash
# 清空数据库 -> 重新应用 Migrations -> 重新应用 Seed
supabase db reset
```

### 3. 停止服务
```bash
supabase stop
```

---

## ⚠️ 常见问题

**Q: Python 脚本报错 `PGRST202` 找不到函数？**
A: 这是因为 SQL 函数缓存未刷新。请进入 Studio -> Settings -> API -> 点击 **"Reload schema cache"**。

**Q: 搜索功能报错 `vector` 类型不存在？**
A: 确保迁移脚本中包含了 `create extension if not exists vector;`，并且 Docker 容器已正确加载该插件。

**Q: 启动时 Docker 报错端口占用？**
A: 检查 `54322` (DB) 或 `54321` (API) 端口是否被占用，或尝试 `supabase stop` 后重试。