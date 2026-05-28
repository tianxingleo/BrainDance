# 文档一致性审查变更日志

> 审查日期：2026-04-30
> 审查范围：docs/ 下所有 markdown 文件，与根 README.md 及代码实际状态比对
> 审查基准：根 `README.md`（最新）、`docs/03-API参考/API接口指南.md`（较新）、`docs/03-API参考/数据库设计.md`（较新）

---

## 一、整体评估

docs/ 目录存在 **两套并行的文档结构**，且内容更新不同步：

| 目录 | 状态 | 说明 |
|------|------|------|
| `docs/01-入门指南/`、`docs/02-架构设计/`、`docs/03-API参考/` | 较新（近期同步过） | 推荐使用 |
| `docs/1.总览/`、`docs/2.架构说明/`、`docs/3.API文档/` | 过时 | 早期版本，缺少大量新增功能 |
| `docs/INDEX.md` | 过时 | 引用旧目录结构 |
| `docs/API_DOC.md`（根级副本） | 较新（v1.2） | 与 `03-API参考/API接口指南.md` 基本一致 |
| `docs/3.API文档/API_DOC.md` | 严重过时（v1.0） | 仍引用 Python HTTP 服务，缺少多种任务类型 |
| `docs/3.API文档/数据库设计.md` | 过时 | 缺少 `memory_poses`、`community_posts`、`worker_nodes` 表 |
| `docs/LOCAL_DEPLOYMENT.md` | 过时 | 仅覆盖 search-models，不涉及 Agent 链路 |

---

## 二、逐文件不一致清单

### 2.1 严重过时（已直接修复）

#### (1) `docs/3.API文档/API_DOC.md` -- 严重过时

- **问题**：版本号 v1.0，声称采用 "Supabase (BaaS) + Python (微服务) 混合架构"，仍列出 Python API `http://127.0.0.1:8000`
- **实际状态**（来源：`README.md` 技术架构章节；`supabase/functions/` 目录）：
  - 已完全采用 Supabase BaaS 架构，语义搜索通过 Edge Function (Deno) 实现，无独立 Python API 服务
- **问题**：task_type 只列了 `video_3dgs`、`single_image_sam3d`、`single_image_sharp`
- **实际状态**（来源：`README.md` "当前已接入的主要任务类型"；`ai_engine/3dgs/src/pipelines/` 目录）：
  - 缺少 `multi_image`、`da3_feed_forward_3dgs`、`da3_sugar`/`da3+sugar`、`da3_2dgs`/`da3+2dgs`、`sparse2dgs`
- **问题**：processing_tasks 表缺少 `display_name`、`task_params` 字段
- **实际状态**（来源：`docs/03-API参考/API接口指南.md` 3.1 节）
- **问题**：model_assets 表缺少 `user_id`、`objects`、`preview_img_path`、`meta_info` 字段
- **问题**：缺少 `community_posts`、`worker_nodes`、`memory_poses` 表
- **问题**：Storage 只有一个 bucket，缺少 `braindance-models`
- **处理**：已直接修复（见下方修改记录）

#### (2) `docs/3.API文档/数据库设计.md` -- 过时

- **问题**：概览表只列 4 张表，缺少 `memory_poses`、`community_posts`、`worker_nodes`
- **问题**：processing_tasks DDL 缺少 `display_name`、`task_type`、`task_params` 字段
- **问题**：缺少 `braindance-models` Storage bucket 的文档
- **问题**：变更历史停留在 2026-01-18
- **处理**：已直接修复（见下方修改记录）

### 2.2 内容不完整（需要更新但影响较小）

#### (3) `docs/02-架构设计/系统架构.md`

- **问题**：最后更新日期 2026-01-20
- **问题**：Edge Functions 只提到 "Semantic Search" + "API Key 保护"
- **实际状态**（来源：`README.md` 技术架构章节；`supabase/functions/` 目录）：
  - 当前 Edge Functions 分层：`search-models`（基础搜索）、`agent-recall`（统一 Agent 入口）、`spatial-search-agent`（LangChain 实验入口）、`time-compare-agent`（时间对比）
- **问题**：Pipeline 只列 `video_3dgs.py` 和 `single_image_sam3d.py`
- **实际状态**（来源：`README.md` "当前已接入的主要任务类型"）：
  - 还包括 `da3_sugar`、`da3_2dgs`、`sparse2dgs`、`da3_feed_forward_3dgs`、`single_image_sharp` 等
- **问题**：PostgreSQL 表只列 `processing_tasks`、`model_assets`、`rag_docs`、`tasks`
- **实际状态**（来源：`README.md` 数据流章节）：
  - 还包括 `memory_poses`、`community_posts`、`worker_nodes`、`related_model_links`、`memory_collections`
- **问题**：基础设施列出 MinIO/S3、Redis 为"可选"
- **实际状态**（来源：`README.md` 技术栈速览）：
  - 不再使用 MinIO/S3、Redis，完全依赖 Supabase Storage 和 PostgreSQL
- **问题**：相关文档链接使用旧路径 `../3.API文档/API接口.md`
- **建议修改**：将链接更新为 `../03-API参考/API接口指南.md`

#### (4) `docs/02-架构设计/项目架构.md`

- **问题**：PostgreSQL 表只列 4 张（`processing_tasks`、`model_assets`、`rag_docs`、`tasks`）
- **实际状态**（来源：`README.md` 数据流章节）：还包括 `memory_poses`、`community_posts`、`worker_nodes`
- **问题**：Pipeline 类型只列 3 种
- **实际状态**：已接入 8+ 种（参见 README "当前已接入的主要任务类型"）
- **问题**：核心模块表列出的路径（如 `src/modules/image_proc.py`）可能已过时
- **问题**：技术栈基础设施列出 MinIO/S3、Redis
- **建议修改**：补全新增表、Pipeline 类型，移除 MinIO/S3、Redis 引用

#### (5) `docs/02-架构设计/技术栈清单.md`

- **问题**：Edge Functions 只列 `search-models`
- **实际状态**（来源：`supabase/functions/` 目录）：
  - 还包括 `agent-recall`、`spatial-search-agent`、`time-compare-agent`
- **问题**：本地问答微调部分标记为"实验"，未提及已在 Flutter 中落地的端侧 RAG 链路
- **实际状态**（来源：`README.md` "端侧本地问答模型下载、选择与受约束问答"）
- **问题**：最后更新日期 2026-01-23

#### (6) `docs/1.总览/项目架构.md`

- **问题**：与 `docs/02-架构设计/项目架构.md` 内容高度重复，且同样缺失新增表和 Pipeline
- **问题**：相关文档链接使用旧路径
- **建议**：考虑重定向到 `docs/02-架构设计/项目架构.md` 或标记为废弃

#### (7) `docs/2.架构说明/系统架构.md`

- **问题**：与 `docs/02-架构设计/系统架构.md` 内容高度重复
- **问题**：Pipeline 只列 `video_3dgs` 和 `single_image_sam3d`
- **建议**：标记为废弃，指向 `docs/02-架构设计/系统架构.md`

### 2.3 索引与导航文件

#### (8) `docs/INDEX.md`

- **问题**：目录树完全过时，使用旧目录结构（`1.总览/`、`2.架构说明/`、`3.API文档/`、`4.测试/`、`5.贡献/` 等）
- **实际状态**：当前目录结构为 `01-入门指南/` 到 `09-LangChain专题/`
- **问题**：缺少 `04-本地问答与微调/`、`09-LangChain专题/` 等新目录
- **问题**：Supabase Functions 目录只列 `search-models/`
- **实际状态**（来源：`supabase/functions/` 目录）：还包括 `agent-recall/`、`spatial-search-agent/`、`time-compare-agent/`、`_shared/agent-core/`
- **处理**：已直接修复

#### (9) `docs/README.md`

- **问题**：常用命令速查中 conda 环境名写 `gs_linux_backup`
- **实际状态**（来源：`README.md` 快速开始章节）：应使用 `braindance`
- **问题**：文档版本号 v1.0，最后更新 2026-03-22
- **处理**：已直接修复 conda 环境名

### 2.4 入门文档

#### (10) `docs/01-入门指南/快速开始.md`

- **问题**：最低 GPU 要求写 "RTX 3060 或更高，至少 8GB 显存"
- **实际状态**（来源：`README.md` 测试环境章节）：
  - AI Engine 最低配置：NVIDIA RTX 5070 12GB
  - 推荐配置：NVIDIA L20 46GB x2
- **问题**：验证数据库表只列 `processing_tasks`、`model_assets`、`rag_docs`、`tasks`
- **实际状态**：还应包括 `memory_poses`、`community_posts`、`worker_nodes`
- **问题**：缺少 Dashboard 启动步骤
- **实际状态**（来源：`README.md` 启动步骤章节）：
  - 应增加 Dashboard 启动：`cd dashboard && npm install && npm run dev`
- **问题**：本地问答微调参考文档路径使用旧格式 `docs/开发文档/`
- **实际状态**：应为 `docs/04-本地问答与微调/`
- **建议修改**：更新 GPU 要求、补充数据库表列表、增加 Dashboard 步骤

#### (11) `docs/01-入门指南/本地部署.md`

- **问题**：仅覆盖 `search-models` Edge Function 的本地部署
- **问题**：未涉及 `agent-recall` 等新 Edge Function
- **问题**：向量维度标注为 1536（需确认是否准确）
- **建议修改**：补充 `agent-recall` 本地运行说明

### 2.5 功能描述缺失

以下在 README 中已有详细描述但 docs/ 下尚未覆盖的功能点：

#### (12) Flutter 移动端 Recall 页面

- **来源**：`README.md` "移动端主链路已经包括" 章节
- **缺失**：docs/ 中无专门的 Flutter Recall 功能描述文档
- **实际能力**：
  - `agent-recall` 驱动的 Agent 检索
  - 多轮续聊
  - 流式步骤面板（SSE/NDJSON）
  - 端侧本地问答模型下载、选择与受约束问答

#### (13) Agent Recall 统一入口

- **来源**：`README.md` Agent Recall 章节
- **已有文档**：`docs/09-LangChain专题/` 中有详细记录，但架构文档未引用
- **缺失**：`docs/02-架构设计/` 系列文档未更新 Agent 能力描述

#### (14) Dashboard Worker/任务观测

- **来源**：`README.md` "启动步骤" 章节、`dashboard/` 目录
- **已有文档**：`dashboard/` 目录存在
- **缺失**：`docs/` 中无专门的 Dashboard 功能描述

#### (15) 3DGS Viewer AR 模式 / Marker 锚点定位

- **来源**：最近 git commit `6d8e5f5 feat(viewer): 新增基于图片锚点的 Marker AR 查看模式`
- **缺失**：docs/ 中无 AR 模式文档
- **说明**：这是非常新的功能（当前分支 `tianxingleo-ar`），可能尚未合并到 main

#### (16) 本地 Qwen3 RAG/LLM

- **来源**：`README.md` "端侧本地问答模型下载" 章节、`ai_engine/finetune_qwen3/` 目录
- **已有文档**：`docs/04-本地问答与微调/` 中有大量记录
- **问题**：架构文档（`docs/02-架构设计/`）中未体现此能力

---

## 三、已直接修改的文件

以下文件已在本审查中直接修改：

| 文件 | 修改内容 |
|------|---------|
| `docs/3.API文档/API_DOC.md` | 全面重写，与 `docs/03-API参考/API接口指南.md` 对齐 |
| `docs/3.API文档/数据库设计.md` | 补全新增表、字段、bucket，与 `docs/03-API参考/数据库设计.md` 对齐 |
| `docs/INDEX.md` | 更新目录结构、函数列表，与实际 docs/ 目录对齐 |
| `docs/README.md` | 修正 conda 环境名为 `braindance`，更新最后更新日期 |

---

## 四、建议后续处理

1. **废弃标记**：在 `docs/1.总览/`、`docs/2.架构说明/`、`docs/3.API文档/` 的 README 或顶部添加指向新目录的提示
2. **架构文档更新**：更新 `docs/02-架构设计/系统架构.md` 和 `项目架构.md` 中的 Edge Functions 列表、Pipeline 列表、数据库表列表
3. **新增功能文档**：为 AR 模式、Dashboard 观测等新功能创建描述文档
4. **目录去重**：`docs/1.总览/` 与 `docs/02-架构设计/`、`docs/3.API文档/` 与 `docs/03-API参考/` 存在大量重复，建议清理或添加废弃提示
