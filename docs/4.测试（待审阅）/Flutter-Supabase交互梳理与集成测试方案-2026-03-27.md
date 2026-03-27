# Flutter 与 Supabase 交互梳理及集成测试方案

更新时间：2026-03-27

## 1. 文档目标

本文面向 BrainDance 仓库当前代码现状，重点梳理 `app/` 下 Flutter 项目与 Supabase 的实际交互方式，并给出一套可落地的集成测试方案。

本文只以当前代码为准，不以历史设计口径为准。

## 2. 交互总览

当前 Flutter 与 Supabase 的交互可以分成 5 类：

1. `Auth`
2. `Database`
3. `Storage`
4. `Realtime`
5. `Edge Functions`

入口初始化位于 `app/lib/main.dart:48-62`，应用启动后会读取 `.env`，再通过 `Supabase.initialize()` 初始化客户端。

Supabase 配置位于 `app/lib/configs/supabase_config.dart`，这里定义了：

- `SUPABASE_URL`
- `SUPABASE_KEY / SUPABASE_SECRET_KEY / SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY`
- `isAdminMode`
- 端侧模型分发 bucket 与默认对象路径

其中 `isAdminMode` 是一个非常关键的测试维度：

- 普通模式：按登录态和 RLS 路径访问
- 管理员模式：如果注入的是 secret / service role key，前端会绕过登录页，直接进入应用

这意味着集成测试至少要覆盖两套运行口径：

- `RLS 用户模式`
- `Admin 浏览模式`

## 3. Flutter 直连 Supabase 的交互清单

## 3.1 Auth

### 3.1.1 登录与注册

文件：`app/lib/pages/login.dart:30-58`

交互：

- `auth.signUp(email, password)`
- `auth.signInWithPassword(email, password)`

用途：

- 注册账号
- 登录后进入主应用

测试关注点：

- 成功登录后是否进入 `/`
- 错误密码时是否显示错误
- `Admin Mode` 下是否不走登录流程

### 3.1.2 登录状态驱动页面刷新

文件：`app/lib/pages/task_list.dart:70-89`

交互：

- `auth.onAuthStateChange`

用途：

- 任务列表在登录/登出时刷新或清空

测试关注点：

- 登出后任务页是否清空并报“未登录”
- 重新登录后是否恢复查询

## 3.2 Database

当前 Flutter 直接访问的核心表主要有 3 张：

1. `processing_tasks`
2. `model_assets`
3. `community_posts`

### 3.2.1 processing_tasks

表定义起点：`supabase/migrations/20260118144558_init_schema.sql`

增量字段：

- `task_type` / `task_params`
- `display_name`

Flutter 读写位置如下。

#### A. 任务创建

文件：

- `app/lib/pages/generate/generate_submission.dart:362-400`
- `app/lib/pages/generate/generate_submission.dart:470-506`
- `app/lib/pages/video_submit.dart:116-130`

交互：

- `insert into processing_tasks`

典型写入字段：

- `scene_id`
- `user_id`
- `status`
- `task_type`
- `task_params`
- `display_name`

业务意义：

- 图片转 3D 任务提交
- 视频转 3D 任务提交
- 双链路视频任务提交

#### B. 任务列表读取

文件：`app/lib/pages/task_list.dart:104-143`

交互：

- `select * from processing_tasks`
- 普通用户模式按 `user_id` 过滤
- 管理员模式直接读取全部

业务意义：

- 任务页按状态分组展示
- 展示日志摘要

#### C. Recall 页处理中任务读取

文件：`app/lib/pages/recall/recall_data_sync.dart:78-109`

交互：

- `select * from processing_tasks where status = 'processing'`

业务意义：

- Recall 页顶部处理中任务卡片
- 任务日志跟随更新

#### D. Recall 页补 display_name

文件：`app/lib/pages/recall/recall_data_sync.dart:158-183`

交互：

- 用 `scene_id` 回查 `processing_tasks(scene_id, display_name)`

业务意义：

- `model_assets` 里当前未稳定承担统一展示名来源，前端仍要从任务表补齐显示名

#### E. 模型详情读取任务信息

文件：`app/lib/pages/recall/recall_model_actions.dart:147-186`

交互：

- `select created_at, updated_at, task_type, quality_score from processing_tasks`

业务意义：

- Recall 模型详情弹窗展示任务元数据

#### F. 模型重命名

文件：`app/lib/pages/recall/recall_model_actions.dart:228-281`

交互：

- `update processing_tasks set display_name = ? where scene_id = ?`

业务意义：

- 前端当前通过更新任务表的 `display_name` 来驱动 Recall 展示名

#### G. Viewer 同名兄弟模型查询

文件：`app/lib/services/viewer_navigation.dart:39-97`

交互：

- 先查 `processing_tasks(scene_id, display_name)`
- 再基于同 `display_name` 找到同名版本集合

业务意义：

- Time Peeling 时间剥离视图

### 3.2.2 model_assets

表定义起点：`supabase/migrations/20260118144558_init_schema.sql`

增量字段：

- `display_name`
- 近期 agent 相关字段：`place_id`、`memory_thread_id`、`version_label`、`summary_title`、`agent_meta`

Flutter 读写位置如下。

#### A. Recall 主列表读取

文件：`app/lib/pages/recall/recall_data_sync.dart:144-207`

交互：

- `select id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info, created_at from model_assets order by created_at desc`

业务意义：

- Recall 页的核心模型列表
- 本地索引同步输入源

#### B. 任务页点开已完成任务时查询模型文件

文件：`app/lib/pages/task_list.dart:353-368`

交互：

- 根据 `scene_id` 查询 `model_assets.ply_path`

业务意义：

- 取得 viewer 入口的模型文件地址

#### C. Community 可分享模型读取

文件：`app/lib/pages/community/repository.dart:91-125`

交互：

- `select id, scene_id, description, ply_path, preview_img_path from model_assets`
- 已登录时按 `user_id` 过滤

业务意义：

- 发帖时选择要分享的模型

#### D. Recall 模型删除

文件：`app/lib/pages/recall/recall_model_actions.dart:467-540`

交互：

- `delete from model_assets where id = ? and user_id = ?`

业务意义：

- 删除自己的云端模型记录

注意：

- 删除数据库前会先递归删除对应 Storage 目录

#### E. 本地 AI 轮询本人模型签名

文件：`app/lib/pages/recall/recall_local_ai.dart:104-120`

交互：

- `select id, scene_id, created_at from model_assets where user_id = ?`

业务意义：

- 检测自己是否新增/减少模型，触发 Recall 刷新

### 3.2.3 community_posts

表定义：`supabase/migrations/20260317170000_create_community_posts.sql`

Flutter 读写位置如下。

#### A. 社区帖子列表读取

文件：`app/lib/pages/community/repository.dart:11-89`

交互：

- 读取 `community_posts`
- 联表 `model_assets`

业务意义：

- 社区流展示
- 从帖子回溯到模型和封面图

#### B. 社区发帖

文件：`app/lib/pages/community/repository.dart:131-164`

交互：

- `insert into community_posts`

业务意义：

- 将当前模型发布到社区

注意：

- 插入失败时会退化为 `_localDrafts`，因此测试要区分“服务端成功”和“本地乐观降级”两种结果

## 3.3 Storage

### 3.3.1 原始素材上传

文件：

- `app/lib/pages/generate/generate_submission.dart:564-604`
- `app/lib/pages/video_submit.dart:81-114`

交互：

- 不是通过 `supabase.storage.from(...).upload()` 上传
- 而是直接用 `Dio POST /storage/v1/object/braindance-assets/{user_id}/{scene_id}/raw/...`

用途：

- 上传 `image.png`
- 上传 `video.mp4`

测试关注点：

- Authorization / apikey 头是否正确
- 大文件上传是否返回 200/201
- 上传后对象路径是否可在 bucket 中看到

### 3.3.2 模型和姿态文件公开地址拼装

文件：

- `app/lib/services/viewer_navigation.dart:8-30`
- `app/lib/pages/recall/recall_data_sync.dart:112-141`
- `app/lib/pages/community/repository.dart:167-199`

交互：

- `storage.from('braindance-assets').getPublicUrl(path)`

用途：

- 生成 `ply_path`
- 推导 `webgl_poses.json`
- 生成封面/模型访问链接

### 3.3.3 Recall 删除云端模型时递归删文件

文件：`app/lib/pages/recall/recall_model_actions.dart:410-499`

交互：

- `storage.list(path: ...)`
- `storage.remove(storageFiles)`

用途：

- 删除整个模型目录

### 3.3.4 本地 AI 模型目录扫描

文件：`app/lib/services/local_model_catalog_service.dart:37-122`

交互：

- 扫描 `braindance-models`
- 也兼容扫描 `braindance-assets`
- 读取 `catalog/model_catalog.json`
- `storage.list()` 枚举 `.gguf`

用途：

- 端侧模型下载列表

测试关注点：

- bucket 不存在时能否优雅降级
- catalog 缺失时是否还能从目录扫描回退

## 3.4 Realtime

当前 Flutter 只直接监听 `processing_tasks`。

### 3.4.1 Recall 页面 Realtime

文件：`app/lib/pages/recall/recall_data_sync.dart:4-59`

交互：

- `channel('public:processing_tasks:recall')`
- `onPostgresChanges(event: all, table: processing_tasks)`

用途：

- 处理中的任务实时更新
- 从 `processing -> completed/failed` 时立即移出

### 3.4.2 全局任务通知 Realtime

文件：`app/lib/services/task_notification_service.dart:52-142`

交互：

- `channel('public:processing_tasks')`
- 监听 `update`

用途：

- 全局完成/失败任务通知

测试关注点：

- 从非目标状态切到 `completed`/`failed` 时是否只提醒一次
- 进入任务页后是否正确标记为已通知

## 3.5 Edge Functions

## 3.5.1 search-models

Flutter 调用：

- `app/lib/pages/recall/recall_search.dart:135-152`

函数入口：

- `supabase/functions/search-models/index.ts`
- `supabase/functions/search-models/shared.ts`

Flutter 行为：

- `functions.invoke('search-models', body: {'query': query})`

函数侧实际数据库交互：

- `rpc('match_memory_poses')`
- `select model_assets(id, scene_id, display_name)` 补显示名

结论：

- 虽然 Flutter 没有直接读 `memory_poses`，但云端搜索链路已经依赖它

## 3.5.2 agent-recall

Flutter 调用：

- `app/lib/services/agent_recall_service.dart:536-709`
- `app/lib/pages/recall/recall_search.dart:154-297`

Flutter 行为：

- 流式模式：手工 `POST /functions/v1/agent-recall?stream=1`
- 回退模式：`functions.invoke('agent-recall')`

函数入口：

- `supabase/functions/agent-recall/index.ts`
- `supabase/functions/_shared/agent-core/spatialAgent.ts`
- `supabase/functions/_shared/agent-core/assetTools.ts`
- `supabase/functions/_shared/agent-core/memoryTools.ts`

函数侧实际数据库交互范围：

- `processing_tasks`
- `model_assets`
- `memory_poses`
- `related_model_links`
- `memory_collections`
- `memory_collection_items`
- `match_memory_poses` RPC
- `match_model_assets` RPC

函数侧还可能产生写入：

- 元数据批量更新 `model_assets`
- 创建专题 `memory_collections`
- 维护专题条目 `memory_collection_items`
- 归组线程 `model_assets.place_id / memory_thread_id / version_label`
- 创建创作任务 `processing_tasks`

结论：

- Agent 相关能力对 Flutter 来说是“函数调用”
- 对数据库来说是“多表读写编排”
- 这部分必须单独做集成测试，不能只测 UI

## 3.5.3 text-to-image

Flutter 调用：

- `app/lib/pages/generate/generate_submission.dart:422-467`

函数入口：

- `supabase/functions/text-to-image/index.ts`

函数侧交互：

- 调 DashScope 异步生图接口
- 不直接写 Supabase 数据库

结论：

- 这是外部依赖型集成点
- 测试时建议用桩或录制回放，不建议每次真调模型

## 3.5.4 confirm-text-image

Flutter 调用：

- `app/lib/pages/generate/generate_submission.dart:285-345`

函数入口：

- `supabase/functions/confirm-text-image/index.ts`

函数侧实际数据库/存储交互：

- 校验用户 token
- 下载远程图片
- 上传图片到 `braindance-assets`
- `insert into processing_tasks`

结论：

- 这是“函数内代前端完成存储与写库”的复合事务链路
- 必须做端到端验证

## 4. 当前 Flutter 实际依赖的 Supabase 对象

### 4.1 Flutter 直接依赖

- `auth.users` 相关认证能力
- `processing_tasks`
- `model_assets`
- `community_posts`
- Storage bucket：`braindance-assets`
- Storage bucket：`braindance-models`
- Realtime：`processing_tasks`

### 4.2 Flutter 通过 Edge Function 间接依赖

- `memory_poses`
- `related_model_links`
- `memory_collections`
- `memory_collection_items`
- `match_memory_poses`
- `match_model_assets`

## 5. 当前现状中的测试难点

## 5.1 管理员模式与普通用户模式行为不同

同一套 UI，在 `service_role / secret key` 下会绕过登录并可能绕过 RLS。

如果只测一种模式，会漏掉：

- 登录流程
- 用户隔离
- RLS 真实权限问题

## 5.2 上传链路是“REST Storage + DB 写库”分两步

图片/视频任务提交不是单事务：

1. 先上传 Storage
2. 再写 `processing_tasks`

这意味着测试里要检查“半成功”场景：

- 文件上传成功但写库失败
- 写库成功前页面退出

## 5.3 Recall 页面既有轮询也有 Realtime

Recall 当前混合使用：

- 首次查询
- 5 秒轮询部分状态
- Realtime 监听 `processing_tasks`

如果只做静态查询测试，无法验证真实刷新行为。

## 5.4 Agent 测试不能只看最终回答

`agent-recall` 还承载：

- 流式事件顺序
- `tool_trace`
- `top_candidates`
- `follow_up`
- `session_state`

这部分应视为协议级集成测试。

## 6. 推荐的集成测试范围

建议把集成测试分成 4 层。

### 6.1 第 1 层：Flutter 直连 Supabase 基础链路

覆盖目标：

- 登录/注册
- 任务提交
- 任务列表读取
- Recall 模型读取
- 社区帖子读写
- Storage URL 推导

### 6.2 第 2 层：Flutter + Realtime

覆盖目标：

- Recall 处理中任务跟随更新
- 全局通知跟随状态变化

### 6.3 第 3 层：Flutter + Edge Functions

覆盖目标：

- `search-models`
- `agent-recall`
- `confirm-text-image`

说明：

- `text-to-image` 建议默认用 mock / stub
- 否则测试成本和外部不稳定性过高

### 6.4 第 4 层：Agent / 搜索函数协议回归

覆盖目标：

- Agent 流式事件顺序
- 非流式 fallback
- 预览与执行模式差异
- 写工具副作用是否落库

## 7. 推荐测试环境

## 7.1 Supabase 本地栈

建议使用仓库自带 `supabase/` 目录本地启动：

```bash
cd supabase
supabase start
```

建议测试环境要求：

- 使用本地 PostgreSQL + Realtime + Storage + Edge Runtime
- 单独准备 `.env.test`
- 不复用开发者个人线上项目

## 7.2 Flutter 集成测试目录

当前 `app/` 下还没有 `integration_test/`，只有：

- `app/test/widget_test.dart`
- `app/test_invoke.dart`

建议新增：

```text
app/integration_test/
  auth_flow_test.dart
  task_submission_test.dart
  recall_sync_test.dart
  community_flow_test.dart
  edge_function_flow_test.dart
```

## 7.3 测试数据准备

建议用 3 类固定账号：

- `user_a@test.local`
- `user_b@test.local`
- `admin@test.local`

建议准备 4 组固定场景：

- `scene_processing_001`
- `scene_completed_001`
- `scene_same_name_v1`
- `scene_same_name_v2`

建议准备 2 个固定 bucket 路径族：

- `braindance-assets/{user_id}/{scene_id}/raw/*`
- `braindance-assets/{user_id}/{scene_id}/output/*`

还应准备：

- 至少 1 组 `memory_poses`
- 至少 1 组 `community_posts`
- 至少 1 组 `memory_collections`

## 8. 建议的测试用例矩阵

## 8.1 Auth 用例

### 用例 A1：普通用户登录成功

步骤：

1. 启动 Flutter
2. 输入测试账号密码
3. 点击登录

断言：

- 成功跳转首页
- `Supabase.instance.client.auth.currentSession` 非空

### 用例 A2：错误密码登录失败

断言：

- 页面不跳转
- 出现错误提示

### 用例 A3：Admin Mode 直接进入应用

前提：

- `.env` 使用 secret/service-role key

断言：

- 首屏直接进入 `/`
- 不出现登录表单

## 8.2 任务提交用例

### 用例 T1：图片任务提交成功

覆盖代码：

- `generate_submission.dart` 图片上传
- `processing_tasks` 写入

断言：

- Storage 中出现 `raw/image.png`
- `processing_tasks` 新增 1 条记录
- 字段 `task_type` 与所选类型一致

### 用例 T2：视频任务提交成功

断言：

- Storage 中出现 `raw/video.mp4`
- `processing_tasks.task_type` 正确
- `task_params` 正确落库

### 用例 T3：视频提交页 display_name 正确写入

断言：

- `processing_tasks.display_name` 等于输入值

### 用例 T4：未登录时提交任务会先要求登录

断言：

- 被导航到登录页
- 登录完成后重新提交成功

## 8.3 Recall 与模型列表用例

### 用例 R1：Recall 首次加载模型列表

断言：

- `model_assets` 被成功读取
- `processing_tasks.display_name` 被正确合并到模型卡片

### 用例 R2：处理中任务实时出现

前提：

- 测试过程中直接更新数据库，把某任务状态改为 `processing`

断言：

- Recall 页出现处理中任务
- 日志列表同步刷新

### 用例 R3：处理中任务完成后从 Recall 消失

前提：

- 将同一任务状态从 `processing` 更新为 `completed`

断言：

- Realtime 收到更新
- 任务从处理中区域移除

### 用例 R4：点击已完成任务能打开 viewer 所需模型地址

断言：

- 会查询 `model_assets.ply_path`
- 能生成公开 URL

### 用例 R5：同名场景兄弟版本可被查询出来

断言：

- `viewer_navigation.dart` 能基于 `display_name` 找出同名版本集合

## 8.4 社区用例

### 用例 C1：读取社区流

断言：

- `community_posts` 列表正常返回
- 联表模型字段正常映射

### 用例 C2：创建社区帖子成功

断言：

- `community_posts` 新增记录
- 列表刷新后可见新帖子

### 用例 C3：发帖失败时本地草稿兜底

做法：

- 人为让 `community_posts` 插入失败

断言：

- UI 中仍出现本地乐观帖子
- 但数据库无新增记录

## 8.5 Storage 用例

### 用例 S1：Recall 云端模型删除成功

断言：

- 目标目录文件从 `braindance-assets` 删除
- `model_assets` 对应记录被删除

### 用例 S2：只能删除自己的模型

断言：

- `user_a` 无法删除 `user_b` 的模型

### 用例 S3：本地 AI 模型目录可扫描

断言：

- 当 `catalog/model_catalog.json` 存在时可读取 catalog
- 当 catalog 缺失时仍能通过 `storage.list()` 找到 `.gguf`

## 8.6 Edge Function 用例

### 用例 E1：search-models 返回可解析结果

断言：

- 返回 `success=true`
- `results` 为列表
- Recall 搜索结果可正常渲染

### 用例 E2：agent-recall 流式返回完整事件

断言：

- 至少包含 `ping/status/.../done`
- `done` 中包含 `answer`
- 若有候选，则 `top_candidates` 结构完整

### 用例 E3：agent-recall 流式失败后走 fallback

做法：

- 故意制造流式链路异常

断言：

- Flutter 自动回退到 `functions.invoke('agent-recall')`
- 仍能得到最终答案或明确错误

### 用例 E4：confirm-text-image 端到端成功

断言：

- 函数返回 `scene_id`
- Storage 出现图片
- `processing_tasks` 新增对应记录

### 用例 E5：agent 写工具预览与执行分离

前提：

- 选取会触发元数据修改的问句

断言：

- `preview` 模式下数据库不发生真实写入
- `execute` 模式下写入实际发生

## 9. 推荐实现方案

## 9.1 Flutter 侧

建议使用 Flutter 官方 `integration_test` 包。

推荐能力：

- 驱动真实页面交互
- 直接验证导航、表单、列表、提示
- 可在测试中调用辅助 HTTP / Supabase 校验接口

建议补充一个测试工具层，例如：

```text
app/integration_test/support/
  test_env.dart
  test_accounts.dart
  supabase_assertions.dart
  storage_assertions.dart
  seed_client.dart
```

职责建议：

- `test_env.dart`：读取 `.env.test`
- `test_accounts.dart`：封装测试账号登录
- `supabase_assertions.dart`：断言表记录是否存在
- `storage_assertions.dart`：断言 bucket 对象是否存在
- `seed_client.dart`：测试前后准备/清理数据

## 9.2 数据准备与清理

建议通过单独脚本做测试前置和回收，而不是把建数逻辑散落到每个测试里。

建议新增：

```text
tests/fixtures/
  supabase_seed.sql
  cleanup.sql
```

或者：

```text
tests/scripts/
  seed_supabase_test_data.sh
  cleanup_supabase_test_data.sh
```

要求：

- 每个测试场景使用可重复的 `scene_id`
- 每轮执行前先清理旧数据
- 避免污染开发数据

## 9.3 Edge Function 测试补充

对于 `agent-recall`，建议同时保留两类测试：

1. Flutter 端集成测试
2. 函数级 CLI / HTTP 回归测试

仓库现状里，Agent 联调推荐脚本是：

```bash
python ai_engine/finetune_qwen3/scripts/agent_recall_debug_cli.py \
  --query "请你找一下洛天依相关的模型" \
  --execution-mode preview \
  --show-request \
  --show-response-meta \
  --show-event-timeline \
  --show-full-result
```

这类脚本适合验证：

- 事件顺序
- tool trace
- top candidates
- follow up
- session state

Flutter 集成测试则重点验证：

- 页面是否正确消费这些事件
- UI 是否在流式过程中正确更新

## 10. 推荐执行顺序

建议按以下顺序建设：

1. 先补 `integration_test` 基础框架与 `.env.test`
2. 再做 Auth + 任务提交两条主链路
3. 再做 Recall 列表与 Realtime
4. 再做 Community
5. 最后做 `agent-recall` 与 `confirm-text-image`

原因：

- Auth + 提交链路覆盖了最核心的上传与写库路径
- Recall 覆盖了最复杂的读库与实时同步
- Agent 是最高成本链路，适合放在基础设施稳定后补齐

## 11. 建议的验收标准

一套可接受的 Flutter-Supabase 集成测试基线，至少应满足：

1. 普通用户登录成功与失败用例都可自动执行
2. 图片/视频任务提交后，Storage 与 `processing_tasks` 都能被自动断言
3. Recall 页面能自动验证模型列表读取和处理中任务同步
4. 社区发帖成功链路能自动验证
5. `search-models` 与 `agent-recall` 至少各有 1 条自动化回归
6. 测试执行前后可自动清理数据

## 12. 结论

从当前代码看，Flutter 并不是只把 Supabase 当成“登录后端”使用，而是同时深度依赖了：

- 用户认证
- 任务主表
- 模型资产表
- 社区帖子表
- Storage 文件系统
- Realtime 状态推送
- 多个 Edge Functions

其中最核心的业务主链路是：

1. 登录
2. 上传素材到 Storage
3. 写入 `processing_tasks`
4. Worker 处理后回写 `model_assets`
5. Recall / Task List / Community 再读取这些结果

因此最合理的集成测试策略不是只测单个页面，而是围绕这条主链路构建端到端用例，再针对 Realtime 与 Agent 补协议级回归。
