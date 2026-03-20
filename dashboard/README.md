# BrainDance Dashboard

`dashboard/` 是一个基于 `Vue 3 + Vite + TypeScript` 的运维看板，用来观察 BrainDance 当前的任务状态、存储情况和 Realtime 连接状态。

## 当前功能

结合 `src/App.vue`，当前页面主要提供：

- 任务总览：`pending / processing / completed / failed`
- 任务筛选：按状态、任务类型和关键词过滤
- 成功率、平均质量分、平均处理时长
- 最近任务趋势图
- Worker 在线状态推断
- Worker 实例注册、心跳与在线数量统计
- 对指定 Worker 发起优雅暂停（远程 `Ctrl+C` 风格）
- Realtime 订阅状态
- 数据表数量统计
- Storage bucket 探测与容量统计
- Edge Functions 探测

目前只有一个主页面，核心逻辑集中在：

- `src/App.vue`
- `src/lib/supabase.ts`

## 环境变量

先复制模板：

```bash
cd dashboard
cp .env.example .env
```

然后按实际环境填写：

```env
VITE_SUPABASE_URL=http://127.0.0.1:54321
VITE_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
VITE_STORAGE_BUCKETS=braindance-assets
VITE_SUPABASE_EDGE_FUNCTIONS=search-models,test-timeout
```

说明：

- `VITE_SUPABASE_URL` 和 `VITE_SUPABASE_ANON_KEY` 为必填
- `VITE_STORAGE_BUCKETS` 用于在无法列桶时按已知桶名探测
- `VITE_SUPABASE_EDGE_FUNCTIONS` 用于前端探测函数健康状态

## 本地运行

```bash
cd dashboard
npm install
npm run dev
```

默认访问地址：

```text
http://localhost:5173
```

## 构建

```bash
npm run build
```

构建产物位于 `dashboard/dist/`。

## 使用前提

这个看板默认直连 Supabase，因此它依赖以下对象已经存在：

- `processing_tasks`
- `model_assets`
- `memory_poses`
- `rag_docs`
- `tasks`
- `worker_nodes`
- `braindance-assets` bucket

如果本地数据库还没初始化，请先参考 [supabase/README.md](/home/ltx/projects/BrainDance/supabase/README.md) 启动 Supabase。

## 限制说明

- 这是内部状态看板，不是独立业务后台
- 当前没有单独的后端中间层，前端直接使用 Supabase Anon Key
- 页面展示能力以 `src/App.vue` 的当前实现为准，README 不额外承诺未落地功能
