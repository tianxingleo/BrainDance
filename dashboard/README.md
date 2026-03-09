# BrainDance Dashboard

独立前端看板（`Vue 3 + Vite + TypeScript`），直接使用 Supabase Anon Key 读取业务状态，无需中间层。

## 功能模块

- 节点状态：根据最新任务更新时间推断在线/离线
- 任务队列：展示最近任务、状态标签与进度条
- 资产统计：模型总数、memory pose 总数、任务成功率
- 趋势图：最近 24 小时任务创建量（ECharts）
- 实时刷新：订阅 `processing_tasks` / `model_assets` / `memory_poses`

## 本地运行

```bash
cd dashboard
cp .env.example .env
# 编辑 .env，填入真实 Supabase URL 与 Anon Key
npm install
npm run dev
```

默认访问地址：`http://localhost:5173`

## 环境变量

```bash
VITE_SUPABASE_URL=http://127.0.0.1:54321
VITE_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
# 可选：列桶无权限时按已知桶名探测
VITE_STORAGE_BUCKETS=braindance-assets
# 可选：Edge Functions 探测列表
VITE_SUPABASE_EDGE_FUNCTIONS=search-models,test-timeout
```

## 打包部署

```bash
npm run build
```

构建产物在 `dashboard/dist/`，可直接挂载到 Nginx、Vercel、Netlify 等静态托管平台。
