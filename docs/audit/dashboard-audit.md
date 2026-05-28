# Dashboard 审计报告

## 审计概要

- 审计时间：2026-04-30
- 审计范围：`dashboard/src` 全部源文件及根目录配置文件
- 发现问题数：23 个
- P0: 2 个 | P1: 5 个 | P2: 9 个 | P3: 7 个

### 审计文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `dashboard/src/App.vue` | 2015 | 主页面：全部业务逻辑、数据获取、UI 渲染 |
| `dashboard/src/main.ts` | 7 | 应用入口 |
| `dashboard/src/lib/supabase.ts` | 17 | Supabase 客户端初始化 |
| `dashboard/src/lib/task-insights.ts` | 279 | 任务日志解析与阶段推断 |
| `dashboard/src/components/TaskLogDrawer.vue` | 392 | 任务日志抽屉组件 |
| `dashboard/src/components/HelloWorld.vue` | 41 | 脚手架默认组件（未使用） |
| `dashboard/src/style.css` | 929 | 全局样式 |
| `dashboard/src/vite-env.d.ts` | 9 | 类型声明 |
| `dashboard/package.json` | 29 | 依赖配置 |
| `dashboard/vite.config.ts` | 7 | Vite 构建配置 |
| `dashboard/tsconfig.json` | 7 | TS 配置（引用子配置） |
| `dashboard/tsconfig.app.json` | 16 | 应用 TS 编译选项 |

---

## 功能现状

### 已实现功能

1. **任务总览面板**：成功率、队列数量、Worker 状态、存储状态、Edge 函数状态、用户/资产数量、平均质量/耗时
2. **任务趋势图**：支持 24h / 7d / 30d / 全部四种时间范围的折线图
3. **任务状态饼图**：排队中/处理中/已完成/失败四态占比
4. **任务队列表格**：前 20 条任务列表，含状态标签、进度条、日志摘要、质量分
5. **处理中模型卡片**：展示 processing 状态任务的阶段、Worker、进度、最新日志
6. **失败任务时间线**：展示最近 6 条失败任务及日志摘要
7. **Worker 集群管理**：Worker 表格 + 优雅暂停 / 中断 / 恢复控制按钮
8. **Storage 桶状态**：桶列表、可见性、对象数、体积估算
9. **数据库概览**：各表行数柱状图 + 时间维度活跃度指标
10. **用户列表**：聚合用户活跃度、Top 用户、近期活跃用户
11. **Edge Functions 探测**：对配置的函数名发送 OPTIONS 探测
12. **实时数据更新**：Supabase Realtime 订阅 4 张表 + 可配置间隔的轮询
13. **任务日志抽屉**：展示单任务完整日志时间线、进度、元信息
14. **主题切换**：暗色/亮色模式 + 强调色自定义

### 缺少的功能

- 无登录/认证机制
- 无路由系统
- 无分页（任务队列仅显示前 20 条）
- 无任务重试/取消操作
- 无导出功能
- 底部 Dock 3 个标签（趋势/资源/设置）无实际功能
- 无 404 页面
- 无错误边界 / 空状态兜底

---

## P0 级问题

### P0-1: Dashboard 无认证机制，worker_nodes 表使用开发用全开策略

**文件**：`dashboard/src/App.vue` 第 627-653 行、`supabase/migrations/20260320143000_create_worker_nodes.sql` 第 61-67 行

**问题描述**：

Dashboard 整体没有登录/认证层，任何知道 URL 的人即可访问全部功能。同时 `worker_nodes` 表的 RLS 策略为：

```sql
create policy "Allow all for dev on worker_nodes"
on "public"."worker_nodes"
as permissive
for all
to public
using (true)
with check (true);
```

该策略允许 anon 角色对 `worker_nodes` 表执行任意 CRUD 操作。Dashboard 通过 Supabase anon key 发起 UPDATE 写入 `desired_state`、`control_note`、`control_requested_at`（App.vue 第 632-645 行），任何拥有 Supabase URL 和 anon key 的人（前端代码中可见）都可以直接控制 Worker 的暂停、中断和恢复。

**影响**：
- 生产环境中任何人都可以中断正在运行的 Worker 和任务
- 属于策略名称明确标注的 "dev" 策略，不应存在于生产环境
- 即使不通过 Dashboard，也可以通过 Supabase API 直接操作

**建议**：
- 为 Dashboard 增加 Supabase Auth 登录
- 将 worker_nodes 的 RLS 策略改为仅允许特定角色（如 `service_role` 或特定 admin 用户）写入
- Dashboard 的 Worker 控制操作应通过 Edge Function（带权限校验）间接执行

---

### P0-2: `fetchAllRows` 在每次刷新时全表扫描三张表，数据量增长后将导致严重性能退化

**文件**：`dashboard/src/App.vue` 第 834-859 行（定义）、第 1168-1171 行（调用）

**问题描述**：

每次 `refreshDashboard` 调用时，代码通过 `fetchAllRows` 分页拉取 `processing_tasks`、`model_assets`、`tasks` 三张表的全部行（`user_id, created_at`），用于构建 `userSummaries`。`fetchAllRows` 以 1000 条为一页循环拉取，无上限。

当三张表总计超过数万行时：
- 每次刷新将产生数十次 HTTP 请求
- 传输大量不必要的数据
- 极易触发 Supabase 的连接池/速率限制
- 前端内存占用持续增长

**影响**：
- 当前数据量可能尚可承受，但随业务增长将快速退化
- 结合自动刷新（最低 15 秒间隔），可能在短时间内产生大量请求
- 即使数据未变化也会重复拉取

**建议**：
- 在后端创建数据库视图或 RPC 函数，直接在 SQL 层聚合用户统计
- 或使用 Supabase 的 `group` / `aggregate` 功能减少传输量
- 对用户列表数据增加独立的缓存/刷新策略（如仅手动刷新或更长的自动间隔）

---

## P1 级问题

### P1-1: App.vue 单文件超过 2000 行，职责严重耦合

**文件**：`dashboard/src/App.vue`（2015 行）

**问题描述**：

App.vue 承担了全部业务逻辑，包括：类型定义（15 个 interface）、数据获取（11 个并行请求）、状态管理（40+ 个 ref/computed）、图表配置（3 个 ECharts option）、工具函数（15+ 个）、模板渲染（700+ 行 HTML）。已远超合理的单文件规模。

**影响**：
- 极难定位和修改特定功能
- 无法对任何子功能进行独立测试
- TypeScript 编译和 IDE 分析性能下降
- 多人协作时极易产生合并冲突

**建议**：
拆分为以下模块：
- `composables/useTaskData.ts` — 任务数据获取与状态管理
- `composables/useWorkerControl.ts` — Worker 控制逻辑
- `composables/useStorageStats.ts` — 存储桶探测
- `composables/useEdgeChecks.ts` — Edge 函数探测
- `composables/useRealtime.ts` — Realtime 订阅管理
- `components/OverviewCards.vue` — 总览卡片
- `components/TaskTrendChart.vue` — 任务趋势图
- `components/WorkerTable.vue` — Worker 表格
- `components/StorageTable.vue` — 存储表格
- `types/dashboard.ts` — 共享类型定义
- `utils/format.ts` — 格式化工具函数

---

### P1-2: `TaskStatus` 类型联合被 `| string` 破坏

**文件**：`dashboard/src/lib/task-insights.ts` 第 1 行

**问题描述**：

```typescript
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | string
```

`| string` 使 TypeScript 将整个类型退化为 `string`，字面量联合完全失效。任何字符串都能通过类型检查，IDE 的自动补全和类型提示也无法正常工作。

**影响**：
- 无法在编译时捕获无效状态值
- IDE 无法提供状态选项提示
- `statusMap`（App.vue 第 151-156 行）的 key 匹配变得无意义

**建议**：
移除 `| string`，改为严格联合类型。若确实需要支持未知状态，使用可辨识联合或 `& {}` 技巧保持提示能力：

```typescript
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed'
```

---

### P1-3: 缺少 `.env.example` 文件，环境变量未文档化

**文件**：缺失 `dashboard/.env.example`

**问题描述**：

Dashboard 依赖以下环境变量，但没有任何 `.env.example` 文件记录：

- `VITE_SUPABASE_URL` — Supabase 项目 URL
- `VITE_SUPABASE_ANON_KEY` — Supabase anon key
- `VITE_SUPABASE_EDGE_FUNCTIONS` — 逗号分隔的 Edge Function 名称列表
- `VITE_STORAGE_BUCKETS` — 逗号分隔的已知 Storage 桶名称

其中 `VITE_SUPABASE_EDGE_FUNCTIONS` 和 `VITE_STORAGE_BUCKETS` 在代码中完全没有任何提示，新开发者无从得知。

**影响**：
- 新开发者 clone 项目后无法顺利启动 Dashboard
- Edge Functions 探测和 Storage 回退探测功能默认不生效

**建议**：
创建 `dashboard/.env.example`：

```
VITE_SUPABASE_URL=https://xxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJxxx...
VITE_SUPABASE_EDGE_FUNCTIONS=agent-recall,search-models
VITE_STORAGE_BUCKETS=braindance-assets
```

---

### P1-4: Worker 状态不完整，缺少 `starting` 和 `error` 状态处理

**文件**：`dashboard/src/App.vue` 第 571-586 行

**问题描述**：

数据库 schema（`20260320143000_create_worker_nodes.sql` 第 6 行）定义 Worker 状态为：`starting / idle / busy / stopping / offline / error`。但 Dashboard 仅处理了 4 种状态：

```typescript
const getWorkerStatusTag = (status: string) => {
  if (status === 'idle') return 'success'
  if (status === 'busy') return 'warning'
  if (status === 'stopping') return 'danger'
  if (status === 'offline') return 'info'
  return 'info'  // starting 和 error 都走这里
}
const getWorkerStatusLabel = (worker: WorkerNode) => {
  if (isWorkerRowOnline(worker)) {
    if (worker.status === 'busy') return '执行中'
    if (worker.status === 'stopping') return '停止中'
    if (worker.status === 'idle') return '空闲'
    return '在线'  // starting 走这里，语义不准确
  }
  return worker.status === 'offline' ? '已离线' : '失联'  // error 走"失联"
}
```

`starting` 状态的 Worker 显示为"在线"，`error` 状态的 Worker 显示为"失联"，均不能准确反映真实情况。

**建议**：
增加 `starting` 和 `error` 的显式处理：

```typescript
if (status === 'starting') return 'info'    // 启动中
if (status === 'error') return 'danger'     // 异常
```

---

### P1-5: `quality_score` 默认值 0 被计入平均分计算

**文件**：`dashboard/src/App.vue` 第 169-178 行、`supabase/migrations/20260118144558_init_schema.sql` 第 34 行

**问题描述**：

数据库定义 `quality_score integer default 0`，而前端计算平均分时仅过滤 `typeof score === 'number'`：

```typescript
const scores = taskRows.value
  .map((item) => item.quality_score)
  .filter((score): score is number => typeof score === 'number')
```

`quality_score` 为 `number | null` 类型（App.vue 第 26 行），数据库默认值 0 是有效数字，会被计入平均分。大量未评分的任务（quality_score = 0）会将平均分严重拉低。

**建议**：
过滤掉 0 值（假设 0 表示未评分）：

```typescript
.filter((score): score is number => typeof score === 'number' && score > 0)
```

或与后端协商将未评分字段设为 `null`。

---

## P2 级问题

### P2-1: 任务趋势图仅基于最近 500 条任务

**文件**：`dashboard/src/App.vue` 第 1148-1151 行

**问题描述**：

任务查询 `.limit(500)` 将结果限制为最近 500 条。任务趋势图的"全部"时间范围选项只能反映这 500 条数据的分布，当总任务数超过 500 时，历史趋势会出现断崖式下降，产生误导。

**建议**：
- 为趋势图使用独立的时间桶聚合查询（如 RPC 函数）
- 或在 UI 上标注"趋势仅基于最近 500 条任务"

---

### P2-2: Storage 桶扫描有 4000 对象硬上限

**文件**：`dashboard/src/App.vue` 第 1019 行

**问题描述**：

`scanBucket` 函数中 `maxScan = 4000`，超过 4000 个对象的桶会被截断。截断是静默的，不向用户提示数据不完整，导致 Storage 体积和对象数统计不准确。

**建议**：
- 截断时在统计结果中标注"数据可能不完整"
- 考虑使用 Supabase Storage 的 admin API 获取精确统计

---

### P2-3: Realtime 订阅无断线重连和错误处理

**文件**：`dashboard/src/App.vue` 第 1243-1251 行

**问题描述**：

`bindChannel` 仅将订阅状态写入 `channelState`，但未监听 `CHANNEL_ERROR` 事件，也没有在断线时主动重新订阅。用户只能通过左上角状态指示器看到"连接中"，但无法知晓连接失败的原因。

```typescript
const channel = supabase
  .channel(channelName)
  .on('postgres_changes', { event: '*', schema: 'public', table: tableName }, scheduleRefresh)
  .subscribe((status) => {
    channelState.value[tableName] = status
  })
```

**建议**：
- 监听 `system` 事件以捕获断线
- 实现退避重连逻辑
- 连接失败时通过 ElMessage 通知用户

---

### P2-4: `HelloWorld.vue` 为未使用的脚手架默认组件

**文件**：`dashboard/src/components/HelloWorld.vue`（41 行）

**问题描述**：

该文件是 Vite + Vue 脚手架生成的默认模板组件，包含计数器示例和外部链接，未被任何文件引用。

**建议**：删除该文件。

---

### P2-5: Worker 控制的 `desired_state: 'interrupt'` 未在数据库 schema 注释中记录

**文件**：`supabase/migrations/20260320143000_create_worker_nodes.sql` 第 27 行

**问题描述**：

数据库注释写的是 `run / pause`：

```sql
comment on column "public"."worker_nodes"."desired_state" is 'run / pause，dashboard 通过该字段请求 worker 优雅退出';
```

但 Dashboard 和 Worker 代码均支持 `'interrupt'` 值（App.vue 第 629 行，`worker.py` 第 210 行）。数据库 schema 与实际使用的值不一致。

**建议**：更新注释为 `'run / pause / interrupt'`。

---

### P2-6: 底部 Dock 导航项无实际功能

**文件**：`dashboard/src/App.vue` 第 1373-1391 行

**问题描述**：

底部 Dock 有 4 个导航项（概览/趋势/资源/设置），但除"概览"外，其余 3 个无点击事件、无路由、无实际功能，仅作为装饰性 UI 存在。

**建议**：
- 要么实现对应的视图/路由
- 要么移除未实现的导航项，避免误导用户

---

### P2-7: 错误检查遗漏部分并行请求的失败

**文件**：`dashboard/src/App.vue` 第 1173-1180 行

**问题描述**：

`refreshDashboard` 中 11 个并行请求的错误检查仅覆盖前 5 个：

```typescript
if (tasksRes.error || workerRes.error || processingTaskCountRes.error || assetCountRes.error || poseCountRes.error) {
```

`task24hRes`、`asset7dRes`、`ragCountRes`、`taskTableCountRes` 的错误被静默忽略。如果这些请求失败，Dashboard 会显示过时或默认值（如 0），用户无法得知数据不完整。

**建议**：将所有请求的错误统一收集和展示。

---

### P2-8: 任务队列表格无分页

**文件**：`dashboard/src/App.vue` 第 282 行、第 1650 行

**问题描述**：

```typescript
const taskQueue = computed(() => filteredTasks.value.slice(0, 20))
```

任务队列硬编码显示前 20 条，无分页控件。当过滤后的任务超过 20 条时，用户无法查看其余任务。

**建议**：使用 Element Plus 的 `el-pagination` 组件实现分页，或增加"加载更多"按钮。

---

### P2-9: 内联样式未提取到 scoped CSS

**文件**：`dashboard/src/App.vue` 第 1739 行、第 1783 行、第 1814 行

**问题描述**：

模板中有多处内联 `style` 属性：

```html
<div style="margin-top: 10px;">
<div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
<div class="header-meta" style="margin-top: 12px;">
```

与已有 scoped CSS 体系不一致，降低可维护性。

**建议**：提取为 scoped CSS class。

---

## P3 级问题

### P3-1: `vite-env.d.ts` 手动声明 `@iconify/vue` 类型覆盖官方类型

**文件**：`dashboard/src/vite-env.d.ts` 第 3-9 行

**问题描述**：

`@iconify/vue` v5 已自带 TypeScript 类型声明。手动声明的 `DefineComponent<{ icon: string }>` 过于简化，缺少 `width`、`height`、`color`、`inline` 等官方支持的 props 类型。

**建议**：移除手动声明，依赖官方类型。若官方类型有兼容问题，应单独处理。

---

### P3-2: `style.css` 全局样式文件过大（929 行）

**文件**：`dashboard/src/style.css`

**问题描述**：

全局样式包含大量组件级样式（如 `.processing-card`、`.drawer-*`、`.timeline-*`），这些应由各组件自行管理。全局文件职责过重会导致样式冲突风险和难以追踪的级联影响。

**建议**：将组件特定样式迁移到对应 Vue 组件的 `<style scoped>` 中，全局文件仅保留 CSS 变量、reset、通用工具类。

---

### P3-3: `hexToRgba` 函数无输入校验

**文件**：`dashboard/src/App.vue` 第 348-363 行

**问题描述**：

```typescript
const value = Number.parseInt(full, 16)
```

如果 `accentColor` 为无效 hex 值，`parseInt` 返回 `NaN`，产生 `rgba(NaN, NaN, NaN, alpha)` 这样的无效 CSS 值。虽然 `el-color-picker` 通常会约束输入，但缺少防御性校验。

**建议**：增加 `isNaN` 检查，返回 fallback 颜色。

---

### P3-4: 模板中大量重复的"概览卡片"与"告警卡片"数据来源重叠

**文件**：`dashboard/src/App.vue` 第 699-772 行（overviewCards）、第 774-823 行（alertRows）

**问题描述**：

`overviewCards` 和 `alertRows` 两个 computed 有 4 个相同的指标（成功率、失败任务、Worker、Edge），各自独立定义，更新逻辑和文案需要同步维护。一处修改容易遗漏另一处。

**建议**：提取共享的指标定义为单一数据源，两个列表从中派生。

---

### P3-5: 无 `tsconfig.node.json` 的内容审查（未在审计范围，但需注意）

**文件**：`dashboard/tsconfig.node.json`（未读取）

**问题描述**：

`tsconfig.json` 引用了 `tsconfig.node.json`，但该文件未在本次审计范围内读取。需确认其配置与 `vite.config.ts` 的兼容性。

---

### P3-6: `tasks` 表的 `user_id` 为 UUID 类型，但 Dashboard 以 string 方式处理

**文件**：`dashboard/src/App.vue` 第 864 行、`supabase/migrations/20260118144558_init_schema.sql` 第 56 行

**问题描述**：

数据库定义 `tasks.user_id uuid`，但 Dashboard 将其视为 `string` 进行聚合。虽然 Supabase JS SDK 会将 UUID 序列化为字符串返回，且在 JavaScript 中 UUID 字符串的比较和 Map key 使用不受影响，但类型定义上存在不一致。

**建议**：在类型注释中说明 UUID 序列化为字符串的原因，或统一为 `string & { __brand: 'UUID' }` 品牌类型。

---

### P3-7: Edge Function 探测使用 OPTIONS 方法，可能遗漏运行时错误

**文件**：`dashboard/src/App.vue` 第 958-966 行

**问题描述**：

Edge Function 探测使用 `OPTIONS` 方法（CORS 预检），仅测试网关可达性，不测试函数的实际执行能力。函数可能部署成功但运行时出错（如环境变量缺失、依赖问题），OPTIONS 请求仍返回 200。

**建议**：对于关键函数，考虑增加一个轻量级 GET/POST 探测端点（如 health check），或在 UI 中标注"仅测试网关可达性"。

---

## 功能缺失清单

| 缺失功能 | 优先级 | 说明 |
|----------|--------|------|
| 用户认证 | 高 | Dashboard 完全无 auth，任何人可访问和操作 |
| Worker 写入权限收敛 | 高 | 需将 worker_nodes 的写入限制为授权角色 |
| 用户列表服务端聚合 | 高 | 全表扫描三张表构建用户统计，需迁移到 SQL/RPC |
| 任务队列分页 | 中 | 当前仅展示前 20 条，无法浏览更多 |
| 任务重试/取消操作 | 中 | 仅能查看，无法对任务执行运维操作 |
| 任务详情页 | 中 | 仅通过 Drawer 展示日志，缺少独立详情视图 |
| 数据导出 | 低 | 无法导出任务列表、用户列表、Worker 状态等 |
| 路由系统 | 低 | 无 vue-router，无法深链到特定视图 |
| 移动端适配 | 低 | CSS 有响应式断点，但交互体验不适合移动端 |
| 通知/告警推送 | 低 | 异常仅在页面内展示，离线时无法感知 |

---

## 建议新建 Issue 清单

1. **`[Dashboard][P0] 添加用户认证，接入 Supabase Auth`** — 为 Dashboard 增加登录页，使用 Supabase Auth 保护所有页面
2. **`[Dashboard][P0] 收敛 worker_nodes 写入权限`** — 将 RLS 策略从 "Allow all for dev" 改为仅允许授权角色写入，Dashboard 通过 Edge Function 间接控制 Worker
3. **`[Dashboard][P1] 将用户统计聚合迁移到后端 RPC`** — 消除 `fetchAllRows` 全表扫描，在 Supabase 侧创建 `get_user_activity_summary` RPC 函数
4. **`[Dashboard][P1] 拆分 App.vue 为多个 composables 和组件`** — 按功能域拆分 2000+ 行的单文件
5. **`[Dashboard][P1] 修复 TaskStatus 类型定义，移除 `| string``** — 恢复类型联合的约束能力
6. **`[Dashboard][P1] 新增 `.env.example` 文件`** — 文档化所有必需的环境变量
7. **`[Dashboard][P1] 补全 Worker 状态处理（starting / error）`** — 显式处理所有数据库定义的 Worker 状态
8. **`[Dashboard][P1] 修复 quality_score 平均分计算（过滤默认值 0）`** — 避免未评分任务拉低平均分
9. **`[Dashboard][P2] 为任务趋势图增加独立聚合查询`** — 避免仅基于 500 条任务的误导性趋势
10. **`[Dashboard][P2] 增加 Realtime 断线重连与错误通知`** — 提升实时连接的健壮性
11. **`[Dashboard][P2] 为任务队列增加分页`** — 使用 el-pagination 展示全部筛选结果
12. **`[Dashboard][P2] 删除未使用的 HelloWorld.vue`** — 清理脚手架残留
13. **`[Dashboard][P3] 重构全局 CSS，迁移组件样式到 scoped`** — 减少 style.css 体积和级联风险
