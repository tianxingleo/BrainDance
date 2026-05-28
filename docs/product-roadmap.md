# BrainDance 产品扩展路线图

> 最后更新：2026-05-28
>
> 本文档反映项目当前实际状态。所有条目均基于代码仓库实际核查，非规划口径。

## 1. 当前项目已有能力总结

### Flutter App (app/)
- 登录与会话管理（Supabase Auth）
- 视频上传与任务创建（支持 10+ 种 task_type，工厂模式分发）
- 任务状态实时监控（Supabase Realtime）
- Recall 资产页（模型网格展示、搜索、排序、分享到社区）
- Agent Recall 对话（流式 SSE/NDJSON、多轮续聊、候选确认）
- WebView 3DGS 模型查看器（orbit 旋转模式，含焦距管理和恢复）
- 端侧本地 AI 问答（Qwen3-1.7B GGUF，下载 + 推理，含端侧搜索摘要构建）
- 社区功能（发布/浏览帖子、模型下载、地图展示）
- 前置/后置摄像头自由切换录制
- EEG 信号录制页面
- 深色/浅色主题切换 + 多语言（中/英，含语言设置延迟应用以避免启动崩溃）
- Time Peeling 时间轴 UI 组件（按时间分组展示同场景多版本模型）
- 登出功能（设置页清除缓存处）

### AI Worker (ai_engine/)
- 3DGS 重建 Pipeline：video_3dgs、video_dual_chain、single_image_sam3d、single_image_sharp、da3_feed_forward_3dgs、da3_sugar、da3_2dgs、sparse2dgs、multi_image（共 10 种 task_type 注册于 PipelineFactory）
- Dual Chain 架构：快链（SHARP/SAM3D 自动路由）+ 慢链（video_3dgs）并行
- AI 质检（Qwen-VL）：自动评分、对象标注、描述生成
- Worker 注册与心跳（worker_nodes 表）
- 原子抢单（`claim_next_pending_task` RPC，`FOR UPDATE SKIP LOCKED`）
- 优雅暂停/中断/恢复控制
- 结果上传到 Supabase Storage + model_assets 入库
- 预处理不确定进度条

### Supabase 后端 (supabase/)
- PostgreSQL 17 + pgvector 向量搜索
- 7 个 Edge Function：search-models、agent-recall、spatial-search-agent、time-compare-agent、text-to-image、confirm-text-image、test-timeout
- Agent Core 共享编排（spatialAgent.ts）：5 种模式（spatial_search、asset_metadata、time_compare、creative、memory_graph）
- Storage 管理（braindance-assets、braindance-models）
- Realtime 状态同步
- 全表 RLS 安全加固（21 个 migration，覆盖所有核心表）
- Storage 全开策略已删除，改为按用户路径隔离 + anon 只读
- 原子抢单 RPC + SECURITY DEFINER 辅助函数
- 用户长期记忆表（user_long_term_memory）
- display_name 同步触发器（model_assets → processing_tasks）

### Dashboard (dashboard/)
- 任务总览面板（成功率、队列、Worker 状态、存储状态）
- 任务趋势图 + 状态饼图
- Worker 集群管理（暂停/中断/恢复）
- 任务日志抽屉
- Storage 桶状态 + 数据库概览
- **Supabase Auth 登录/登出**（邮箱 + 密码，未认证时显示登录表单）

### 3DGS Viewer (3dgs_viewer/)
- 辅助脚本：位姿评估、标签、导出、同步
- **my-3dgs-viewer**：桌面端 3DGS 查看器，自由/轨道/电影三种相机模式
- **vr-3dgs-viewer**：独立 VR 查看器，Desktop/Stereo/WebXR 三种预览模式，支持 SteamVR 手柄交互（抓取/缩放/HUD 点选）、本地模型加载、旁观窗口、空间标记导航
- **spark-3dgs-viewer**：Spark 查看器前端，含 Marker AR 纸板锚定模式

### 测试 (tests/)
- Python 测试套件（search API、benchmark、本地问答、Agent Recall batch 等）
- HTTP 冒烟测试脚本（agent_recall_stream、search_models、confirm_text_image）
- 集成测试基础设施（seed SQL、bootstrap/cleanup 脚本、Flutter 集成测试启动脚本）
- 系统测试用例文档

### 文档 (docs/)
- 9 个专题目录覆盖入门、部署、LangChain、本地问答等
- AGENTS.md / CLAUDE.md 工程规范
- 比赛开发文档模板（已填至 v1.2.2）

---

## 2. 还缺哪些核心产品能力

> 已完成项已从此表移除：~~安全加固（RLS）~~ ✅、~~Dashboard 认证~~ ✅、~~Worker 抢单原子化~~ ✅、~~删除 Storage 全开策略~~ ✅

| 缺失能力 | 影响 | 优先级 |
|---------|------|:------:|
| Flutter P0 崩溃修复（5 处） | 演示时切换主题/语言/流式中途切模式可能闪退 | P0 |
| 任务失败原因展示（quality_reason） | 任务失败时用户看不到原因 | P1 |
| 跨模块 task_type 枚举统一 | `da3+sugar` 加号格式仍存在，维护成本高 | P1 |
| 端到端测试覆盖 | 改动容易引入回归 | P1 |
| Dashboard App.vue 拆分 | 单文件 2012 行，维护困难 | P2 |
| time-compare-agent 前端对接 | Edge Function 已存在但 Flutter 未接入 | P2 |
| processing_tasks updated_at 自动更新触发器 | Dashboard 时间排序不准确 | P2 |
| Agent 高级功能前端对接 | Agent 记忆字段已定义但未消费 | P2 |
| 性能监控与告警 | 演示时 Worker 挂了无感知 | P2 |
| 数据导出/备份 | 数据丢失无恢复手段 | P2 |
| 用户引导/Onboarding | 新用户不知道怎么用 | P3 |
| 模型分享优化 | 社区功能基础但体验粗糙 | P3 |

---

## 3. 1 天内可完成的小功能

### 3.1 任务失败原因展示

- **状态**：未完成
- **用户价值**：用户知道为什么任务失败，减少困惑
- **技术实现路径**：Flutter TaskList 页面读取 `quality_reason` 字段并展示，Dashboard 也展示
- **涉及模块**：app/lib/pages/task_list/、dashboard/src/App.vue
- **风险**：极低
- **验收标准**：失败任务卡片展示失败原因文本
- **建议 Issue 标题**：`feat(app): 任务列表展示失败原因 (quality_reason)`

### 3.2 processing_tasks updated_at 自动更新

- **状态**：未完成
- **用户价值**：Dashboard 时间排序准确
- **技术实现路径**：添加 migration 创建 `moddatetime` 触发器
- **涉及模块**：supabase/migrations/
- **风险**：极低
- **验收标准**：Worker 更新任务状态后 updated_at 自动刷新
- **建议 Issue 标题**：`fix(supabase): processing_tasks.updated_at 自动更新触发器`

### 3.3 Flutter P0 崩溃修复（5 处）

- **状态**：未完成
- **用户价值**：消除 5 个已知崩溃点
- **技术实现路径**：参考 Flutter 审计报告 P0-1 到 P0-5 的修复方案
- **涉及模块**：app/lib/main.dart、recall_search.dart、recall_model_actions.dart、download_event_bus.dart
- **风险**：低（防御性修改）
- **验收标准**：flutter analyze 通过，切换主题/语言不崩溃，Agent 流式中途切模式不崩溃
- **建议 Issue 标题**：`fix(app): 修复 Flutter 审计发现的 5 个 P0 级崩溃问题`

### ~~3.4 Worker 任务抢单原子化~~ ✅ 已完成

已通过 migration `20260430100004_create_claim_task_rpc.sql` 实现。Worker 调用 `claim_next_pending_task` RPC，内部使用 `FOR UPDATE SKIP LOCKED` 原子抢单。

### 3.5 Dashboard App.vue 拆分（首阶段）

- **状态**：未完成（当前 2012 行）
- **用户价值**：降低维护成本
- **技术实现路径**：将 App.vue 按功能拆为 5-6 个 composable + 子组件
- **涉及模块**：dashboard/src/
- **风险**：低（纯重构，不改功能）
- **验收标准**：拆分后功能不变，npm run build 通过
- **建议 Issue 标题**：`refactor(dashboard): 拆分 App.vue 为 composable + 子组件`

---

## 4. 3 天内可完成的中等功能

### ~~4.1 Supabase RLS 安全加固~~ ✅ 已完成

已通过多个 migration 文件实现：
- `20260430100001_enable_model_assets_rls.sql` — model_assets 启用 RLS + 用户策略
- `20260430100002_enable_rag_docs_rls.sql` — rag_docs 启用 RLS
- `20260430100003_fix_worker_nodes_rls.sql` — worker_nodes RLS 收紧
- `20260430100000_drop_storage_enable_all_policy.sql` — 删除 Storage 全开策略
- `20260517000000_add_model_assets_write_policies.sql` — model_assets 写入策略
- `20260517083500_add_braindance_models_read_policies.sql` — models 桶只读策略
- memory_poses、community_posts、memory_links 等表均已启用 RLS

**残留风险**：init schema 中仍有 `anon` 的 `TRUNCATE` 权限授予（model_assets、processing_tasks 等），后续需清理。

### ~~4.2 Dashboard 认证 + Worker 控制权限~~ ✅ 已完成

Dashboard 已实现 Supabase Auth 登录/登出：
- 未认证时显示登录表单（邮箱 + 密码）
- 认证后显示完整运维面板
- 使用 `supabase.auth.signInWithPassword` / `signOut`
- `onAuthStateChange` 监听器管理认证状态

### 4.3 跨模块 task_type 枚举统一

- **状态**：未完成
- **用户价值**：新增 task_type 时不需要四处同步
- **技术实现路径**：
  1. 在 Supabase 中创建 ENUM 类型
  2. Worker factory.py 统一从常量引用
  3. Flutter 统一 task_type 映射表
  4. 废弃加号格式 `da3+sugar` → `da3_sugar`
- **涉及模块**：supabase/migrations/、ai_engine/3dgs/src/core/、app/lib/pages/
- **风险**：低
- **验收标准**：所有模块使用统一枚举，拼写错误会被编译期/运行时捕获
- **建议 Issue 标题**：`refactor: 统一 task_type 枚举定义，废弃加号格式`

### 4.4 Agent 高级功能前端对接（time_compare + memory_graph）

- **状态**：部分完成（后端 Edge Function 已存在，Flutter 未接入）
- **当前进展**：`time-compare-agent` Edge Function 已实现完整的自然语言时间对比功能；Flutter 端已有 `TimePeelingList` UI 组件展示时间轴，但尚未调用 time-compare-agent
- **剩余工作**：
  1. Flutter Recall 页添加时间对比入口（选择两个模型）
  2. 调用 time-compare-agent 获取对比结果
  3. 展示变化点列表和可视化
- **涉及模块**：app/lib/pages/recall/、supabase/functions/time-compare-agent/
- **风险**：中（需要前端 UI 设计 + Agent 返回格式对接）
- **验收标准**：选择同一地点的两个版本模型，能看到变化描述
- **建议 Issue 标题**：`feat(app): 对接 time-compare-agent，实现时间对比展示`

---

## 5. 需要较大设计的长期功能

### 5.1 端到端集成测试框架

- **用户价值**：改动有回归保障，演示前快速验证
- **技术实现路径**：
  1. Flutter integration_test 已有基础框架，需要补充更多场景
  2. Supabase Edge Function 测试（Deno test）
  3. Worker Pipeline 冒烟测试（mock Supabase）
  4. CI 集成（GitHub Actions 已有基础）
- **涉及模块**：tests/、app/integration_test/、.github/workflows/
- **风险**：低（但耗时）
- **验收标准**：CI 跑通 Flutter analyze + Worker compile + Edge Function deno check + 集成测试
- **建议 Issue 标题**：`feat(ci): 建立端到端集成测试框架`

### 5.2 实时性能监控与告警

- **用户价值**：演示时知道系统健康状态，出问题能快速定位
- **技术实现路径**：
  1. Worker 上报 GPU 使用率、任务处理耗时
  2. Dashboard 添加实时监控面板
  3. Edge Function 添加请求耗时日志
- **涉及模块**：ai_engine/3dgs/、dashboard/src/、supabase/functions/
- **风险**：中
- **建议 Issue 标题**：`feat(monitor): Worker + Edge Function 性能监控与 Dashboard 展示`

### 5.3 空间锚点可视化（3D 场景内标记和导航）

- **用户价值**：从"能看 3D"升级到"能在 3D 里导航和搜索"
- **技术实现路径**：
  1. Viewer 支持在 3D 场景中叠加 memory_poses 锚点
  2. 用户点击锚点可查看该视角的描述和标签
  3. Agent 搜索结果可自动 fly_to 到对应锚点位置
- **涉及模块**：app/lib/pages/webgl_viewer.dart、3dgs_viewer/
- **风险**：高（需要 WebGL 开发）
- **建议 Issue 标题**：`feat(viewer): 3D 场景内空间锚点标记与导航`

---

## 6. 最能提升比赛/展示效果的功能

> 基于当前已完成的安全加固和认证，重新排序优先级

按影响力排序：

1. **Flutter P0 崩溃修复（5 处）** — 演示时闪退是致命问题
2. **任务全流程可靠性** — 从提交到看到 3D 模型，一镜到底不失败
3. **3D 模型加载稳定性** — Viewer 不崩、不黑屏、流畅旋转
4. **Agent Recall 演示脚本** — 准备一个稳定的演示 query，确保不翻车
5. **任务失败原因展示** — 失败时给出原因而非空白，展示工程完整度
6. **时间对比功能展示** — 项目核心卖点之一，"同一空间不同时间"（后端已就绪，需前端接入）
7. **端侧 AI 演示** — 断网场景下的本地问答，独特亮点
8. **Dashboard 美化** — 评委看到专业运维面板会加分（已有认证）

---

## 7. 最能提升代码质量的功能

按 ROI 排序：

1. **Flutter P0 崩溃修复（5 处）** — 投入小收益大
2. **Dashboard App.vue 拆分** — 2012 行降到可维护状态
3. **跨模块 task_type 枚举统一** — 消除 `da3+sugar` 加号格式隐患
4. ~~**Worker 任务抢单原子化**~~ ✅ 已完成
5. ~~**Supabase RLS 加固**~~ ✅ 已完成
6. **Flutter 错误处理补全** — 10+ 处静默 catch 改为有意义的错误展示
7. **init schema 残留权限清理** — `anon` 的 `TRUNCATE` 授予应收回

---

## 8. 不应该现在做的功能

| 功能 | 为什么不做 |
|------|---------|
| ksplat 格式支持 | 当前 Pipeline 不生成 ksplat，Viewer 已支持但无实际需求 |
| memory_graph 图谱可视化 | 前端消费链路太长，需要先完成 Agent 记忆字段对接 |
| 大规模并发优化 | 当前单机部署，并发不是瓶颈 |
| 多租户/团队协作 | 比赛场景不需要，个人使用足够 |
| 移动端 AR 叠加 | 技术复杂度高，投入产出比低 |
| CI/CD 自动部署 | 当前手动部署足够，比赛期间稳定性优先 |

---

## 9. 建议的 Sprint 计划

> 原计划中 Sprint 1 的安全与稳定性工作已基本完成（RLS、认证、抢单原子化），以下为基于当前实际状态的更新计划。

### Sprint 1 (Day 1-2): 崩溃修复与基础体验

**目标**：消除演示时的"会炸"风险

- [ ] 修复 Flutter 5 个 P0 崩溃
- [ ] processing_tasks updated_at 触发器
- [ ] 任务失败原因展示（Flutter + Dashboard）
- [ ] Flutter 错误处理补全（静默 catch → 有意义提示）

### Sprint 2 (Day 3-5): 功能补全与演示准备

**目标**：让演示流程完整流畅

- [ ] 准备演示脚本（稳定 query + 预置数据）
- [ ] 3D 模型加载稳定性（Viewer 不崩、不黑屏、错误恢复）
- [ ] 端侧 AI 断网演示场景准备
- [ ] Dashboard 美化（图表交互、空状态、加载态）

### Sprint 3 (Day 6-10): 亮点功能与打磨

**目标**：展示差异化能力

- [ ] time-compare-agent 前端对接
- [ ] task_type 枚举统一
- [ ] Dashboard App.vue 拆分
- [ ] 补充关键路径集成测试
- [ ] 文档更新（README 同步、部署手册、比赛文档更新至 Beta 版）
- [ ] init schema 残留权限清理（anon TRUNCATE）

---

## 10. 建议创建的 Issue 清单

> 已完成项已标记 ~~删除线~~，不需要再创建 Issue。

按优先级排序：

| 优先级 | Issue 标题 | 标签 | 状态 |
|:------:|-----------|------|:----:|
| P0 | `fix(app): 修复 Flutter 审计发现的 5 个 P0 级崩溃问题` | bug, flutter | 待做 |
| ~~P0~~ | ~~`security(supabase): 全表 RLS 安全加固`~~ | ~~security, supabase~~ | ✅ |
| ~~P0~~ | ~~`fix(worker): 任务抢单原子化，消除竞态条件`~~ | ~~bug, worker~~ | ✅ |
| ~~P0~~ | ~~`security(supabase): 删除 Storage 全开策略`~~ | ~~security, supabase~~ | ✅ |
| ~~P1~~ | ~~`feat(dashboard): 添加认证层，收紧 Worker 控制权限`~~ | ~~feature, dashboard~~ | ✅ |
| P1 | `feat(app): 任务列表展示失败原因 (quality_reason)` | feature, flutter | 待做 |
| P1 | `refactor(dashboard): 拆分 App.vue 为 composable + 子组件` | refactor, dashboard | 待做 |
| P1 | `refactor: 统一 task_type 枚举定义` | refactor, cross-module | 待做 |
| P2 | `feat(app): 对接 time-compare-agent，实现时间对比展示` | feature, flutter, agent | 待做 |
| P2 | `fix(app): 补全 10+ 处静默 catch，添加有意义的错误展示` | bug, flutter | 待做 |
| P2 | `fix(supabase): processing_tasks.updated_at 自动更新触发器` | fix, supabase | 待做 |
| P2 | `feat(ci): 建立端到端集成测试框架` | feature, ci | 待做 |
| P2 | `refactor(supabase): Edge Function 请求字段命名风格统一` | refactor, supabase | 待做 |
| P3 | `security(supabase): 收回 init schema 中 anon 的 TRUNCATE 权限` | security, supabase | 待做 |
| P3 | `feat(viewer): 3D 场景内空间锚点标记与导航` | feature, viewer | 待做 |
| P3 | `feat(monitor): Worker + Edge Function 性能监控` | feature, monitoring | 待做 |
| P3 | `feat(app): 添加任务重试按钮` | feature, flutter | 待做 |
