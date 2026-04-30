# BrainDance 产品扩展路线图

## 1. 当前项目已有能力总结

### Flutter App (app/)
- 登录与会话管理（Supabase Auth）
- 视频上传与任务创建（支持 8 种 task_type）
- 任务状态实时监控（Supabase Realtime）
- Recall 资产页（模型网格展示、搜索、排序）
- Agent Recall 对话（流式 SSE/NDJSON、多轮续聊、候选确认）
- WebView 3DGS 模型查看器（orbit 旋转模式）
- 端侧本地 AI 问答（Qwen3-1.7B GGUF，下载 + 推理）
- 社区功能（发布/浏览帖子）
- EEG 信号录制页面
- 深色/浅色主题切换 + 多语言（中/英）

### AI Worker (ai_engine/)
- 3DGS 重建 Pipeline：video_3dgs、single_image_sam3d、single_image_sharp、da3 系列、sparse2dgs
- Dual Chain 架构：快链（SHARP/SAM3D）+ 慢链（video_3dgs）并行
- AI 质检（Qwen-VL）：自动评分、对象标注、描述生成
- Worker 注册与心跳（worker_nodes 表）
- 优雅暂停/中断/恢复控制
- 结果上传到 Supabase Storage + model_assets 入库

### Supabase 后端 (supabase/)
- PostgreSQL + pgvector 向量搜索
- 7 个 Edge Function：search-models、agent-recall、spatial-search-agent、time-compare-agent、text-to-image、confirm-text-image、test-timeout
- Agent Core 共享编排（spatialAgent.ts）：5 种模式（spatial_search、asset_metadata、time_compare、creative、memory_graph）
- Storage 管理（braindance-assets、braindance-models）
- Realtime 状态同步

### Dashboard (dashboard/)
- 任务总览面板（成功率、队列、Worker 状态、存储状态）
- 任务趋势图 + 状态饼图
- Worker 集群管理（暂停/中断/恢复）
- 任务日志抽屉
- Storage 桶状态 + 数据库概览

### 3DGS Viewer (3dgs_viewer/)
- 辅助脚本：位姿评估、标签、导出、同步
- Orbit 相机旋转模式（Flutter WebView 集成）

### 文档 (docs/)
- 9 个专题目录覆盖入门、部署、LangChain、本地问答等
- AGENTS.md / CLAUDE.md 工程规范

---

## 2. 还缺哪些核心产品能力

| 缺失能力 | 影响 | 优先级 |
|---------|------|:------:|
| 安全加固（RLS、认证） | 生产环境不可部署 | P0 |
| 任务失败重试 | 用户体验差，演示风险高 | P1 |
| Dashboard 认证 | 任何人可控制 Worker | P0 |
| 跨模块契约统一 | 维护成本高，容易出隐蔽 bug | P1 |
| 端到端测试覆盖 | 改动容易引入回归 | P1 |
| Agent 高级功能前端对接 | Agent 记忆字段已定义但未消费 | P2 |
| 性能监控与告警 | 演示时 Worker 挂了无感知 | P2 |
| 数据导出/备份 | 数据丢失无恢复手段 | P2 |
| 用户引导/Onboarding | 新用户不知道怎么用 | P2 |
| 模型分享优化 | 社区功能基础但体验粗糙 | P3 |

---

## 3. 1 天内可完成的小功能

### 3.1 任务失败原因展示

- **用户价值**：用户知道为什么任务失败，减少困惑
- **技术实现路径**：Flutter TaskList 页面读取 `quality_reason` 字段并展示，Dashboard 也展示
- **涉及模块**：app/lib/pages/task_list/、dashboard/src/App.vue
- **风险**：极低
- **验收标准**：失败任务卡片展示失败原因文本
- **建议 Issue 标题**：`feat(app): 任务列表展示失败原因 (quality_reason)`

### 3.2 processing_tasks updated_at 自动更新

- **用户价值**：Dashboard 时间排序准确
- **技术实现路径**：添加 migration 创建 `moddatetime` 触发器
- **涉及模块**：supabase/migrations/
- **风险**：极低
- **验收标准**：Worker 更新任务状态后 updated_at 自动刷新
- **建议 Issue 标题**：`fix(supabase): processing_tasks.updated_at 自动更新触发器`

### 3.3 Flutter P0 崩溃修复（5 处）

- **用户价值**：消除 5 个已知崩溃点
- **技术实现路径**：参考 Flutter 审计报告 P0-1 到 P0-5 的修复方案
- **涉及模块**：app/lib/main.dart、recall_search.dart、recall_model_actions.dart、download_event_bus.dart
- **风险**：低（防御性修改）
- **验收标准**：flutter analyze 通过，切换主题/语言不崩溃，Agent 流式中途切模式不崩溃
- **建议 Issue 标题**：`fix(app): 修复 Flutter 审计发现的 5 个 P0 级崩溃问题`

### 3.4 Worker 任务抢单原子化

- **用户价值**：多 Worker 场景不会重复处理同一任务
- **技术实现路径**：创建 Supabase RPC 函数 `claim_task()`，使用 `UPDATE ... WHERE status='pending' RETURNING *` 实现原子抢单
- **涉及模块**：supabase/migrations/、ai_engine/3dgs/src/core/worker.py
- **风险**：中（需要同时改 SQL 和 Python）
- **验收标准**：两个 Worker 同时轮询不会抢到同一个任务
- **建议 Issue 标题**：`fix(worker): 任务抢单原子化，消除竞态条件`

### 3.5 Dashboard App.vue 拆分（首阶段）

- **用户价值**：降低维护成本
- **技术实现路径**：将 App.vue（2015 行）按功能拆为 5-6 个 composable + 子组件
- **涉及模块**：dashboard/src/
- **风险**：低（纯重构，不改功能）
- **验收标准**：拆分后功能不变，npm run build 通过
- **建议 Issue 标题**：`refactor(dashboard): 拆分 App.vue 为 composable + 子组件`

---

## 4. 3 天内可完成的中等功能

### 4.1 Supabase RLS 安全加固

- **用户价值**：生产环境可安全部署，满足比赛评审的安全关注点
- **技术实现路径**：按照 Supabase 安全审计报告，逐个修复 P0/P1 问题
  1. model_assets / rag_docs 启用 RLS
  2. 删除 Storage 全开策略
  3. worker_nodes / processing_tasks / community_posts 收紧策略
  4. 新表（memory_links 等）启用 RLS
- **涉及模块**：supabase/migrations/、supabase/functions/
- **风险**：中（收紧 RLS 可能影响现有功能，需要逐表验证）
- **验收标准**：anon 角色不能读取其他用户数据，Worker（service_role）功能不受影响
- **建议 Issue 标题**：`security(supabase): 全表 RLS 安全加固`

### 4.2 Dashboard 认证 + Worker 控制权限

- **用户价值**：防止未授权用户操控 Worker
- **技术实现路径**：
  1. Dashboard 添加 Supabase Auth 登录页
  2. Worker 控制操作改为通过带认证的 Edge Function 间接执行
  3. worker_nodes 写入权限限制为 service_role
- **涉及模块**：dashboard/src/、supabase/functions/、supabase/migrations/
- **风险**：中
- **验收标准**：未登录用户无法访问 Dashboard，已登录用户只能查看不能控制 Worker（除非是 admin）
- **建议 Issue 标题**：`feat(dashboard): 添加认证层，收紧 Worker 控制权限`

### 4.3 跨模块 task_type 枚举统一

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

- **用户价值**：演示时展示"同一空间不同时间对比"和"记忆图谱"两个亮点功能
- **技术实现路径**：
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

按影响力排序：

1. **安全加固（RLS + Dashboard 认证）** — 评委关注安全时不会暴露
2. **时间对比功能展示** — 项目核心卖点之一，"同一空间不同时间"
3. **Agent Recall 演示脚本** — 准备一个稳定的演示 query，确保不翻车
4. **3D 模型加载稳定性** — Viewer 不崩、不黑屏、流畅旋转
5. **任务全流程可靠性** — 从提交到看到 3D 模型，一镜到底不失败
6. **Dashboard 美化** — 评委看到专业运维面板会加分
7. **端侧 AI 演示** — 断网场景下的本地问答，独特亮点

---

## 7. 最能提升代码质量的功能

按 ROI 排序：

1. **Flutter P0 崩溃修复（5 处）** — 投入小收益大
2. **Dashboard App.vue 拆分** — 2015 行降到可维护状态
3. **跨模块 task_type 枚举统一** — 消除未来维护隐患
4. **Worker 任务抢单原子化** — 多 Worker 场景必改
5. **Supabase RLS 加固** — 技术债转技术资产
6. **Flutter 错误处理补全** — 10+ 处静默 catch 改为有意义的错误展示

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

### Sprint 1 (Day 1-2): 安全与稳定性

**目标**：消除演示时的"会炸"风险

- [ ] 修复 Flutter 5 个 P0 崩溃
- [ ] Supabase RLS 加固（至少 P0 级的 4 个问题）
- [ ] Worker 任务抢单原子化
- [ ] 删除 Storage 全开策略
- [ ] processing_tasks updated_at 触发器

### Sprint 2 (Day 3-5): 功能补全与体验

**目标**：让演示流程完整流畅

- [ ] 任务失败原因展示（Flutter + Dashboard）
- [ ] Dashboard 认证 + Worker 控制权限
- [ ] task_type 枚举统一
- [ ] Dashboard App.vue 拆分
- [ ] Flutter 错误处理补全（静默 catch → 有意义提示）
- [ ] 准备演示脚本（稳定 query + 预置数据）

### Sprint 3 (Day 6-10): 亮点功能与打磨

**目标**：展示差异化能力

- [ ] time-compare-agent 前端对接
- [ ] Dashboard 美化（图表交互、空状态、加载态）
- [ ] Viewer 稳定性提升（加载进度、错误恢复）
- [ ] 端侧 AI 断网演示场景准备
- [ ] 补充关键路径集成测试
- [ ] 文档更新（README 同步、部署手册）

---

## 10. 建议创建的 Issue 清单

按优先级排序：

| 优先级 | Issue 标题 | 标签 |
|:------:|-----------|------|
| P0 | `fix(app): 修复 Flutter 审计发现的 5 个 P0 级崩溃问题` | bug, flutter |
| P0 | `security(supabase): 全表 RLS 安全加固` | security, supabase |
| P0 | `fix(worker): 任务抢单原子化，消除竞态条件` | bug, worker |
| P0 | `security(supabase): 删除 Storage 全开策略` | security, supabase |
| P1 | `feat(dashboard): 添加认证层，收紧 Worker 控制权限` | feature, dashboard |
| P1 | `feat(app): 任务列表展示失败原因 (quality_reason)` | feature, flutter |
| P1 | `refactor(dashboard): 拆分 App.vue 为 composable + 子组件` | refactor, dashboard |
| P1 | `refactor: 统一 task_type 枚举定义` | refactor, cross-module |
| P2 | `feat(app): 对接 time-compare-agent，实现时间对比展示` | feature, flutter, agent |
| P2 | `fix(app): 补全 10+ 处静默 catch，添加有意义的错误展示` | bug, flutter |
| P2 | `feat(ci): 建立端到端集成测试框架` | feature, ci |
| P2 | `refactor(supabase): Edge Function 请求字段命名风格统一` | refactor, supabase |
| P3 | `feat(viewer): 3D 场景内空间锚点标记与导航` | feature, viewer |
| P3 | `feat(monitor): Worker + Edge Function 性能监控` | feature, monitoring |
| P3 | `feat(app): 添加任务重试按钮` | feature, flutter |
