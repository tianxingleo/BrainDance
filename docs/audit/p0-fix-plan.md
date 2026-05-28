# P0 修复计划：去重、分组与实施矩阵

> 基于 6 份审计报告（Flutter / AI Worker / Dashboard / Supabase / 跨模块契约 / 建议汇总）
> 生成时间：2026-04-30
> 原始 P0 数量：16 个 → 去重后：14 个

---

## 一、去重合并

| 原始编号 | 来源报告 | 问题 | 去重结果 |
|---------|---------|------|---------|
| Supabase P0-3 | supabase-security-audit | Storage "Enable all" 全开策略 | ✅ 保留 |
| 契约 #8 | contract-audit | Storage 全开策略覆盖 user folder 策略 | ❌ 与上条重复，合并 |
| Dashboard P0-1 | dashboard-audit | 无认证 + worker_nodes 全开 RLS | 拆为两个独立修复：Dashboard 认证 + worker_nodes RLS |
| Supabase P0-4 | supabase-security-audit | worker_nodes "Allow all for dev" | 与 Dashboard P0-1 中的 RLS 部分合并 |

**去重结论**：16 → 14 个独立修复项

---

## 二、按修复域分组

### A 组：Supabase / RLS / Storage 安全（4 个）

| 序号 | 问题标题 | 来源报告 | 涉及文件 |
|-----|---------|---------|---------|
| A1 | Storage "Enable all" 全开策略 | supabase P0-3 + 契约 #8 | `supabase/migrations/20260118144558_init_schema.sql:390-396` |
| A2 | model_assets 未启用 RLS | supabase P0-1 | `supabase/migrations/20260118144558_init_schema.sql:8-21` |
| A3 | rag_docs 未启用 RLS | supabase P0-2 | `supabase/migrations/20260118144558_init_schema.sql:46-51` |
| A4 | worker_nodes "Allow all for dev" | supabase P0-4 + dashboard P0-1 | `supabase/migrations/20260320143000_create_worker_nodes.sql:61-67` |

### B 组：Flutter 崩溃与资源泄漏（5 个）

| 序号 | 问题标题 | 来源报告 | 涉及文件 |
|-----|---------|---------|---------|
| B1 | WidgetsBindingObserver 重复注册 | flutter P0 #1 | `app/lib/main.dart:95-101` |
| B2 | downloadEventBus StreamController 泄漏 | flutter P0 #2 | `app/lib/services/download_event_bus.dart:27` |
| B3 | MaterialApp.builder child! 强制解包 | flutter P0 #3 | `app/lib/main.dart:179` |
| B4 | transformMatrix 类型转换不安全 | flutter P0 #4 | `app/lib/pages/recall/recall_model_actions.dart:48` |
| B5 | Agent 流式 onDone 缺少 null 保护 | flutter P0 #5 | `app/lib/pages/recall/recall_search.dart:267-279` |

### C 组：AI Worker 任务调度（4 个）

| 序号 | 问题标题 | 来源报告 | 涉及文件 |
|-----|---------|---------|---------|
| C1 | 任务抢单竞态条件 | worker P0-01 | `ai_engine/3dgs/src/core/worker.py:452-459,780-783` |
| C2 | mask_guided.py 空壳必崩 | worker P0-02 | `ai_engine/3dgs/src/pipelines/mask_guided.py`, `ai_engine/3dgs/src/core/factory.py` |
| C3 | da3_2dgs_pipeline Unix-only pty | worker P0-03 | `ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py:2` |
| C4 | knowledge_base.py 硬性 import | worker P0-04 | `ai_engine/3dgs/src/modules/knowledge_base.py:7-8` |

### D 组：Dashboard 认证与性能（2 个）

| 序号 | 问题标题 | 来源报告 | 涉及文件 |
|-----|---------|---------|---------|
| D1 | Dashboard 无认证 | dashboard P0-1 | `dashboard/src/App.vue:627-653`, `dashboard/src/lib/supabase.ts` |
| D2 | 全表扫描性能退化 | dashboard P0-2 | `dashboard/src/App.vue:834-859,1168-1171` |

### E 组：跨模块契约（已合并到 A1）

Storage 全开策略问题已合并到 A1，不再独立存在。

---

## 三、依赖关系与修复顺序

```
A1 (Storage 全开) ← 无依赖，最先修
A2 (model_assets RLS) ← 无依赖，可与 A1 并行
A3 (rag_docs RLS) ← 无依赖，可与 A1 并行
A4 (worker_nodes RLS) ← 无依赖，但与 D1 有关联

B1-B5 (Flutter 崩溃) ← 无依赖，可与 A 组并行

C2 (mask_guided) ← 无依赖，简单
C3 (pty 跨平台) ← 无依赖，简单
C4 (import 保护) ← 无依赖，简单
C1 (任务抢单原子化) ← 风险高，需单独审

D1 (Dashboard 认证) ← 依赖 A4（RLS 收紧后 Dashboard 需适配）
D2 (全表扫描) ← 无依赖，可与 D1 并行
```

### 推荐并行组

| 轮次 | 并行修复 | 原因 |
|-----|---------|------|
| 第 1 轮 | A1 + A2 + A3 | Supabase 安全，互相独立，全是 migration |
| 第 2 轮 | B1 + B2 + B3+B4+B5 | Flutter 修复，全是防御性修改 |
| 第 3 轮 | A4 + C2 + C3 + C4 | worker_nodes RLS + Worker 简单修复 |
| 第 4 轮 | C1 | 任务抢单原子化，高风险，单独审 |
| 第 5 轮 | D1 + D2 | Dashboard，依赖 A4 完成 |

---

## 四、每个 P0 的详细修复矩阵

### A1: Storage "Enable all" 全开策略

| 项 | 内容 |
|----|------|
| **问题标题** | 删除 Storage "Enable all storage access" 全开策略 |
| **来源报告** | supabase-security-audit P0-3 + contract-audit #8 |
| **涉及文件** | `supabase/migrations/20260118144558_init_schema.sql:390-396` |
| **最小修复范围** | 新增 1 个 migration，DROP POLICY "Enable all storage access" on storage.objects |
| **验收方式** | `SELECT * FROM storage.policies WHERE name = 'Enable all storage access'` 应返回空 |
| **Claude Code 自动修** | ✅ 适合（纯 SQL migration） |
| **建议 worktree** | `supabase-storage-policy-fix` |
| **建议 PR 标题** | `fix(supabase): 删除 Storage 全开策略，恢复 user folder 权限边界` |
| **回滚方式** | 重新执行 `CREATE POLICY "Enable all storage access" ON storage.objects FOR ALL TO public USING (true) WITH CHECK (true)` |

### A2: model_assets 未启用 RLS

| 项 | 内容 |
|----|------|
| **问题标题** | model_assets 表启用 RLS 并添加 user_id 隔离策略 |
| **来源报告** | supabase-security-audit P0-1 |
| **涉及文件** | `supabase/migrations/20260118144558_init_schema.sql:8-21` |
| **最小修复范围** | 新增 1 个 migration：启用 RLS + 4 条 policy（SELECT/INSERT/UPDATE/DELETE 基于 user_id） |
| **验收方式** | 匿名用户 SELECT 返回空；用户 A 无法访问用户 B 的数据；service_role 可全量访问 |
| **Claude Code 自动修** | ✅ 适合（纯 SQL migration，但需注意 user_id 为 text 类型） |
| **建议 worktree** | `supabase-model-assets-rls` |
| **建议 PR 标题** | `fix(supabase): 启用 model_assets RLS，添加 user_id 行级隔离` |
| **回滚方式** | `ALTER TABLE model_assets DISABLE ROW LEVEL SECURITY; DROP POLICY ...` |
| **注意事项** | user_id 为 text 而非 uuid，需用 `auth.uid()::text` 转换 |

### A3: rag_docs 未启用 RLS

| 项 | 内容 |
|----|------|
| **问题标题** | rag_docs 表启用 RLS，限制为通过 Edge Function 查询 |
| **来源报告** | supabase-security-audit P0-2 |
| **涉及文件** | `supabase/migrations/20260118144558_init_schema.sql:46-51` |
| **最小修复范围** | 新增 1 个 migration：启用 RLS + anon 只读策略（用于 search Edge Function 的 service_role 访问） |
| **验收方式** | 匿名用户无法直接 SELECT rag_docs；service_role 可正常访问 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 worktree** | `supabase-rag-docs-rls` |
| **建议 PR 标题** | `fix(supabase): 启用 rag_docs RLS，限制直接访问` |
| **回滚方式** | `ALTER TABLE rag_docs DISABLE ROW LEVEL SECURITY; DROP POLICY ...` |

### A4: worker_nodes "Allow all for dev" 策略

| 项 | 内容 |
|----|------|
| **问题标题** | 收紧 worker_nodes RLS，从全开改为 service_role only |
| **来源报告** | supabase-security-audit P0-4 + dashboard-audit P0-1 |
| **涉及文件** | `supabase/migrations/20260320143000_create_worker_nodes.sql:61-67` |
| **最小修复范围** | 新增 1 个 migration：替换全开策略为 authenticated SELECT + service_role ALL |
| **验收方式** | 匿名用户无法 UPDATE desired_state；认证用户可 SELECT；service_role 可 ALL |
| **Claude Code 自动修** | ✅ 适合（但 Dashboard 需适配，见 D1） |
| **建议 worktree** | `supabase-worker-nodes-rls` |
| **建议 PR 标题** | `fix(supabase): 收紧 worker_nodes RLS，限制 anon 写入` |
| **回滚方式** | 恢复原全开策略 |

### B1: WidgetsBindingObserver 重复注册

| 项 | 内容 |
|----|------|
| **问题标题** | MyApp 改为 StatefulWidget，在 initState 注册 observer，在 dispose 移除 |
| **来源报告** | flutter-audit P0 #1 |
| **涉及文件** | `app/lib/main.dart:95-101` |
| **最小修复范围** | MyApp 从 StatelessWidget 改为 StatefulWidget，移除 build() 中的 addObserver 调用 |
| **验收方式** | `flutter analyze` 无错误；多次切换深色模式后 observer 列表不增长 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 worktree** | `flutter-observer-leak` |
| **建议 PR 标题** | `fix(flutter): 修复 WidgetsBindingObserver 重复注册和未移除的内存泄漏` |
| **需手动验证** | 是（需运行 app，多次切换深色/浅色模式） |

### B2: downloadEventBus StreamController 泄漏

| 项 | 内容 |
|----|------|
| **问题标题** | 为 downloadEventBus 添加关闭方法或使用 provider 管理生命周期 |
| **来源报告** | flutter-audit P0 #2 |
| **涉及文件** | `app/lib/services/download_event_bus.dart:27` |
| **最小修复范围** | 添加 `disposeEventBus()` 方法，在 app 退出时调用 |
| **验收方式** | `flutter analyze` 无错误；长时间运行反复进出 Recall 页面，内存不持续增长 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 worktree** | `flutter-streamcontroller-dispose` |
| **建议 PR 标题** | `fix(flutter): 添加 downloadEventBus 关闭方法，修复内存泄漏` |
| **需手动验证** | 是（需长时间运行 app） |

### B3: MaterialApp.builder child! 强制解包

| 项 | 内容 |
|----|------|
| **问题标题** | child! 改为 child ?? const SizedBox.shrink() |
| **来源报告** | flutter-audit P0 #3 |
| **涉及文件** | `app/lib/main.dart:179` |
| **最小修复范围** | 1 行代码修改 |
| **验收方式** | `flutter analyze` 无错误 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 PR 标题** | `fix(flutter): MaterialApp.builder child! 改为防御性处理` |

### B4: transformMatrix 类型转换不安全

| 项 | 内容 |
|----|------|
| **问题标题** | transformMatrix 强制类型转换改为安全转换 |
| **来源报告** | flutter-audit P0 #4 |
| **涉及文件** | `app/lib/pages/recall/recall_model_actions.dart:48` |
| **最小修复范围** | `e as num` 改为 `(e as num?)?.toDouble() ?? 0.0` |
| **验收方式** | `flutter analyze` 无错误 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 PR 标题** | `fix(flutter): transformMatrix 类型转换改为安全转换` |

### B5: Agent 流式 onDone 缺少 null 保护

| 项 | 内容 |
|----|------|
| **问题标题** | onDone 回调中增加 null 保护 |
| **来源报告** | flutter-audit P0 #5 |
| **涉及文件** | `app/lib/pages/recall/recall_search.dart:267-279` |
| **最小修复范围** | 在 onDone 中增加 `if (_agentChatMessage == null) return;` |
| **验收方式** | `flutter analyze` 无错误；快速切换搜索模式时不崩溃 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 PR 标题** | `fix(flutter): Agent 流式 onDone 回调增加 null 保护` |
| **需手动验证** | 是（需运行 app，快速切换搜索模式） |

### C1: 任务抢单竞态条件

| 项 | 内容 |
|----|------|
| **问题标题** | 使用 Supabase RPC 实现原子性任务抢单 |
| **来源报告** | ai-worker-audit P0-01 |
| **涉及文件** | `ai_engine/3dgs/src/core/worker.py:452-459,780-783` + 新增 Supabase migration |
| **最小修复范围** | 新增 RPC 函数 `claim_pending_task()`；修改 worker.py 调用 RPC 替代 select+update |
| **验收方式** | 两个 worker 同时启动，只有一个 claim 同一任务 |
| **Claude Code 自动修** | ⚠️ 部分适合（SQL 可自动，Python 调用修改需审） |
| **建议 worktree** | `worker-atomic-claim` |
| **建议 PR 标题** | `fix(worker): 实现原子性任务抢单，防止多 Worker 竞态` |
| **回滚方式** | 恢复原 select+update 逻辑 |
| **风险** | 高——涉及并发和任务状态，需仔细测试 |

### C2: mask_guided.py 空壳必崩

| 项 | 内容 |
|----|------|
| **问题标题** | 从 PipelineFactory 移除空壳 MultiImagePipeline 或实现完整逻辑 |
| **来源报告** | ai-worker-audit P0-02 |
| **涉及文件** | `ai_engine/3dgs/src/pipelines/mask_guided.py`, `ai_engine/3dgs/src/core/factory.py:3` |
| **最小修复范围** | 从 factory.py 移除 "multi_image" 注册，或改为抛 NotImplementedError |
| **验收方式** | `python -c "from src.core.factory import PipelineFactory; print(PipelineFactory.create('multi_image'))"` 应报错而非 NameError |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 PR 标题** | `fix(worker): 移除空壳 MultiImagePipeline 注册，防止 NameError` |

### C3: da3_2dgs_pipeline Unix-only pty

| 项 | 内容 |
|----|------|
| **问题标题** | 替换 pty 为 subprocess.Popen + 跨平台实时输出方案 |
| **来源报告** | ai-worker-audit P0-03 |
| **涉及文件** | `ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py:2` |
| **最小修复范围** | 将 `import pty` 改为条件导入，在 `_run_cmd` 内部用 subprocess 替代 pty |
| **验收方式** | Windows 上能 import 该模块不报错 |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 PR 标题** | `fix(worker): da3_2dgs_pipeline 替换 Unix-only pty 为跨平台方案` |

### C4: knowledge_base.py 硬性 import

| 项 | 内容 |
|----|------|
| **问题标题** | 添加 try-catch 保护 openai/supabase 导入 |
| **来源报告** | ai-worker-audit P0-04 |
| **涉及文件** | `ai_engine/3dgs/src/modules/knowledge_base.py:7-8` |
| **最小修复范围** | 2 行 import 改为 try-catch 模式 |
| **验收方式** | 缺少 openai 包时整个 3dgs 包仍可正常 import |
| **Claude Code 自动修** | ✅ 适合 |
| **建议 PR 标题** | `fix(worker): knowledge_base.py 添加 try-catch 保护导入` |

### D1: Dashboard 无认证

| 项 | 内容 |
|----|------|
| **问题标题** | 为 Dashboard 增加最小认证保护 |
| **来源报告** | dashboard-audit P0-1 |
| **涉及文件** | `dashboard/src/App.vue:627-653`, `dashboard/src/lib/supabase.ts` |
| **最小修复范围** | 添加 Supabase Auth 登录页 + 路由守卫 |
| **验收方式** | 未登录时无法访问 Dashboard；登录后功能正常 |
| **Claude Code 自动修** | ⚠️ 部分适合（需新增组件和路由，但方案明确） |
| **建议 worktree** | `dashboard-auth-query-hardening` |
| **建议 PR 标题** | `feat(dashboard): 添加 Supabase Auth 登录保护` |
| **回滚方式** | 移除认证组件和路由守卫 |
| **风险** | 中——可能影响 Dashboard 现有功能，需适配 RLS 变更 |

### D2: 全表扫描性能退化

| 项 | 内容 |
|----|------|
| **问题标题** | 消除 fetchAllRows 全表扫描，改为后端聚合 |
| **来源报告** | dashboard-audit P0-2 |
| **涉及文件** | `dashboard/src/App.vue:834-859,1168-1171` |
| **最小修复范围** | 添加 limit + 分页，或创建 Supabase RPC 做服务端聚合 |
| **验收方式** | Dashboard 加载时间不随数据量线性增长 |
| **Claude Code 自动修** | ⚠️ 部分适合（前端修改容易，RPC 需要设计） |
| **建议 PR 标题** | `perf(dashboard): 消除全表扫描，添加分页和限制` |

---

## 五、合并顺序建议

```
 1. docs/audit/p0-fix-plan.md（本文档）
 2. A1: Storage 全开策略删除
 3. A2: model_assets RLS
 4. A3: rag_docs RLS
 5. A4: worker_nodes RLS 收紧
 6. B1+B3: Flutter main.dart 修复（observer + child!）
 7. B2: Flutter downloadEventBus 修复
 8. B4+B5: Flutter 类型转换 + null 保护
 9. C2+C3+C4: Worker 简单修复
10. C1: Worker 任务抢单原子化（高风险，单独审）
11. D1: Dashboard 认证
12. D2: Dashboard 全表扫描优化
```

---

## 六、每个修复的验收输出模板

每个修复完成后必须输出：

```
## 修改摘要
- ...

## 修改文件
- ...

## 验证命令
- ...

## 手动验证
- ...

## 风险点
- ...

## 回滚方式
- ...

## 是否触碰危险区域
- Supabase migration: 是/否
- RLS/Storage policy: 是/否
- Auth/secret: 是/否
- Worker task claim: 是/否
```
