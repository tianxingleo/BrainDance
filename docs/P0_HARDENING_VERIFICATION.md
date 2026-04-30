# P0 Hardening 验收包

> 覆盖提交范围：`c552f0c` → `5982b07`（共 5 个提交）
> 生成日期：2026-04-30
> 状态：**待验收，禁止在未完成本文档所有检查项前合并到 main**

---

## 0. 合并顺序建议

按风险从低到高，分批合并：

| 批次 | 内容 | 风险 | 前置条件 |
|------|------|------|----------|
| 1 | Flutter 崩溃修复（`app/` 改动） | 低 | `flutter analyze` 通过 |
| 2 | Worker 跨平台修复（`ai_engine/` 改动） | 低 | Python 语法检查通过 |
| 3 | `claim_next_pending_task` RPC | 中 | 并发测试通过 |
| 4 | `get_user_activity_summary` RPC | 低 | SQL 验证通过 |
| 5 | `rag_docs` RLS（仅限 authenticated SELECT） | 中 | 确认 agent-core 不直接查 rag_docs |
| 6 | `worker_nodes` RLS（删除 dev 全开策略） | 中 | Worker 用 service_role 确认 |
| 7 | `model_assets` RLS | **高** | **必须先完成 backfill，见第 2 节** |
| 8 | Storage 策略删除 | **高** | **确认 Flutter 上传路径用 authenticated** |
| 9 | Dashboard 认证 | 中 | 手动登录/登出验证通过 |

---

## 1. Supabase 迁移验收

### 1.1 `20260430100000` — 删除 Storage 全开策略

**目标**：删除允许 anon 完全控制 storage.objects 的策略。

**执行前验证**（在 Supabase SQL Editor 运行）：

```sql
-- 确认策略存在
SELECT policyname, cmd, roles
FROM pg_policies
WHERE tablename = 'objects'
  AND schemaname = 'storage'
  AND policyname ILIKE '%all%';
```

**执行后验证**：

```sql
-- 确认策略已删除
SELECT count(*) FROM pg_policies
WHERE tablename = 'objects'
  AND schemaname = 'storage'
  AND policyname ILIKE '%all%';
-- 期望：0

-- 确认 authenticated 用户仍有上传权限（检查现有策略）
SELECT policyname, cmd, roles
FROM pg_policies
WHERE tablename = 'objects' AND schemaname = 'storage';
```

**回滚 SQL**：

```sql
-- 如果 Flutter 上传失败，临时恢复（仅用于紧急回滚，不应长期保留）
CREATE POLICY "Enable all storage access for all users"
  ON storage.objects
  FOR ALL
  TO public
  USING (true)
  WITH CHECK (true);
```

**风险提示**：Flutter 上传图片时必须携带有效 JWT（authenticated 角色）。如果 Flutter 端使用 anon key 上传，删除此策略后上传会 403。执行前必须确认 Flutter 上传代码使用的是 `supabase.auth.currentSession` 而非匿名访问。

---

### 1.2 `20260430100001` — model_assets RLS

**⚠️ 高风险：必须先执行 backfill，否则历史数据对所有用户不可见。**

**步骤 1：执行前数据摸底**（手动在 SQL Editor 运行，不进入迁移）：

```sql
-- 统计 NULL 和 default_user 记录数量
SELECT
  CASE
    WHEN user_id IS NULL THEN 'NULL'
    WHEN user_id = 'default_user' THEN 'default_user'
    ELSE 'has_user_id'
  END AS category,
  count(*) AS cnt
FROM public.model_assets
GROUP BY 1;

-- 查看这些记录的样本（确认是否有价值保留）
SELECT id, name, user_id, created_at
FROM public.model_assets
WHERE user_id IS NULL OR user_id = 'default_user'
ORDER BY created_at DESC
LIMIT 20;
```

**步骤 2：backfill（二选一，执行迁移前手动运行）**：

```sql
-- 方案 A：将历史数据归属到某个真实用户（替换 <real-user-uuid>）
UPDATE public.model_assets
SET user_id = '<real-user-uuid>'
WHERE user_id IS NULL OR user_id = 'default_user';

-- 方案 B：如果确认这些记录是测试数据，可以删除
-- DELETE FROM public.model_assets WHERE user_id IS NULL OR user_id = 'default_user';
```

**步骤 3：执行迁移后验证**：

```sql
-- 确认 RLS 已启用
SELECT relname, relrowsecurity
FROM pg_class
WHERE relname = 'model_assets';
-- 期望：relrowsecurity = true

-- 确认策略数量
SELECT count(*) FROM pg_policies WHERE tablename = 'model_assets';
-- 期望：4（SELECT/INSERT/UPDATE/DELETE 各一条）

-- 确认索引已创建
SELECT indexname FROM pg_indexes
WHERE tablename = 'model_assets' AND indexname = 'idx_model_assets_user_id';
-- 期望：1 行

-- 用 anon 角色测试（应返回 0 行）
SET ROLE anon;
SELECT count(*) FROM public.model_assets;
RESET ROLE;
```

**回滚 SQL**：

```sql
ALTER TABLE public.model_assets DISABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view own model assets" ON public.model_assets;
DROP POLICY IF EXISTS "Users can insert own model assets" ON public.model_assets;
DROP POLICY IF EXISTS "Users can update own model assets" ON public.model_assets;
DROP POLICY IF EXISTS "Users can delete own model assets" ON public.model_assets;
```

---

### 1.3 `20260430100002` — rag_docs RLS

**关键结论**：经过代码搜索确认，`supabase/functions/_shared/agent-core/` 中所有 TypeScript 文件均**不直接查询 `rag_docs` 表**。agent-core 只操作 `model_assets`。`rag_docs` 的读取路径需单独确认（可能通过 pgvector 相似度搜索的 RPC 函数访问）。

**执行前验证**：

```sql
-- 确认 rag_docs 当前无 RLS
SELECT relname, relrowsecurity FROM pg_class WHERE relname = 'rag_docs';
-- 期望：relrowsecurity = false（迁移前）

-- 确认哪些 RPC 函数会查询 rag_docs
SELECT routine_name, routine_definition
FROM information_schema.routines
WHERE routine_schema = 'public'
  AND routine_definition ILIKE '%rag_docs%';
```

**执行后验证**：

```sql
-- 确认 RLS 已启用
SELECT relname, relrowsecurity FROM pg_class WHERE relname = 'rag_docs';
-- 期望：relrowsecurity = true

-- 确认 anon 无法直接 SELECT
SET ROLE anon;
SELECT count(*) FROM public.rag_docs;
-- 期望：报错 permission denied 或返回 0（取决于策略）
RESET ROLE;

-- 确认 service_role 仍可访问（Worker 路径）
-- 在 Supabase Dashboard > SQL Editor 以 service_role 执行：
SELECT count(*) FROM public.rag_docs;
-- 期望：正常返回行数
```

**回滚 SQL**：

```sql
ALTER TABLE public.rag_docs DISABLE ROW LEVEL SECURITY;
GRANT SELECT ON public.rag_docs TO anon;
```

---

### 1.4 `20260430100003` — worker_nodes RLS

**关键问题**：Worker 连接 Supabase 使用的是 `service_role` key 还是 `anon` key？

**执行前确认**（检查 Worker 配置）：

```bash
# 检查 Worker 环境变量配置
grep -r "SUPABASE_KEY\|SUPABASE_SERVICE\|supabase_key" ai_engine/3dgs/ --include="*.py" --include="*.env*" -l
```

**执行后验证**：

```sql
-- 确认旧的全开策略已删除
SELECT policyname FROM pg_policies
WHERE tablename = 'worker_nodes'
  AND policyname ILIKE '%dev%';
-- 期望：0 行

-- 确认新策略存在
SELECT policyname, cmd, roles FROM pg_policies WHERE tablename = 'worker_nodes';

-- 用 anon 角色测试 SELECT（应该可以）
SET ROLE anon;
SELECT count(*) FROM public.worker_nodes;
RESET ROLE;

-- 用 anon 角色测试 INSERT（应该失败）
SET ROLE anon;
INSERT INTO public.worker_nodes (id, status) VALUES ('test', 'idle');
-- 期望：permission denied
RESET ROLE;
```

**回滚 SQL**：

```sql
CREATE POLICY "Allow all for dev"
  ON public.worker_nodes
  FOR ALL
  TO public
  USING (true)
  WITH CHECK (true);
```

---

### 1.5 `20260430100004` — claim_next_pending_task RPC

**已修复**：`SET search_path = public` 已在本次验收包中补充到迁移文件。

**并发安全验证**（需要 pgbench 或手动并发测试）：

```sql
-- 方法 1：单会话验证基本功能
-- 先插入测试任务
INSERT INTO public.processing_tasks (id, status, created_at)
VALUES
  ('test-task-1', 'pending', now() - interval '2 minutes'),
  ('test-task-2', 'pending', now() - interval '1 minute'),
  ('test-task-3', 'pending', now());

-- 调用 RPC，应返回最早的任务（test-task-1）
SELECT * FROM public.claim_next_pending_task();
-- 期望：返回 test-task-1，status = 'processing'

-- 再次调用，应返回 test-task-2
SELECT * FROM public.claim_next_pending_task();
-- 期望：返回 test-task-2

-- 无 pending 任务时应返回 NULL
SELECT * FROM public.claim_next_pending_task();
-- 期望：返回 NULL（所有字段为 NULL）

-- 清理测试数据
DELETE FROM public.processing_tasks WHERE id LIKE 'test-task-%';
```

```sql
-- 方法 2：验证 SKIP LOCKED 行为（需要两个并发会话）
-- 会话 A：
BEGIN;
SELECT * FROM public.claim_next_pending_task();
-- 不要 COMMIT，保持事务开启

-- 会话 B（同时执行）：
SELECT * FROM public.claim_next_pending_task();
-- 期望：返回不同的任务（SKIP LOCKED 跳过了会话 A 锁定的行）
-- 或者如果只有一个 pending 任务，返回 NULL

-- 会话 A：
ROLLBACK;
```

```sql
-- 方法 3：验证 search_path 安全性
-- 确认函数定义包含 SET search_path
SELECT prosrc, proconfig
FROM pg_proc
WHERE proname = 'claim_next_pending_task';
-- 期望：proconfig 包含 'search_path=public'
```

**回滚 SQL**：

```sql
DROP FUNCTION IF EXISTS public.claim_next_pending_task();
```

---

### 1.6 `20260430100005` — get_user_activity_summary RPC

**执行后验证**：

```sql
-- 确认函数存在
SELECT routine_name FROM information_schema.routines
WHERE routine_schema = 'public' AND routine_name = 'get_user_activity_summary';

-- 调用函数，确认返回结构正确
SELECT * FROM public.get_user_activity_summary() LIMIT 5;
-- 期望：返回 user_id, total_tasks, tasks_24h, tasks_7d, total_assets, assets_7d, last_active

-- 确认 FULL OUTER JOIN 正确处理只有任务/只有资产的用户
SELECT
  user_id,
  total_tasks,
  total_assets,
  last_active
FROM public.get_user_activity_summary()
WHERE total_tasks = 0 OR total_assets = 0
LIMIT 5;
-- 期望：coalesce 正确填充 0，不出现 NULL
```

**回滚 SQL**：

```sql
DROP FUNCTION IF EXISTS public.get_user_activity_summary();
```

---

## 2. Flutter 崩溃修复验收

### 2.1 验证方法

```bash
cd app
flutter analyze
# 期望：No issues found!（或仅有 info 级别提示）
```

### 2.2 各修复点说明

| 文件 | 修复内容 | 验证方式 |
|------|----------|----------|
| `lib/main.dart` | StatelessWidget → StatefulWidget，正确注册/注销 observer | `flutter analyze` 通过；热重载后无 "setState called after dispose" |
| `lib/services/download_event_bus.dart` | 新增 `disposeDownloadEventBus()` | `flutter analyze` 通过 |
| `lib/pages/recall/recall_model_actions.dart` | `(e as num)` → `(e is num) ? e.toDouble() : 0.0` | 传入非数字 transform 矩阵时不崩溃 |
| `lib/pages/recall/recall_search.dart` | onDone 回调加 `if (!mounted \|\| _agentChatMessage == null) return` | 快速切换页面时不崩溃 |

### 2.3 手动验证路径

1. 启动 App，进入 Recall 页面
2. 发起一次 Agent 查询，在流式响应返回过程中快速切换到其他页面
3. 期望：不崩溃，无 "setState called after dispose" 错误
4. 返回 Recall 页面，重新发起查询，期望正常工作

---

## 3. Worker 修复验收

### 3.1 Python 语法检查

```bash
cd ai_engine/3dgs
python -c "
import ast, pathlib
files = [
    'src/pipelines/da3_2dgs_pipeline.py',
    'src/pipelines/image_to_3d.py',
    'src/modules/knowledge_base.py',
    'src/core/worker.py',
]
for f in files:
    try:
        ast.parse(pathlib.Path(f).read_text(encoding='utf-8'))
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'FAIL: {f}: {e}')
"
```

### 3.2 跨平台导入检查

```bash
# 确认 pty 模块已从 da3_2dgs_pipeline.py 移除
grep -n "import pty\|import select" ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py
# 期望：无输出

# 确认 knowledge_base.py 的 try-except 导入
grep -A3 "try:" ai_engine/3dgs/src/modules/knowledge_base.py
# 期望：看到 try/except ImportError 包裹 openai 和 supabase 导入
```

### 3.3 Worker 启动冒烟测试（无 GPU 环境）

```bash
conda activate Braindance
cd ai_engine/3dgs

# 测试 Worker 模块可以导入（不实际启动）
python -c "
import sys
sys.path.insert(0, 'src')
# 测试 knowledge_base 在缺少依赖时的优雅降级
from modules.knowledge_base import KnowledgeBase
print('KnowledgeBase import: OK')
"

# 测试 image_to_3d NotImplementedError
python -c "
import sys
sys.path.insert(0, 'src')
from pipelines.image_to_3d import MultiImagePipeline
p = MultiImagePipeline.__new__(MultiImagePipeline)
try:
    p.run()
    print('FAIL: should have raised NotImplementedError')
except NotImplementedError as e:
    print(f'OK: NotImplementedError raised: {e}')
"
```

### 3.4 Worker 使用 claim_next_pending_task 的验证

Worker 现在通过 RPC 抢单，不再使用两步 select+update。验证方式：

```bash
# 检查 worker.py 中不再有旧的两步抢单逻辑
grep -n "status.*pending\|WHERE status" ai_engine/3dgs/src/core/worker.py
# 期望：不出现直接 UPDATE processing_tasks SET status = 'processing' WHERE status = 'pending'

# 确认使用 RPC
grep -n "claim_next_pending_task" ai_engine/3dgs/src/core/worker.py
# 期望：至少 1 行
```

---

## 4. Dashboard 认证验收

### 4.1 手动验证步骤

1. **未登录访问**：打开 Dashboard，期望显示登录页面，不显示数据
2. **错误密码**：输入错误密码，期望显示错误提示，不崩溃
3. **正确登录**：输入正确的 admin 邮箱和密码，期望跳转到 Dashboard 主页
4. **会话持久化**：刷新页面，期望保持登录状态（不重新跳转到登录页）
5. **登出**：点击登出按钮，期望返回登录页，数据清空
6. **Token 过期**：等待 token 过期（或手动在 Supabase 吊销 session），期望自动跳转到登录页

### 4.2 关键安全检查

```bash
# 确认 Dashboard 使用 anon key，不使用 service_role key
grep -r "service_role\|SERVICE_ROLE" dashboard/src/ dashboard/.env* 2>/dev/null
# 期望：无输出（Dashboard 绝不能持有 service_role key）

# 确认 supabase.ts 中 persistSession = true
grep "persistSession\|autoRefreshToken" dashboard/src/lib/supabase.ts
# 期望：两者均为 true
```

### 4.3 RPC 调用验证

```bash
# 确认 App.vue 不再有全表扫描
grep -n "fetchAllRows\|\.from('processing_tasks')\|\.from('model_assets')\|\.from('tasks')" dashboard/src/App.vue
# 期望：无输出（已替换为 RPC 调用）

# 确认使用 RPC
grep -n "get_user_activity_summary" dashboard/src/App.vue
# 期望：至少 1 行
```

---

## 5. 已知风险与后续待办

| 编号 | 风险描述 | 严重程度 | 处理建议 |
|------|----------|----------|----------|
| R-01 | model_assets 中存在 NULL/default_user 历史记录，启用 RLS 后不可见 | 高 | 执行迁移前必须完成 backfill（见第 1.2 节） |
| R-02 | Storage 策略删除后，Flutter 上传若使用 anon 会 403 | 高 | 确认 Flutter 上传代码携带 JWT 后再执行 |
| R-03 | worker_nodes RLS 中 Worker 使用 anon key 时只能 SELECT | 中 | 确认 Worker 用 service_role（绕过 RLS）或 authenticated |
| R-04 | rag_docs RLS 限制 authenticated SELECT，若有 RPC 函数以 anon 查询会失败 | 中 | 执行前检查所有查询 rag_docs 的 RPC 函数的 SECURITY 设置 |
| R-05 | Dashboard 登录页无速率限制，存在暴力破解风险 | 低 | 后续在 Supabase Auth 配置中启用 rate limiting |
| R-06 | claim_next_pending_task 在 processing_tasks 无索引时全表扫描 | 低 | 后续为 (status, created_at) 添加复合索引 |

---

## 6. 验收通过标准

以下所有项目必须全部通过，才允许合并对应批次：

- [ ] `flutter analyze` 零错误
- [ ] Python 语法检查全部 OK
- [ ] model_assets backfill 已执行，NULL/default_user 记录已处理
- [ ] claim_next_pending_task 并发测试通过（SKIP LOCKED 行为符合预期）
- [ ] Dashboard 手动登录/登出/刷新验证通过
- [ ] Dashboard 代码中无 service_role key
- [ ] Storage 策略删除前已确认 Flutter 上传使用 authenticated
- [ ] 所有 Supabase 迁移的"执行后验证"SQL 均返回预期结果
