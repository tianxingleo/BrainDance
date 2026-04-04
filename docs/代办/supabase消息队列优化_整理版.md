# Supabase 消息队列优化方案

## 背景

如果 1000 个用户产生 10000 条任务，而所有客户端都在监听同一张大表，并且后端 Worker 还在笨拙地遍历全表，那么系统一定会崩溃。

Supabase (PostgreSQL) 的性能虽然强大，但如果不加优化地"生用"，确实会遇到**性能瓶颈**和**垃圾数据堆积**的问题。

针对"海量数据筛选"和"回收机制"，需要引入 **索引优化 (Indexing)**、**部分订阅 (Partial Subscription)** 和 **TTL (自动过期)** 策略。

---

## 1. 解决遍历效率低：索引优化 (Indexes)

PostgreSQL 如果没有索引，查找 `status='pending'` 确实是全表扫描 (O(N))。如果有 100万条历史数据，Worker 每次取任务都要扫描全表，CPU 会直接炸开。

**解决方案：为查询字段建立"联合索引"。**

Worker 最常做的查询是：`SELECT * FROM processing_tasks WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1`。

请在 Supabase SQL Editor 执行：

```sql
-- 1. 为状态和创建时间建立联合索引 (Worker 找活干飞快)
CREATE INDEX idx_tasks_status_created 
ON processing_tasks (status, created_at);

-- 2. 为用户ID建立索引 (前端查列表飞快)
CREATE INDEX idx_tasks_user_id 
ON processing_tasks (user_id);
```

**效果**：即使表里有 1 亿条数据，Worker 查找下一条任务的时间也是 **毫秒级 (O(log N))**，完全不需要遍历。

---

## 2. 解决前端筛选效率低：服务端过滤 (Filter)

前端 Flutter 监听 Realtime 时，绝对**不能**监听整张表。

**错误做法 (让客户端炸裂)**：

```dart
// 错误：监听整张表，然后自己在手机内存里 filter
supabase.from('processing_tasks').stream(primaryKey: ['id']).listen((data) {
  // 手机会收到全服所有人的任务更新，流量爆炸
});
```

**正确做法 (服务端过滤)**：

Supabase Realtime 支持在**服务器端**过滤数据。通过 RLS (行级安全策略) 或 `filter` 参数，只推送当前用户的数据。

```dart
// 正确：只订阅 user_id = 我 的行
supabase.channel('my_tasks')
  .onPostgresChanges(
    event: PostgresChangeEvent.all,
    schema: 'public',
    table: 'processing_tasks',
    filter: PostgresChangeFilter(
      type: PostgresChangeFilterType.eq,
      column: 'user_id',
      value: myUserId, // 关键：服务器只会给你推这条通道的数据
    ),
    callback: (payload) { ... }
  )
  .subscribe();
```

或者直接使用 `.stream().eq('user_id', uid)`，Supabase 底层会处理好过滤。

---

## 3. 解决没有回收机制：TTL (自动过期)

如果用户上传了 10000 个任务，一个月后这些 `completed` 或 `failed` 的记录就成了垃圾数据，不仅占空间，还拖慢索引。

PostgreSQL 原生支持扩展 **`pg_cron`** (Supabase 支持)，可以写一个定时任务，每天凌晨自动清理旧数据。

**在 SQL Editor 执行：**

```sql
-- 1. 开启 pg_cron 扩展 (如果没开)
create extension if not exists pg_cron;

-- 2. 创建一个定时任务：每天凌晨 3 点清理 7 天前的临时任务记录
-- 注意：只删 processing_tasks 表里的记录，不删 model_assets (资产库)
select cron.schedule(
  'cleanup-old-tasks', -- 任务名
  '0 3 * * *',         -- Cron 表达式 (每天 03:00)
  $$
    DELETE FROM processing_tasks 
    WHERE created_at < now() - interval '7 days'
    AND status IN ('completed', 'failed'); -- 只删已结束的
  $$
);
```

**效果**：任务表永远保持轻量级（只保留最近 7 天的记录），历史包袱会被自动甩掉。这才是真正的企业级"回收机制"。

---

## 4. 终极优化：Worker 抢占锁 (防止重复处理)

如果扩展到 **多台 Worker 机器**，可能会出现两台机器同时抢到同一个 `pending` 任务的情况。

这时候需要用 PostgreSQL 的 `FOR UPDATE SKIP LOCKED` 特性。这是最高效的队列实现方式。

修改 Worker 获取任务的逻辑：

**Worker 代码逻辑修改 (Python):**

```python
# 稍微改一下 Worker 获取任务的 SQL
# 利用 RPC (远程函数) 来原子性地 "获取并锁定" 一个任务

def get_next_task(self):
    # 调用在数据库预定义的函数 (下面会写 SQL)
    response = self.supabase.rpc('pop_next_task').execute()
    if response.data:
        return response.data[0] # 拿到了独占的任务
    return None
```

**对应的 SQL 函数 (需在 Supabase 执行):**

```sql
create or replace function pop_next_task()
returns setof processing_tasks
language plpgsql
as $$
declare
  selected_task processing_tasks;
begin
  -- 核心黑科技：SKIP LOCKED
  -- 1. 找到一个 pending 任务
  -- 2. 瞬间锁住它 (FOR UPDATE)
  -- 3. 如果别的 Worker 锁了，直接跳过 (SKIP LOCKED)，找下一个
  
  SELECT * INTO selected_task
  FROM processing_tasks
  WHERE status = 'pending'
  ORDER BY created_at ASC
  LIMIT 1
  FOR UPDATE SKIP LOCKED; -- 这一行是多机并发的关键

  -- 如果找到了，立刻把它标记为 processing，防止别人抢
  IF found THEN
    UPDATE processing_tasks
    SET status = 'processing', updated_at = now()
    WHERE id = selected_task.id;
    RETURN NEXT selected_task;
  END IF;
end;
$$;
```

---

## 总结

对于 1000 用户 x 10000 数据的场景：

1. **查询慢？** -> **加索引 (`CREATE INDEX`)**，将 O(N) 降为 O(log N)。
2. **推送多？** -> **前端过滤 (`filter='user_id=...'`)**，只推送用户自己的几条数据，流量极小。
3. **数据堆积？** -> **定时清理 (`pg_cron`)**，自动删除 7 天前的废弃记录，保持表瘦身。
4. **并发冲突？** -> **原子锁 (`SKIP LOCKED`)**，支持未来横向扩展多台 GPU 服务器。

这套架构支撑 10 万日活用户都没问题。
