-- P0 修复: 创建原子性任务抢单 RPC 函数
-- 使用 FOR UPDATE SKIP LOCKED 实现并发安全的任务分发
-- 多个 Worker 同时调用时，每个任务只会被一个 Worker 领取
-- Worker 使用 service_role 调用，绕过 RLS

CREATE OR REPLACE FUNCTION public.claim_next_pending_task()
RETURNS processing_tasks
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    claimed processing_tasks%ROWTYPE;
BEGIN
    UPDATE processing_tasks
    SET status = 'processing',
        updated_at = now(),
        logs = '[]'::jsonb
    WHERE id = (
        SELECT id FROM processing_tasks
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING * INTO claimed;

    RETURN claimed;
END;
$$;

COMMENT ON FUNCTION public.claim_next_pending_task() IS '原子性抢单：选取最早的 pending 任务并锁定为 processing，使用 SKIP LOCKED 避免多 Worker 竞态';
