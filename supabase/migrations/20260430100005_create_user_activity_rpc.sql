-- P0 修复: 创建用户活跃度聚合 RPC，替代 Dashboard 全表扫描
-- 将 processing_tasks / model_assets / tasks 三张表的全量拉取改为服务端聚合
-- 返回每个用户的任务数、模型数、最近活跃时间等统计

CREATE OR REPLACE FUNCTION public.get_user_activity_summary()
RETURNS TABLE(user_id text, total_tasks bigint, tasks_24h bigint, tasks_7d bigint, total_assets bigint, assets_7d bigint, last_active timestamptz)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  WITH
    task_stats AS (
      SELECT
        user_id,
        count(*) as total_tasks,
        count(*) FILTER (WHERE created_at >= now() - interval '24 hours') as tasks_24h,
        count(*) FILTER (WHERE created_at >= now() - interval '7 days') as tasks_7d,
        max(created_at) as last_task
      FROM processing_tasks
      GROUP BY user_id
    ),
    asset_stats AS (
      SELECT
        user_id,
        count(*) as total_assets,
        count(*) FILTER (WHERE created_at >= now() - interval '7 days') as assets_7d,
        max(created_at) as last_asset
      FROM model_assets
      GROUP BY user_id
    )
  SELECT
    coalesce(t.user_id, a.user_id) as user_id,
    coalesce(t.total_tasks, 0) as total_tasks,
    coalesce(t.tasks_24h, 0) as tasks_24h,
    coalesce(t.tasks_7d, 0) as tasks_7d,
    coalesce(a.total_assets, 0) as total_assets,
    coalesce(a.assets_7d, 0) as assets_7d,
    greatest(t.last_task, a.last_asset) as last_active
  FROM task_stats t
  FULL OUTER JOIN asset_stats a ON t.user_id = a.user_id
  ORDER BY last_active DESC NULLS LAST;
END;
$$;

COMMENT ON FUNCTION public.get_user_activity_summary() IS '服务端聚合用户活跃度，替代 Dashboard 全表扫描三张表';
