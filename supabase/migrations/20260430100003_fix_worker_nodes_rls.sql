-- P0 修复: 收紧 worker_nodes 表的 RLS 策略
-- 原策略 "Allow all for dev" 允许 anon 完全控制 Worker（暂停/中断/恢复）
-- 新策略：anon 只读（Dashboard 状态展示），authenticated 可读写，service_role 全量
-- 注意：此修改会导致 Dashboard 暂时无法控制 Worker，直到 Dashboard 接入认证

-- 删除全开策略
DROP POLICY IF EXISTS "Allow all for dev on worker_nodes" ON public.worker_nodes;

-- anon 可以 SELECT（Dashboard 未认证时仍可查看 Worker 状态）
CREATE POLICY "Anon can read worker_nodes"
  ON public.worker_nodes
  AS PERMISSIVE
  FOR SELECT
  TO anon
  USING (true);

-- authenticated 用户可以读写（Dashboard 接入认证后使用）
CREATE POLICY "Authenticated can read worker_nodes"
  ON public.worker_nodes
  AS PERMISSIVE
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated can insert worker_nodes"
  ON public.worker_nodes
  AS PERMISSIVE
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated can update worker_nodes"
  ON public.worker_nodes
  AS PERMISSIVE
  FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated can delete worker_nodes"
  ON public.worker_nodes
  AS PERMISSIVE
  FOR DELETE
  TO authenticated
  USING (true);

-- 更新注释，补充 interrupt 值
COMMENT ON COLUMN "public"."worker_nodes"."desired_state" IS 'run / pause / interrupt，dashboard 通过该字段请求 worker 优雅退出或中断';
