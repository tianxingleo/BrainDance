-- P0 修复: 启用 rag_docs 表的 RLS，限制直接访问
-- rag_docs 没有 user_id 字段，无法做用户级隔离
-- 设计决策：anon 无权访问，authenticated 只读，写入通过 Worker（service_role）完成
-- 查询通过 Edge Function（使用 service_role）间接执行

-- 启用 RLS
ALTER TABLE public.rag_docs ENABLE ROW LEVEL SECURITY;

-- authenticated 用户只能 SELECT（用于直接查询场景）
CREATE POLICY "Authenticated users can read rag_docs"
  ON public.rag_docs
  AS PERMISSIVE
  FOR SELECT
  TO authenticated
  USING (true);

-- 撤销 anon 对 rag_docs 的所有权限
-- service_role 和 postgres 不受 RLS 影响，Worker 正常写入
REVOKE ALL ON public.rag_docs FROM anon;
