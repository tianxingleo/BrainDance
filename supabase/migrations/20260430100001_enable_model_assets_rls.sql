-- P0 修复: 启用 model_assets 表的 RLS，添加基于 user_id 的行级隔离策略
-- model_assets.user_id 为 text 类型，需用 auth.uid()::text 转换
-- Worker 使用 service_role 绕过 RLS，不受影响
--
-- 风险说明：
--   现有数据中存在 user_id IS NULL 或 user_id = 'default_user' 的历史记录。
--   启用 RLS 后，这些记录对所有 authenticated 用户不可见（auth.uid()::text 不匹配）。
--   执行本迁移前，必须先确认这些记录的归属，并执行下方的 backfill 步骤。
--
-- 必须先执行 backfill（在 Supabase SQL Editor 中手动运行，迁移不自动执行）：
--   UPDATE public.model_assets
--   SET user_id = '<真实 user_id>'
--   WHERE user_id IS NULL OR user_id = 'default_user';
--
-- 如果历史数据确认无需保留，可改为：
--   DELETE FROM public.model_assets WHERE user_id IS NULL OR user_id = 'default_user';

-- 启用 RLS
ALTER TABLE public.model_assets ENABLE ROW LEVEL SECURITY;

-- 用户只能查看自己的模型资产
CREATE POLICY "Users can view own model assets"
  ON public.model_assets
  AS PERMISSIVE
  FOR SELECT
  TO public
  USING (user_id = auth.uid()::text);

-- 用户只能插入自己的模型资产
CREATE POLICY "Users can insert own model assets"
  ON public.model_assets
  AS PERMISSIVE
  FOR INSERT
  TO public
  WITH CHECK (user_id = auth.uid()::text);

-- 用户只能更新自己的模型资产
CREATE POLICY "Users can update own model assets"
  ON public.model_assets
  AS PERMISSIVE
  FOR UPDATE
  TO public
  USING (user_id = auth.uid()::text)
  WITH CHECK (user_id = auth.uid()::text);

-- 用户只能删除自己的模型资产
CREATE POLICY "Users can delete own model assets"
  ON public.model_assets
  AS PERMISSIVE
  FOR DELETE
  TO public
  USING (user_id = auth.uid()::text);

-- 添加 user_id 索引，优化按用户查询性能
CREATE INDEX IF NOT EXISTS idx_model_assets_user_id ON public.model_assets(user_id);
