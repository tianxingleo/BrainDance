-- 为 Recall / Agent 统一入口补充时间对比、专题归档与弱图谱基础字段
-- 执行时间: 2026-03-26

ALTER TABLE public.model_assets
ADD COLUMN IF NOT EXISTS place_id uuid,
ADD COLUMN IF NOT EXISTS memory_thread_id uuid,
ADD COLUMN IF NOT EXISTS version_label text,
ADD COLUMN IF NOT EXISTS summary_title text,
ADD COLUMN IF NOT EXISTS event_label text,
ADD COLUMN IF NOT EXISTS agent_meta jsonb DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.model_assets.place_id
IS '同一物理地点的稳定标识，用于时间对比与地点版本链归组';

COMMENT ON COLUMN public.model_assets.memory_thread_id
IS '同一地点下多次扫描形成的记忆线程标识';

COMMENT ON COLUMN public.model_assets.version_label
IS '面向用户展示的版本标签，如 2024-毕业前、2025-搬家后';

COMMENT ON COLUMN public.model_assets.summary_title
IS 'Agent 生成的简短回忆标题，不直接替代 display_name';

COMMENT ON COLUMN public.model_assets.event_label
IS '轻量事件标签，如 毕业前、搬家后、春节、装修后';

COMMENT ON COLUMN public.model_assets.agent_meta
IS 'Agent 维护的轻量结构化元数据，如命名来源、整理状态、趋势标签';

CREATE INDEX IF NOT EXISTS idx_model_assets_place_created_at
ON public.model_assets (place_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_assets_thread_created_at
ON public.model_assets (memory_thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_assets_user_place_created_at
ON public.model_assets (user_id, place_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_assets_user_thread_created_at
ON public.model_assets (user_id, memory_thread_id, created_at DESC);
