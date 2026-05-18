-- 用户长期记忆：聚合搜索偏好与最近搜索历史
-- 用于 Agent 跨会话个性化检索

CREATE TABLE IF NOT EXISTS public.user_long_term_memory (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,

  -- 聚合偏好
  preferred_regions text[] DEFAULT '{}',
  preferred_asset_types text[] DEFAULT '{}',
  preferred_time_ranges text[] DEFAULT '{}',
  preferred_objects text[] DEFAULT '{}',

  -- 最近搜索日志 (最多 10 条)
  recent_searches jsonb DEFAULT '[]'::jsonb,

  -- 元数据
  search_count integer DEFAULT 0,
  last_updated_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),

  CONSTRAINT uq_user_long_term_memory_user_id UNIQUE (user_id)
);

COMMENT ON TABLE public.user_long_term_memory
IS '用户长期记忆：聚合搜索偏好与最近搜索历史，用于 Agent 个性化检索';

COMMENT ON COLUMN public.user_long_term_memory.preferred_regions
IS '用户历史搜索中高频出现的区域/地点偏好（最多 5 个）';

COMMENT ON COLUMN public.user_long_term_memory.preferred_objects
IS '用户历史搜索中高频出现的物体类型偏好（最多 8 个）';

COMMENT ON COLUMN public.user_long_term_memory.recent_searches
IS '最近 10 次搜索摘要，每项: {query, mode, topResultSummary, regions, objects, timestamp}';

CREATE INDEX IF NOT EXISTS idx_user_long_term_memory_user_id
ON public.user_long_term_memory (user_id);
