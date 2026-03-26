-- 新增弱图谱关系表与记忆专题表
-- 执行时间: 2026-03-26

CREATE TABLE IF NOT EXISTS public.related_model_links (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_model_id uuid NOT NULL REFERENCES public.model_assets(id) ON DELETE CASCADE,
  target_model_id uuid NOT NULL REFERENCES public.model_assets(id) ON DELETE CASCADE,
  relation_type text NOT NULL,
  score double precision DEFAULT 0,
  meta jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now())
);

COMMENT ON TABLE public.related_model_links
IS '记录模型之间的弱关系，如 same_place、same_thread、before_after、same_event';

CREATE INDEX IF NOT EXISTS idx_related_model_links_source_relation
ON public.related_model_links (source_model_id, relation_type);

CREATE INDEX IF NOT EXISTS idx_related_model_links_target_relation
ON public.related_model_links (target_model_id, relation_type);

CREATE TABLE IF NOT EXISTS public.memory_collections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  title text NOT NULL,
  description text,
  cover_model_id uuid REFERENCES public.model_assets(id) ON DELETE SET NULL,
  collection_type text DEFAULT 'manual',
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  updated_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now())
);

COMMENT ON TABLE public.memory_collections
IS '用户整理出的记忆专题、时间线或主题归档集合';

CREATE TABLE IF NOT EXISTS public.memory_collection_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  collection_id uuid NOT NULL REFERENCES public.memory_collections(id) ON DELETE CASCADE,
  model_id uuid NOT NULL REFERENCES public.model_assets(id) ON DELETE CASCADE,
  sort_order integer DEFAULT 0,
  note text,
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  UNIQUE (collection_id, model_id)
);

COMMENT ON TABLE public.memory_collection_items
IS '记忆专题中的模型成员与排序信息';

CREATE INDEX IF NOT EXISTS idx_memory_collection_items_collection_sort
ON public.memory_collection_items (collection_id, sort_order);

CREATE INDEX IF NOT EXISTS idx_memory_collection_items_model_id
ON public.memory_collection_items (model_id);
