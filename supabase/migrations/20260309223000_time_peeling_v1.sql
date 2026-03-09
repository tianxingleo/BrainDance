-- Time Peeling V1 schema and RPC

-- 1) Logical spaces
CREATE TABLE IF NOT EXISTS public.memory_spaces (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  title text,
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  updated_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.memory_spaces ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'memory_spaces' AND policyname = 'Users can view their own memory spaces'
  ) THEN
    CREATE POLICY "Users can view their own memory spaces"
      ON public.memory_spaces FOR SELECT
      USING (auth.uid()::text = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'memory_spaces' AND policyname = 'Users can insert their own memory spaces'
  ) THEN
    CREATE POLICY "Users can insert their own memory spaces"
      ON public.memory_spaces FOR INSERT
      WITH CHECK (auth.uid()::text = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'memory_spaces' AND policyname = 'Users can update their own memory spaces'
  ) THEN
    CREATE POLICY "Users can update their own memory spaces"
      ON public.memory_spaces FOR UPDATE
      USING (auth.uid()::text = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'memory_spaces' AND policyname = 'Users can delete their own memory spaces'
  ) THEN
    CREATE POLICY "Users can delete their own memory spaces"
      ON public.memory_spaces FOR DELETE
      USING (auth.uid()::text = user_id);
  END IF;
END $$;

-- 2) Time captures in one space
CREATE TABLE IF NOT EXISTS public.space_captures (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  space_id uuid NOT NULL REFERENCES public.memory_spaces(id) ON DELETE CASCADE,
  user_id text NOT NULL,
  scene_id text NOT NULL,
  captured_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  status text NOT NULL DEFAULT 'processing',
  align_to_capture_id uuid REFERENCES public.space_captures(id) ON DELETE SET NULL,
  alignment_matrix jsonb NOT NULL DEFAULT '[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]'::jsonb,
  alignment_score double precision,
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.space_captures ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'space_captures' AND policyname = 'Users can view their own space captures'
  ) THEN
    CREATE POLICY "Users can view their own space captures"
      ON public.space_captures FOR SELECT
      USING (auth.uid()::text = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'space_captures' AND policyname = 'Users can insert their own space captures'
  ) THEN
    CREATE POLICY "Users can insert their own space captures"
      ON public.space_captures FOR INSERT
      WITH CHECK (auth.uid()::text = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'space_captures' AND policyname = 'Users can update their own space captures'
  ) THEN
    CREATE POLICY "Users can update their own space captures"
      ON public.space_captures FOR UPDATE
      USING (auth.uid()::text = user_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'space_captures' AND policyname = 'Users can delete their own space captures'
  ) THEN
    CREATE POLICY "Users can delete their own space captures"
      ON public.space_captures FOR DELETE
      USING (auth.uid()::text = user_id);
  END IF;
END $$;

-- 3) Backward-compatible columns
ALTER TABLE public.processing_tasks
  ADD COLUMN IF NOT EXISTS space_id uuid REFERENCES public.memory_spaces(id) ON DELETE SET NULL;

ALTER TABLE public.model_assets
  ADD COLUMN IF NOT EXISTS space_id uuid REFERENCES public.memory_spaces(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS capture_id uuid REFERENCES public.space_captures(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS captured_at timestamptz;

-- 4) Indexes
CREATE INDEX IF NOT EXISTS idx_space_captures_space_captured_at
  ON public.space_captures(space_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_assets_capture_id
  ON public.model_assets(capture_id);

CREATE INDEX IF NOT EXISTS idx_model_assets_space_captured_at
  ON public.model_assets(space_id, captured_at DESC);

-- 5) Timeline RPC
CREATE OR REPLACE FUNCTION public.get_space_captures(p_space_id uuid)
RETURNS TABLE (
  capture_id uuid,
  scene_id text,
  captured_at timestamptz,
  status text,
  model_url text,
  alignment_matrix jsonb,
  alignment_score double precision
)
LANGUAGE sql
SECURITY DEFINER
AS $$
  SELECT
    c.id AS capture_id,
    c.scene_id,
    c.captured_at,
    c.status,
    a.ply_path AS model_url,
    c.alignment_matrix,
    c.alignment_score
  FROM public.space_captures c
  LEFT JOIN public.model_assets a ON a.capture_id = c.id
  WHERE c.space_id = p_space_id
    AND (auth.uid() IS NULL OR c.user_id = auth.uid()::text)
  ORDER BY c.captured_at DESC;
$$;

GRANT EXECUTE ON FUNCTION public.get_space_captures(uuid) TO anon, authenticated, service_role;

-- 6) Enrich existing search RPC with space/capture dimensions
CREATE OR REPLACE FUNCTION public.match_memory_poses(
  query_embedding public.vector(1536),
  match_threshold double precision,
  match_count integer,
  filter_start timestamp with time zone DEFAULT NULL::timestamp with time zone,
  filter_end timestamp with time zone DEFAULT NULL::timestamp with time zone
)
RETURNS TABLE (
  id uuid,
  scene_id text,
  description text,
  ply_path text,
  created_at timestamp with time zone,
  user_id text,
  space_id uuid,
  capture_id uuid,
  captured_at timestamp with time zone,
  similarity double precision,
  matched_frames jsonb
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  WITH matched AS (
    SELECT
      m.id AS frame_id,
      m.model_id,
      m.image_name,
      m.transform_matrix,
      1 - (m.embedding <=> query_embedding) AS similarity,
      a.scene_id,
      a.description,
      a.ply_path,
      a.created_at,
      a.user_id,
      a.space_id,
      a.capture_id,
      a.captured_at
    FROM public.memory_poses m
    JOIN public.model_assets a ON m.model_id = a.id
    WHERE 1 - (m.embedding <=> query_embedding) > match_threshold
      AND (filter_start IS NULL OR a.created_at >= filter_start)
      AND (filter_end IS NULL OR a.created_at <= filter_end)
      AND (auth.uid() IS NULL OR a.user_id = auth.uid()::text)
  )
  SELECT
    matched.model_id AS id,
    matched.scene_id,
    matched.description,
    matched.ply_path,
    matched.created_at,
    matched.user_id,
    MAX(matched.space_id) AS space_id,
    MAX(matched.capture_id) AS capture_id,
    MAX(matched.captured_at) AS captured_at,
    MAX(matched.similarity) AS similarity,
    jsonb_agg(
      jsonb_build_object(
        'image_name', matched.image_name,
        'transform_matrix', matched.transform_matrix,
        'similarity', matched.similarity
      ) ORDER BY matched.similarity DESC
    ) AS matched_frames
  FROM matched
  GROUP BY matched.model_id, matched.scene_id, matched.description, matched.ply_path, matched.created_at, matched.user_id
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;
