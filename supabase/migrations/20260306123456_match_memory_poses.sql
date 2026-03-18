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
      a.user_id
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
