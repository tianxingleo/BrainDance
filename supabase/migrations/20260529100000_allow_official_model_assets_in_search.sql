-- Allow detection/query tools to read model_assets rows where is_official = true
-- across users.
--
-- Rationale: 检索/查询类工具需要能命中"官方资产"（is_official=true）即使其
-- user_id 与当前调用者不同。写工具仍然按 user_id 严格隔离，由调用方代码保证。
--
-- 本迁移只触达两个语义检索 RPC：match_memory_poses / match_model_assets。
-- - 新增 RETURNS TABLE 中暴露 is_official，便于上层做"自有 OR 官方"判定。
-- - WHERE 子句放开 auth.uid() 限制：允许命中自己的或 is_official=true 的行。
-- - 模型表的 is_official 列已在线上存在，因此本迁移不重新声明该列。

DROP FUNCTION IF EXISTS "public"."match_memory_poses"(
  "public"."vector",
  double precision,
  integer,
  timestamp with time zone,
  timestamp with time zone
);

CREATE OR REPLACE FUNCTION "public"."match_memory_poses"(
  "query_embedding" "public"."vector",
  "match_threshold" double precision,
  "match_count" integer,
  "filter_start" timestamp with time zone DEFAULT NULL::timestamp with time zone,
  "filter_end" timestamp with time zone DEFAULT NULL::timestamp with time zone
) RETURNS TABLE(
  "id" "uuid",
  "scene_id" "text",
  "description" "text",
  "ply_path" "text",
  "created_at" timestamp with time zone,
  "user_id" "text",
  "is_official" boolean,
  "similarity" double precision,
  "matched_frames" "jsonb"
)
LANGUAGE "plpgsql" SECURITY DEFINER
AS $_$
DECLARE
  _limit integer;
BEGIN
  _limit := GREATEST(match_count * 50, 200);

  RETURN QUERY EXECUTE '
    WITH nearest AS MATERIALIZED (
      SELECT
        m.id AS frame_id,
        m.model_id,
        m.image_name,
        m.transform_matrix,
        m.embedding <=> $1 AS distance
      FROM public.memory_poses m
      WHERE m.embedding IS NOT NULL
      ORDER BY m.embedding <=> $1
      LIMIT $6
    ),
    filtered AS (
      SELECT
        n.frame_id,
        n.model_id,
        n.image_name,
        n.transform_matrix,
        1 - n.distance AS similarity
      FROM nearest n
      WHERE 1 - n.distance > $2
    ),
    joined AS (
      SELECT
        f.frame_id,
        f.model_id,
        f.image_name,
        f.transform_matrix,
        f.similarity,
        a.scene_id,
        a.description,
        a.ply_path,
        a.created_at,
        a.user_id,
        COALESCE(a.is_official, false) AS is_official
      FROM filtered f
      JOIN public.model_assets a ON f.model_id = a.id
      WHERE ($4::timestamptz IS NULL OR a.created_at >= $4)
        AND ($5::timestamptz IS NULL OR a.created_at <= $5)
        AND (
          auth.uid() IS NULL
          OR a.user_id = auth.uid()::text
          OR COALESCE(a.is_official, false) = true
        )
    )
    SELECT
      joined.model_id AS id,
      joined.scene_id,
      joined.description,
      joined.ply_path,
      joined.created_at,
      joined.user_id,
      joined.is_official,
      MAX(joined.similarity) AS similarity,
      jsonb_agg(
        jsonb_build_object(
          ''image_name'', joined.image_name,
          ''transform_matrix'', joined.transform_matrix,
          ''similarity'', joined.similarity
        ) ORDER BY joined.similarity DESC
      ) AS matched_frames
    FROM joined
    GROUP BY joined.model_id, joined.scene_id, joined.description, joined.ply_path, joined.created_at, joined.user_id, joined.is_official
    ORDER BY similarity DESC
    LIMIT $3
  ' USING query_embedding, match_threshold, match_count, filter_start, filter_end, _limit;
END;
$_$;

ALTER FUNCTION "public"."match_memory_poses"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) OWNER TO "postgres";

GRANT ALL ON FUNCTION "public"."match_memory_poses"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) TO "anon";
GRANT ALL ON FUNCTION "public"."match_memory_poses"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) TO "authenticated";
GRANT ALL ON FUNCTION "public"."match_memory_poses"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) TO "service_role";


DROP FUNCTION IF EXISTS "public"."match_model_assets"(
  "public"."vector",
  double precision,
  integer,
  timestamp with time zone,
  timestamp with time zone
);

CREATE OR REPLACE FUNCTION "public"."match_model_assets"(
  "query_embedding" "public"."vector",
  "match_threshold" double precision,
  "match_count" integer,
  "filter_start" timestamp with time zone DEFAULT NULL::timestamp with time zone,
  "filter_end" timestamp with time zone DEFAULT NULL::timestamp with time zone
) RETURNS TABLE(
  "id" "uuid",
  "scene_id" "text",
  "description" "text",
  "ply_path" "text",
  "created_at" timestamp with time zone,
  "user_id" "text",
  "is_official" boolean,
  "similarity" double precision
)
LANGUAGE "plpgsql" SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    model_assets.id,
    model_assets.scene_id,
    model_assets.description,
    model_assets.ply_path,
    model_assets.created_at,
    model_assets.user_id,
    COALESCE(model_assets.is_official, false) AS is_official,
    1 - (model_assets.embedding <=> query_embedding) AS similarity
  FROM model_assets
  WHERE 1 - (model_assets.embedding <=> query_embedding) > match_threshold
    AND (filter_start IS NULL OR model_assets.created_at >= filter_start)
    AND (filter_end IS NULL OR model_assets.created_at <= filter_end)
  ORDER BY model_assets.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

ALTER FUNCTION "public"."match_model_assets"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) OWNER TO "postgres";

GRANT ALL ON FUNCTION "public"."match_model_assets"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) TO "anon";
GRANT ALL ON FUNCTION "public"."match_model_assets"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) TO "authenticated";
GRANT ALL ON FUNCTION "public"."match_model_assets"(
  "public"."vector", double precision, integer,
  timestamp with time zone, timestamp with time zone
) TO "service_role";
