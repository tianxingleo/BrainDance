create extension if not exists "pg_trgm" with schema "public";

create extension if not exists "vector" with schema "public";

create sequence "public"."rag_docs_id_seq";


  create table "public"."model_assets" (
    "id" uuid not null default gen_random_uuid(),
    "scene_id" text not null,
    "user_id" text,
    "source_task_id" uuid,
    "description" text,
    "objects" text[],
    "tags" text[],
    "embedding" public.vector(1536),
    "ply_path" text,
    "preview_img_path" text,
    "meta_info" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone not null default timezone('utc'::text, now())
      );



  create table "public"."processing_tasks" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" text not null,
    "scene_id" text not null,
    "status" text default 'pending'::text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "logs" jsonb default '[]'::jsonb,
    "tags" text[] default '{}'::text[],
    "quality_score" integer default 0,
    "quality_reason" text,
    "subject" text,
    "category" text,
    "description" text,
    "keywords" text[] default '{}'::text[]
      );


alter table "public"."processing_tasks" enable row level security;


  create table "public"."rag_docs" (
    "id" bigint not null default nextval('public.rag_docs_id_seq'::regclass),
    "content" text,
    "metadata" jsonb,
    "embedding" public.vector(1536)
      );



  create table "public"."tasks" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "source_path" text not null,
    "status" text default 'pending'::text,
    "worker_id" text,
    "result_data" jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
      );


alter table "public"."tasks" enable row level security;

alter sequence "public"."rag_docs_id_seq" owned by "public"."rag_docs"."id";

CREATE INDEX model_assets_embedding_idx ON public.model_assets USING hnsw (embedding public.vector_cosine_ops);

CREATE UNIQUE INDEX model_assets_pkey ON public.model_assets USING btree (id);

CREATE UNIQUE INDEX processing_tasks_pkey ON public.processing_tasks USING btree (id);

CREATE UNIQUE INDEX rag_docs_pkey ON public.rag_docs USING btree (id);

CREATE UNIQUE INDEX tasks_pkey ON public.tasks USING btree (id);

CREATE UNIQUE INDEX unique_scene_id ON public.model_assets USING btree (scene_id);

alter table "public"."model_assets" add constraint "model_assets_pkey" PRIMARY KEY using index "model_assets_pkey";

alter table "public"."processing_tasks" add constraint "processing_tasks_pkey" PRIMARY KEY using index "processing_tasks_pkey";

alter table "public"."rag_docs" add constraint "rag_docs_pkey" PRIMARY KEY using index "rag_docs_pkey";

alter table "public"."tasks" add constraint "tasks_pkey" PRIMARY KEY using index "tasks_pkey";

alter table "public"."model_assets" add constraint "unique_scene_id" UNIQUE using index "unique_scene_id";

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.match_model_assets(query_embedding public.vector, match_threshold double precision, match_count integer, filter_start timestamp with time zone DEFAULT NULL::timestamp with time zone, filter_end timestamp with time zone DEFAULT NULL::timestamp with time zone)
 RETURNS TABLE(id uuid, scene_id text, description text, ply_path text, created_at timestamp with time zone, similarity double precision)
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
begin
  return query
  select
    model_assets.id,
    model_assets.scene_id,
    model_assets.description,
    model_assets.ply_path,
    model_assets.created_at,
    1 - (model_assets.embedding <=> query_embedding) as similarity
  from model_assets
  where 1 - (model_assets.embedding <=> query_embedding) > match_threshold
  -- 时间过滤逻辑
  and (filter_start is null or model_assets.created_at >= filter_start)
  and (filter_end is null or model_assets.created_at <= filter_end)
  order by model_assets.embedding <=> query_embedding
  limit match_count;
end;
$function$
;

grant delete on table "public"."model_assets" to "anon";

grant insert on table "public"."model_assets" to "anon";

grant references on table "public"."model_assets" to "anon";

grant select on table "public"."model_assets" to "anon";

grant trigger on table "public"."model_assets" to "anon";

grant truncate on table "public"."model_assets" to "anon";

grant update on table "public"."model_assets" to "anon";

grant delete on table "public"."model_assets" to "authenticated";

grant insert on table "public"."model_assets" to "authenticated";

grant references on table "public"."model_assets" to "authenticated";

grant select on table "public"."model_assets" to "authenticated";

grant trigger on table "public"."model_assets" to "authenticated";

grant truncate on table "public"."model_assets" to "authenticated";

grant update on table "public"."model_assets" to "authenticated";

grant delete on table "public"."model_assets" to "postgres";

grant insert on table "public"."model_assets" to "postgres";

grant references on table "public"."model_assets" to "postgres";

grant select on table "public"."model_assets" to "postgres";

grant trigger on table "public"."model_assets" to "postgres";

grant truncate on table "public"."model_assets" to "postgres";

grant update on table "public"."model_assets" to "postgres";

grant delete on table "public"."model_assets" to "service_role";

grant insert on table "public"."model_assets" to "service_role";

grant references on table "public"."model_assets" to "service_role";

grant select on table "public"."model_assets" to "service_role";

grant trigger on table "public"."model_assets" to "service_role";

grant truncate on table "public"."model_assets" to "service_role";

grant update on table "public"."model_assets" to "service_role";

grant delete on table "public"."processing_tasks" to "anon";

grant insert on table "public"."processing_tasks" to "anon";

grant references on table "public"."processing_tasks" to "anon";

grant select on table "public"."processing_tasks" to "anon";

grant trigger on table "public"."processing_tasks" to "anon";

grant truncate on table "public"."processing_tasks" to "anon";

grant update on table "public"."processing_tasks" to "anon";

grant delete on table "public"."processing_tasks" to "authenticated";

grant insert on table "public"."processing_tasks" to "authenticated";

grant references on table "public"."processing_tasks" to "authenticated";

grant select on table "public"."processing_tasks" to "authenticated";

grant trigger on table "public"."processing_tasks" to "authenticated";

grant truncate on table "public"."processing_tasks" to "authenticated";

grant update on table "public"."processing_tasks" to "authenticated";

grant delete on table "public"."processing_tasks" to "postgres";

grant insert on table "public"."processing_tasks" to "postgres";

grant references on table "public"."processing_tasks" to "postgres";

grant select on table "public"."processing_tasks" to "postgres";

grant trigger on table "public"."processing_tasks" to "postgres";

grant truncate on table "public"."processing_tasks" to "postgres";

grant update on table "public"."processing_tasks" to "postgres";

grant delete on table "public"."processing_tasks" to "service_role";

grant insert on table "public"."processing_tasks" to "service_role";

grant references on table "public"."processing_tasks" to "service_role";

grant select on table "public"."processing_tasks" to "service_role";

grant trigger on table "public"."processing_tasks" to "service_role";

grant truncate on table "public"."processing_tasks" to "service_role";

grant update on table "public"."processing_tasks" to "service_role";

grant delete on table "public"."rag_docs" to "anon";

grant insert on table "public"."rag_docs" to "anon";

grant references on table "public"."rag_docs" to "anon";

grant select on table "public"."rag_docs" to "anon";

grant trigger on table "public"."rag_docs" to "anon";

grant truncate on table "public"."rag_docs" to "anon";

grant update on table "public"."rag_docs" to "anon";

grant delete on table "public"."rag_docs" to "authenticated";

grant insert on table "public"."rag_docs" to "authenticated";

grant references on table "public"."rag_docs" to "authenticated";

grant select on table "public"."rag_docs" to "authenticated";

grant trigger on table "public"."rag_docs" to "authenticated";

grant truncate on table "public"."rag_docs" to "authenticated";

grant update on table "public"."rag_docs" to "authenticated";

grant delete on table "public"."rag_docs" to "postgres";

grant insert on table "public"."rag_docs" to "postgres";

grant references on table "public"."rag_docs" to "postgres";

grant select on table "public"."rag_docs" to "postgres";

grant trigger on table "public"."rag_docs" to "postgres";

grant truncate on table "public"."rag_docs" to "postgres";

grant update on table "public"."rag_docs" to "postgres";

grant delete on table "public"."rag_docs" to "service_role";

grant insert on table "public"."rag_docs" to "service_role";

grant references on table "public"."rag_docs" to "service_role";

grant select on table "public"."rag_docs" to "service_role";

grant trigger on table "public"."rag_docs" to "service_role";

grant truncate on table "public"."rag_docs" to "service_role";

grant update on table "public"."rag_docs" to "service_role";

grant delete on table "public"."tasks" to "anon";

grant insert on table "public"."tasks" to "anon";

grant references on table "public"."tasks" to "anon";

grant select on table "public"."tasks" to "anon";

grant trigger on table "public"."tasks" to "anon";

grant truncate on table "public"."tasks" to "anon";

grant update on table "public"."tasks" to "anon";

grant delete on table "public"."tasks" to "authenticated";

grant insert on table "public"."tasks" to "authenticated";

grant references on table "public"."tasks" to "authenticated";

grant select on table "public"."tasks" to "authenticated";

grant trigger on table "public"."tasks" to "authenticated";

grant truncate on table "public"."tasks" to "authenticated";

grant update on table "public"."tasks" to "authenticated";

grant delete on table "public"."tasks" to "postgres";

grant insert on table "public"."tasks" to "postgres";

grant references on table "public"."tasks" to "postgres";

grant select on table "public"."tasks" to "postgres";

grant trigger on table "public"."tasks" to "postgres";

grant truncate on table "public"."tasks" to "postgres";

grant update on table "public"."tasks" to "postgres";

grant delete on table "public"."tasks" to "service_role";

grant insert on table "public"."tasks" to "service_role";

grant references on table "public"."tasks" to "service_role";

grant select on table "public"."tasks" to "service_role";

grant trigger on table "public"."tasks" to "service_role";

grant truncate on table "public"."tasks" to "service_role";

grant update on table "public"."tasks" to "service_role";


  create policy "Allow all for dev"
  on "public"."processing_tasks"
  as permissive
  for all
  to public
using (true);



  create policy "Enable all access for local dev"
  on "public"."tasks"
  as permissive
  for all
  to public
using (true);



  create policy "Allow user delete own folder"
  on "storage"."objects"
  as permissive
  for delete
  to public
using (((bucket_id = 'braindance-assets'::text) AND ((auth.uid())::text = (storage.foldername(name))[1])));



  create policy "Allow user upload own folder"
  on "storage"."objects"
  as permissive
  for insert
  to public
with check (((bucket_id = 'braindance-assets'::text) AND ((auth.uid())::text = (storage.foldername(name))[1])));



  create policy "Allow user view own folder"
  on "storage"."objects"
  as permissive
  for select
  to public
using (((bucket_id = 'braindance-assets'::text) AND ((auth.uid())::text = (storage.foldername(name))[1])));



  create policy "Enable all storage access"
  on "storage"."objects"
  as permissive
  for all
  to public
using (true);



