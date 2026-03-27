-- BrainDance 集成测试最小种子数据

create table if not exists public.community_posts (
    id uuid primary key default gen_random_uuid(),
    user_id text,
    model_asset_id uuid references public.model_assets(id) on delete set null,
    model_name text,
    title text not null,
    caption text not null default '',
    place_name text not null,
    latitude double precision not null,
    longitude double precision not null,
    cover_image_url text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

insert into public.processing_tasks (
  id, user_id, scene_id, status, display_name, task_type, task_params, description
) values
  (
    '10000000-0000-0000-0000-000000000001',
    'it_user_a',
    'it_minimal_scene_001',
    'completed',
    'IT 最小样例 001',
    'video_3dgs',
    '{}'::jsonb,
    'integration minimal task'
  )
on conflict (id) do update set
  status = excluded.status,
  display_name = excluded.display_name,
  updated_at = now();

insert into public.model_assets (
  id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info
) values
  (
    '20000000-0000-0000-0000-000000000001',
    'it_minimal_scene_001',
    'it_user_a',
    'integration minimal asset',
    array['chair','desk'],
    array['it','minimal'],
    'it_user_a/it_minimal_scene_001/output/point_cloud.ply',
    'it_user_a/it_minimal_scene_001/output/preview.txt',
    '{"source":"integration_minimal"}'::jsonb
  )
on conflict (id) do update set
  description = excluded.description,
  tags = excluded.tags,
  ply_path = excluded.ply_path;

insert into public.community_posts (
  id, user_id, model_asset_id, model_name, title, caption, place_name, latitude, longitude, cover_image_url
) values
  (
    '30000000-0000-0000-0000-000000000001',
    'it_user_a',
    '20000000-0000-0000-0000-000000000001',
    'it_minimal_scene_001',
    'IT Community Minimal 001',
    'integration minimal post',
    'Integration Place',
    30.0,
    120.0,
    'it_user_a/it_minimal_scene_001/output/preview.txt'
  )
on conflict (id) do update set
  caption = excluded.caption,
  updated_at = now();
