-- BrainDance 集成测试清理脚本
-- 删除 it_* 前缀测试数据，允许重复执行。

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

delete from public.community_posts
where user_id like 'it_%'
   or model_name like 'it_%'
   or title like 'IT %';

delete from public.memory_poses
where model_id in (
  select id from public.model_assets where scene_id like 'it_%'
);

delete from public.model_assets
where scene_id like 'it_%'
   or user_id like 'it_%'
   or ply_path like 'it_%';

delete from public.processing_tasks
where scene_id like 'it_%'
   or user_id like 'it_%'
   or display_name like 'IT %';
