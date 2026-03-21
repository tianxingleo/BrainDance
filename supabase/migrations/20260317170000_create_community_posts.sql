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

create index if not exists community_posts_created_at_idx
    on public.community_posts (created_at desc);

create index if not exists community_posts_place_idx
    on public.community_posts (place_name);

create index if not exists community_posts_geo_idx
    on public.community_posts (latitude, longitude);

alter table public.community_posts enable row level security;

drop policy if exists "Allow all for dev community posts" on public.community_posts;
create policy "Allow all for dev community posts"
on public.community_posts
for all
to public
using (true)
with check (true);
