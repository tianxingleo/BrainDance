create table if not exists "public"."worker_nodes" (
  "worker_id" text not null,
  "hostname" text,
  "pid" integer,
  "status" text not null default 'starting'::text,
  "current_task_id" uuid,
  "current_scene_id" text,
  "desired_state" text not null default 'run'::text,
  "control_note" text,
  "control_requested_at" timestamp with time zone,
  "last_heartbeat" timestamp with time zone not null default timezone('utc'::text, now()),
  "started_at" timestamp with time zone not null default timezone('utc'::text, now()),
  "stopped_at" timestamp with time zone,
  "metadata" jsonb not null default '{}'::jsonb
);

create unique index if not exists "worker_nodes_pkey" on "public"."worker_nodes" using btree ("worker_id");
create index if not exists "worker_nodes_last_heartbeat_idx" on "public"."worker_nodes" using btree ("last_heartbeat" desc);
create index if not exists "worker_nodes_status_idx" on "public"."worker_nodes" using btree ("status");

alter table "public"."worker_nodes" add constraint "worker_nodes_pkey" primary key using index "worker_nodes_pkey";

alter table "public"."worker_nodes" enable row level security;

comment on table "public"."worker_nodes" is 'AI Engine worker 注册、心跳与控制表';
comment on column "public"."worker_nodes"."status" is 'starting / idle / busy / stopping / offline / error';
comment on column "public"."worker_nodes"."desired_state" is 'run / pause，dashboard 通过该字段请求 worker 优雅退出';

grant delete on table "public"."worker_nodes" to "anon";
grant insert on table "public"."worker_nodes" to "anon";
grant references on table "public"."worker_nodes" to "anon";
grant select on table "public"."worker_nodes" to "anon";
grant trigger on table "public"."worker_nodes" to "anon";
grant truncate on table "public"."worker_nodes" to "anon";
grant update on table "public"."worker_nodes" to "anon";

grant delete on table "public"."worker_nodes" to "authenticated";
grant insert on table "public"."worker_nodes" to "authenticated";
grant references on table "public"."worker_nodes" to "authenticated";
grant select on table "public"."worker_nodes" to "authenticated";
grant trigger on table "public"."worker_nodes" to "authenticated";
grant truncate on table "public"."worker_nodes" to "authenticated";
grant update on table "public"."worker_nodes" to "authenticated";

grant delete on table "public"."worker_nodes" to "postgres";
grant insert on table "public"."worker_nodes" to "postgres";
grant references on table "public"."worker_nodes" to "postgres";
grant select on table "public"."worker_nodes" to "postgres";
grant trigger on table "public"."worker_nodes" to "postgres";
grant truncate on table "public"."worker_nodes" to "postgres";
grant update on table "public"."worker_nodes" to "postgres";

grant delete on table "public"."worker_nodes" to "service_role";
grant insert on table "public"."worker_nodes" to "service_role";
grant references on table "public"."worker_nodes" to "service_role";
grant select on table "public"."worker_nodes" to "service_role";
grant trigger on table "public"."worker_nodes" to "service_role";
grant truncate on table "public"."worker_nodes" to "service_role";
grant update on table "public"."worker_nodes" to "service_role";

create policy "Allow all for dev on worker_nodes"
on "public"."worker_nodes"
as permissive
for all
to public
using (true)
with check (true);
