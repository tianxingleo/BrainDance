-- 为任务队列表增加展示名称字段
-- 执行时间: 2026-03-07

ALTER TABLE "public"."processing_tasks"
ADD COLUMN IF NOT EXISTS "display_name" text;

COMMENT ON COLUMN "public"."processing_tasks"."display_name"
IS '任务展示名称，用于前端列表显示';
