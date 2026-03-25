-- 为模型资产补充正式展示名称字段
-- 执行时间: 2026-03-25

ALTER TABLE "public"."model_assets"
ADD COLUMN IF NOT EXISTS "display_name" text;

COMMENT ON COLUMN "public"."model_assets"."display_name"
IS '模型资产展示名称，用于 Recall / Agent / Collection 页面统一显示';
