-- 添加 processing_tasks 表的 task_type 和 task_fields 字段
-- 用于支持不同的任务类型（video_3dgs, single_image_sam3d, single_image_sharp）
-- 执行时间: 2026-01-21

-- 1. 添加 task_type 字段
-- 用途: 标识任务类型，支持视频转3DGS和单图转3DGS
-- 默认值: 'video_3dgs' (向后兼容)
ALTER TABLE "public"."processing_tasks"
ADD COLUMN "task_type" text DEFAULT 'video_3dgs'::text;

-- 2. 添加 task_params 字段
-- 用途: 存储任务特定参数，JSON格式
-- 示例:
--   - single_image_sam3d: {"mask_path": "custom/mask.png"}
--   - single_image_sharp: {}  (无额外参数)
--   - video_3dgs: {}  (无额外参数)
ALTER TABLE "public"."processing_tasks"
ADD COLUMN "task_params" jsonb DEFAULT '{}'::jsonb;

-- 3. 添加索引以提高查询性能
-- 常见查询模式: 根据 task_type 和 status 查询待处理任务
CREATE INDEX IF NOT EXISTS "processing_tasks_task_type_idx"
ON "public"."processing_tasks" (task_type);

CREATE INDEX IF NOT EXISTS "processing_tasks_status_task_type_idx"
ON "public"."processing_tasks" (status, task_type);

-- 4. 添加注释（文档说明）
COMMENT ON COLUMN "public"."processing_tasks"."task_type" IS '任务类型: video_3dgs(GS), single_image_sam3d(单图转3视频转3DDGS-SAM3D), single_image_sharp(单图转3DGS-SHARP)';
COMMENT ON COLUMN "public"."processing_tasks"."task_params" IS '任务参数，JSON格式，不同任务类型有不同参数';

-- 5. 更新现有数据
-- 将已有的 pending 任务设置为默认 task_type
UPDATE "public"."processing_tasks"
SET task_type = 'video_3dgs'
WHERE task_type IS NULL;

-- 6. 添加约束（可选，确保 task_type 只能是有效值）
-- 注意: 如果需要严格约束，可以取消注释以下行
-- ALTER TABLE "public"."processing_tasks"
-- ADD CONSTRAINT "task_type_check"
-- CHECK (task_type IN ('video_3dgs', 'single_image_sam3d', 'single_image_sharp'));

-- 7. 授予权限（确保 anon 和 authenticated 角色可以读写这些字段）
GRANT SELECT(task_type, task_params) ON TABLE "public"."processing_tasks" TO "anon";
GRANT SELECT(task_type, task_params) ON TABLE "public"."processing_tasks" TO "authenticated";
GRANT UPDATE(task_type, task_params) ON TABLE "public"."processing_tasks" TO "service_role";
