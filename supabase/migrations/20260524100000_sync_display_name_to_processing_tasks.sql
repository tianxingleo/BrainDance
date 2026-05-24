-- 将 model_assets.display_name 自动同步到 processing_tasks.display_name
--
-- 背景:
--   * model_assets 是模型命名的权威源 (Recall / Agent rename_model_asset 都写这里)
--   * processing_tasks.display_name 仅用于任务列表与 "处理中" 视图的展示
--   * 过去仅 Flutter 端 recall_model_actions 做 best-effort 双写,
--     Agent / batch_patch / write_model_assets 等路径全部漏写,
--     导致任务列表名字与 Recall / Community 不一致
--
-- 方案:
--   1. processing_tasks.scene_id 加索引 (触发器 update 与 Realtime 反查都用)
--   2. 在 model_assets 上挂 AFTER UPDATE OF display_name 触发器,
--      自动同步到同一 scene_id 的所有 processing_tasks 行
--   3. 一次性回填: 仅填 task 端为空、model_assets 端有名字的记录,
--      保留 video_submit 入口已经写入的用户输入

-- 1. 索引 (幂等)
CREATE INDEX IF NOT EXISTS "processing_tasks_scene_id_idx"
  ON "public"."processing_tasks" ("scene_id");

-- 2. 触发器函数
--    SECURITY DEFINER + 显式 search_path: 让函数以 owner 身份写 processing_tasks,
--    绕过调用方 RLS, 同时防止 search_path 注入。
CREATE OR REPLACE FUNCTION "public"."sync_display_name_to_processing_tasks"()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  -- 外层: 名字没变就什么也不做, 避免无意义写入触发 Realtime
  IF NEW.display_name IS DISTINCT FROM OLD.display_name THEN
    -- 内层: 只更新真正不一致的行, 避免对已经同步的 task 重复写入
    UPDATE "public"."processing_tasks"
       SET display_name = NEW.display_name
     WHERE scene_id = NEW.scene_id
       AND display_name IS DISTINCT FROM NEW.display_name;
  END IF;
  RETURN NEW;
END;
$$;

COMMENT ON FUNCTION "public"."sync_display_name_to_processing_tasks"()
IS '当 model_assets.display_name 变更时, 自动同步到所有同 scene_id 的 processing_tasks 行。';

-- 3. 触发器 (先 drop 再 create, 保证幂等)
--    AFTER UPDATE OF display_name: 仅在 display_name 列被 UPDATE 时触发,
--    其它字段更新 (Worker upsert ply_path / tags 等) 不会进入该触发器。
--    特意不挂 INSERT: model_assets 首次创建是 Worker upsert, 那时 display_name 为 NULL,
--    不应该把 video_submit 已经填到 task 上的用户输入覆盖成 NULL。
DROP TRIGGER IF EXISTS "trg_sync_display_name_to_tasks" ON "public"."model_assets";

CREATE TRIGGER "trg_sync_display_name_to_tasks"
  AFTER UPDATE OF "display_name" ON "public"."model_assets"
  FOR EACH ROW
  EXECUTE FUNCTION "public"."sync_display_name_to_processing_tasks"();

-- 4. 一次性回填
--    仅在 task 端为空、且 model_assets 端非空时填充,
--    保留用户在 video_submit 提交时已经写入的标题。
UPDATE "public"."processing_tasks" pt
   SET display_name = ma.display_name
  FROM "public"."model_assets" ma
 WHERE pt.scene_id = ma.scene_id
   AND ma.display_name IS NOT NULL
   AND btrim(ma.display_name) <> ''
   AND (pt.display_name IS NULL OR btrim(pt.display_name) = '');
