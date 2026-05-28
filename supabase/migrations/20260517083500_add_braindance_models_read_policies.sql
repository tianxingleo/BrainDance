-- 允许端侧使用 anon/publishable key 读取模型发布 bucket 的元数据。
-- 模型文件本身通过 public object URL 下载；这里补的是列表发现能力，
-- 让 Flutter 可以扫描新上传的 GGUF，而不必每次都手动同步 catalog。

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'buckets'
      AND policyname = 'anon_can_read_braindance_models_bucket_meta'
  ) THEN
    CREATE POLICY anon_can_read_braindance_models_bucket_meta
      ON storage.buckets
      FOR SELECT
      TO anon
      USING (id = 'braindance-models');
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'objects'
      AND policyname = 'anon_can_list_braindance_model_objects'
  ) THEN
    CREATE POLICY anon_can_list_braindance_model_objects
      ON storage.objects
      FOR SELECT
      TO anon
      USING (bucket_id = 'braindance-models');
  END IF;
END $$;
