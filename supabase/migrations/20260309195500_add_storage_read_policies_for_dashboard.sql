-- Dashboard storage read policies for public bucket `braindance-assets`
-- Why: anon/publishable key often cannot list buckets/objects without explicit SELECT policies.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'buckets'
      AND policyname = 'anon_can_read_braindance_bucket_meta'
  ) THEN
    CREATE POLICY anon_can_read_braindance_bucket_meta
      ON storage.buckets
      FOR SELECT
      TO anon
      USING (id = 'braindance-assets');
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'objects'
      AND policyname = 'anon_can_list_braindance_objects'
  ) THEN
    CREATE POLICY anon_can_list_braindance_objects
      ON storage.objects
      FOR SELECT
      TO anon
      USING (bucket_id = 'braindance-assets');
  END IF;
END $$;
