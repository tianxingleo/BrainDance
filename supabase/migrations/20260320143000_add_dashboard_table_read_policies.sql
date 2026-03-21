-- Dashboard reads these tables directly with the publishable/anon key.
-- Keep RLS enabled, but explicitly allow read-only access for dashboard pages.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'processing_tasks'
      AND policyname = 'dashboard_read_processing_tasks'
  ) THEN
    CREATE POLICY dashboard_read_processing_tasks
      ON public.processing_tasks
      FOR SELECT
      TO anon, authenticated
      USING (true);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'memory_poses'
      AND policyname = 'dashboard_read_memory_poses'
  ) THEN
    CREATE POLICY dashboard_read_memory_poses
      ON public.memory_poses
      FOR SELECT
      TO anon, authenticated
      USING (true);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'tasks'
      AND policyname = 'dashboard_read_tasks'
  ) THEN
    CREATE POLICY dashboard_read_tasks
      ON public.tasks
      FOR SELECT
      TO anon, authenticated
      USING (true);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'model_assets'
      AND policyname = 'dashboard_read_model_assets'
  ) THEN
    CREATE POLICY dashboard_read_model_assets
      ON public.model_assets
      FOR SELECT
      TO anon, authenticated
      USING (true);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'rag_docs'
      AND policyname = 'dashboard_read_rag_docs'
  ) THEN
    CREATE POLICY dashboard_read_rag_docs
      ON public.rag_docs
      FOR SELECT
      TO anon, authenticated
      USING (true);
  END IF;
END $$;
