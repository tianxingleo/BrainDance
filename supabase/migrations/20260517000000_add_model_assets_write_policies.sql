-- model_assets RLS was never enabled in init schema, but dashboard_read
-- migration assumed it was on and added a SELECT-only policy.  Without
-- UPDATE / DELETE policies the recall page rename and delete-cloud-model
-- operations break silently (PostgREST returns empty array with 200).

-- Ensure RLS is enabled (idempotent).
ALTER TABLE public.model_assets ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to update their own rows.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename  = 'model_assets'
      AND policyname = 'authenticated_update_own_model_assets'
  ) THEN
    CREATE POLICY authenticated_update_own_model_assets
      ON public.model_assets
      FOR UPDATE
      TO authenticated
      USING ((auth.uid())::text = user_id);
  END IF;
END $$;

-- Allow authenticated users to delete their own rows.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename  = 'model_assets'
      AND policyname = 'authenticated_delete_own_model_assets'
  ) THEN
    CREATE POLICY authenticated_delete_own_model_assets
      ON public.model_assets
      FOR DELETE
      TO authenticated
      USING ((auth.uid())::text = user_id);
  END IF;
END $$;
