-- Create memory_poses table for spatial semantic anchors
CREATE TABLE IF NOT EXISTS public.memory_poses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES public.model_assets(id) ON DELETE CASCADE,
    image_name TEXT NOT NULL,
    transform_matrix JSONB NOT NULL,
    tag TEXT,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);
-- Enable RLS
ALTER TABLE public.memory_poses ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own memory poses"
    ON public.memory_poses FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.model_assets
            WHERE model_assets.id = memory_poses.model_id
            AND model_assets.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can insert their own memory poses"
    ON public.memory_poses FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.model_assets
            WHERE model_assets.id = memory_poses.model_id
            AND model_assets.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can update their own memory poses"
    ON public.memory_poses FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.model_assets
            WHERE model_assets.id = memory_poses.model_id
            AND model_assets.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can delete their own memory poses"
    ON public.memory_poses FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.model_assets
            WHERE model_assets.id = memory_poses.model_id
            AND model_assets.user_id = auth.uid()::text
        )
    );

-- Create index for vector search
CREATE INDEX IF NOT EXISTS memory_poses_embedding_idx ON public.memory_poses USING hnsw (embedding vector_cosine_ops);
