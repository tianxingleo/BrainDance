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
📥 [接收任务] 场景ID: test_scene_sharp_1768839315 | 任务ID: a6699632-ad25-49c1-8196-13091c369ee8
[test_scene_sharp_1768839315] 正在从云端下载资源...
🔧 检测到任务类型: single_image_sharp
[test_scene_sharp_1768839315] 下载单张图片...
Storage endpoint URL should have a trailing slash.
❌ 任务处理失败: 资源下载失败 (路径: test_user/test_scene_sharp_1768839315/raw/image.png): {'statusCode': 500, 'error': Internal, 'message': Internal Server Error}
🗑️ 已删除临时文件: test_scene_sharp_1768839315.png
.............................................................
📥 [接收任务] 场景ID: scene_party_001 | 任务ID: 08e875a6-17bf-4512-9b9f-9fd4432fa405
[scene_party_001] 正在从云端下载资源...
🔧 检测到任务类型: video_3dgs
[scene_party_001] 下载视频...
❌ 任务处理失败: 资源下载失败 (路径: test1/scene_party_001/raw/video.mp4): {'statusCode': 500, 'error': Internal, 'message': Internal Server Error}
🗑️ 已删除临时文件: scene_party_001.mp4
....................................
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
