from src.pipelines.video_3dgs import Video3DGSPipeline
from src.pipelines.image_to_3d import MultiImagePipeline
from src.pipelines.single_image_sam3d import SingleImageSAM3DPipeline
from src.pipelines.single_image_sharp import SingleImageSharpPipeline
from src.pipelines.da3_feed_forward_pipeline import DA3FeedForwardPipeline

class PipelineFactory:
    @staticmethod
    def get_pipeline(task_type: str, context: dict):
        pipelines = {
            "video_3dgs": Video3DGSPipeline,
            # video_dual_chain 由 worker 负责编排（快链+慢链），这里兜底映射到传统视频链
            "video_dual_chain": Video3DGSPipeline,
            "multi_image": MultiImagePipeline,
            "single_image_sam3d": SingleImageSAM3DPipeline,
            "single_image_sharp": SingleImageSharpPipeline,
            "da3_feed_forward_3dgs": DA3FeedForwardPipeline,
        }
        
        pipeline_class = pipelines.get(task_type)
        if not pipeline_class:
            raise ValueError(f"未知的任务类型: {task_type}")
            
        return pipeline_class(context)
