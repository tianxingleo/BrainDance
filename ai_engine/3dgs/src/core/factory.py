from src.pipelines.video_3dgs import Video3DGSPipeline
from src.pipelines.image_to_3d import MultiImagePipeline
from src.pipelines.single_image_sam3d import SingleImageSAM3DPipeline

class PipelineFactory:
    @staticmethod
    def get_pipeline(task_type: str, context: dict):
        pipelines = {
            "video_3dgs": Video3DGSPipeline,
            "multi_image": MultiImagePipeline,
            "single_image_sam3d": SingleImageSAM3DPipeline,
        }
        
        pipeline_class = pipelines.get(task_type)
        if not pipeline_class:
            raise ValueError(f"未知的任务类型: {task_type}")
            
        return pipeline_class(context)