from src.pipelines.video_3dgs import Video3DGSPipeline
from src.pipelines.image_to_3d import MultiImagePipeline

class PipelineFactory:
    @staticmethod
    def get_pipeline(task_type: str, context: dict):
        pipelines = {
            "video_3dgs": Video3DGSPipeline,
            "multi_image": MultiImagePipeline,
            # 未来可以加 "single_image_tripo" 等
        }
        
        pipeline_class = pipelines.get(task_type)
        if not pipeline_class:
            raise ValueError(f"未知的任务类型: {task_type}")
            
        return pipeline_class(context)