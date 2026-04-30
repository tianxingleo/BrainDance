from src.core.pipeline_base import BasePipeline
from typing import Dict, Any

class MultiImagePipeline(BasePipeline):
    def run(self, input_path: str, params: Dict[str, Any]):
        raise NotImplementedError(
            "MultiImagePipeline 尚未实现，请使用 single_image_sam3d / single_image_sharp / video_3dgs 等已实现的 task_type"
        )