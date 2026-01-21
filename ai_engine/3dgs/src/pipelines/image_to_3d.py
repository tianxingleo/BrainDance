from src.core.pipeline_base import BasePipeline
from typing import Dict, Any

class MultiImagePipeline(BasePipeline):
    def run(self, input_path: str, params: Dict[str, Any]):
        self.log("🖼️ 启动多图重建流水线...")
        
        # 这里的逻辑不一样：不需要抽帧，直接解压缩 input_path (假设是 zip)
        # 解压到 workspace
        
        use_mask = params.get('use_mask', False)
        if use_mask:
            self.log("🎭 检测到 Mask 需求，正在运行 SAM 分割...")
            # 调用 src/modules/ai_segmentor.py
        
        # 运行 Colmap/Glomap
        # 运行 Nerfstudio
        
        return final_ply_path, metadata