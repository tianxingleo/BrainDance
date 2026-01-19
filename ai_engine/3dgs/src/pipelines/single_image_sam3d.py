from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.pipeline_base import BasePipeline
from src.config import PipelineConfig
from src.modules.sam3d_engine.core import SAM3DEngine

class SingleImageSAM3DPipeline(BasePipeline):
    """单图 -> 3DGS 流水线 (基于 SAM3D)"""
    
    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log(f"🚀 启动 SAM3D 单图重建任务: {input_path}")
        
        work_dir = Path(self.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        config = PipelineConfig()
        repo_path = params.get('repo_path', str(config.sam3d_repo_path))
        model_dir = params.get('model_dir', str(config.sam3d_checkpoint_dir))
        
        self.log(f"⚙️ 初始化 SAM3D 引擎 (Repo: {repo_path})")
        engine = SAM3DEngine(repo_path=repo_path, model_dir=model_dir)
        
        custom_mask = params.get('mask_path')
        self.log("🔥 开始生成 3DGS 模型...")
        ply_path = engine.run(
            image_path=input_path,
            output_dir=str(work_dir),
            mask_path=custom_mask
        )
        
        self.log(f"✅ 3DGS 模型生成完毕: {ply_path}")
        return ply_path, {"engine": "sam3d", "original_image": input_path}
