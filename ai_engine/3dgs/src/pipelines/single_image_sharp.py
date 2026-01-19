from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.pipeline_base import BasePipeline
from src.config import PipelineConfig
from src.modules.sharp_engine import SharpEngine


class SingleImageSharpPipeline(BasePipeline):
    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log(f"启动 SHARP 单图重建任务: {input_path}")

        work_dir = Path(self.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        config = PipelineConfig()
        repo_path = params.get('repo_path', str(config.sharp_repo_path))

        self.log(f"初始化 SHARP 引擎 (Repo: {repo_path})")
        engine = SharpEngine(repo_path=repo_path)

        self.log("开始生成 3DGS 模型...")
        generated_ply = engine.run(input_path, str(work_dir))

        self.log(f"3DGS 模型生成完毕: {generated_ply}")

        return generated_ply, {"engine": "sharp", "original_image": input_path}
