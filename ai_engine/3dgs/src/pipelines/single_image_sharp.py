from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.pipeline_base import BasePipeline
from src.config import PipelineConfig
from src.modules.sharp_engine import SharpEngine
from src.utils.ply_utils import rotate_gaussian_ply_x180
# 🟢 引入场景分析器，用于单图 RAG 注入
from src.modules.scene_analyzer import SceneAnalyzer
import os
try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None


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

        self.log("正在修正 SHARP 输出模型朝向（绕 X 轴旋转 180°）...")
        generated_ply = rotate_gaussian_ply_x180(generated_ply)
        self.log(f"模型朝向修正完成: {generated_ply}")

        # 使用基类提供的 helper 执行 RAG 分析与上传/入库（方法内部保证容错）
        metadata = {"engine": "sharp", "original_image": input_path, "preview_img_path": input_path}
        try:
            rag_meta = self.run_rag_analysis(input_path)
            metadata.update(rag_meta)
        except Exception:
            pass

        try:
            self.upload_and_record(generated_ply, metadata, params)
        except Exception:
            pass

        return generated_ply, metadata
