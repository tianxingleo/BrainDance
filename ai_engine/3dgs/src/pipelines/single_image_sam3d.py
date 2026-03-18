from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.pipeline_base import BasePipeline
from src.config import PipelineConfig
from src.modules.sam3d_engine.core import SAM3DEngine
from src.modules.scene_analyzer import SceneAnalyzer
import os
try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None

class SingleImageSAM3DPipeline(BasePipeline):
    """单图 -> 3DGS 流水线 (基于 SAM3D)"""
    
    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log(f"🚀 启动 SAM3D 单图重建任务: {input_path}")
        
        work_dir = Path(self.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        config = PipelineConfig()
        repo_path = params.get('repo_path', str(config.sam3d_repo_path))
        model_dir = params.get('model_dir', str(config.shared_model_dir))
        
        self.log(f"⚙️ 初始化 SAM3D 引擎 (Repo: {repo_path}, Model: {model_dir})")
        engine = SAM3DEngine(repo_path=repo_path, model_dir=model_dir)
        
        custom_mask = params.get('mask_path')
        self.log("🔥 开始生成 3DGS 模型...")
        ply_path = engine.run(
            image_path=input_path,
            output_dir=str(work_dir),
            mask_path=custom_mask
        )
        
        self.log(f"✅ 3DGS 模型生成完毕: {ply_path}")
        # ==================== 🟢 [新增] 单图 RAG 语义注入 ====================
        rag_meta = {}
        try:
            cfg = PipelineConfig()
            analyzer = SceneAnalyzer(cfg)
            self.log("🧠 [RAG] 开始对单张图片进行语义分析...")
            analysis = analyzer.analyze_single_image(input_path)

            rag_meta = {
                "ai_description": analysis.get("description", ""),
                "ai_tags": analysis.get("tags", []),
                "ai_objects": analysis.get("objects", []),
                "ai_score": analysis.get("score", 0),
                "ai_reason": analysis.get("reason", "")
            }
            self.log(f"    -> 🏷️ Tags: {rag_meta['ai_tags']}")
            self.log("    -> ✅ 单图 RAG 分析完成")

        except Exception as e:
            self.log(f"    -> ⚠️ RAG 分析失败，已跳过: {e}", level="WARN")

        metadata = {"engine": "sam3d", "original_image": input_path, "preview_img_path": input_path}
        metadata.update(rag_meta)

        # 使用基类的 helper 进行 RAG 语义注入与上传（封装了 supabase 的容错）
        try:
            rag_meta = self.run_rag_analysis(input_path)
            metadata.update(rag_meta)
        except Exception:
            pass

        try:
            self.upload_and_record(ply_path, metadata, params)
        except Exception:
            # upload_and_record 内部已捕获异常，不应抛出
            pass

        return ply_path, metadata
