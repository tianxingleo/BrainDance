from pathlib import Path
from typing import Dict, Any, Tuple
from src.core.pipeline_base import BasePipeline
from src.config import PipelineConfig
from src.modules.sam3d_engine.core import SAM3DEngine
from src.modules.scene_analyzer import SceneAnalyzer
from src.utils.ply_utils import rotate_ply_vertices_euler
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
        # 兼容旧参数：历史上 model_dir 同时被用作 checkpoint 目录
        checkpoint_dir = (
            params.get('checkpoint_dir')
            or params.get('sam3d_checkpoint_dir')
            or params.get('model_dir')
            or str(config.sam3d_checkpoint_dir)
        )
        mask_model_dir = (
            params.get('mask_model_dir')
            or params.get('shared_model_dir')
            or str(config.shared_model_dir)
        )
        
        self.log(
            f"⚙️ 初始化 SAM3D 引擎 (Repo: {repo_path}, "
            f"Checkpoint: {checkpoint_dir}, MaskModel: {mask_model_dir})"
        )
        engine = SAM3DEngine(
            repo_path=repo_path,
            checkpoint_dir=checkpoint_dir,
            mask_model_dir=mask_model_dir,
            model_dir=params.get('model_dir'),
        )
        
        custom_mask = params.get('mask_path')
        self.log("🔥 开始生成 3DGS 模型...")
        ply_path = engine.run(
            image_path=input_path,
            output_dir=str(work_dir),
            mask_path=custom_mask
        )

        # 生成后执行朝向修正：默认绕 X 轴旋转 -90°，让单图模型更接近输入照片视角。
        rotate_enabled = params.get("post_rotate_enabled", True)
        if rotate_enabled:
            rx = float(params.get("post_rotate_deg_x", -90.0))
            ry = float(params.get("post_rotate_deg_y", 0.0))
            rz = float(params.get("post_rotate_deg_z", 0.0))
            try:
                self.log(f"🧭 [Post] 正在旋转模型: rx={rx}, ry={ry}, rz={rz}")
                ply_path = rotate_ply_vertices_euler(
                    ply_path=ply_path,
                    rx_deg=rx,
                    ry_deg=ry,
                    rz_deg=rz,
                )
                self.log("    -> ✅ 模型朝向修正完成")
            except Exception as e:
                self.log(f"    -> ⚠️ 模型旋转失败，保留原始结果: {e}", level="WARN")
        
        self.log(f"✅ 3DGS 模型生成完毕: {ply_path}")
        # ==================== 🟢 [新增] 单图 RAG 语义注入 ====================
        rag_meta = {}
        try:
            cfg = PipelineConfig()
            analyzer = SceneAnalyzer(cfg)
            self.log("🧠 [RAG] 开始对单张图片进行语义分析...")
            analysis = analyzer.analyze_single_image(input_path, log_callback=self.log)

            if analysis.get("ok"):
                rag_meta = {
                    "ai_description": analysis.get("description", ""),
                    "ai_tags": analysis.get("tags", []),
                    "ai_objects": analysis.get("objects", []),
                    "ai_reason": analysis.get("reason", "")
                }
                if analysis.get("score") is not None:
                    rag_meta["ai_score"] = analysis.get("score")
                self.log(f"    -> 🏷️ Tags: {rag_meta.get('ai_tags', [])}")
                self.log("    -> ✅ 单图 RAG 分析完成")
            else:
                self.log(
                    f"    -> ⚠️ 单图 RAG 分析未产出有效结果，跳过元数据回填: {analysis.get('reason', 'Unknown')}",
                    level="WARN",
                )

        except Exception as e:
            self.log(f"    -> ⚠️ RAG 分析失败，已跳过: {e}", level="WARN")

        metadata = {"engine": "sam3d", "original_image": input_path, "preview_img_path": input_path}
        metadata.update(rag_meta)

        # 单图流水线已经显式执行过一次 RAG，失败时不要重复请求同一张图。

        try:
            self.upload_and_record(ply_path, metadata, params)
        except Exception:
            # upload_and_record 内部已捕获异常，不应抛出
            pass

        return ply_path, metadata
