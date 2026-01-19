import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# 引入之前的模块
from .mocks import inject_rtx50_mocks
from .memory import force_cpu_load
from .utils import generate_cpu_config
# 🟢 引入新写的 MaskGenerator
from .masking import MaskGenerator

class SAM3DEngine:
    def __init__(self, repo_path: str, model_dir: str = None):
        """
        :param repo_path: sam-3d-objects 仓库路径
        :param model_dir: AI 模型路径 (yolo/sam)，默认为 repo_path 同级
        """
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / "checkpoints/hf/pipeline.yaml"
        
        # 1. 注入环境
        inject_rtx50_mocks()
        self._setup_path()
        os.chdir(self.repo_path)
        
        # 2. 初始化抠图器
        # 如果未指定 model_dir，默认假设在当前脚本运行目录的 models 下，或者你指定的公共目录
        if model_dir is None:
            # 假设模型在 demo 根目录
            model_dir = Path(__file__).parent.parent 
        self.mask_generator = MaskGenerator(model_dir=model_dir)

    def _setup_path(self):
        # 确保仓库根目录和 notebook 子目录都在 sys.path 中
        repo_str = str(self.repo_path)
        notebook_str = str(self.repo_path / "notebook")
        
        if notebook_str not in sys.path:
            sys.path.insert(0, notebook_str)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def run(self, image_path: str, output_dir: str, mask_path: str = None):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 🟢 自动抠图逻辑 Start ---
        if mask_path and Path(mask_path).exists():
            print(f"🎭 使用指定 Mask: {mask_path}")
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            # 调用 MaskGenerator 自动生成
            # 默认使用 smart 模式，如果模型不在会自动降级
            mask = self.mask_generator.get_mask(image_path, method="smart")
        # --- 🟢 自动抠图逻辑 End ---

        # 准备配置
        cpu_config = generate_cpu_config(self.config_path)
        
        # 延迟 import (避免过早加载 torch)
        try:
            from inference import Inference
        except ImportError:
            raise ImportError(f"无法从 {self.repo_path} 导入 inference，请检查路径")

        # 初始化模型 (Force CPU RAM)
        print("🧠 [SAM3D] 初始化生成模型...")
        with force_cpu_load():
            inference = Inference(str(cpu_config))
            pipeline = inference._pipeline
            pipeline.device = torch.device('cuda')

        # 读取图片
        pil_image = Image.open(image_path).convert("RGBA")
        image_rgb = np.array(pil_image)[:, :, :3]

        try:
            # Stage 1
            print("🚀 [Stage 1] 生成结构...")
            self._move_stage1_to_gpu(pipeline)
            stage1_out = pipeline.run(
                image=image_rgb, 
                mask=mask,  # 🟢 传入生成的 Mask
                stage1_only=True, 
                seed=42
            )
            
            # Stage 2
            print("🚀 [Stage 2] 生成 3DGS...")
            self._switch_stage1_to_stage2(pipeline)
            
            original_sample = pipeline.sample_sparse_structure
            pipeline.sample_sparse_structure = lambda *args, **kwargs: stage1_out
            
            try:
                output = pipeline.run(
                    image=image_rgb, 
                    mask=mask, # 🟢 传入生成的 Mask
                    stage1_only=False, 
                    seed=42,
                    with_mesh_postprocess=False, 
                    with_texture_baking=False
                )
            finally:
                pipeline.sample_sparse_structure = original_sample

            # 保存
            if "gs" in output:
                ply_path = output_dir / f"{image_path.stem}_3dgs.ply"
                output["gs"].save_ply(str(ply_path))
                print(f"💾 结果已保存: {ply_path}")
                return str(ply_path)
            
        finally:
            torch.cuda.empty_cache()
            
    def _move_stage1_to_gpu(self, pipeline):
        torch.cuda.empty_cache()
        
        # 必须检查 value 是否不为 None
        if "ss_generator" in pipeline.models and pipeline.models["ss_generator"] is not None:
            pipeline.models["ss_generator"].to('cuda')
            
        if "ss_decoder" in pipeline.models and pipeline.models["ss_decoder"] is not None:
            pipeline.models["ss_decoder"].to('cuda')

        if "ss_encoder" in pipeline.models and pipeline.models["ss_encoder"] is not None:
            pipeline.models["ss_encoder"].to('cuda')

        if "ss_condition_embedder" in pipeline.condition_embedders:
            embedder = pipeline.condition_embedders["ss_condition_embedder"]
            if embedder is not None:
                embedder.to('cuda')

    def _switch_stage1_to_stage2(self, pipeline):
        # 1. 卸载 Stage 1
        if "ss_generator" in pipeline.models and pipeline.models["ss_generator"] is not None:
            pipeline.models["ss_generator"].cpu()
            
        if "ss_decoder" in pipeline.models and pipeline.models["ss_decoder"] is not None:
            pipeline.models["ss_decoder"].cpu()

        if "ss_encoder" in pipeline.models and pipeline.models["ss_encoder"] is not None:
            pipeline.models["ss_encoder"].cpu()
            
        if "ss_condition_embedder" in pipeline.condition_embedders:
            embedder = pipeline.condition_embedders["ss_condition_embedder"]
            if embedder is not None:
                embedder.cpu()
                
        torch.cuda.empty_cache()
        
        # 2. 加载 Stage 2
        if "slat_generator" in pipeline.models and pipeline.models["slat_generator"] is not None:
            pipeline.models["slat_generator"].to('cuda')
            
        if "slat_decoder_gs" in pipeline.models and pipeline.models["slat_decoder_gs"] is not None:
            pipeline.models["slat_decoder_gs"].to('cuda')

        if "slat_condition_embedder" in pipeline.condition_embedders:
            embedder = pipeline.condition_embedders["slat_condition_embedder"]
            if embedder is not None:
                embedder.to('cuda')
