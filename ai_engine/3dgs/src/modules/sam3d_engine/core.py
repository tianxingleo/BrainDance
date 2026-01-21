import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

# 引入之前的模块
from .mocks import inject_rtx50_mocks
from .memory import force_cpu_load
from .utils import generate_cpu_config
# 🟢 引入新写的 MaskGenerator
from .masking import MaskGenerator

class SAM3DEngine:
    def __init__(self, repo_path: str, model_dir: Optional[str] = None):
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
        # 如果未指定 model_dir，默认指向 demo 根目录 (即 sam3d_engine 的上两级目录)
        if model_dir is None:
            # core.py 在 demo/SAM3d/sam3d_engine/ 下，所以 parent.parent.parent 是 demo/
            model_dir = Path(__file__).parent.parent.parent 
        self.mask_generator = MaskGenerator(model_dir=model_dir)

    def _setup_path(self):
        # 确保仓库根目录和 notebook 子目录都在 sys.path 中
        repo_str = str(self.repo_path)
        notebook_str = str(self.repo_path / "notebook")
        
        if notebook_str not in sys.path:
            sys.path.insert(0, notebook_str)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def run(self, image_path: str, output_dir: str, mask_path: Optional[str] = None):
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
            
        # 💾 [调试] 保存抠出的 Mask 供查看
        mask_save_path = output_dir / f"{image_path.stem}_mask.png"
        Image.fromarray(mask).save(mask_save_path)
        print(f"🎭 Mask 已保存至: {mask_save_path}")
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

        # ==================== 🛠️ [外科手术] 切除 Mesh 解码器 ====================
        # 原因：我们只生成 3DGS，不需要 Mesh。且 Mesh 解码依赖 Kaolin，环境 Mock 会报错。
        # 操作：用一个 dummy 替换掉它，跳过复杂计算。
        print("🔪 [System] 正在移除冗余的 Mesh 解码器以防止崩溃...")
        class DummyMeshDecoder(torch.nn.Module):
            def forward(self, x, **kwargs):
                return None 

        if "slat_decoder_mesh" in pipeline.models:
            pipeline.models["slat_decoder_mesh"] = DummyMeshDecoder()
            pipeline.models["slat_decoder_mesh"].to('cpu') 
        # ===========================================================================

        # ==================== 🛠️ [外科手术] 屏蔽后处理函数 ====================
        # 原因：Pipeline 还是会尝试读取 mesh 数据，我们需要让它闭嘴。
        # 操作：把 postprocess_slat_output 替换，并在调用原逻辑前删除 mesh 键。
        print("🔧 [System] 正在 Patch 后处理管线...")
        import types
        original_postprocess = pipeline.postprocess_slat_output

        def safe_postprocess(self, outputs, *args, **kwargs):
            # 🛡️ 核心修复：彻底移除 mesh 键，强制 Pipeline 跳过所有针对网格的后处理。
            # 这不仅能防止崩溃，还能确保原始逻辑能够成功走到 "gaussian" -> "gs" 的处理部分。
            if "mesh" in outputs:
                del outputs["mesh"]
            return original_postprocess(outputs, *args, **kwargs)

        pipeline.postprocess_slat_output = types.MethodType(safe_postprocess, pipeline)
        # ===========================================================================

        # 读取图片并限制分辨率 (显存保护)
        pil_image = Image.open(image_path).convert("RGBA")
        orig_w, orig_h = pil_image.size
        
        target_size = 400
        if max(orig_w, orig_h) > target_size:
            scale = target_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"    📉 [显存保护] 图片降采样至: {new_w} x {new_h}")
            
        image_rgb = np.array(pil_image)[:, :, :3]
        
        # 确保 Mask 尺寸与处理后的图片一致
        if mask.shape[0] != image_rgb.shape[0] or mask.shape[1] != image_rgb.shape[1]:
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.NEAREST)
            mask = np.array(mask_pil)

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
            
            # 🟢 [修复] 显式指定解码格式为 gaussian，跳过不需要的格式
            pipeline.decode_formats = ["gaussian"]
            
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

        # 🟢 [移除] 不再需要搬运 slat_decoder_mesh，因为它已被替换为 Dummy 且保持在 CPU
        
        if "slat_condition_embedder" in pipeline.condition_embedders:
            embedder = pipeline.condition_embedders["slat_condition_embedder"]
            if embedder is not None:
                embedder.to('cuda')
