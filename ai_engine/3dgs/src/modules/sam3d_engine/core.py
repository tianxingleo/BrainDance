import json
import os
import sys
import gc
import torch
import types
import numpy as np
import importlib.util
import builtins
import shutil
import subprocess
from pathlib import Path
from PIL import Image
from typing import Optional, Iterable

# 引入之前的模块
from .mocks import inject_rtx50_mocks
from .memory import force_cpu_load
from .utils import generate_cpu_config
# 🟢 引入新写的 MaskGenerator
from .masking import MaskGenerator

class SAM3DEngine:
    DIRECT_PIPELINE_VRAM_THRESHOLD_GB = 16.0

    def __init__(
        self,
        repo_path: str,
        checkpoint_dir: Optional[str] = None,
        mask_model_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        """
        :param repo_path: sam-3d-objects 仓库路径
        :param checkpoint_dir: SAM3D checkpoint 目录（可传 .../checkpoints 或 .../checkpoints/hf）
        :param mask_model_dir: AI 抠图模型路径 (yolo/sam)
        :param model_dir: 兼容旧参数，历史上被用于传 checkpoint 路径
        """
        self.repo_path = Path(repo_path)
        self.config_path = self._resolve_config_path(
            checkpoint_dir=checkpoint_dir,
            legacy_model_dir=model_dir,
        )
        
        # 1. 注入环境
        inject_rtx50_mocks()
        self._setup_path()
        os.chdir(self.repo_path)
        
        # 2. 初始化抠图器
        if mask_model_dir is None:
            # 与历史行为兼容：老参数 model_dir 有时被拿来当抠图模型目录
            mask_model_dir = model_dir
        if mask_model_dir is None:
            # core.py 在 src/modules/sam3d_engine/ 下；回退到 3dgs 根目录
            mask_model_dir = Path(__file__).resolve().parents[3]
        self.mask_generator = MaskGenerator(model_dir=mask_model_dir)

    def _resolve_config_path(self, checkpoint_dir: Optional[str], legacy_model_dir: Optional[str]) -> Path:
        search_roots = []
        if checkpoint_dir:
            search_roots.append(Path(checkpoint_dir).expanduser())

        env_checkpoint_dir = os.getenv("SAM3D_CHECKPOINT_DIR", "").strip()
        if env_checkpoint_dir:
            search_roots.append(Path(env_checkpoint_dir).expanduser())

        if legacy_model_dir:
            search_roots.append(Path(legacy_model_dir).expanduser())

        # 默认路径：支持仓库内和外部模型目录两种结构
        search_roots.extend(
            [
                self.repo_path / "checkpoints/hf",
                self.repo_path / "checkpoints",
                self.repo_path,
            ]
        )
        search_roots.extend(self._discover_system_checkpoint_roots())

        candidates = list(self._iter_config_candidates(search_roots))
        for path in candidates:
            if path.is_file():
                return path

        tried = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            "找不到 SAM3D 配置文件 pipeline.yaml。\n"
            "请检查 `checkpoint_dir` / `SAM3D_CHECKPOINT_DIR` 是否正确。\n"
            "已尝试路径:\n"
            f"{tried}"
        )

    def _discover_system_checkpoint_roots(self):
        roots = []
        # 常见模型目录
        for base in (
            Path("/ltx-data/BrainDance/models"),
            Path("/ltx-data/BrainDance/ai_engine/models"),
            Path("/ltx-data/models"),
            Path.home() / "braindance_workspace/models",
            Path.home() / "workspace/ai",
        ):
            if not base.exists():
                continue
            roots.extend(
                [
                    base / "sam3d/checkpoints",
                    base / "sam3d/checkpoints/hf",
                    base / "checkpoints/sam3d",
                    base,
                ]
            )
        roots.extend(self._discover_editable_install_checkpoint_roots())
        return roots

    def _discover_editable_install_checkpoint_roots(self):
        roots = []

        # 优先复用本机已经通过 editable install 安装的 sam3d_objects 源目录。
        # 这类目录通常已经带有完整的 checkpoints/hf 资源。
        for direct_url in Path.home().glob(".local/lib/python*/site-packages/sam3d_objects-*.dist-info/direct_url.json"):
            if not direct_url.is_file():
                continue
            try:
                payload = json.loads(direct_url.read_text(encoding="utf-8"))
                url = payload.get("url", "")
                if url.startswith("file://"):
                    repo_path = Path(url.removeprefix("file://")).expanduser()
                    roots.extend(
                        [
                            repo_path / "checkpoints/hf",
                            repo_path / "checkpoints",
                            repo_path,
                        ]
                    )
            except Exception:
                pass

        try:
            spec = importlib.util.find_spec("sam3d_objects")
            if spec and spec.origin:
                package_root = Path(spec.origin).resolve().parent
                repo_path = package_root.parent
                roots.extend(
                    [
                        repo_path / "checkpoints/hf",
                        repo_path / "checkpoints",
                        repo_path,
                    ]
                )
        except Exception:
            pass

        return roots

    def _iter_config_candidates(self, roots: Iterable[Path]):
        seen = set()
        for root in roots:
            root = root.resolve()
            if root.suffix == ".yaml":
                candidate_list = [root]
            else:
                candidate_list = [
                    root / "pipeline.yaml",
                    root / "hf/pipeline.yaml",
                    root / "checkpoints/hf/pipeline.yaml",
                ]
            for candidate in candidate_list:
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                yield candidate

    def _setup_path(self):
        # 确保仓库根目录和 notebook 子目录都在 sys.path 中
        repo_str = str(self.repo_path)
        notebook_str = str(self.repo_path / "notebook")
        
        if notebook_str not in sys.path:
            sys.path.insert(0, notebook_str)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _load_inference_class(self):
        """
        优先复用上游 notebook/inference.py 的公开接口。
        若其在导入阶段因为 kaolin / gradio 等可视化依赖失败，则回退到仅保留推理能力的轻量封装。
        """
        try:
            from inference import Inference

            return Inference
        except Exception as exc:
            notebook_path = self.repo_path / "notebook" / "inference.py"
            if not notebook_path.is_file():
                raise ImportError(
                    f"无法找到 SAM3D 推理入口: {notebook_path}"
                ) from exc

            print(
                "⚠️ [SAM3D] notebook/inference.py 导入失败，"
                f"将回退到轻量推理封装: {type(exc).__name__}: {exc}"
            )
            return self._build_lightweight_inference_class()

    def _build_lightweight_inference_class(self):
        from hydra.utils import get_method, instantiate
        from omegaconf import DictConfig, ListConfig, OmegaConf
        from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

        whitelist_filters = [
            lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
        ]
        blacklist_filters = [
            lambda target: get_method(target)
            in {
                builtins.exec,
                builtins.eval,
                builtins.__import__,
                os.kill,
                os.system,
                os.putenv,
                os.remove,
                os.removedirs,
                os.rmdir,
                os.fchdir,
                os.setuid,
                os.fork,
                os.forkpty,
                os.killpg,
                os.rename,
                os.renames,
                os.truncate,
                os.replace,
                os.unlink,
                os.fchmod,
                os.fchown,
                os.chmod,
                os.chown,
                os.chroot,
                os.lchown,
                os.getcwd,
                os.chdir,
                shutil.rmtree,
                shutil.move,
                shutil.chown,
                subprocess.Popen,
                builtins.help,
            },
        ]

        def check_target(target: str):
            if any(filt(target) for filt in whitelist_filters) and not any(
                filt(target) for filt in blacklist_filters
            ):
                return
            raise RuntimeError(
                f"target '{target}' 不允许被 hydra 实例化，请检查配置来源。"
            )

        def check_hydra_safety(config: DictConfig):
            to_check = [config]
            while to_check:
                node = to_check.pop()
                if isinstance(node, DictConfig):
                    to_check.extend(list(node.values()))
                    if "_target_" in node:
                        check_target(node["_target_"])
                elif isinstance(node, ListConfig):
                    to_check.extend(list(node))

        class LightweightInference:
            """只保留推理初始化能力，避免 notebook 侧可视化依赖阻塞主链路。"""

            def __init__(self, config_file: str, compile: bool = False):
                config = OmegaConf.load(config_file)
                config.rendering_engine = "pytorch3d"
                config.compile_model = compile
                config.workspace_dir = os.path.dirname(config_file)
                check_hydra_safety(config)
                self._pipeline: InferencePipelinePointMap = instantiate(config)

        return LightweightInference

    def run(self, image_path: str, output_dir: str, mask_path: Optional[str] = None):
        inference = None
        pipeline = None
        output = None
        image_rgb = None

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

        # 延迟 import (避免过早加载 torch)
        try:
            Inference = self._load_inference_class()
        except Exception as exc:
            raise ImportError(
                "SAM3D 推理入口加载失败。"
                f"\n仓库路径: {self.repo_path}"
                f"\nnotebook 路径: {self.repo_path / 'notebook' / 'inference.py'}"
                f"\n原始异常: {type(exc).__name__}: {exc}"
            ) from exc

        total_vram_gb = self._get_total_vram_gb()
        use_direct_pipeline = total_vram_gb is not None and total_vram_gb > self.DIRECT_PIPELINE_VRAM_THRESHOLD_GB

        if use_direct_pipeline:
            print(
                f"🧠 [SAM3D] 检测到 GPU 总显存 {total_vram_gb:.2f}GB > "
                f"{self.DIRECT_PIPELINE_VRAM_THRESHOLD_GB:.0f}GB，跳过显存优化，走原生直连 Pipeline..."
            )
            inference = Inference(str(self.config_path))
            pipeline = inference._pipeline
        else:
            if total_vram_gb is None:
                print("🧠 [SAM3D] 未能获取 GPU 总显存，默认启用显存优化 Pipeline...")
            else:
                print(
                    f"🧠 [SAM3D] 检测到 GPU 总显存 {total_vram_gb:.2f}GB <= "
                    f"{self.DIRECT_PIPELINE_VRAM_THRESHOLD_GB:.0f}GB，启用显存优化 Pipeline..."
                )
            cpu_config = generate_cpu_config(self.config_path)
            with force_cpu_load():
                inference = Inference(str(cpu_config))
                pipeline = inference._pipeline
                pipeline.device = torch.device("cuda")

        self._patch_pipeline_for_gaussian_only(pipeline)

        image_rgb, mask = self._prepare_inputs(
            image_path=image_path,
            mask=mask,
            enable_vram_optimization=not use_direct_pipeline,
        )

        try:
            if use_direct_pipeline:
                output = self._run_direct_pipeline(pipeline, image_rgb, mask)
            else:
                output = self._run_memory_optimized_pipeline(pipeline, image_rgb, mask)

            # 保存
            if "gs" in output:
                ply_path = output_dir / f"{image_path.stem}_3dgs.ply"
                output["gs"].save_ply(str(ply_path))
                print(f"💾 结果已保存: {ply_path}")
                return str(ply_path)

            raise RuntimeError("SAM3D 输出中未找到 gs 结果")
            
        finally:
            self._release_pipeline_resources(pipeline)
            del output
            del image_rgb
            del pipeline
            del inference
            del mask
            gc.collect()
            self._release_cuda_memory("run-finally")

    def _get_total_vram_gb(self) -> Optional[float]:
        if not torch.cuda.is_available():
            return None

        try:
            device_index = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_index).total_memory
            return total_memory / (1024 ** 3)
        except Exception as exc:
            print(f"⚠️ [SAM3D] 获取 GPU 显存失败，将回退到显存优化模式: {exc}")
            return None

    def _patch_pipeline_for_gaussian_only(self, pipeline):
        # 只保留 Gaussian 输出，避免 mesh 相关依赖和后处理崩溃。
        print("🔪 [System] 正在移除冗余的 Mesh 解码器以防止崩溃...")

        class DummyMeshDecoder(torch.nn.Module):
            def forward(self, x, **kwargs):
                return None

        if "slat_decoder_mesh" in pipeline.models:
            pipeline.models["slat_decoder_mesh"] = DummyMeshDecoder()
            pipeline.models["slat_decoder_mesh"].to("cpu")

        print("🔧 [System] 正在 Patch 后处理管线...")
        original_postprocess = pipeline.postprocess_slat_output

        def safe_postprocess(self, outputs, *args, **kwargs):
            if "mesh" in outputs:
                del outputs["mesh"]
            return original_postprocess(outputs, *args, **kwargs)

        pipeline.postprocess_slat_output = types.MethodType(safe_postprocess, pipeline)
        pipeline.decode_formats = ["gaussian"]

    def _prepare_inputs(self, image_path: Path, mask: np.ndarray, enable_vram_optimization: bool):
        pil_image = Image.open(image_path).convert("RGBA")
        orig_w, orig_h = pil_image.size

        if enable_vram_optimization:
            target_size = 1920
            if max(orig_w, orig_h) > target_size:
                scale = target_size / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                print(f"    📉 [显存保护] 图片降采样至: {new_w} x {new_h}")

        image_rgb = np.array(pil_image)[:, :, :3]

        if mask.shape[0] != image_rgb.shape[0] or mask.shape[1] != image_rgb.shape[1]:
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.NEAREST)
            mask = np.array(mask_pil)

        return image_rgb, mask

    def _run_direct_pipeline(self, pipeline, image_rgb: np.ndarray, mask: np.ndarray):
        print("🚀 [Direct Pipeline] 直接运行 SAM3D Pipeline...")
        return pipeline.run(
            image=image_rgb,
            mask=mask,
            stage1_only=False,
            seed=42,
            with_mesh_postprocess=False,
            with_texture_baking=False,
        )

    def _run_memory_optimized_pipeline(self, pipeline, image_rgb: np.ndarray, mask: np.ndarray):
        print("🚀 [Stage 1] 生成结构...")
        self._move_stage1_to_gpu(pipeline)
        stage1_out = pipeline.run(
            image=image_rgb,
            mask=mask,
            stage1_only=True,
            seed=42,
        )

        print("🚀 [Stage 2] 生成 3DGS...")
        self._switch_stage1_to_stage2(pipeline)

        original_sample = pipeline.sample_sparse_structure
        pipeline.sample_sparse_structure = lambda *args, **kwargs: stage1_out

        try:
            return pipeline.run(
                image=image_rgb,
                mask=mask,
                stage1_only=False,
                seed=42,
                with_mesh_postprocess=False,
                with_texture_baking=False,
            )
        finally:
            pipeline.sample_sparse_structure = original_sample
            del stage1_out
            
    def _move_stage1_to_gpu(self, pipeline):
        if torch.cuda.is_available():
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
                
        if torch.cuda.is_available():
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

    def _release_pipeline_resources(self, pipeline):
        if pipeline is None:
            return

        # 强制把模型组件从 GPU 挪回 CPU，避免显存残留占用到下个任务。
        try:
            models = getattr(pipeline, "models", None)
            if isinstance(models, dict):
                for module in models.values():
                    self._safe_module_to_cpu(module)
        except Exception as exc:
            print(f"⚠️ [SAM3D] 释放 models 失败: {exc}")

        try:
            embedders = getattr(pipeline, "condition_embedders", None)
            if isinstance(embedders, dict):
                for module in embedders.values():
                    self._safe_module_to_cpu(module)
        except Exception as exc:
            print(f"⚠️ [SAM3D] 释放 condition_embedders 失败: {exc}")

        try:
            if hasattr(pipeline, "device"):
                pipeline.device = torch.device("cpu")
        except Exception:
            pass

    def _safe_module_to_cpu(self, module):
        if module is None:
            return
        try:
            if hasattr(module, "to"):
                module.to("cpu")
            elif hasattr(module, "cpu"):
                module.cpu()
        except Exception:
            pass

    def _release_cuda_memory(self, phase: str):
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print(f"🧹 [SAM3D] 显存清理完成: {phase}")
