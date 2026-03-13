import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

from src.core.pipeline_base import BasePipeline
from src.config import PipelineConfig
from src.modules.image_proc import ImageProcessor
from src.modules.scene_analyzer import SceneAnalyzer
from src.modules.da3_runner import DA3Runner
from src.utils.common import format_duration


class DA3SuGaRPipeline(BasePipeline):
    """
    视频 -> DA3 -> SuGaR 流水线。
    """

    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log("🎬 启动 DA3 + SuGaR 流水线...")
        self.log(f"📄 输入文件: {input_path}")

        video_path_obj = Path(input_path)
        cfg = PipelineConfig(
            project_name=self.scene_id,
            video_path=video_path_obj,
            mapper_type="da3",
        )
        cfg.project_dir = Path(self.work_dir)
        if params.get("da3_repo_path"):
            cfg.da3_repo_path = Path(str(params["da3_repo_path"]))
        if "enable_scene_analysis" in params:
            cfg.enable_scene_analysis = bool(params["enable_scene_analysis"])
        if params.get("skip_scene_analysis"):
            cfg.enable_scene_analysis = False

        global_start_time = time.time()
        pipeline_metadata: Dict[str, Any] = {}

        img_processor = ImageProcessor(cfg, log_callback=self.log)
        scene_analyzer = SceneAnalyzer(cfg)
        da3_runner = DA3Runner(cfg, log_callback=self.log)

        self.log("🎬 [1/3] 开始视频抽帧与图片预处理...")
        cfg.project_dir.mkdir(parents=True, exist_ok=True)

        dest_video_path = cfg.project_dir / video_path_obj.name
        if not dest_video_path.exists():
            shutil.copy(str(video_path_obj), str(dest_video_path))

        temp_dir = cfg.project_dir / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(dest_video_path),
                    "-vf",
                    "fps=1,scale=1920:1920:force_original_aspect_ratio=decrease:flags=lanczos",
                    "-q:v",
                    "2",
                    "-map_metadata",
                    "-1",
                    str(temp_dir / "frame_%05d.jpg"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg 抽帧失败: {e}") from e

        img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)

        raw_images_dir = cfg.project_dir / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = sorted(list(temp_dir.glob("*")))
        limit = cfg.max_images
        if len(all_imgs) > limit:
            indices = np.linspace(0, len(all_imgs) - 1, limit, dtype=int)
            all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]

        for img in all_imgs:
            shutil.copy2(str(img), str(raw_images_dir / img.name))
        shutil.rmtree(temp_dir)
        self.log(f"    -> 图片准备完成，共 {len(all_imgs)} 张")

        if cfg.enable_scene_analysis:
            self.log(f"🧐 [AI 质检] 阈值: {cfg.min_quality_score} 分")
            passed, score, reason, tags, description, objects = scene_analyzer.run(
                raw_images_dir,
                log_callback=lambda msg: self.log(msg),
            )
            pipeline_metadata.update(
                {
                    "ai_score": score,
                    "ai_tags": tags,
                    "ai_reason": reason,
                    "ai_description": description,
                    "ai_objects": objects,
                }
            )
            if not passed:
                err_msg = f"AI 质检不通过 ({score}分 < {cfg.min_quality_score}分): {reason}"
                self.log(err_msg, level="ERROR")
                raise RuntimeError(err_msg)

        self.log("⚙️ [2/3] 正在执行 DA3 位姿与深度解算...")
        if not da3_runner.run():
            err_msg = "❌ Pipeline 中断：DA3 解算失败"
            self.log(err_msg, level="ERROR")
            raise RuntimeError(err_msg)

        da3_output_dir = cfg.data_dir / "colmap" / "da3_output"
        if not da3_output_dir.exists():
            raise FileNotFoundError(f"DA3 输出目录不存在: {da3_output_dir}")

        self.log("🧠 [3/3] 开始执行 SuGaR 训练脚本...")
        da3_repo = self._resolve_da3_repo(cfg, params)
        sugar_repo = self._resolve_sugar_repo(params)
        script_path = da3_repo / "da3_to_sugar_pipeline.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"找不到脚本: {script_path}")

        scene_name = str(params.get("sugar_scene_name", self.scene_id))
        regularization = str(params.get("regularization", "dn_consistency"))
        refinement_time = str(params.get("refinement_time", "short"))
        high_poly = self._to_bool_str(params.get("high_poly", False))
        fast_mode = self._to_bool_str(params.get("fast_mode", True))

        cmd = [
            "bash",
            str(script_path),
            str(da3_output_dir),
            scene_name,
            regularization,
            refinement_time,
            high_poly,
            fast_mode,
        ]

        env = os.environ.copy()
        env["DA3_DIR"] = str(da3_repo)
        env["SUGAR_DIR"] = str(sugar_repo)

        if "gpu_index" in params:
            env["CUDA_VISIBLE_DEVICES"] = str(params["gpu_index"])
        elif not env.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)

        if params.get("sugar_conda_env"):
            env["CONDA_ENV"] = str(params["sugar_conda_env"])
            env["USE_CONDA"] = "true"
        if params.get("conda_base"):
            env["CONDA_BASE"] = str(params["conda_base"])

        self.log(f"    => 运行脚本: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(da3_repo),
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                self.log(f"    | {line}")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"SuGaR 脚本执行失败，退出码: {process.returncode}")

        final_ply_path = self._find_best_ply(sugar_repo, scene_name)
        if final_ply_path is None:
            final_ply_path = self._export_coarse_checkpoint_if_needed(sugar_repo, scene_name, env)
        if final_ply_path is None or not final_ply_path.exists():
            raise FileNotFoundError("SuGaR 执行完成，但未找到可交付的 PLY 输出")

        self.log(f"💾 导出 PLY 完成: {final_ply_path}")
        self.log(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")

        try:
            self.upload_and_record(str(final_ply_path), pipeline_metadata, params)
        except Exception:
            pass

        return str(final_ply_path), pipeline_metadata

    @staticmethod
    def _to_bool_str(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip().lower()

    def _resolve_da3_repo(self, cfg: PipelineConfig, params: Dict[str, Any]) -> Path:
        candidate_list: List[Path] = []
        if params.get("da3_repo_path"):
            candidate_list.append(Path(str(params["da3_repo_path"])))
        if os.getenv("DA3_REPO_PATH"):
            candidate_list.append(Path(os.getenv("DA3_REPO_PATH", "")))
        candidate_list.extend(
            [
                cfg.da3_repo_path,
                Path("/ltx-data/Depth-Anything-3"),
            ]
        )
        for candidate in candidate_list:
            if candidate and (candidate / "da3_to_sugar_pipeline.sh").exists():
                return candidate
        raise FileNotFoundError("未找到可用的 DA3 仓库路径（缺少 da3_to_sugar_pipeline.sh）")

    def _resolve_sugar_repo(self, params: Dict[str, Any]) -> Path:
        candidate_list: List[Path] = []
        if params.get("sugar_repo_path"):
            candidate_list.append(Path(str(params["sugar_repo_path"])))
        if os.getenv("SUGAR_REPO_PATH"):
            candidate_list.append(Path(os.getenv("SUGAR_REPO_PATH", "")))
        candidate_list.extend([Path("/ltx-data/SuGaR"), Path("/home/ltx/projects/SuGaR")])

        for candidate in candidate_list:
            if candidate and (candidate / "train_fast.py").exists():
                return candidate
        raise FileNotFoundError("未找到可用的 SuGaR 仓库路径（缺少 train_fast.py）")

    def _find_best_ply(self, sugar_repo: Path, scene_name: str) -> Optional[Path]:
        candidates: List[Path] = []
        candidates.extend((sugar_repo / "output" / "refined_ply" / scene_name).glob("*.ply"))
        candidates.extend((sugar_repo / "output" / scene_name).rglob("*coarse_gaussians.ply"))
        candidates.extend((sugar_repo / "output" / scene_name).rglob("point_cloud.ply"))
        candidates.extend((sugar_repo / "output").glob(f"{scene_name}*.ply"))
        valid = [p for p in candidates if p.exists()]
        if not valid:
            return None
        return max(valid, key=lambda p: p.stat().st_mtime)

    def _export_coarse_checkpoint_if_needed(self, sugar_repo: Path, scene_name: str, env: Dict[str, str]) -> Optional[Path]:
        export_script = sugar_repo / "export_coarse_ckpt_to_ply.py"
        if not export_script.exists():
            return None

        ckpts = list((sugar_repo / "output" / scene_name).rglob("*.pt"))
        if not ckpts:
            return None
        ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
        out_ply = ckpt.with_name(f"{ckpt.stem}_coarse_gaussians.ply")
        cmd = ["python", str(export_script), "--ckpt", str(ckpt), "--out", str(out_ply)]
        self.log(f"    -> 未找到PLY，尝试从checkpoint导出: {ckpt.name}")
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            self.log(f"    -> checkpoint导出失败: {proc.stdout}", level="WARN")
            return None
        return out_ply if out_ply.exists() else None
