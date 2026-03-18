import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import BASE_DIR
from src.config import PipelineConfig
from src.core.pipeline_base import BasePipeline
from src.modules.da3_runner import DA3Runner
from src.modules.image_proc import ImageProcessor
from src.modules.scene_analyzer import SceneAnalyzer
from src.utils.common import format_duration


class DA3TwoDGSPipeline(BasePipeline):
    """
    【视频 -> DA3 -> 2DGS】流水线
    逻辑：视频抽帧 -> (可选) AI质检 -> DA3 位姿/深度解算 -> 2DGS 训练 -> 导出 PLY

    注意：2DGS 目标是替代 Nerfstudio 3DGS 训练链路，依赖较长视频序列。
    不支持单图或少量图片输入。
    """

    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log("🎬 启动 DA3 + 2DGS 视频流水线...")
        self.log(f"📄 输入文件: {input_path}")

        start_time = time.time()
        video_path = Path(input_path)
        if not video_path.exists():
            raise FileNotFoundError(f"输入视频不存在: {video_path}")

        if video_path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise RuntimeError("da3_2dgs 仅支持视频输入（mp4/mov/avi/mkv/webm）。")

        cfg = PipelineConfig(
            project_name=self.scene_id,
            mapper_type="da3",
            video_path=video_path,
        )
        cfg.project_dir = Path(self.work_dir)

        # 参数（优先 task_params）
        iterations = int(params.get("iterations", params.get("training_iterations", 30000)))
        gpu_index = int(params.get("gpu_index", 1))
        render_after_train = bool(params.get("render_after_train", False))

        # 抽帧/数据规模参数（强调视频序列）
        extract_fps = float(params.get("extract_fps", 2.0))
        max_edge = int(params.get("max_edge", 1920))
        blur_keep_ratio = float(params.get("blur_keep_ratio", 0.85))
        max_images = int(params.get("max_images", cfg.max_images))
        min_images = int(params.get("min_images", 24))

        enable_scene_analysis = bool(params.get("enable_scene_analysis", False))

        # 2DGS 仓库路径：task_params > env > 常见默认路径
        dgs_repo = self._resolve_2dgs_repo(params.get("dgs_repo_path"))
        dgs_data_dir = cfg.project_dir / "two_dgs" / "data" / self.scene_id
        dgs_output_dir = cfg.project_dir / "two_dgs" / "output" / self.scene_id

        cfg.project_dir.mkdir(parents=True, exist_ok=True)
        raw_images_dir = cfg.project_dir / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)

        # 1) 视频抽帧与筛选
        self.log("🎞️ [1/4] 视频抽帧与图片筛选...")
        extracted_count = self._extract_video_to_images(
            video_path=video_path,
            raw_images_dir=raw_images_dir,
            extract_fps=extract_fps,
            max_edge=max_edge,
            blur_keep_ratio=blur_keep_ratio,
            max_images=max_images,
            cfg=cfg,
        )

        if extracted_count < min_images:
            raise RuntimeError(
                f"可用帧数不足：{extracted_count} < {min_images}。"
                "2DGS 需要较长视频序列，请提供更多连续视角。"
            )

        image_files = self._list_images(raw_images_dir)
        self.log(f"    -> 图片准备完成，共 {len(image_files)} 张")

        pipeline_metadata: Dict[str, Any] = {
            "engine": "da3_2dgs",
            "input_path": str(video_path),
            "training_iterations": iterations,
            "preview_img_path": str(image_files[0]),
            "frame_count": len(image_files),
        }

        # 2) 可选 AI 质检
        if enable_scene_analysis:
            self.log("🧐 [2/4] 执行 AI 质检...")
            analyzer = SceneAnalyzer(cfg)
            passed, score, reason, tags, description, objects = analyzer.run(
                raw_images_dir, log_callback=lambda msg: self.log(msg)
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
                raise RuntimeError(f"AI 质检不通过 ({score}): {reason}")
            self.log(f"    -> 质检通过: {score} 分")
        else:
            self.log("🧐 [2/4] 跳过 AI 质检（enable_scene_analysis=false）")

        # 3) DA3 解算
        self.log("⚙️ [3/4] 执行 DA3 位姿与深度解算...")
        da3_runner = DA3Runner(cfg, log_callback=self.log)
        if not da3_runner.run():
            raise RuntimeError("DA3 解算失败")
        self.log("    -> DA3 解算完成")

        # 4) 2DGS 训练
        self.log("🧠 [4/4] 执行 2DGS 训练...")
        self._prepare_2dgs_dataset(cfg, dgs_data_dir)
        self._run_2dgs_training(
            dgs_repo=dgs_repo,
            dgs_data_dir=dgs_data_dir,
            dgs_output_dir=dgs_output_dir,
            iterations=iterations,
            gpu_index=gpu_index,
            render_after_train=render_after_train,
        )

        final_ply = dgs_output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not final_ply.exists():
            raise FileNotFoundError(f"2DGS 输出 PLY 不存在: {final_ply}")

        pipeline_metadata.update(
            {
                "dgs_repo_path": str(dgs_repo),
                "dgs_data_dir": str(dgs_data_dir),
                "dgs_output_dir": str(dgs_output_dir),
                "gpu_index": gpu_index,
            }
        )

        self.log(f"💾 输出完成: {final_ply}")
        self.log(f"⏱️ 总耗时: {format_duration(time.time() - start_time)}")
        return str(final_ply), pipeline_metadata

    def _resolve_2dgs_repo(self, repo_path_param: Any) -> Path:
        candidates: List[Path] = []
        if repo_path_param:
            candidates.append(Path(str(repo_path_param)))

        env_repo = os.getenv("DGS2_REPO_PATH", "").strip()
        if env_repo:
            candidates.append(Path(env_repo))

        candidates.extend(
            [
                BASE_DIR / "src/libs/2d-gaussian-splatting",
                Path("/ltx-data/2d-gaussian-splatting"),
                Path("/home/ltx/projects/2d-gaussian-splatting"),
            ]
        )

        for p in candidates:
            if p.exists() and (p / "train.py").exists():
                return p
        raise FileNotFoundError(
            "找不到 2d-gaussian-splatting 仓库，请通过 task_params.dgs_repo_path 或 DGS2_REPO_PATH 配置。"
        )

    def _extract_video_to_images(
        self,
        *,
        video_path: Path,
        raw_images_dir: Path,
        extract_fps: float,
        max_edge: int,
        blur_keep_ratio: float,
        max_images: int,
        cfg: PipelineConfig,
    ) -> int:
        # 清理旧内容
        for item in raw_images_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            else:
                shutil.rmtree(item, ignore_errors=True)

        temp_dir = raw_images_dir.parent / "temp_extract"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 复制视频到工作区，保持目录一致性
        dest_video_path = raw_images_dir.parent / video_path.name
        if not dest_video_path.exists():
            shutil.copy2(video_path, dest_video_path)

        self.log(
            f"    -> FFmpeg 抽帧: fps={extract_fps}, max_edge={max_edge}, "
            "lanczos 重采样"
        )
        vf = f"fps={extract_fps},scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease:flags=lanczos"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(dest_video_path),
                    "-vf",
                    vf,
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
            raise RuntimeError(f"FFmpeg 抽帧失败: {e}")

        img_processor = ImageProcessor(cfg, log_callback=self.log)
        img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=blur_keep_ratio)

        all_imgs = sorted([p for p in temp_dir.glob("*") if p.is_file()])
        if len(all_imgs) > max_images:
            indices = np.linspace(0, len(all_imgs) - 1, max_images, dtype=int)
            all_imgs = [all_imgs[i] for i in sorted(set(indices))]

        for idx, img in enumerate(all_imgs, start=1):
            shutil.copy2(img, raw_images_dir / f"frame_{idx:05d}.jpg")

        shutil.rmtree(temp_dir, ignore_errors=True)
        return len(all_imgs)

    def _prepare_2dgs_dataset(self, cfg: PipelineConfig, dgs_data_dir: Path):
        sparse_src = cfg.data_dir / "colmap" / "sparse" / "0"
        images_src = cfg.data_dir / "images"

        if not sparse_src.exists():
            raise FileNotFoundError(f"COLMAP sparse 不存在: {sparse_src}")
        if not images_src.exists():
            raise FileNotFoundError(f"images 目录不存在: {images_src}")

        if dgs_data_dir.exists():
            shutil.rmtree(dgs_data_dir)
        (dgs_data_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (dgs_data_dir / "images").mkdir(parents=True, exist_ok=True)

        for bin_file in sparse_src.glob("*.bin"):
            shutil.copy2(bin_file, dgs_data_dir / "sparse" / "0" / bin_file.name)

        for img in self._list_images(images_src):
            shutil.copy2(img, dgs_data_dir / "images" / img.name)

        copied = len(self._list_images(dgs_data_dir / "images"))
        if copied < 2:
            raise RuntimeError("2DGS 数据集图片不足（<2），无法训练。")
        self.log(f"    -> 2DGS 数据准备完成: {copied} 张图")

    def _run_2dgs_training(
        self,
        *,
        dgs_repo: Path,
        dgs_data_dir: Path,
        dgs_output_dir: Path,
        iterations: int,
        gpu_index: int,
        render_after_train: bool,
    ):
        dgs_output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["QT_QPA_PLATFORM"] = "offscreen"

        train_cmd = [
            "python",
            "train.py",
            "-s",
            str(dgs_data_dir),
            "-m",
            str(dgs_output_dir),
            "--iterations",
            str(iterations),
            "--save_iterations",
            str(iterations),
            "--test_iterations",
            str(iterations),
        ]
        self._run_cmd(train_cmd, cwd=dgs_repo, env=env, desc=f"2DGS 训练 ({iterations} iter)")

        if render_after_train:
            render_cmd = [
                "python",
                "render.py",
                "-m",
                str(dgs_output_dir),
                "--iteration",
                str(iterations),
                "--skip_test",
                "--skip_mesh",
            ]
            self._run_cmd(render_cmd, cwd=dgs_repo, env=env, desc="2DGS 渲染")

    def _run_cmd(self, cmd: List[str], cwd: Path, env: Dict[str, str], desc: str):
        self.log(f"🚀 {desc}")
        self.log(f"    => {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if any(
                k in line
                for k in [
                    "ITER",
                    "Evaluating",
                    "Saving Gaussians",
                    "Training complete",
                    "ERROR",
                    "Error",
                ]
            ):
                self.log(f"    | {line}")
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    def _list_images(self, folder: Path) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])
