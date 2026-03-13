import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import PipelineConfig
from src.config import BASE_DIR
from src.core.pipeline_base import BasePipeline
from src.modules.da3_runner import DA3Runner
from src.modules.scene_analyzer import SceneAnalyzer
from src.utils.common import format_duration


class DA3TwoDGSPipeline(BasePipeline):
    """
    【少量图片 -> DA3 -> 2DGS】流水线
    逻辑：图片整理 -> (可选) AI质检 -> DA3 位姿/深度解算 -> 2DGS 训练 -> 导出 PLY
    """

    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log("🖼️ 启动 DA3 + 2DGS 流水线...")
        self.log(f"📄 输入文件: {input_path}")

        start_time = time.time()
        input_obj = Path(input_path)
        if not input_obj.exists():
            raise FileNotFoundError(f"输入不存在: {input_obj}")

        cfg = PipelineConfig(
            project_name=self.scene_id,
            mapper_type="da3",
        )
        cfg.project_dir = Path(self.work_dir)

        # 参数（优先 task_params）
        iterations = int(params.get("iterations", params.get("training_iterations", 7000)))
        max_images = int(params.get("max_images", 60))
        keep_ratio = float(params.get("keep_ratio", 1.0))
        enable_scene_analysis = bool(params.get("enable_scene_analysis", False))
        gpu_index = int(params.get("gpu_index", 1))
        render_after_train = bool(params.get("render_after_train", False))

        # 2DGS 仓库路径：task_params > env > 常见默认路径
        dgs_repo = self._resolve_2dgs_repo(params.get("dgs_repo_path"))
        dgs_data_dir = cfg.project_dir / "two_dgs" / "data" / self.scene_id
        dgs_output_dir = cfg.project_dir / "two_dgs" / "output" / self.scene_id

        cfg.project_dir.mkdir(parents=True, exist_ok=True)
        raw_images_dir = cfg.project_dir / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)

        # 1) 准备输入图片
        self.log("📦 [1/4] 整理输入图片...")
        self._prepare_images(
            source=input_obj,
            raw_images_dir=raw_images_dir,
            max_images=max_images,
            keep_ratio=keep_ratio,
        )
        image_files = self._list_images(raw_images_dir)
        if len(image_files) < 2:
            raise RuntimeError("图片数量不足，至少需要 2 张图片才能进行 DA3 + 2DGS。")
        self.log(f"    -> 图片准备完成，共 {len(image_files)} 张")

        pipeline_metadata: Dict[str, Any] = {
            "engine": "da3_2dgs",
            "input_path": str(input_obj),
            "training_iterations": iterations,
            "preview_img_path": str(image_files[0]),
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
        raise FileNotFoundError("找不到 2d-gaussian-splatting 仓库，请通过 task_params.dgs_repo_path 或 DGS2_REPO_PATH 配置。")

    def _prepare_images(self, source: Path, raw_images_dir: Path, max_images: int, keep_ratio: float):
        # 清理旧内容
        for item in raw_images_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            else:
                shutil.rmtree(item, ignore_errors=True)

        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(raw_images_dir)
        elif source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            shutil.copy2(source, raw_images_dir / source.name)
        elif source.is_dir():
            for img in self._list_images(source):
                shutil.copy2(img, raw_images_dir / img.name)
        else:
            raise RuntimeError("输入文件格式不支持，需为 .zip、单张图片或图片目录。")

        images = self._list_images(raw_images_dir)
        if not images:
            raise RuntimeError("未找到可用图片，请检查 zip/目录内容。")

        # 统一重命名，保证顺序稳定
        tmp_dir = raw_images_dir / "_normalized"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for idx, src in enumerate(images, start=1):
            dst = tmp_dir / f"frame_{idx:05d}.jpg"
            shutil.copy2(src, dst)

        for item in raw_images_dir.iterdir():
            if item == tmp_dir:
                continue
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            else:
                shutil.rmtree(item, ignore_errors=True)

        keep_count = len(list(tmp_dir.glob("*")))
        if keep_ratio > 0 and keep_ratio < 1:
            keep_count = max(2, int(keep_count * keep_ratio))
        keep_count = min(keep_count, max_images)

        normalized = sorted(tmp_dir.glob("*"))[:keep_count]
        for idx, src in enumerate(normalized, start=1):
            dst = raw_images_dir / f"frame_{idx:05d}.jpg"
            shutil.move(str(src), str(dst))
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
            if any(k in line for k in ["ITER", "Evaluating", "Saving Gaussians", "Training complete", "ERROR", "Error"]):
                self.log(f"    | {line}")
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    def _list_images(self, folder: Path) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])
