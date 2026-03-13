import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.core.pipeline_base import BasePipeline
from src.utils.common import format_duration


class Sparse2DGSPipeline(BasePipeline):
    """
    【少量图片 -> Sparse2DGS】流水线
    输入约定：Worker 下载的 raw/images.zip（或单图兜底）
    流程：解压图片 -> COLMAP 解算 -> Sparse2DGS 训练 -> 返回 point_cloud.ply
    """

    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        self.log("🧩 启动 Sparse2DGS 多图重建流水线...")
        self.log(f"📄 输入资源: {input_path}")

        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        sparse2dgs_repo = Path(
            params.get("sparse2dgs_repo_path")
            or os.getenv("SPARSE2DGS_REPO_PATH", "/ltx-data/Sparse2DGS")
        )
        conda_env = str(params.get("conda_env") or os.getenv("SPARSE2DGS_CONDA_ENV", "Braindance"))
        iterations = int(params.get("iterations", 7000))
        resolution = int(params.get("resolution", 2))
        depth_ratio = float(params.get("depth_ratio", 1.0))
        lambda_dist = int(float(params.get("lambda_dist", 1000)))
        colmap_matcher = str(params.get("colmap_matcher", "exhaustive_matcher"))
        colmap_mapper = str(params.get("colmap_mapper", "mapper"))

        if not sparse2dgs_repo.exists():
            raise FileNotFoundError(f"Sparse2DGS 仓库不存在: {sparse2dgs_repo}")
        if not (sparse2dgs_repo / "train.py").exists():
            raise FileNotFoundError(f"缺少训练脚本: {sparse2dgs_repo / 'train.py'}")

        work_dir = Path(self.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        images_dir = work_dir / "input_images"
        colmap_dir = work_dir / "colmap_output"
        sparse_scene_dir = work_dir / "sparse2dgs_data" / self.scene_id
        output_scene_dir = work_dir / "sparse2dgs_output" / self.scene_id

        images_dir.mkdir(parents=True, exist_ok=True)
        colmap_dir.mkdir(parents=True, exist_ok=True)
        sparse_scene_dir.mkdir(parents=True, exist_ok=True)
        output_scene_dir.mkdir(parents=True, exist_ok=True)

        # 1) 准备图像
        self._prepare_images(input_file, images_dir)
        image_count = self._count_images(images_dir)
        if image_count < 3:
            raise RuntimeError(f"可用图片数量不足（当前 {image_count} 张），至少需要 3 张")
        self.log(f"🖼️ 图片准备完成: {image_count} 张")

        # 2) COLMAP 位姿解算
        self.log("🗺️ 开始 COLMAP 位姿解算...")
        sparse0_dir = self._run_colmap(images_dir, colmap_dir, colmap_matcher, colmap_mapper)
        self.log(f"✅ COLMAP 解算完成: {sparse0_dir}")

        # 3) 组装 Sparse2DGS 数据结构
        self.log("📦 组装 Sparse2DGS 输入目录...")
        self._prepare_sparse2dgs_scene(images_dir, sparse0_dir, sparse_scene_dir)
        dataset_link = self._ensure_sparse2dgs_dataset_link(sparse2dgs_repo, sparse_scene_dir)

        # 4) 训练 Sparse2DGS
        self.log(
            f"🧠 开始训练 Sparse2DGS (iter={iterations}, r={resolution}, depth_ratio={depth_ratio})..."
        )
        try:
            self._run_sparse2dgs_training(
                sparse2dgs_repo=sparse2dgs_repo,
                conda_env=conda_env,
                source_scene_dir=sparse_scene_dir,
                output_scene_dir=output_scene_dir,
                iterations=iterations,
                resolution=resolution,
                depth_ratio=depth_ratio,
                lambda_dist=lambda_dist,
            )
        finally:
            # 避免污染 Sparse2DGS 自带 dtu_sparse 目录
            if dataset_link and dataset_link.is_symlink():
                try:
                    dataset_link.unlink()
                except OSError:
                    pass

        final_ply = output_scene_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not final_ply.exists():
            raise FileNotFoundError(f"训练结束但未找到输出点云: {final_ply}")

        preview = output_scene_dir / "vis" / f"iteration{iterations}.jpg"
        metadata = {
            "engine": "sparse2dgs",
            "image_count": image_count,
            "iterations": iterations,
            "preview_img_path": str(preview) if preview.exists() else "",
            "input_type": "images_zip",
        }

        self.log(f"✅ Sparse2DGS 完成: {final_ply}")
        self.log(f"⏱️ 总耗时: {format_duration(time.time() - start_time)}")
        return str(final_ply), metadata

    def _prepare_images(self, input_file: Path, images_dir: Path):
        if input_file.suffix.lower() == ".zip":
            with zipfile.ZipFile(input_file, "r") as zf:
                zf.extractall(images_dir)
        elif input_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.copy2(str(input_file), str(images_dir / input_file.name))
        else:
            raise RuntimeError(
                f"不支持的输入类型: {input_file.suffix}，请上传 raw/images.zip 或单图"
            )

        # 展平目录，统一放到 images_dir 根下
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        moved = 0
        for p in list(images_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in valid_ext and p.parent != images_dir:
                target = images_dir / p.name
                if target.exists():
                    target = images_dir / f"{p.stem}_{moved}{p.suffix.lower()}"
                shutil.move(str(p), str(target))
                moved += 1

        # 删除空目录
        for p in sorted(images_dir.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    def _count_images(self, images_dir: Path) -> int:
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        return sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext)

    def _run_command(self, cmd, step_name: str, cwd: Path = None, env: Dict[str, str] = None):
        self.log(f"    -> {step_name}: {' '.join(map(str, cmd))}")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        keywords = (
            "ERROR",
            "Error",
            "error",
            "WARNING",
            "warning",
            "Training progress",
            "Traceback",
            "RuntimeError",
            "FileNotFoundError",
            "Optimizing",
            "✅",
            "❌",
        )
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if any(k in line for k in keywords):
                self.log(f"    | {line}")

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"{step_name} 执行失败 (exit={process.returncode})")

    def _run_colmap(
        self,
        images_dir: Path,
        colmap_dir: Path,
        matcher: str,
        mapper: str,
    ) -> Path:
        database_path = colmap_dir / "database.db"
        sparse_root = colmap_dir / "sparse"
        sparse_root.mkdir(parents=True, exist_ok=True)

        colmap_bin = shutil.which("colmap") or "/usr/local/bin/colmap"
        if not Path(colmap_bin).exists():
            raise FileNotFoundError("找不到 colmap 可执行文件，请先安装 COLMAP")

        self._run_command(
            [
                colmap_bin,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(images_dir),
                "--ImageReader.single_camera",
                "1",
                "--FeatureExtraction.use_gpu",
                "1",
            ],
            "COLMAP 特征提取",
        )

        self._run_command(
            [
                colmap_bin,
                matcher,
                "--database_path",
                str(database_path),
                "--FeatureMatching.use_gpu",
                "1",
            ],
            "COLMAP 图像匹配",
        )

        # primary mapper
        mapper_cmd = [
            colmap_bin,
            mapper,
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_root),
        ]
        try:
            self._run_command(mapper_cmd, f"COLMAP {mapper}")
        except Exception:
            if mapper != "mapper":
                # fallback 到标准 mapper
                self.log("⚠️ 指定 mapper 失败，回退到 COLMAP mapper")
                self._run_command(
                    [
                        colmap_bin,
                        "mapper",
                        "--database_path",
                        str(database_path),
                        "--image_path",
                        str(images_dir),
                        "--output_path",
                        str(sparse_root),
                    ],
                    "COLMAP mapper(回退)",
                )
            else:
                raise

        sparse0_dir = sparse_root / "0"
        if not sparse0_dir.exists():
            # 某些版本可能直接写在 sparse 根目录
            required = ["cameras.bin", "images.bin", "points3D.bin"]
            if all((sparse_root / name).exists() for name in required):
                sparse0_dir.mkdir(parents=True, exist_ok=True)
                for name in required:
                    shutil.move(str(sparse_root / name), str(sparse0_dir / name))
            else:
                raise RuntimeError("COLMAP 未生成有效 sparse/0 结果")
        return sparse0_dir

    def _prepare_sparse2dgs_scene(self, images_dir: Path, sparse0_dir: Path, scene_dir: Path):
        target_images = scene_dir / "images"
        target_sparse = scene_dir / "sparse" / "0"
        target_images.mkdir(parents=True, exist_ok=True)
        target_sparse.mkdir(parents=True, exist_ok=True)

        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                shutil.copy2(str(p), str(target_images / p.name))

        for name in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = sparse0_dir / name
            if not src.exists():
                raise FileNotFoundError(f"COLMAP 稀疏结果缺少文件: {src}")
            shutil.copy2(str(src), str(target_sparse / name))

        # 生成 CLMVSNet 需要的 cam_*.txt
        self._generate_dtu_camera_txts(
            sparse0_dir=target_sparse,
            images_dir=target_images,
            scene_dir=scene_dir,
        )

    def _ensure_sparse2dgs_dataset_link(self, sparse2dgs_repo: Path, scene_dir: Path):
        dataset_root = sparse2dgs_repo / "dtu_sparse"
        dataset_root.mkdir(parents=True, exist_ok=True)
        target = dataset_root / scene_dir.name

        if target.exists():
            # 已存在则直接复用（常见于重试任务）
            return target

        try:
            target.symlink_to(scene_dir, target_is_directory=True)
            self.log(f"🔗 已创建数据集链接: {target} -> {scene_dir}")
        except OSError:
            shutil.copytree(scene_dir, target)
            self.log(f"📁 符号链接失败，已复制场景到: {target}")
        return target

    def _generate_dtu_camera_txts(self, sparse0_dir: Path, images_dir: Path, scene_dir: Path):
        txt_dir = scene_dir / "colmap_txt"
        txt_dir.mkdir(parents=True, exist_ok=True)

        colmap_bin = shutil.which("colmap") or "/usr/local/bin/colmap"
        self._run_command(
            [
                colmap_bin,
                "model_converter",
                "--input_path",
                str(sparse0_dir),
                "--output_path",
                str(txt_dir),
                "--output_type",
                "TXT",
            ],
            "COLMAP 模型转 TXT",
        )

        cameras = self._parse_cameras_txt(txt_dir / "cameras.txt")
        points_xyz = self._parse_points3d_txt(txt_dir / "points3D.txt")
        image_entries = self._parse_images_txt(txt_dir / "images.txt")

        for entry in image_entries:
            image_name = entry["name"]
            stem = Path(image_name).stem
            camera_id = entry["camera_id"]
            qvec = entry["qvec"]
            tvec = entry["tvec"]

            if camera_id not in cameras:
                continue
            K, colmap_w, colmap_h = cameras[camera_id]
            img_path = images_dir / image_name
            if img_path.exists():
                # 修正内参与重建图像分辨率一致
                import cv2

                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 0 and h > 0 and (w != colmap_w or h != colmap_h):
                        sx = w / float(colmap_w)
                        sy = h / float(colmap_h)
                        K = K.copy()
                        K[0, 0] *= sx
                        K[1, 1] *= sy
                        K[0, 2] *= sx
                        K[1, 2] *= sy

            R = self._qvec_to_rotmat(qvec)
            T = np.array(tvec, dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = T

            dp_min, dp_max = self._estimate_depth_range(points_xyz, R, T)

            cam_txt = scene_dir / f"cam_{stem}.txt"
            with open(cam_txt, "w", encoding="utf-8") as f:
                for row in K:
                    f.write(f"{row[0]} {row[1]} {row[2]}\n")
                f.write("\n")
                for row in w2c:
                    f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")
                f.write("\n")
                f.write(f"{dp_min} {dp_max}\n")

    def _parse_cameras_txt(self, camera_txt: Path):
        cameras = {}
        with open(camera_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width, height = int(parts[2]), int(parts[3])
                params = list(map(float, parts[4:]))
                K = np.eye(3, dtype=np.float64)
                if model == "SIMPLE_PINHOLE":
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                elif model == "PINHOLE":
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                else:
                    # SIMPLE_RADIAL / RADIAL / OPENCV 等都至少包含 f, cx, cy
                    fx = fy = params[0]
                    cx = params[1] if len(params) > 1 else width / 2.0
                    cy = params[2] if len(params) > 2 else height / 2.0
                K[0, 0], K[1, 1] = fx, fy
                K[0, 2], K[1, 2] = cx, cy
                cameras[camera_id] = (K, width, height)
        return cameras

    def _parse_points3d_txt(self, points_txt: Path):
        pts = []
        with open(points_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not pts:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    def _parse_images_txt(self, images_txt: Path):
        entries = []
        with open(images_txt, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        # 每个 image 两行：第一行为元信息，第二行为 points2d
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 10:
                continue
            entries.append(
                {
                    "image_id": int(parts[0]),
                    "qvec": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                    "tvec": [float(parts[5]), float(parts[6]), float(parts[7])],
                    "camera_id": int(parts[8]),
                    "name": parts[9],
                }
            )
        return entries

    def _qvec_to_rotmat(self, qvec):
        qw, qx, qy, qz = qvec
        return np.array(
            [
                [
                    1 - 2 * qy * qy - 2 * qz * qz,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx * qx - 2 * qz * qz,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx * qx - 2 * qy * qy,
                ],
            ],
            dtype=np.float64,
        )

    def _estimate_depth_range(self, points_xyz: np.ndarray, R: np.ndarray, T: np.ndarray):
        if points_xyz.shape[0] == 0:
            return 0.1, 10.0
        # 对性能友好：最多采样 2000 点
        stride = max(1, points_xyz.shape[0] // 2000)
        sample = points_xyz[::stride]
        z = (R @ sample.T).T[:, 2] + T[2]
        z = z[z > 1e-4]
        if z.size == 0:
            return 0.1, 10.0
        dp_min = max(0.01, float(np.percentile(z, 5) * 0.8))
        dp_max = max(dp_min + 0.1, float(np.percentile(z, 95) * 1.2))
        return dp_min, dp_max

    def _run_sparse2dgs_training(
        self,
        *,
        sparse2dgs_repo: Path,
        conda_env: str,
        source_scene_dir: Path,
        output_scene_dir: Path,
        iterations: int,
        resolution: int,
        depth_ratio: float,
        lambda_dist: int,
    ):
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            "train.py",
            "-s",
            str(source_scene_dir),
            "-m",
            str(output_scene_dir),
            "--iterations",
            str(iterations),
            "-r",
            str(resolution),
            "--depth_ratio",
            str(depth_ratio),
            "--lambda_dist",
            str(int(lambda_dist)),
            "--test_iterations",
            "-1",
            "--save_iterations",
            str(iterations),
            "--quiet",
        ]

        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "1"

        self._run_command(cmd, "Sparse2DGS 训练", cwd=sparse2dgs_repo, env=env)
