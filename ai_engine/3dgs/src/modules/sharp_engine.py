import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


class SharpEngine:
    def __init__(self, repo_path: str):
        """
        :param repo_path: ml-sharp 仓库的根目录
        """
        self.repo_path = Path(repo_path)
        self._check_environment()

    def _check_environment(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(f"[SharpEngine] 找不到仓库: {self.repo_path}")

        if shutil.which("sharp") is None:
            raise RuntimeError(
                f"[SharpEngine] 未找到 'sharp' 命令！\n"
                f"请确保已在 {self.repo_path} 下运行过 'pip install -e .'"
            )

    def run(self, image_path: str, output_dir: str) -> str:
        """
        执行推理
        :param image_path: 输入图片路径
        :param output_dir: 输出目录
        :return: 生成的 ply 文件路径
        """
        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SharpEngine] 启动推理...")
        print(f"    输入: {image_path.name}")
        print(f"    输出: {output_dir}")

        cmd = [
            "sharp", "predict",
            "-i", str(image_path),
            "-o", str(output_dir)
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(self.repo_path),
                env=env
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Sharp 推理失败，退出码: {e.returncode}")

        ply_files = list(output_dir.rglob("*.ply"))

        if not ply_files:
            raise RuntimeError("Sharp 运行成功但未生成 .ply 文件")

        final_ply = ply_files[0]
        print(f"[SharpEngine] 生成成功: {final_ply.name}")

        return str(final_ply)
