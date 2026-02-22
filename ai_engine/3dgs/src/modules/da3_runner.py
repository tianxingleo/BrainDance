# src/modules/da3_runner.py
# 功能：使用Depth Anything 3处理图像并进行位姿估计和三维重建
# 实现：调用DA3及COLMAP格式转换脚本替代原有的GLOMAP解算
# 逻辑：1. DA3 Streaming处理 2. 输出转COLMAP文本格式 3. 文本转二进制格式 4. 生成transforms.json 5. 质量检查
import os
import shutil
import subprocess
import json
from pathlib import Path

# 引入项目配置
from src.config import PipelineConfig

class DA3Runner:
    def __init__(self, cfg: PipelineConfig, log_callback=None):
        self.cfg = cfg
        self.log_callback = log_callback or print
        
        self.da3_repo_path = self.cfg.da3_repo_path
        self.da3_streaming_cmd = self.da3_repo_path / "da3_streaming/da3_streaming.py"
        self.da3_convert_cmd = self.da3_repo_path / "convert_da3_to_colmap.py"
        self.da3_binary_cmd = self.da3_repo_path / "colmap_text_to_binary.py"
        
        if not self.da3_streaming_cmd.exists():
            raise FileNotFoundError(f"❌ 找不到 DA3 主程序: {self.da3_streaming_cmd}")
        
        self.log_callback(f"    -> 🎯 锁定引擎: DA3={self.da3_streaming_cmd}")
        
        self.env = os.environ.copy()
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
        self.env["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # 将 DA3 的 repo 目录及其 src 子目录加到 PYTHONPATH 中以防止模块找不到
        pythonpath = self.env.get("PYTHONPATH", "")
        da3_src_path = self.da3_repo_path / "src"
        self.env["PYTHONPATH"] = f"{str(self.da3_repo_path)}{os.pathsep}{str(da3_src_path)}{os.pathsep}{pythonpath}"

    def run(self):
        """执行 DA3 完整流程"""
        self.log_callback(f"📐 [2/4] DA3 位姿解算 (Depth Anything 3 + Streaming)")

        # 路径准备
        raw_images_dir = self.cfg.project_dir / "raw_images"
        dest_images_dir = self.cfg.images_dir
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        for img in raw_images_dir.glob("*"):
            if not (dest_images_dir / img.name).exists():
                shutil.copy2(str(img), str(dest_images_dir / img.name))

        colmap_output_dir = self.cfg.data_dir / "colmap"
        da3_output_dir = colmap_output_dir / "da3_output"
        colmap_output_dir.mkdir(parents=True, exist_ok=True)
        da3_output_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir = colmap_output_dir / "sparse"

        try:
            # 清理
            if sparse_dir.exists(): shutil.rmtree(sparse_dir)
            sparse_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.transforms_file.exists(): self.cfg.transforms_file.unlink()

            # Step 1: DA3 Streaming
            da3_config = self.da3_repo_path / "da3_streaming/configs/base_config.yaml"
            self._run_cmd([
                "python", str(self.da3_streaming_cmd),
                "--image_dir", str(raw_images_dir),
                "--config", str(da3_config),
                "--output_dir", str(da3_output_dir)
            ], "Step 1: DA3 Streaming 处理")

            # Step 2: 转换为 COLMAP 文本格式
            self._run_cmd([
                "python", str(self.da3_convert_cmd),
                "--base_dir", str(da3_output_dir),
                "--output_dir", str(colmap_output_dir)
            ], "Step 2: 转换为 COLMAP 文本格式")

            # Step 3: 转换为 COLMAP 二进制格式
            # convert_da3_to_colmap.py 在 output_dir 下会创建 sparse/0
            colmap_sparse_0_dir = colmap_output_dir / "sparse" / "0"
            self._run_cmd([
                "python", str(self.da3_binary_cmd),
                str(colmap_sparse_0_dir)
            ], "Step 3: 转换为 COLMAP 二进制格式")

            # Step 4: 生成 transforms.json
            self._run_cmd([
                "ns-process-data", "images",
                "--data", str(dest_images_dir),
                "--output-dir", str(self.cfg.data_dir),
                "--skip-colmap",
                "--skip-image-processing",
                "--num-downscales", "0"
            ], "Step 4: 生成 transforms.json")

            # Step 5: 检查
            if self._check_quality(raw_images_dir):
                self.log_callback(f"    ✨ DA3 流程成功！")
                return True

        except Exception as e:
            self.log_callback(f"    ❌ DA3 流程失败: {e}")
            return False
        return False

    def _run_cmd(self, cmd, desc):
        """内部工具：执行命令 (含环境隔离逻辑)"""
        self.log_callback(f"🚀 {desc}...")

        cmd_env = self.env.copy()

        # 设置工作目录：DA3 streaming 需要在 DA3 repo 根目录运行以访问 ./weights
        cwd = None
        if "da3_streaming.py" in cmd[1]:
            cwd = str(self.da3_repo_path)

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cmd_env, cwd=cwd
            )
            for line in process.stdout:
                line = line.strip()
                if not line: continue
                
                # 针对进度条等特殊输出的过滤逻辑
                if any(k in line for k in ["Error", "Warning", "Elapsed", "Matching", "Processing"]):
                     self.log_callback(f"    | {line}")
                elif "100%" in line:
                     self.log_callback(f"    | {line}")
                elif "frame_" in line and "npz" in line:
                     # 避免 np.savez 刷屏，可以根据需要保留
                     pass
                else:
                    # 默认也输出，确保不丢关键信息
                    if len(line) < 200: # 避免输出太长的二进制或乱码
                        self.log_callback(f"    | {line}")

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.CalledProcessError as e:
            self.log_callback(f"❌ 命令执行崩溃: {cmd[0]} (代码 {e.returncode})")
            raise e

    def _check_quality(self, raw_images_dir):
        if not self.cfg.transforms_file.exists(): return False
        with open(self.cfg.transforms_file, 'r') as f: meta = json.load(f)
        reg_count = len(meta["frames"])
        total_count = len(list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png")))
        ratio = reg_count / total_count if total_count > 0 else 0
        self.log_callback(f"    📊 融合匹配率 (由 DA3 计算): {ratio:.2%} ({reg_count}/{total_count})")
        return ratio > 0.1  # DA3的阈值设低一点，因为它是密集估计
