# src/modules/glomap_runner.py
# 功能：实现GLOMAP位姿解算功能，处理图像的特征提取、匹配和三维重建
# 实现：调用COLMAP和GLOMAP可执行文件，执行完整的位姿解算流程
# 逻辑：1. 特征提取 2. 顺序匹配 3. 全局重建 4. 目录结构修正 5. 生成transforms.json 6. 质量检查
# 包含：GlomapRunner类、GLOMAP流程控制、COLMAP接口调用、质量检查算法
import os
import shutil
import subprocess
import json
from pathlib import Path

# 引入项目配置
from src.config import PipelineConfig

class GlomapRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        
        # 1. 查找 COLMAP (优先使用 Conda 环境自带的！)
        self.colmap_exe = shutil.which("colmap")
        if not self.colmap_exe:
            if os.path.exists("/usr/local/bin/colmap"):
                self.colmap_exe = "/usr/local/bin/colmap"
        
        # 2. 查找 GLOMAP
        self.glomap_exe = shutil.which("glomap")
        if not self.glomap_exe:
            if os.path.exists("/usr/local/bin/glomap"):
                self.glomap_exe = "/usr/local/bin/glomap"

        if not self.colmap_exe or not self.glomap_exe:
            raise FileNotFoundError("❌ 缺少 colmap 或 glomap 可执行文件")

        print(f"    -> 🎯 锁定引擎: COLMAP={self.colmap_exe}")
        print(f"    -> 🎯 锁定引擎: GLOMAP={self.glomap_exe}")
        
        self.env = os.environ.copy()
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def run(self):
        """执行 GLOMAP 完整流程"""
        print(f"\n📐 [2/4] GLOMAP 位姿解算 (Global Mapping)")

        # 路径准备
        raw_images_dir = self.cfg.project_dir / "raw_images"
        dest_images_dir = self.cfg.images_dir
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        for img in raw_images_dir.glob("*"):
            if not (dest_images_dir / img.name).exists():
                shutil.copy2(str(img), str(dest_images_dir / img.name))

        colmap_output_dir = self.cfg.data_dir / "colmap"
        colmap_output_dir.mkdir(parents=True, exist_ok=True)
        database_path = colmap_output_dir / "database.db"
        sparse_dir = colmap_output_dir / "sparse"

        try:
            # 清理
            if database_path.exists(): database_path.unlink()
            if sparse_dir.exists(): shutil.rmtree(sparse_dir)
            sparse_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.transforms_file.exists(): self.cfg.transforms_file.unlink()

            # Step 1: 特征提取
            self._run_cmd([
                self.colmap_exe, "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--ImageReader.camera_model", "OPENCV",
                "--ImageReader.single_camera", "1"
            ], "Step 1: 特征提取 (COLMAP)")

            # Step 2: 顺序匹配
            self._run_cmd([
                self.colmap_exe, "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "25"
            ], "Step 2: 顺序匹配 (COLMAP)")

            # Step 3: 全局重建
            print(f"    -> 🚀 启动 GLOMAP 引擎...")
            self._run_cmd([
                self.glomap_exe, "mapper",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--output_path", str(sparse_dir)
            ], "Step 3: 全局映射 (GLOMAP)")

            # Step 4: 目录修正
            self._fix_directory_structure(sparse_dir)

            # Step 5: 生成 json
            self._run_cmd([
                "ns-process-data", "images",
                "--data", str(dest_images_dir),
                "--output-dir", str(self.cfg.data_dir),
                "--skip-colmap",
                "--skip-image-processing",
                "--num-downscales", "0"
            ], "生成 transforms.json")

            # Step 6: 检查
            if self._check_quality(raw_images_dir):
                print(f"    ✨ GLOMAP 流程成功！")
                return True

        except Exception as e:
            print(f"    ❌ GLOMAP 流程失败: {e}")
            return False
        return False

    def _run_cmd(self, cmd, desc):
        """内部工具：执行命令 (含环境隔离逻辑)"""
        print(f"🚀 {desc}...")
        
        # 🔥 环境隔离逻辑 🔥
        cmd_env = self.env.copy()
        exe_path = cmd[0]
        # 如果是系统程序 (/usr/local/bin/glomap)，清除 LD_LIBRARY_PATH 防止 Conda 干扰
        if exe_path.startswith("/usr") or exe_path.startswith("/bin"):
            if "LD_LIBRARY_PATH" in cmd_env:
                del cmd_env["LD_LIBRARY_PATH"]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cmd_env
            )
            for line in process.stdout:
                if any(k in line for k in ["Error", "Warning", "Elapsed", "image pairs"]):
                    print(f"    | {line.strip()}")
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行崩溃: {cmd[0]} (代码 {e.returncode})")
            raise e

    def _fix_directory_structure(self, sparse_root):
        target_dir_0 = sparse_root / "0"
        target_dir_0.mkdir(parents=True, exist_ok=True)
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        required_files_txt = ["cameras.txt", "images.txt", "points3D.txt"]
        model_found = False
        for root, dirs, files in os.walk(sparse_root):
            if all(f in files for f in required_files):
                src = Path(root)
                if src != target_dir_0:
                    for f in required_files:
                        if (target_dir_0/f).exists(): (target_dir_0/f).unlink()
                        shutil.move(str(src/f), str(target_dir_0/f))
                model_found = True
                break
            if all(f in files for f in required_files_txt):
                src = Path(root)
                if src != target_dir_0:
                    for f in required_files_txt:
                        if (target_dir_0/f).exists(): (target_dir_0/f).unlink()
                        shutil.move(str(src/f), str(target_dir_0/f))
                model_found = True
                break
        if not model_found: raise RuntimeError("GLOMAP 未生成有效的稀疏模型文件！")

    def _check_quality(self, raw_images_dir):
        if not self.cfg.transforms_file.exists(): return False
        with open(self.cfg.transforms_file, 'r') as f: meta = json.load(f)
        reg_count = len(meta["frames"])
        total_count = len(list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png")))
        ratio = reg_count / total_count if total_count > 0 else 0
        print(f"    📊 匹配率: {ratio:.2%} ({reg_count}/{total_count})")
        return ratio > 0.2
