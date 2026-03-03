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
        self.colmap_use_gpu = os.getenv("COLMAP_USE_GPU", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.colmap_exe = self._resolve_executable("colmap", "COLMAP_BIN")
        self.glomap_exe = self._resolve_executable("glomap", "GLOMAP_BIN")

        missing = []
        if not self.colmap_exe:
            missing.append("colmap")
        if not self.glomap_exe:
            missing.append("glomap")
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                f"❌ 缺少可执行文件: {missing_text}。"
                f"请安装后重试，或在 .env 中设置 COLMAP_BIN/GLOMAP_BIN 为绝对路径。"
            )

        print(f"    -> 🎯 锁定引擎: COLMAP={self.colmap_exe}")
        print(f"    -> 🎯 锁定引擎: GLOMAP={self.glomap_exe}")
        print(f"    -> ⚙️ COLMAP GPU 开关: {'开启' if self.colmap_use_gpu else '关闭'}")
        
        self.env = os.environ.copy()
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def _resolve_executable(self, cmd_name: str, env_key: str):
        env_path = os.getenv(env_key, "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists() and os.access(str(p), os.X_OK):
                return str(p)

        by_path = shutil.which(cmd_name)
        if by_path:
            return by_path

        common_path = Path("/usr/local/bin") / cmd_name
        if common_path.exists() and os.access(str(common_path), os.X_OK):
            return str(common_path)

        return None

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
            # 注意：启用 retry_cpu=True 以便在 CUDA runtime 不匹配时自动切换到 CPU
            use_gpu_str = "GPU" if self.colmap_use_gpu else "CPU"
            self._run_cmd([
                self.colmap_exe, "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--ImageReader.camera_model", "OPENCV",
                "--ImageReader.single_camera", "1",
                "--FeatureExtraction.use_gpu", "1" if self.colmap_use_gpu else "0",
                "--FeatureExtraction.gpu_index", "0" if self.colmap_use_gpu else "-1"
            ], f"Step 1: 特征提取 (COLMAP {use_gpu_str})", retry_cpu=self.colmap_use_gpu)

            # Step 2: 顺序匹配
            # COLMAP 3.13.0+ 中匹配 GPU 开关为 --FeatureMatching.use_gpu
            use_gpu_str = "GPU" if self.colmap_use_gpu else "CPU"
            self._run_cmd([
                self.colmap_exe, "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "25",
                "--FeatureMatching.use_gpu", "1" if self.colmap_use_gpu else "0",
                "--FeatureMatching.gpu_index", "0" if self.colmap_use_gpu else "-1"
            ], f"Step 2: 顺序匹配 (COLMAP {use_gpu_str})", retry_cpu=self.colmap_use_gpu)

            # Step 3: 全局重建
            print(f"    -> 🚀 启动 GLOMAP 引擎...")
            self._run_cmd([
                self.glomap_exe, "mapper",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--output_path", str(sparse_dir),
                "--GlobalPositioning.use_gpu", "1" if self.colmap_use_gpu else "0",
                "--BundleAdjustment.use_gpu", "1" if self.colmap_use_gpu else "0",
                "--GlobalPositioning.gpu_index", "0" if self.colmap_use_gpu else "-1",
                "--BundleAdjustment.gpu_index", "0" if self.colmap_use_gpu else "-1",
            ], "Step 3: 全局映射 (GLOMAP)", retry_cpu=self.colmap_use_gpu)

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

    def _run_cmd(self, cmd, desc, retry_cpu=False):
        """内部工具：执行命令 (含环境隔离逻辑)"""
        print(f"🚀 {desc}...")
        
        # 🔥 环境隔离逻辑 🔥
        cmd_env = self.env.copy()
        exe_path = cmd[0]
        # 清除 LD_LIBRARY_PATH 防止 Conda 库与系统 CUDA 库冲突导致 SIGABRT
        # 同时覆盖系统程序和 Conda 环境内的程序（colmap/glomap）
        colmap_or_glomap = any(name in exe_path for name in ["colmap", "glomap"])
        system_bin = exe_path.startswith("/usr") or exe_path.startswith("/bin")
        if system_bin or colmap_or_glomap:
            # 🟢 [修复] 解决 GLOMAP SIGABRT (-6) 崩溃问题
            # 根本原因：Conda 环境中的 libgomp.so.1 或 libstdc++.so.6 与系统库冲突。
            # 方案：执行 GLOMAP 时，强制屏蔽 Conda 相关库路径，只允许使用系统库。
            for env_var in ["LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH"]:
                if env_var in cmd_env:
                    del cmd_env[env_var]
            
            # 移除 PATH 中的 Conda 路径，防止 subprocess 内部逻辑意外加载
            current_path = cmd_env.get("PATH", "")
            clean_paths = [p for p in current_path.split(os.pathsep) if "miniconda" not in p]
            cmd_env["PATH"] = os.pathsep.join(clean_paths)
            
            # 显式禁止加载任何非系统库
            cmd_env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib"

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cmd_env
            )
            for line in process.stdout:
                if any(k in line for k in ["Error", "Warning", "Elapsed", "image pairs", "CUDA error"]):
                    print(f"    | {line.strip()}")
            process.wait()

            if process.returncode != 0:
                # 🟢 [优化] 如果是 GPU 模式执行失败且允许重试，则自动切换 CPU 模式
                if retry_cpu:
                    print(f"    ⚠️ GPU 执行失败，尝试自动切换至 CPU 模式重试...")
                    self.colmap_use_gpu = False
                    new_cmd = list(cmd)
                    # 查找是否存在 use_gpu 参数并将其设为 0
                    found_gpu_flag = False
                    for i, arg in enumerate(new_cmd):
                        if "use_gpu" in arg:
                            if i + 1 < len(new_cmd):
                                new_cmd[i+1] = "0"
                                found_gpu_flag = True
                        if "gpu_index" in arg:
                            if i + 1 < len(new_cmd):
                                new_cmd[i+1] = "-1"
                    
                    if found_gpu_flag:
                        return self._run_cmd(new_cmd, f"{desc} (CPU 降级模式)", retry_cpu=False)
                
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
