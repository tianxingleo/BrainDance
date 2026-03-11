# src/modules/nerf_engine.py
# 功能：实现Nerfstudio训练引擎功能，执行3DGS训练和模型导出
# 实现：调用ns-train和ns-export命令，进行splatfacto模型训练和PLY导出
# 逻辑：1. 计算场景参数(Collider) 2. 执行splatfacto训练 3. 导出PLY模型 4. 点云后处理(切割)
# 包含：NerfstudioEngine类、训练方法、导出方法、点云后处理算法
import os
import shutil
import subprocess
from pathlib import Path

# --- 项目引用 ---
from src.config import PipelineConfig

# 关键：引入计算 Collider 的几何算法
from src.utils.geometry import analyze_and_calculate_adaptive_collider
# 关键：引入点云切割算法
from src.utils.ply_utils import perform_percentile_culling

class NerfstudioEngine:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = cfg.project_dir / "outputs"
        # 准备环境变量
        self.env = os.environ.copy()
        self.env["QT_QPA_PLATFORM"] = "offscreen"
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def train(self):
        """执行 splatfacto 训练"""
        print(f"\n🔥 [4/4] 开始训练 (Splatfacto)")
        
        # 确保输出目录及其父目录存在，防止 nerfstudio 内部创建失败
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.cfg.project_name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.cfg.project_name / "splatfacto").mkdir(parents=True, exist_ok=True)

        # 1. 计算场景参数 (Collider) - 直接调用之前的全局函数
        collider_args, scene_type = analyze_and_calculate_adaptive_collider(
            self.cfg.transforms_file,
            force_cull=self.cfg.force_spherical_culling,
            radius_scale=self.cfg.scene_radius_scale
        )
        self.scene_type = scene_type # 存下来给导出步骤用
        is_da3_mapper = (self.cfg.mapper_type or "").strip().lower() == "da3"
        stop_split_at = min(
            self.cfg.training_iterations,
            max(2000, self.cfg.training_iterations - 3000)
        )
        if is_da3_mapper:
            stop_split_at = min(
                stop_split_at,
                max(1500, min(4000, self.cfg.training_iterations // 3))
            )

        # 2. 组装命令
        cmd = [
            "ns-train", "splatfacto",
            "--data", str(self.cfg.data_dir),
            "--output-dir", str(self.output_dir),
            "--experiment-name", self.cfg.project_name,
            "--pipeline.model.random-init", "False",
            "--pipeline.model.background-color", "random",
            # 0.05 会把很多尚未稳定但几何上有价值的高斯提前删掉，容易越训越碎。
            "--pipeline.model.cull-alpha-thresh", "0.005",
            "--pipeline.model.stop-split-at", str(stop_split_at),
            # 视频序列里相邻帧高度相关，用 FPS 采样能减少局部视角过采样带来的撕裂。
            "--pipeline.datamanager.train-cameras-sampling-strategy", "fps",
            *collider_args,
            "--max-num-iterations", str(self.cfg.training_iterations),
            "--vis", "viewer+tensorboard",
            "--viewer.quit-on-train-completion", "True",
        ]

        if is_da3_mapper:
            cmd.extend([
                # DA3 初始化通常已经有较完整几何，后期继续频繁 refine/split 更容易越训越碎。
                "--pipeline.model.warmup-length", "1000",
                "--pipeline.model.refine-every", "200",
                "--pipeline.model.reset-alpha-every", "60",
                "--pipeline.model.densify-grad-thresh", "0.0004",
                "--pipeline.model.densify-size-thresh", "0.02",
                "--pipeline.model.split-screen-size", "0.08",
                "--pipeline.model.stop-screen-size-at", "2000",
                "--pipeline.model.use-scale-regularization", "True",
            ])

        cmd.extend([
            "nerfstudio-data",
            "--downscale-factor", "1",
            "--auto-scale-poses", "False",
        ])
        
        # 3. 执行
        subprocess.run(cmd, check=True, env=self.env)

    def export(self):
        """导出 ply 并进行后处理"""
        print(f"\n💾 正在导出...")
        # 找到最新的 config.yml
        search_path = self.output_dir / self.cfg.project_name / "splatfacto"
        try:
            run_dirs = sorted(list(search_path.glob("*")))
            config_path = run_dirs[-1] / "config.yml"
        except IndexError:
            print("❌ 未找到训练结果 config.yml")
            return None

        # 导出命令
        subprocess.run([
            "ns-export", "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(self.cfg.project_dir)
        ], check=True, env=self.env)

        # 导出相机 (用于空间语义锚点)
        cameras_export_dir = self.cfg.project_dir / "cameras_export"
        if cameras_export_dir.exists():
            shutil.rmtree(str(cameras_export_dir))
        cameras_export_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run([
                "ns-export", "cameras", "--load-config", str(config_path), 
                "--output-dir", str(cameras_export_dir)
            ], check=True, env=self.env)
        except Exception as e:
            print(f"⚠️ 无法使用 ns-export cameras 导出相机, 可能是版本不支持: {e}")

        # 后处理：点云切割
        raw_ply = self.cfg.project_dir / "point_cloud.ply"
        if not raw_ply.exists(): raw_ply = self.cfg.project_dir / "splat.ply"
        cleaned_ply = self.cfg.project_dir / "point_cloud_cleaned.ply"
        final_ply = raw_ply

        # 判断是否需要切割 (物体模式 or 强制切割)
        need_cull = (self.scene_type == "object" or self.cfg.force_spherical_culling)
        
        if need_cull and raw_ply.exists():
            # 调用之前的全局函数
            success = perform_percentile_culling(
                raw_ply, 
                self.cfg.transforms_file, 
                cleaned_ply,
                keep_percentile=self.cfg.keep_percentile
            )
            if success:
                final_ply = cleaned_ply

        # 复制结果到 results 目录
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        target_path = results_dir / f"{self.cfg.project_name}.ply"
        shutil.copy2(str(final_ply), str(target_path))
        
        return target_path
