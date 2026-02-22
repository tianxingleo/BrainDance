import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# 引入基类
from src.core.pipeline_base import BasePipeline

# 引入配置
from src.config import PipelineConfig

# 引入所有业务模块
from src.modules.image_proc import ImageProcessor
from src.modules.scene_analyzer import SceneAnalyzer
from src.modules.da3_runner import DA3Runner

# 引入辅助工具
from src.utils.common import format_duration

class DA3FeedForwardPipeline(BasePipeline):
    """
    【视频 -> 3DGS (前馈直接生成)】特殊流水线
    逻辑：视频抽帧 -> (可选) AI质检 -> DA3-Streaming解算 -> 直接基于反投影构建GS模型并行导出
    特点：无需经过 Nerfstudio 训练，由 DA3 直接提供密集点云构建 3DGS!
    """

    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.log("🎬 启动 DA3 Feed-Forward (直接反投影) 3DGS 流水线...")
        self.log(f"📄 输入文件: {input_path}")

        # ==========================================
        # 0. 初始化配置 (动态构建 Config)
        # ==========================================
        video_path_obj = Path(input_path)
        
        cfg = PipelineConfig(
            project_name=self.scene_id,
            video_path=video_path_obj,
            mapper_type="da3"  # 强制绑定 DA3
        )
        cfg.project_dir = Path(self.work_dir)
        
        # 参数调整
        cfg.iterations = 0 # 实际上不需要 Nerfstudio 训练
        
        global_start_time = time.time()
        pipeline_metadata = {}

        # ==========================================
        # 1. 实例化核心模块 
        # ==========================================
        img_processor = ImageProcessor(cfg, log_callback=self.log)
        scene_analyzer = SceneAnalyzer(cfg)
        da3_runner = DA3Runner(cfg, log_callback=self.log)

        # ==========================================
        # Step 1: 数据准备 (视频抽帧)
        # ==========================================
        self.log("🎬 [1/3] 开始视频抽帧与图片预处理...")
        cfg.project_dir.mkdir(parents=True, exist_ok=True)
        
        dest_video_path = cfg.project_dir / video_path_obj.name
        if not dest_video_path.exists():
            shutil.copy(str(video_path_obj), str(dest_video_path))
        
        temp_dir = cfg.project_dir / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.log("    -> 正在进行 FFmpeg 抽帧 (1 FPS, 720p Lanczos 超采样)...")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(dest_video_path),
                "-vf", "fps=1,scale=1280:720:flags=lanczos",
                "-q:v", "2",
                str(temp_dir / "frame_%05d.jpg")
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg 抽帧失败: {e}")
            
        self.log("    -> FFmpeg 抽帧完成")

        # 图片清洗 (去模糊)
        img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)

        raw_images_dir = cfg.project_dir / "raw_images"
        raw_images_dir.mkdir(parents=True, exist_ok=True)
        
        all_imgs = sorted(list(temp_dir.glob("*")))
        limit = cfg.max_images
        
        if len(all_imgs) > limit:
            indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
            all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
            
        for img in all_imgs:
            shutil.copy2(str(img), str(raw_images_dir / img.name))
            
        shutil.rmtree(temp_dir)
        self.log(f"    -> 图片准备完成，共 {len(all_imgs)} 张")

        # ==========================================
        # Step 1.5: AI 质检
        # ==========================================
        if cfg.enable_scene_analysis:
            self.log(f"🧐 [AI 质检] 阈值: {cfg.min_quality_score} 分")
            passed, score, reason, tags, description, objects = scene_analyzer.run(
                raw_images_dir, 
                log_callback=lambda msg: self.log(msg) 
            )

            status_icon = "✅" if passed else "❌"
            self.log(f"    -> 结果: {status_icon} {score}分 (评价: {reason})")
            self.log(f"    -> 标签: {tags}")

            pipeline_metadata.update({
                "ai_score": score,
                "ai_tags": tags,
                "ai_reason": reason,
                "ai_description": description,
                "ai_objects": objects
            })

            if not passed:
                err_msg = f"AI 质检不通过 ({score}分 < {cfg.min_quality_score}分): {reason}"
                self.log(err_msg, level="ERROR")
                raise RuntimeError(err_msg)

        # ==========================================
        # Step 2: DA3 位姿与深度解算
        # ==========================================
        self.log("⚙️ [2/3] 正在执行 DA3 纯位姿解算...")
        # 依赖于 DA3Runner()，只需运行它得到 DA3 的 `da3_output` 目录
        colmap_output_dir = cfg.data_dir / "colmap"
        da3_output_dir = colmap_output_dir / "da3_output"
        
        if not da3_runner.run():
            err_msg = "❌ Pipeline 中断：DA3 解算失败"
            self.log(err_msg, level="ERROR")
            raise RuntimeError(err_msg)
        self.log("    -> DA3 解算完成")

        # ==========================================
        # Step 3: DA3 Direct Feed-Forward 输出 3DGS
        # ==========================================
        self.log("🧠 [3/3] 开始 DA3 直接反投影构建 3DGS (绕过 Nerfstudio 训练)...")
        # 直接调用 feed_forward_3dgs_from_streaming.py 
        # 要求 DA3 仓库中有此脚本
        
        ff_script = cfg.da3_repo_path / "feed_forward_3dgs_from_streaming.py"
        if not ff_script.exists():
            raise FileNotFoundError(f"找不到 DA3 快速导出脚本: {ff_script}")

        # 设置参数
        feed_forward_out_dir = cfg.project_dir / "output_feed_forward"
        feed_forward_out_dir.mkdir(parents=True, exist_ok=True)
        
        frame_interval = str(params.get("frame_interval", 5))
        conf_threshold = str(params.get("conf_threshold", 0.5))

        cmd = [
            "python", str(ff_script),
            "--streaming-dir", str(da3_output_dir),
            "--output-dir", str(feed_forward_out_dir),
            "--frame-interval", frame_interval,
            "--conf-threshold", conf_threshold
        ]
        
        self.log(f"    => 运行脚本: {' '.join(cmd)}")
        
        # 传递环境变量保证 Python 路径正常
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        da3_src_path = cfg.da3_repo_path / "src"
        env["PYTHONPATH"] = f"{str(cfg.da3_repo_path)}{os.pathsep}{str(da3_src_path)}{os.pathsep}{pythonpath}"

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )
            # 🟢 [实时日志] 读取标准输出并同步回传
            for line in process.stdout:
                line = line.strip()
                if not line: continue
                # 过滤一些极其频繁的进度日志，只保留关键信息
                if any(k in line for k in ["生成", "归一化", "合并", "导出", "大功告成", "Error", "Total points"]):
                    self.log(f"    | {line}")
                elif "Progress" in line or "100%" in line:
                    self.log(f"    | {line}")
            
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.CalledProcessError as e:
            err_msg = f"❌ DA3 反投影执行崩溃 (代码 {e.returncode})"
            self.log(err_msg, level="ERROR")
            raise RuntimeError(err_msg)
        
        # Pipeline expects output to file `0000_perfect_merged.ply` inside `gs_ply` folder
        final_ply_path = feed_forward_out_dir / "gs_ply" / "0000_perfect_merged.ply"
        if not final_ply_path.exists():
             raise FileNotFoundError(f"预期输出文件未找到: {final_ply_path}")

        self.log(f"💾 导出 3DGS PLY 成功: {final_ply_path}")
        self.log(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")
        
        # 上传 PLY 并在 model_assets 中写入记录（非强制，内部容错）
        try:
            self.upload_and_record(str(final_ply_path), pipeline_metadata, params)
        except Exception:
            # upload_and_record 内部已捕获异常，这里保证不抛出
            pass

        return str(final_ply_path), pipeline_metadata

    def cleanup(self):
        """
        重写清理逻辑
        """
        self.log("🧹 正在清理临时文件...")
        pass
