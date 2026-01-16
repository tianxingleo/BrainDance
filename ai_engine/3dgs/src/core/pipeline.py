# [流程] 存放 run_pipeline 函数
# src/core/pipeline.py
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np # 你的代码里用到了 np.linspace

# 1. 引入配置
from src.config import PipelineConfig

# 2. 引入所有模块 (Worker)
from src.modules.image_proc import ImageProcessor
from src.modules.glomap_runner import GlomapRunner
from src.modules.ai_segmentor import AISegmentor
from src.modules.nerf_engine import NerfstudioEngine
from src.modules.scene_analyzer import SceneAnalyzer # 🟢 引入新模块

# 3. 引入辅助工具
from src.utils.common import format_duration

def run_pipeline(cfg: PipelineConfig, log_callback=None):
    """
    log_callback: 一个函数，接受字符串参数，例如 log_callback("Step 1: 开始处理...")
    """
    
    # 定义一个内部辅助函数，同时打印到控制台和发送给回调
    def log(message):
        print(message) # 打印到本地终端
        if log_callback:
            # 加上时间戳让日志更专业
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_callback(f"[{timestamp}] {message}")

    global_start_time = time.time()
    project_name = cfg.project_name
    log(f"🚀 [Pipeline] 启动任务: {project_name}")

    # 初始化返回的元数据 (用于更新数据库)
    pipeline_metadata = {} 
    
    # 1. 实例化所有模块
    img_processor = ImageProcessor(cfg)
    scene_analyzer = SceneAnalyzer(cfg) # 🟢 实例化
    # colmap_runner = ColmapRunner(cfg)
    glomap_runner = GlomapRunner(cfg) 
    ai_segmentor = AISegmentor(cfg)
    nerf_engine = NerfstudioEngine(cfg)

    # ==========================================
    # Step 1: 数据准备
    # ==========================================
    log(f"🎬 [1/4] 开始视频抽帧与图片预处理...")
    # 初始化目录
    if cfg.project_dir.exists(): shutil.rmtree(cfg.project_dir, ignore_errors=True)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(cfg.video_path), str(cfg.project_dir / cfg.video_path.name))
    
    # 抽帧
    temp_dir = cfg.project_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    log(f"    -> 正在进行 FFmpeg 抽帧...")
    subprocess.run(["ffmpeg", "-y", "-i", str(cfg.project_dir / cfg.video_path.name), 
                    "-vf", "fps=10", "-q:v", "2", 
                    str(temp_dir / "frame_%05d.jpg")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"    -> FFmpeg 抽帧完成")
    
    # 清洗
    img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)
    
    # 移动图片到 raw_images
    raw_images_dir = cfg.project_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    all_imgs = sorted(list(temp_dir.glob("*")))
    limit = cfg.max_images
    if len(all_imgs) > limit:
        indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
        all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
    for img in all_imgs: shutil.copy2(str(img), str(raw_images_dir / img.name))
    shutil.rmtree(temp_dir)
    log(f"    -> 图片准备完成，共 {len(all_imgs)} 张")

    # ==========================================
    # Step 1.5: AI 质检 (新增环节)
    # ==========================================
    if cfg.enable_scene_analysis:
        log(f"🧐 [AI 质检] 阈值: {cfg.min_quality_score} 分")
        
        # 接收 6 个返回值
        passed, score, reason, tags, description, objects = scene_analyzer.run(raw_images_dir, log_callback=log)
        
        # 🟢 记录日志
        status_icon = "✅" if passed else "❌"
        log(f"    -> 结果: {status_icon} {score}分 (评价: {reason})")
        log(f"    -> 标签: {tags}")
        
        # 🟢 [关键] 将结果存入 metadata，准备传给 worker
        pipeline_metadata["ai_score"] = score
        pipeline_metadata["ai_tags"] = tags
        pipeline_metadata["ai_reason"] = reason
        pipeline_metadata["ai_description"] = description
        pipeline_metadata["ai_objects"] = objects

        if not passed:
            err_msg = f"AI 质检不通过 ({score}分 < {cfg.min_quality_score}分): {reason}"
            log(err_msg)
            raise RuntimeError(err_msg)

    # ==========================================
    # Step 2: GLOMAP
    # ==========================================
    log(f"⚙️ [2/4] 正在进行位姿解算 (GLOMAP)...")
    if log_callback: 
        log_callback("提示: 解算过程可能较慢，请耐心等待...")
    
    if not glomap_runner.run():
        log("❌ Pipeline 中断：GLOMAP 失败")
        return None, pipeline_metadata
    log(f"    -> 位姿解算完成")

    # ==========================================
    # Step 3: AI
    # ==========================================
    log(f"🤖 [3/4] 正在进行 AI 语义分割...")
    ai_segmentor.run()
    log(f"    -> AI 处理完成")

    # ==========================================
    # Step 4: 训练
    # ==========================================
    log(f"🧠 [4/4] 开始 3DGS 训练...")
    try:
        nerf_engine.train()
        log(f"    -> 训练完成，开始导出...")
        final_ply_path = nerf_engine.export()
        
        log(f"💾 导出 PLY 完成: {final_ply_path}")
        log(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")
        return str(final_ply_path), pipeline_metadata
    except Exception as e:
        log(f"❌ 训练/导出阶段失败: {e}")
        return None, pipeline_metadata
    
    
