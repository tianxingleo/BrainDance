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

# 3. 引入辅助工具
from src.utils.common import format_duration

def run_pipeline(cfg: PipelineConfig):
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Engine] 启动任务: {cfg.project_name}")
    
    # 1. 实例化所有模块
    img_processor = ImageProcessor(cfg)
    # colmap_runner = ColmapRunner(cfg)
    glomap_runner = GlomapRunner(cfg) 
    ai_segmentor = AISegmentor(cfg)
    nerf_engine = NerfstudioEngine(cfg)

    # ==========================================
    # Step 1: 数据准备
    # ==========================================
    # 初始化目录
    if cfg.project_dir.exists(): shutil.rmtree(cfg.project_dir, ignore_errors=True)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(cfg.video_path), str(cfg.project_dir / cfg.video_path.name))
    
    # 抽帧 (这里逻辑简单，直接写这里也行，或者封装进 ImageProcessor)
    temp_dir = cfg.project_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(cfg.project_dir / cfg.video_path.name), 
                    "-vf", "fps=10", "-q:v", "2", 
                    str(temp_dir / "frame_%05d.jpg")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 清洗
    img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)
    
    # 移动图片到 raw_images
    raw_images_dir = cfg.project_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    # 简单的移动逻辑保留在这里，或者也可以移入 ImageProcessor
    all_imgs = sorted(list(temp_dir.glob("*")))
    limit = cfg.max_images
    if len(all_imgs) > limit:
        indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
        all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
    for img in all_imgs: shutil.copy2(str(img), str(raw_images_dir / img.name))
    shutil.rmtree(temp_dir)

    # ==========================================
    # Step 2: GOLMAP
    # ==========================================
    if not glomap_runner.run():
        print("❌ Pipeline 中断：GLOMAP 失败")
        return

    # ==========================================
    # Step 3: AI
    # ==========================================
    # 这里会自动判断是否开启，内部已处理异常
    ai_segmentor.run()

    # ==========================================
    # Step 4 & 5: 训练与导出
    # ==========================================
    try:
        nerf_engine.train()
        final_path = nerf_engine.export()
        print(f"\n🎉 任务完成！结果位于: {final_path}")
    except Exception as e:
        print(f"❌ 训练/导出阶段失败: {e}")

    print(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")
