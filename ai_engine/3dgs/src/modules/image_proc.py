# src/modules/image_proc.py
# 功能：实现图像处理功能，包括模糊图像过滤和图像质量分析
# 实现：使用OpenCV进行图像分析，通过拉普拉斯算子计算图像清晰度
# 逻辑：1. 遍历图像文件夹 2. 计算每张图像的清晰度得分 3. 根据得分过滤模糊图像 4. 控制图像数量
# 包含：ImageProcessor类、智能模糊图像过滤算法、图像质量分析方法
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List

# 引入项目配置
from src.config import PipelineConfig

class ImageProcessor:
    def __init__(self, config: PipelineConfig, log_callback=None):
        self.cfg = config
        self.log_callback = log_callback or print

    def smart_filter_blurry_images(self, image_folder, keep_ratio=0.85):
        """
        [图像清洗算法] 混合策略模糊检测
        """
        self.log_callback(f"🧠 [智能清洗] 正在分析图片质量 (混合策略版)...")
        image_dir = Path(image_folder)
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not images: return
        
        trash_dir = image_dir.parent / "trash_smart"
        trash_dir.mkdir(exist_ok=True)
        
        # --- 这里的代码保持不变，直到 good_images 计算完毕 ---
        img_scores = []
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            grid_h, grid_w = h // 3, w // 3
            max_grid_score = 0
            for r in range(3):
                for c in range(3):
                    roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    score = cv2.Laplacian(roi, cv2.CV_64F).var()
                    if score > max_grid_score: max_grid_score = score
            img_scores.append((img_path, max_grid_score))
            if i % 50 == 0: self.log_callback(f"    -> 分析中... {i}/{len(images)}")
        
        scores = [s[1] for s in img_scores]
        if not scores: return
        quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
        
        good_images = []
        for img_path, score in img_scores:
            if score < quality_threshold:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
            else:
                good_images.append(img_path)
        
        # ======================================================
        # 🔥 核心修改在这里 🔥
        # ======================================================
        # 从配置对象中读取最大图片数量
        max_imgs = self.cfg.max_images  
        
        # 使用 max_imgs 替代原来的 max_images
        if len(good_images) > max_imgs:
            self.log_callback(f"    ⚠️ 图片过多 ({len(good_images)} 张), 正在降采样至 {max_imgs} 张...")
            # np.linspace 生成均匀分布的索引
            indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_imgs, dtype=int))
            for idx, img_path in enumerate(good_images):
                if idx not in indices_to_keep:
                    shutil.move(str(img_path), str(trash_dir / img_path.name))
                    
        self.log_callback(f"✨ 清洗结束，剩余 {len(list(image_dir.glob('*')))} 张。")
