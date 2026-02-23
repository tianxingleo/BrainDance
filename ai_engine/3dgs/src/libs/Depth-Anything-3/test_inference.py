#!/usr/bin/env python3
"""
Depth Anything 3 测试脚本
"""
import os
import glob
import torch
import numpy as np
import cv2
from depth_anything_3.api import DepthAnything3

# 设置环境变量使用 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
print("正在加载模型...")
model = DepthAnything3.from_pretrained('depth-anything/DA3-SMALL')
model = model.to(device=device)
print("模型加载完成！")

# 测试图片
image_paths = sorted(glob.glob('assets/examples/SOH/*.png'))
print(f"找到 {len(image_paths)} 张测试图片")

# 处理每张图片
for i, image_path in enumerate(image_paths):
    print(f"\n处理图片 {i+1}/{len(image_paths)}: {image_path}")

    # 推理
    prediction = model.inference([image_path])

    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.basename(image_path).replace('.png', '')

    # 保存处理后的图片
    processed = prediction.processed_images[0]
    cv2.imwrite(f'{output_dir}/{basename}_processed.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    # 保存深度图（归一化到 0-255）
    depth = prediction.depth[0]
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    cv2.imwrite(f'{output_dir}/{basename}_depth.png', depth_normalized)

    # 保存置信度图
    conf = prediction.conf[0]
    conf_normalized = ((conf - conf.min()) / (conf.max() - conf.min()) * 255).astype(np.uint8)
    cv2.imwrite(f'{output_dir}/{basename}_conf.png', conf_normalized)

    print(f"  深度图形状: {depth.shape}")
    print(f"  深度范围: {depth.min():.4f} - {depth.max():.4f}")
    print(f"  置信度范围: {conf.min():.4f} - {conf.max():.4f}")

print("\n所有图片处理完成！结果保存在 output/ 目录")
