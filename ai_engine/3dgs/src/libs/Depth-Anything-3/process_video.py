#!/usr/bin/env python3
"""
视频深度估计脚本
"""
# 必须在第一个 import 之前设置 HF_ENDPOINT
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import torch
import glob
from depth_anything_3.api import DepthAnything3
import numpy as np
from tqdm import tqdm

# 配置
VIDEO_PATH = '/home/ltx/projects/SuGaR/video.mp4'
OUTPUT_DIR = '/home/ltx/projects/Depth-Anything-3/output/sugar_video'
MODEL_NAME = 'depth-anything/DA3-BASE'  # 可选: SMALL, BASE, LARGE, GIANT
TARGET_FPS = 1  # 每秒抽一帧
RESOLUTION = 720  # 处理分辨率（最长边）

print('=' * 60)
print('视频深度估计处理')
print('=' * 60)
print(f'视频路径: {VIDEO_PATH}')
print(f'输出目录: {OUTPUT_DIR}')
print(f'使用模型: {MODEL_NAME}')
print(f'目标 FPS: {TARGET_FPS}')
print(f'处理分辨率: {RESOLUTION}')
print('=' * 60)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/depth', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/conf', exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / original_fps

print(f'\n视频信息:')
print(f'  原始 FPS: {original_fps:.2f}')
print(f'  总帧数: {total_frames}')
print(f'  时长: {duration:.2f}s')

# 计算采样间隔
frame_interval = int(original_fps / TARGET_FPS)
print(f'  采样间隔: 每 {frame_interval} 帧取 1 帧')
print(f'  预计处理: {total_frames // frame_interval} 帧')

# 加载模型
print(f'\n正在加载模型 {MODEL_NAME}...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DepthAnything3.from_pretrained(MODEL_NAME)
model = model.to(device=device)
print('模型加载完成！')

# 处理视频帧
print('\n开始处理视频帧...')
frame_count = 0
processed_count = 0
times = []

with tqdm(total=total_frames, desc='处理进度') as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 只处理采样帧
        if frame_count % frame_interval == 0:
            # 调整分辨率
            h, w = frame.shape[:2]
            if max(h, w) > RESOLUTION:
                scale = RESOLUTION / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 保存原始帧
            frame_filename = f'{OUTPUT_DIR}/depth/frame_{processed_count:06d}.png'
            cv2.imwrite(frame_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # 深度估计
            import time
            start = time.time()
            prediction = model.inference([frame_rgb])
            torch.cuda.synchronize()
            times.append(time.time() - start)
            
            # 保存深度图
            depth = prediction.depth[0]
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            cv2.imwrite(f'{OUTPUT_DIR}/depth/depth_{processed_count:06d}.png', depth_normalized)
            
            # 保存置信度图
            conf = prediction.conf[0]
            conf_normalized = ((conf - conf.min()) / (conf.max() - conf.min()) * 255).astype(np.uint8)
            cv2.imwrite(f'{OUTPUT_DIR}/conf/conf_{processed_count:06d}.png', conf_normalized)
            
            processed_count += 1
            pbar.update(frame_interval)
            
            # 显示进度信息
            avg_time = sum(times) / len(times)
            fps = 1 / avg_time
            pbar.set_postfix({
                '已处理': processed_count,
                '平均耗时': f'{avg_time*1000:.1f}ms',
                'FPS': f'{fps:.1f}'
            })
        
        frame_count += 1

cap.release()

# 统计信息
print('\n' + '=' * 60)
print('处理完成！')
print('=' * 60)
print(f'总处理帧数: {processed_count}')
print(f'平均推理时间: {sum(times)/len(times)*1000:.1f}ms')
print(f'平均 FPS: {1/(sum(times)/len(times)):.1f}')
print(f'总处理时间: {sum(times):.2f}s')
print(f'\n输出文件:')
print(f'  原始帧: {OUTPUT_DIR}/depth/')
print(f'  深度图: {OUTPUT_DIR}/depth/*.png')
print(f'  置信度: {OUTPUT_DIR}/conf/*.png')

# 生成处理信息
info = {
    'video_path': VIDEO_PATH,
    'model': MODEL_NAME,
    'original_fps': original_fps,
    'target_fps': TARGET_FPS,
    'frame_interval': frame_interval,
    'total_frames': total_frames,
    'processed_frames': processed_count,
    'avg_inference_time_ms': float(sum(times)/len(times)*1000),
    'fps': float(1/(sum(times)/len(times))),
    'total_time_s': float(sum(times)),
}

import json
with open(f'{OUTPUT_DIR}/info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f'\n处理信息已保存: {OUTPUT_DIR}/info.json')
print('=' * 60)
