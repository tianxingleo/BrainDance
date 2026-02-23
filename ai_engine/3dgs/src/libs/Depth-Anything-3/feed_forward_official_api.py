#!/usr/bin/env python3
"""
使用官方 API 一次性处理所有帧
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, '/home/ltx/projects/Depth-Anything-3/src')

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streaming-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output/feed_forward_official')
    parser.add_argument('--model-name', type=str, default='da3-giant')
    parser.add_argument('--frame-interval', type=int, default=10)
    args = parser.parse_args()

    streaming_dir = Path(args.streaming_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    results_dir = streaming_dir / "results_output"
    npz_files = sorted(results_dir.glob("frame_*.npz"))

    print(f"找到 {len(npz_files)} 个 NPZ 文件")

    # 抽帧
    frame_indices = list(range(0, len(npz_files), args.frame_interval))
    print(f"选择 {len(frame_indices)} 帧进行处理")

    # 准备图像列表和 intrinsics
    image_paths = []
    all_intrinsics = []
    for idx in frame_indices:
        # NPZ 文件名是 frame_N.npz，对应的图像是 extracted/frame_NNNNNN.png
        npz_file = npz_files[idx]
        frame_num = int(npz_file.stem.split('_')[1])
        img_path = streaming_dir / "extracted" / f"frame_{frame_num:06d}.png"
        if img_path.exists():
            image_paths.append(str(img_path))
            # 读取 intrinsics
            data = np.load(npz_file)
            all_intrinsics.append(data['intrinsics'])

    print(f"找到 {len(image_paths)} 张图像")

    # 加载位姿（c2w 格式）
    poses_file = streaming_dir / "camera_poses.txt"
    c2w_poses = np.loadtxt(poses_file).reshape(-1, 4, 4)

    # 转换为 w2c 格式
    w2c_poses = np.linalg.inv(c2w_poses)

    # 只保留选中的帧
    selected_w2c = w2c_poses[frame_indices][:len(image_paths)]
    selected_intrinsics = np.array(all_intrinsics)[:len(image_paths)]

    # 坐标系归一化
    print("🔧 执行坐标系归一化...")
    first_w2c_inv = np.linalg.inv(selected_w2c[0])
    for i in range(len(selected_w2c)):
        selected_w2c[i] = first_w2c_inv @ selected_w2c[i]

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    da3 = DepthAnything3(model_name=args.model_name).to(device)
    da3.eval()

    # 使用官方 API 推理
    print("🚀 开始 DA3 推理...")
    with torch.no_grad():
        result = da3.inference(
            image=image_paths[:50],  # 限制最多50帧
            process_res=504,
            infer_gs=True,
            export_format='',
            extrinsics=selected_w2c[:50],  # w2c 格式
            intrinsics=selected_intrinsics[:50],  # intrinsics
        )

    if hasattr(result, 'gaussians') and result.gaussians is not None:
        print(f"✅ 生成 Gaussians 成功!")
        print(f"   means shape: {result.gaussians.means.shape}")
        print(f"   harmonics shape: {result.gaussians.harmonics.shape}")

        # 导出 PLY
        from depth_anything_3.utils.export.gs import export_to_gs_ply
        export_to_gs_ply(
            prediction=result,
            export_dir=str(output_dir),
            gs_views_interval=1
        )
        print(f"✅ PLY 文件已保存至: {output_dir}/gs_ply/")
    else:
        print("❌ 没有生成 Gaussians")

if __name__ == '__main__':
    main()
