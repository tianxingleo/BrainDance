#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/ltx/projects/Depth-Anything-3/src')

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply

def load_streaming_results(streaming_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    results_dir = streaming_dir / "results_output"
    npz_files = sorted(results_dir.glob("frame_*.npz"))
    if not npz_files: raise ValueError("没找到 npz 文件")
    
    first_data = np.load(npz_files[0])
    H, W = first_data['image'].shape[:2]
    N = len(npz_files)

    images = np.zeros((N, H, W, 3), dtype=np.uint8)
    depths = np.zeros((N, H, W), dtype=np.float32)
    confs = np.zeros((N, H, W), dtype=np.float32)
    intrinsics = np.zeros((N, 3, 3), dtype=np.float32)

    print("加载 DA3stream 输出数据...")
    for i, npz_file in enumerate(tqdm(npz_files, desc="加载数据")):
        data = np.load(npz_file)
        images[i] = data['image']
        depths[i] = data['depth']
        confs[i] = data['conf']
        intrinsics[i] = data['intrinsics']

    poses_file = streaming_dir / "camera_poses.txt"
    extrinsics = np.loadtxt(poses_file).reshape(-1, 4, 4)

    min_n = min(N, extrinsics.shape[0])
    return images[:min_n], depths[:min_n], confs[:min_n], extrinsics[:min_n], intrinsics[:min_n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streaming-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output/feed_forward_3dgs')
    parser.add_argument('--model-name', type=str, default='da3-giant')
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--conf-threshold', type=float, default=0.85)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images, depths, confs, extrinsics, intrinsics = load_streaming_results(Path(args.streaming_dir))
    
    # 抽帧
    frame_indices = list(range(0, len(images), args.frame_interval))
    print(f"\n选择 {len(frame_indices)} 帧进行 GS 融合...")

    # ========================================================
    # 🔑 核心数学魔法：坐标系原点归一化 (解决"碎片化"问题)
    # 将第一帧的位姿作为世界中心，避免外部 COLMAP 的大坐标导致截断误差和投影错乱
    # ========================================================
    print("🔧 执行坐标系归一化（将第一帧定为世界原点）...")
    first_ext_inv = np.linalg.inv(extrinsics[frame_indices[0]])
    
    for i in range(len(extrinsics)):
        # 矩阵乘法：新的位姿 = 原始位姿 * 第一帧位姿的逆
        extrinsics[i] = extrinsics[i] @ first_ext_inv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    da3 = DepthAnything3(model_name=args.model_name).to(device)
    da3.eval()

    image_list = [Image.fromarray(images[idx]) for idx in frame_indices]
    filtered_ext = extrinsics[frame_indices]
    filtered_int = intrinsics[frame_indices]

    print("\n🚀 开始安全的前向渲染融合...")
    
    # 直接调用官方高级 API，避免底层组装报错
    with torch.no_grad():
        result = da3.inference(
            image=image_list,
            extrinsics=filtered_ext,  # 传入已经归一化的高精度位姿
            intrinsics=filtered_int,
            align_to_input_ext_scale=False, # 保持物理尺度，不让 DA3 乱改
            infer_gs=True,
            export_dir=str(output_dir),
            export_format='mini_npz'
        )

    if result.gaussians is not None:
        ply_dir = output_dir / "gs_ply"
        ply_dir.mkdir(parents=True, exist_ok=True)
        ply_path = ply_dir / "0000.ply"

        print("💾 正在导出标准 PLY 文件...")
        pred_depth = torch.from_numpy(result.depth).unsqueeze(-1).to(result.gaussians.means.device)
        
        # 官方的保存函数：自带 RGB -> SH 和 scale.log() 的安全处理
        save_gaussian_ply(
            gaussians=result.gaussians,
            save_path=str(ply_path),
            ctx_depth=pred_depth,
            shift_and_scale=False,
            save_sh_dc_only=True,     # 强制仅保存球谐DC，确保颜色渲染兼容
            gs_views_interval=1,
            inv_opacity=True,
            prune_by_depth_percent=0.75, # 剪除超远距离的飞点
            prune_border_gs=True,
            match_3dgs_mcmc_dev=False
        )
        print(f"✓ 完美模型已保存至: {ply_path}")
    else:
        print("⚠️ 未生成高斯球")

if __name__ == '__main__':
    main()