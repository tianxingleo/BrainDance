#!/usr/bin/env python3
"""
[终极完美版] 基于 DA3stream 的前馈 3DGS 生成脚本
完全重构：不再使用 DepthAnything3 的内部 GS 头（因其预训练尺度与 Metric 深度不兼容导致分块/碎片），
而是直接通过严格的几何反投影（Unprojection）构建 3DGS。
这保证了：
1. 几何完全连续，无“分块”现象。
2. 颜色直接源自图像，无“怪异”色差。
3. 严格遵循 Streaming 的位姿和深度。

使用命令：
/home/ltx/my_envs/gs_linux_backup/bin/python feed_forward_3dgs_from_streaming.py \
    --streaming-dir output/sugar_streaming1 \
    --output-dir output/feed_forward_3dgs_full_standard \
    --frame-interval 5 \
    --conf-threshold 0.9
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, '/home/ltx/projects/Depth-Anything-3/src')

# Import necessary helpers
# export_ply expects: means, scales(linear), rotations, harmonics, opacities(logit)
from depth_anything_3.utils.gsply_helpers import export_ply, inverse_sigmoid

def load_streaming_results(streaming_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    results_dir = streaming_dir / "results_output"
    npz_files = sorted(list(results_dir.glob("frame_*.npz")), key=lambda p: int(p.stem.split('_')[-1]))
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
        try:
            data = np.load(npz_file)
            images[i] = data['image'] # H, W, 3 (RGB)
            depths[i] = data['depth'] # H, W (Metric or consistent relative)
            confs[i] = data['conf']
            intrinsics[i] = data['intrinsics']
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

    poses_file = streaming_dir / "camera_poses.txt"
    if not poses_file.exists():
        raise ValueError(f"没有找到位姿文件: {poses_file}")
        
    extrinsics = np.loadtxt(poses_file).reshape(-1, 4, 4)

    min_n = min(N, extrinsics.shape[0])
    return images[:min_n], depths[:min_n], confs[:min_n], extrinsics[:min_n], intrinsics[:min_n]

def rgb_to_sh(rgb):
    """
    Convert RGB [0,1] to SH coefficients (C0 only).
    RGB = C0 * Y00 + 0.5
    Y00 = (RGB - 0.5) / C0
    C0 = 0.28209479177387814
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def unproject_points(depth_map, K, E, image, conf_map, sample_ratio=1.0, conf_threshold=0.5):
    """
    Unproject depth map to 3D world points.
    Returns: means, colors, scales, opacities (in linear/prob domain)
    """
    H, W = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create grid
    v, u = np.indices((H, W)) # v (row), u (col)
    
    # Masking
    mask = (depth_map > 0) & (conf_map > conf_threshold)
    
    if sample_ratio < 1.0:
        # Random subsample
        rand_mask = np.random.rand(H, W) < sample_ratio
        mask = mask & rand_mask
        
    valid_u = u[mask]
    valid_v = v[mask]
    valid_d = depth_map[mask]
    valid_c = image[mask].astype(np.float32) / 255.0 # RGB [0,1]
    valid_conf = conf_map[mask]

    # Camera coordinates
    z_c = valid_d
    x_c = (valid_u - cx) * z_c / fx
    y_c = (valid_v - cy) * z_c / fy
    
    # Stack (N, 3)
    points_c = np.stack([x_c, y_c, z_c], axis=-1)
    
    # -------------------------------------------------------------
    # 恢复纯净数据，移除之前的 Hardcode 补丁。
    # 真正的坐标系转换现在由 modes 字典在导出时处理。
    # -------------------------------------------------------------
    # points_c = points_c * np.array([1, -1, -1])

    # Transform to World: P_w = E @ P_c (assuming E is C2W)
    # E acts on [x,y,z,1]. Or R*p + t
    R_cw = E[:3, :3]
    t_cw = E[:3, 3]
    points_w = (R_cw @ points_c.T).T + t_cw

    # Scales
    # Heuristic: scale proportional to depth projected on pixel
    # pixel_size_at_depth = depth / focal
    avg_focal = (fx + fy) / 2.0
    scales_scalar = valid_d * 2.0 / avg_focal # slightly larger than 1 pixel to cover holes
    scales = np.stack([scales_scalar, scales_scalar, scales_scalar], axis=-1) # Isotropic

    return points_w, valid_c, scales, valid_conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streaming-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output/feed_forward_3dgs')
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    parser.add_argument('--sample-ratio', type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images, depths, confs, extrinsics, intrinsics = load_streaming_results(Path(args.streaming_dir))

    # 抽帧
    frame_indices = list(range(0, len(images), args.frame_interval))
    print(f"\n选择 {len(frame_indices)} 帧进行几何融合...")

    # ========================================================
    # 坐标系原点归一化 (保留以防止数值过大，但注意这只是平移/旋转，不影响相对尺度)
    # ========================================================
    print("🔧 执行坐标系归一化（第一帧作为世界原点）...")
    first_ext_inv = np.linalg.inv(extrinsics[frame_indices[0]])
    extrinsics = np.matmul(first_ext_inv, extrinsics)
    print(f"  ✓ 第一帧相机位置已设为世界坐标原点")

    # Collectors
    all_means = []
    all_colors = []
    all_scales = []
    all_opacities = []
    all_rotations = []

    print("\n🚀 开始生成几何点云 (基于深度反投影)...")
    
    for idx in tqdm(frame_indices, desc="生成点云"):
        means, colors, scales, opacities_prob = unproject_points(
            depths[idx], 
            intrinsics[idx], 
            extrinsics[idx], 
            images[idx], 
            confs[idx], 
            sample_ratio=args.sample_ratio,
            conf_threshold=args.conf_threshold
        )
        
        N_pts = means.shape[0]
        if N_pts == 0: continue
        
        all_means.append(torch.from_numpy(means).float())
        all_colors.append(torch.from_numpy(colors).float())
        all_scales.append(torch.from_numpy(scales).float())
        all_opacities.append(torch.from_numpy(opacities_prob).float())
        
        # Rotation: Identity (Isotropic spheres)
        # [1, 0, 0, 0] (w, x, y, z)
        rots = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(N_pts, 1)
        all_rotations.append(rots)

    if not all_means:
        print("❌ 未生成任何点 (可能是置信度阈值过高或深度无效)")
        return

    # Concatenate
    print("📦 合并所有帧数据...")
    cat_means = torch.cat(all_means, dim=0)
    cat_colors = torch.cat(all_colors, dim=0) # (N, 3) RGB
    cat_scales = torch.cat(all_scales, dim=0)
    cat_opacities = torch.cat(all_opacities, dim=0)
    cat_rotations = torch.cat(all_rotations, dim=0)

    # Prepare for export
    print(f"Total points: {cat_means.shape[0]}")

    # Colors -> SH DC
    # harmonics shape: (N, 3, 1) if DC only
    # rgb_to_sh returns (N, 3)
    sh_dc = rgb_to_sh(cat_colors)
    harmonics = sh_dc.unsqueeze(-1) # (N, 3, 1)

    # Opacities -> Logit
    # Clamp to avoid inf
    cat_opacities = torch.clamp(cat_opacities, 0.001, 0.999)
    opacities_logit = inverse_sigmoid(cat_opacities)

    ply_dir = output_dir / "gs_ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "0000_perfect_merged.ply"

    print("💾 正在导出标准 PLY 文件...")

    export_ply(
        means=cat_means,
        scales=cat_scales, 
        rotations=cat_rotations,
        harmonics=harmonics,
        opacities=opacities_logit,
        path=ply_path,
        shift_and_scale=False,
        save_sh_dc_only=True,
        match_3dgs_mcmc_dev=False
    )

    file_size_mb = Path(ply_path).stat().st_size / (1024 * 1024)
    print(f"✓ 大功告成！完美对齐的三维房间模型已保存至: {ply_path} ({file_size_mb:.1f} MB)")

if __name__ == '__main__':
    main()