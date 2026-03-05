#!/usr/bin/env python3
"""
[修复版] 基于 DA3stream 的前馈 3DGS 生成脚本

关键修复：
1. 正确的坐标系转换：c2w → w2c
2. 移除过度的颜色增强
3. 使用官方 Monkey Patch 技术注入 Streaming 数据
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/ltx/projects/Depth-Anything-3/src')

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.gsply_helpers import export_ply, inverse_sigmoid

def load_streaming_results(streaming_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 DA3stream 输出数据"""
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
    # camera_poses.txt 是 c2w 格式 (cam2world)
    c2w_poses = np.loadtxt(poses_file).reshape(-1, 4, 4)

    # 🔑 关键修复：转换为 w2c (world2cam) 格式
    # DA3 模型期望的是 w2c，不是 c2w！
    w2c_poses = np.linalg.inv(c2w_poses)

    min_n = min(N, w2c_poses.shape[0])
    return images[:min_n], depths[:min_n], confs[:min_n], w2c_poses[:min_n], intrinsics[:min_n]

def concatenate_gaussians(gaussians_list: list) -> Gaussians:
    return Gaussians(
        means=torch.cat([g.means for g in gaussians_list], dim=1),
        harmonics=torch.cat([g.harmonics for g in gaussians_list], dim=1),
        opacities=torch.cat([g.opacities for g in gaussians_list], dim=1),
        scales=torch.cat([g.scales for g in gaussians_list], dim=1),
        rotations=torch.cat([g.rotations for g in gaussians_list], dim=1),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streaming-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output/feed_forward_3dgs_fixed')
    parser.add_argument('--model-name', type=str, default='da3-giant')
    parser.add_argument('--frame-interval', type=int, default=10)
    parser.add_argument('--conf-threshold', type=float, default=0.85)
    parser.add_argument('--sample-ratio', type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据（已经是 w2c 格式）
    images, depths, confs, w2c_extrinsics, intrinsics = load_streaming_results(Path(args.streaming_dir))

    # 抽帧
    frame_indices = list(range(0, len(images), args.frame_interval))
    print(f"\n选择 {len(frame_indices)} 帧进行底层 GS 融合...")

    # ========================================================
    # 🔑 核心修复：坐标系归一化
    # 将第一帧的相机位姿作为世界坐标系的原点
    # ========================================================
    print("🔧 执行坐标系归一化（第一帧作为世界原点）...")
    first_w2c_inv = np.linalg.inv(w2c_extrinsics[frame_indices[0]])
    # 应用变换：T_new = T_first_inv * T_old
    # 这样第一帧就变成了 Identity
    for i in range(len(w2c_extrinsics)):
        w2c_extrinsics[i] = first_w2c_inv @ w2c_extrinsics[i]
    print(f"  ✓ 第一帧相机位置已设为世界坐标原点")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    da3 = DepthAnything3(model_name=args.model_name).to(device)
    da3.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_gaussians = []
    print("\n🚀 开始底层前向渲染融合...")

    # Monkey Patch：注入 Streaming 数据
    original_depth_head = da3.model._process_depth_head

    for idx in tqdm(frame_indices, desc="映射 3DGS"):
        H, W = images[idx].shape[:2]

        img_tensor = transform(Image.fromarray(images[idx])).unsqueeze(0).unsqueeze(0).to(device)
        depth_tensor = torch.from_numpy(depths[idx]).unsqueeze(0).to(device) # (1, H, W)

        # 🔑 重要：使用 w2c 格式的 extrinsics
        ext_tensor = torch.from_numpy(w2c_extrinsics[idx]).unsqueeze(0).unsqueeze(0).float().to(device) # (1, 1, 4, 4)
        intrin_tensor = torch.from_numpy(intrinsics[idx]).unsqueeze(0).unsqueeze(0).float().to(device)

        # Monkey Patch：注入 Streaming 深度和位姿
        def my_depth_head(feats, h, w):
            out = original_depth_head(feats, h, w)
            out.depth = depth_tensor
            return out

        def my_cam_est(feats, h, w, out):
            out.extrinsics = ext_tensor  # w2c 格式
            out.intrinsics = intrin_tensor
            return out

        da3.model._process_depth_head = my_depth_head
        da3.model._process_camera_estimation = my_cam_est

        with torch.no_grad():
            outputs = da3.model(
                img_tensor,
                extrinsics=ext_tensor,  # w2c 格式
                intrinsics=intrin_tensor,
                infer_gs=True
            )
            frame_gs = outputs.gaussians

        da3.model._process_depth_head = original_depth_head

        # 置信度剪枝与降采样
        conf_flat = torch.from_numpy(confs[idx]).view(-1).to(device)
        valid_mask = conf_flat > args.conf_threshold

        if args.sample_ratio < 1.0:
            valid_idx = torch.where(valid_mask)[0]
            num_keep = int(len(valid_idx) * args.sample_ratio)
            keep_idx = valid_idx[torch.randperm(len(valid_idx))[:num_keep]]
            new_mask = torch.zeros_like(valid_mask)
            new_mask[keep_idx] = True
            valid_mask = new_mask

        filtered_gs = Gaussians(
            means=frame_gs.means[:, valid_mask],
            harmonics=frame_gs.harmonics[:, valid_mask],
            opacities=frame_gs.opacities[:, valid_mask],
            scales=frame_gs.scales[:, valid_mask],
            rotations=frame_gs.rotations[:, valid_mask]
        )
        all_gaussians.append(filtered_gs)

    final_gaussians = concatenate_gaussians(all_gaussians)
    num_gaussians = final_gaussians.means.shape[1]
    print(f"\n总计保留高斯球数量: {num_gaussians:,}")

    # ========================================================
    # 🔑 颜色处理：移除过度的增强，使用保守策略
    # ========================================================
    print("🎨 应用保守的颜色增强...")

    harmonics_dc = final_gaussians.harmonics[..., 0:1]  # (1, N, 3, 1)
    print(f"  DC 分量原始范围: [{harmonics_dc.min():.4f}, {harmonics_dc.max():.4f}]")

    # 保守策略：仅轻微增强，避免颜色失真
    # 目标：将动态范围从 ~0.1 扩展到 ~0.3
    enhancement_factor = 2.0
    harmonics_dc_enhanced = harmonics_dc * enhancement_factor

    print(f"  DC 分量增强后范围: [{harmonics_dc_enhanced.min():.4f}, {harmonics_dc_enhanced.max():.4f}]")

    final_gaussians.harmonics[..., 0:1] = harmonics_dc_enhanced
    print(f"  ✓ 颜色已增强 {enhancement_factor}x")

    # 导出 PLY
    ply_dir = output_dir / "gs_ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "0000_fixed.ply"

    print("💾 正在导出 PLY 文件...")

    gs_means = final_gaussians.means[0]
    gs_scales = final_gaussians.scales[0]
    gs_rotations = final_gaussians.rotations[0]
    gs_harmonics = final_gaussians.harmonics[0]
    gs_opacities_inv = inverse_sigmoid(final_gaussians.opacities[0])

    export_ply(
        means=gs_means,
        scales=gs_scales,
        rotations=gs_rotations,
        harmonics=gs_harmonics,
        opacities=gs_opacities_inv,
        path=ply_path,
        shift_and_scale=False,
        save_sh_dc_only=True,
        match_3dgs_mcmc_dev=False
    )

    file_size_mb = Path(ply_path).stat().st_size / (1024 * 1024)
    print(f"✓ 完成！PLY 文件已保存至: {ply_path} ({file_size_mb:.1f} MB)")
    print()
    print("关键修复:")
    print("  1. ✅ c2w → w2c 坐标系转换")
    print("  2. ✅ 第一帧坐标系归一化")
    print("  3. ✅ 保守的颜色增强策略")

if __name__ == '__main__':
    main()
