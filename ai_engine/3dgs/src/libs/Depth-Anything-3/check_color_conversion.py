#!/usr/bin/env python3
"""
检查 DA3 输出的颜色数据
检查原始 harmonics 是什么格式
"""
import sys
sys.path.insert(0, '/home/ltx/projects/Depth-Anything-3/src')

import torch
import numpy as np
from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Gaussians

print("=" * 60)
print("检查 DA3 原始输出的 harmonics 数据格式")
print("=" * 60)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
da3 = DepthAnything3(model_name='da3-giant').to(device)
da3.eval()

# 创建一个简单的测试图像
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载一帧测试
test_image_path = '/home/ltx/projects/Depth-Anything-3/output/sugar_streaming/results_output/frame_0.npz'
data = np.load(test_image_path)
image = data['image']
depth = data['depth']
intrinsics = data['intrinsics']

print(f"\n测试图像: {test_image_path}")
print(f"图像尺寸: {image.shape}")
print(f"深度范围: [{depth.min():.4f}, {depth.max():.4f}]")

# 转换并推理
img_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)
intrin_tensor = torch.from_numpy(intrinsics).unsqueeze(0).unsqueeze(0).float().to(device)

# 简单位姿
extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    outputs = da3.model(
        img_tensor,
        extrinsics=extrinsics,
        intrinsics=intrin_tensor,
        infer_gs=True
    )
    gs = outputs.gaussians

print(f"\n输出 Gaussians 结构:")
print(f"  means shape: {gs.means.shape}")
print(f"  harmonics shape: {gs.harmonics.shape}")
print(f"  opacities shape: {gs.opacities.shape}")

# 检查 harmonics
harmonics = gs.harmonics[0]  # (N, 3, d_sh)
print(f"\nHarmonics 详细信息:")
print(f"  形状: {harmonics.shape}")
print(f"  d_sh (球谐阶数): {harmonics.shape[2]}")

# 提取 DC 分量 (第0阶)
f_dc = harmonics[..., 0]  # (N, 3)
print(f"\nDC 分量 (f_dc) shape: {f_dc.shape}")
print(f"  f_dc[:, 0] (R通道) 范围: [{f_dc[:, 0].min().item():.6f}, {f_dc[:, 0].max().item():.6f}]")
print(f"  f_dc[:, 1] (G通道) 范围: [{f_dc[:, 1].min().item():.6f}, {f_dc[:, 1].max().item():.6f}]")
print(f"  f_dc[:, 2] (B通道) 范围: [{f_dc[:, 2].min().item():.6f}, {f_dc[:, 2].max().item():.6f}]")

# 尝试不同的转换方法
print(f"\n尝试不同的 SH -> RGB 转换方法:")
print(f"  原始 f_dc 均值: [{f_dc[:, 0].mean().item():.6f}, {f_dc[:, 1].mean().item():.6f}, {f_dc[:, 2].mean().item():.6f}]")

# 方法1: f_dc * 0.282095 + 0.5 (标准3DGS)
rgb_method1 = f_dc * 0.282095 + 0.5
print(f"\n方法1 (标准): SH * 0.282095 + 0.5")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method1[:, 0].min().item():.6f}, {rgb_method1[:, 0].max().item():.6f}]")
print(f"    G: [{rgb_method1[:, 1].min().item():.6f}, {rgb_method1[:, 1].max().item():.6f}]")
print(f"    B: [{rgb_method1[:, 2].min().item():.6f}, {rgb_method1[:, 2].max().item():.6f}]")

# 方法2: 直接使用 f_dc (可能已经是RGB?)
print(f"\n方法2 (直接使用): 直接使用 f_dc")
print(f"  RGB 范围:")
print(f"    R: [{f_dc[:, 0].min().item():.6f}, {f_dc[:, 0].max().item():.6f}]")
print(f"    G: [{f_dc[:, 1].min().item():.6f}, {f_dc[:, 1].max().item():.6f}]")
print(f"    B: [{f_dc[:, 2].min().item():.6f}, {f_dc[:, 2].max().item():.6f}]")

# 方法3: sigmoid
rgb_method3 = torch.sigmoid(f_dc)
print(f"\n方法3 (sigmoid): sigmoid(f_dc)")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method3[:, 0].min().item():.6f}, {rgb_method3[:, 0].max().item():.6f}]")
print(f"    G: [{rgb_method3[:, 1].min().item():.6f}, {rgb_method3[:, 1].max().item():.6f}]")
print(f"    B: [{rgb_method3[:, 2].min().item():.6f}, {rgb_method3[:, 2].max().item():.6f}]")

# 方法4: f_dc 直接作为 RGB (clip到 0-1)
rgb_method4 = torch.clamp(f_dc, 0, 1)
print(f"\n方法4 (clamp): clamp(f_dc, 0, 1)")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method4[:, 0].min().item():.6f}, {rgb_method4[:, 0].max().item():.6f}]")
print(f"    G: [{rgb_method4[:, 1].min().item():.6f}, {rgb_method4[:, 1].max().item():.6f}]")
print(f"    B: [{rgb_method4[:, 2].min().item():.6f}, {rgb_method4[:, 2].max().item():.6f}]")

# 对比原始图像颜色
print(f"\n原始图像 RGB 范围:")
print(f"  R: [{image[:, :, 0].min() / 255:.6f}, {image[:, :, 0].max() / 255:.6f}]")
print(f"  G: [{image[:, :, 1].min() / 255:.6f}, {image[:, :, 1].max() / 255:.6f}]")
print(f"  B: [{image[:, :, 2].min() / 255:.6f}, {image[:, :, 2].max() / 255:.6f}]")
