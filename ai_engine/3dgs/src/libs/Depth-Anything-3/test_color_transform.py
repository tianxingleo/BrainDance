#!/usr/bin/env python3
"""
测试不同的颜色转换方法，找出正确的公式
"""
import numpy as np
from plyfile import PlyData

ply_path = '/home/ltx/projects/Depth-Anything-3/output/feed_forward_3dgs_final_normalized/gs_ply/0000_perfect_merged.ply'

plydata = PlyData.read(ply_path)
vertex = plydata['vertex'].data

# 提取 DC 分量
f_dc_0 = vertex['f_dc_0']
f_dc_1 = vertex['f_dc_1']
f_dc_2 = vertex['f_dc_2']

print("=" * 60)
print("原始 f_dc 范围")
print("=" * 60)
print(f"  f_dc_0 (R): [{f_dc_0.min():.6f}, {f_dc_0.max():.6f}]")
print(f"  f_dc_1 (G): [{f_dc_1.min():.6f}, {f_dc_1.max():.6f}]")
print(f"  f_dc_2 (B): [{f_dc_2.min():.6f}, {f_dc_2.max():.6f}]")
print()

# 堆叠为 (N, 3)
f_dc = np.vstack([f_dc_0, f_dc_1, f_dc_2]).T

# 测试不同的转换方法
print("=" * 60)
print("测试不同的颜色转换方法")
print("=" * 60)

# 方法1: 标准转换 (SH -> RGB)
rgb_method1 = f_dc * 0.282095 + 0.5
rgb_method1 = np.clip(rgb_method1, 0, 1)
print(f"\n方法1: 标准转换 (SH * 0.282095 + 0.5)")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method1[:, 0].min():.4f}, {rgb_method1[:, 0].max():.4f}]")
print(f"    G: [{rgb_method1[:, 1].min():.4f}, {rgb_method1[:, 1].max():.4f}]")
print(f"    B: [{rgb_method1[:, 2].min():.4f}, {rgb_method1[:, 2].max():.4f}]")
print(f"  动态范围: R={rgb_method1[:, 0].max() - rgb_method1[:, 0].min():.4f}")

# 方法2: 反向标准转换 (假设 DA3 已经应用了标准转换)
# 需要反向: f_dc = (RGB - 0.5) / 0.282095
# 所以如果 f_dc 是 RGB，那么: RGB = f_dc (直接使用)
rgb_method2 = np.clip(f_dc, 0, 1)
print(f"\n方法2: 直接使用 f_dc (假设 DA3 输出已经是 RGB)")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method2[:, 0].min():.4f}, {rgb_method2[:, 0].max():.4f}]")
print(f"    G: [{rgb_method2[:, 1].min():.4f}, {rgb_method2[:, 1].max():.4f}]")
print(f"    B: [{rgb_method2[:, 2].min():.4f}, {rgb_method2[:, 2].max():.4f}]")
print(f"  动态范围: R={rgb_method2[:, 0].max() - rgb_method2[:, 0].min():.4f}")

# 方法3: 缩放到标准 SH 范围
# 假设 DA3 的 f_dc 是标准 SH 的 k 倍
# 我们需要找到 k，使得转换后的 RGB 范围接近 [0, 1]
# 观察: 当前 f_dc 范围约 [-0.07, 0.11]，转换后约 [0.48, 0.53]
# 如果想要 RGB 范围更大，需要: rgb = f_dc * k + 0.5
# 假设期望 RGB 范围为 [0.2, 0.8]，动态范围 0.6
# 则 k = 0.6 / (f_dc.max() - f_dc.min())
k_r = 0.6 / (f_dc[:, 0].max() - f_dc[:, 0].min())
k_g = 0.6 / (f_dc[:, 1].max() - f_dc[:, 1].min())
k_b = 0.6 / (f_dc[:, 2].max() - f_dc[:, 2].min())
print(f"\n方法3: 自适应缩放 (动态范围=0.6)")
print(f"  缩放因子: k_R={k_r:.2f}, k_G={k_g:.2f}, k_B={k_b:.2f}")
rgb_method3 = f_dc * np.array([k_r, k_g, k_b]) + 0.5
rgb_method3 = np.clip(rgb_method3, 0, 1)
print(f"  RGB 范围:")
print(f"    R: [{rgb_method3[:, 0].min():.4f}, {rgb_method3[:, 0].max():.4f}]")
print(f"    G: [{rgb_method3[:, 1].min():.4f}, {rgb_method3[:, 1].max():.4f}]")
print(f"    B: [{rgb_method3[:, 2].min():.4f}, {rgb_method3[:, 2].max():.4f}]")
print(f"  动态范围: R={rgb_method3[:, 0].max() - rgb_method3[:, 0].min():.4f}")

# 方法4: 归一化到 [0, 1]
rgb_method4 = (f_dc - f_dc.min(axis=0)) / (f_dc.max(axis=0) - f_dc.min(axis=0))
print(f"\n方法4: 归一化到 [0, 1]")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method4[:, 0].min():.4f}, {rgb_method4[:, 0].max():.4f}]")
print(f"    G: [{rgb_method4[:, 1].min():.4f}, {rgb_method4[:, 1].max():.4f}]")
print(f"    B: [{rgb_method4[:, 2].min():.4f}, {rgb_method4[:, 2].max():.4f}]")
print(f"  动态范围: R={rgb_method4[:, 0].max() - rgb_method4[:, 0].min():.4f}")

# 方法5: 假设 DA3 的 harmonics 是 RGB * 0.282095
# 需要反向: f_dc_standard = f_dc / 0.282095
rgb_method5 = (f_dc / 0.282095) * 0.282095 + 0.5  # 这个等于方法1
# 实际应该是: rgb = f_dc / 0.282095 + 0.5
rgb_method5_alt = f_dc / 0.282095 + 0.5
rgb_method5_alt = np.clip(rgb_method5_alt, 0, 1)
print(f"\n方法5: 假设 DA3 = RGB * 0.282095，反向")
print(f"  RGB 范围:")
print(f"    R: [{rgb_method5_alt[:, 0].min():.4f}, {rgb_method5_alt[:, 0].max():.4f}]")
print(f"    G: [{rgb_method5_alt[:, 1].min():.4f}, {rgb_method5_alt[:, 1].max():.4f}]")
print(f"    B: [{rgb_method5_alt[:, 2].min():.4f}, {rgb_method5_alt[:, 2].max():.4f}]")
print(f"  动态范围: R={rgb_method5_alt[:, 0].max() - rgb_method5_alt[:, 0].min():.4f}")

print("\n" + "=" * 60)
print("结论分析:")
print("=" * 60)
print("如果方法1的动态范围太小，说明 DA3 的 harmonics 不是标准 SH 格式")
print("如果方法2的动态范围正常，说明 DA3 的 harmonics 已经是 RGB 格式")
print("如果方法3/4的动态范围正常，说明需要额外的颜色增强")
