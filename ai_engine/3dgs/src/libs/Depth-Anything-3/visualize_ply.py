#!/usr/bin/env python3
"""
检查和可视化 PLY 点云文件
"""
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_ply(ply_path):
    """读取 PLY 文件"""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data

    # 提取数据
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # 提取颜色 (SH DC 系数)
    if 'f_dc_0' in vertex.dtype.names:
        colors = np.vstack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']]).T
        # SH DC 系数需要转换: SH * 0.282095 + 0.5
        colors = colors * 0.282095 + 0.5
        colors = np.clip(colors, 0, 1)
    else:
        colors = None

    return points, colors

def analyze_ply(ply_path):
    """分析 PLY 文件信息"""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # vertex 是一个 PlyElement 对象，需要用 data 属性访问数据
    vertex_data = vertex.data

    print("=" * 60)
    print("PLY 文件信息")
    print("=" * 60)
    print(f"顶点数量: {len(vertex_data):,}")
    print(f"属性字段: {vertex_data.dtype.names}")
    print()

    # 提取坐标统计
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']

    print("坐标范围:")
    print(f"  X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Z: [{z.min():.2f}, {z.max():.2f}]")
    print()

    print("坐标中心:")
    print(f"  X: {x.mean():.2f}")
    print(f"  Y: {y.mean():.2f}")
    print(f"  Z: {z.mean():.2f}")
    print()

    # 检查颜色
    if 'f_dc_0' in vertex_data.dtype.names:
        f_dc_0 = vertex_data['f_dc_0']
        f_dc_1 = vertex_data['f_dc_1']
        f_dc_2 = vertex_data['f_dc_2']

        print("SH DC 系数范围 (原始值):")
        print(f"  f_dc_0: [{f_dc_0.min():.4f}, {f_dc_0.max():.4f}]")
        print(f"  f_dc_1: [{f_dc_1.min():.4f}, {f_dc_1.max():.4f}]")
        print(f"  f_dc_2: [{f_dc_2.min():.4f}, {f_dc_2.max():.4f}]")
        print()

        # 转换为 RGB
        colors = np.vstack([f_dc_0, f_dc_1, f_dc_2]).T
        rgb = colors * 0.282095 + 0.5
        rgb = np.clip(rgb, 0, 1)

        print("转换后的 RGB 范围:")
        print(f"  R: [{rgb[:, 0].min():.4f}, {rgb[:, 0].max():.4f}]")
        print(f"  G: [{rgb[:, 1].min():.4f}, {rgb[:, 1].max():.4f}]")
        print(f"  B: [{rgb[:, 2].min():.4f}, {rgb[:, 2].max():.4f}]")
        print()

def visualize_ply(ply_path, max_points=50000):
    """可视化 PLY 点云（下采样）"""
    print(f"正在读取点云文件: {ply_path}")
    points, colors = read_ply(ply_path)

    num_points = len(points)
    print(f"总点数: {num_points:,}")

    # 下采样
    if num_points > max_points:
        step = num_points // max_points
        points = points[::step]
        colors = colors[::step] if colors is not None else None
        print(f"下采样到: {len(points):,} 点")

    # 创建 3D 图
    fig = plt.figure(figsize=(15, 5))

    # 三视角投影
    views = [
        ('X-Y 平面视图 (俯视)', 0, 1, 2),
        ('X-Z 平面视图 (前视)', 0, 2, 1),
        ('Y-Z 平面视图 (侧视)', 1, 2, 0),
    ]

    for idx, (title, x_idx, y_idx, c_idx) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1)

        if colors is not None:
            ax.scatter(points[:, x_idx], points[:, y_idx],
                      c=colors[:, c_idx], s=0.1, alpha=0.5)
        else:
            ax.scatter(points[:, x_idx], points[:, y_idx],
                      c='gray', s=0.1, alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel(['X', 'Y', 'Z'][x_idx])
        ax.set_ylabel(['X', 'Y', 'Z'][y_idx])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/home/ltx/projects/Depth-Anything-3/output/ply_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n可视化已保存至: {output_path}")

    # 3D 视图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=0.1, alpha=0.3)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='gray', s=0.1, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D 点云可视化')

    output_path_3d = '/home/ltx/projects/Depth-Anything-3/output/ply_analysis_3d.png'
    plt.savefig(output_path_3d, dpi=150)
    print(f"3D 可视化已保存至: {output_path_3d}")

    plt.close('all')

if __name__ == '__main__':
    ply_path = '/home/ltx/projects/Depth-Anything-3/output/feed_forward_3dgs_final_normalized/gs_ply/0000_perfect_merged.ply'

    print("\n" + "="*60)
    print("PLY 点云分析工具")
    print("="*60 + "\n")

    # 分析 PLY 文件
    analyze_ply(ply_path)

    print("\n正在生成可视化...")
    visualize_ply(ply_path, max_points=100000)

    print("\n分析完成！")
