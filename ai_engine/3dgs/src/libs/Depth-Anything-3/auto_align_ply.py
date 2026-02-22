#!/usr/bin/env python3
"""
自动扶正 PLY 点云模型 (Auto-align PLY model)

使用 Open3D 的 RANSAC 平面分割算法自动检测地面，
并将模型旋转使地面对齐到 X-Y 平面 (Z轴朝上)。

用法:
    python auto_align_ply.py <input.ply> [output.ply] [--distance_threshold 0.02] [--translate_to_ground]

依赖:
    pip install open3d numpy
"""

import argparse
import sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("❌ 错误: 请先安装 Open3D:")
    print("   pip install open3d")
    print("   或: conda install -c open3d-admin open3d")
    sys.exit(1)


def auto_align_model(ply_path, output_path,
                     distance_threshold=0.02,
                     ransac_n=3,
                     num_iterations=1000,
                     translate_to_ground=False):
    """
    自动扶正点云模型。
    
    原理:
    1. 使用 RANSAC 分割出最大平面（通常是地面）
    2. 计算该平面的法向量
    3. 将法向量旋转对齐到 Z 轴 (0, 0, 1)
    4. 可选：平移地面到 Z=0
    
    参数:
        ply_path: 输入 PLY 文件路径
        output_path: 输出 PLY 文件路径
        distance_threshold: RANSAC 距离阈值 (米)，越小越严格
        ransac_n: RANSAC 最少采样点数
        num_iterations: RANSAC 迭代次数
        translate_to_ground: 是否将地面平移到 Z=0
    """
    # 1. 读取点云
    print(f"  📂 读取点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = len(pcd.points)
    print(f"     点数: {num_points}")
    
    if num_points < 10:
        print("  ⚠️ 点云太少，无法进行平面分割")
        o3d.io.write_point_cloud(output_path, pcd)
        return False
    
    # 2. RANSAC 分割地面
    print(f"  🔍 RANSAC 平面分割 (threshold={distance_threshold}, iterations={num_iterations})...")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal_len = np.linalg.norm(normal)
    
    print(f"     平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"     法向量: ({a:.4f}, {b:.4f}, {c:.4f})")
    print(f"     地面内点: {len(inliers)} / {num_points} ({100*len(inliers)/num_points:.1f}%)")
    
    if len(inliers) < 0.05 * num_points:
        print("  ⚠️ 警告: 地面内点比例较低，检测到的平面可能不是地面")
    
    # 3. 计算旋转矩阵：将法向量旋转到 Z 轴 (0, 0, 1)
    target_axis = np.array([0, 0, 1])
    
    # 归一化法向量
    if normal_len < 1e-8:
        print("  ⚠️ 法向量接近零向量，跳过旋转")
        o3d.io.write_point_cloud(output_path, pcd)
        return False
    
    normal_unit = normal / normal_len
    
    # 确保法向量朝上 (z分量为正)
    # 如果法向量的 z 分量为负，翻转法向量
    if normal_unit[2] < 0:
        normal_unit = -normal_unit
        print("     (翻转法向量使其朝上)")
    
    # 计算旋转轴和角度
    cos_angle = np.clip(np.dot(normal_unit, target_axis), -1.0, 1.0)
    rotation_axis = np.cross(normal_unit, target_axis)
    rotation_axis_len = np.linalg.norm(rotation_axis)
    
    if rotation_axis_len < 1e-8:
        # 法向量已经与Z轴对齐（或反向）
        if cos_angle > 0:
            print("  ✅ 模型已经是正确朝向，无需旋转")
            R = np.eye(3)
        else:
            # 反向，绕 X 轴旋转 180°
            print("  🔄 模型上下颠倒，旋转 180°")
            R = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ], dtype=np.float64)
    else:
        rotation_angle = np.arccos(cos_angle)
        rotation_axis_unit = rotation_axis / rotation_axis_len
        
        angle_deg = np.degrees(rotation_angle)
        print(f"  🔄 旋转角度: {angle_deg:.2f}°")
        print(f"     旋转轴: ({rotation_axis_unit[0]:.4f}, {rotation_axis_unit[1]:.4f}, {rotation_axis_unit[2]:.4f})")
        
        # 使用 Rodrigues 公式计算旋转矩阵
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis_unit * rotation_angle
        )
    
    # 4. 应用旋转
    print("  🔄 应用旋转变换...")
    pcd.rotate(R, center=(0, 0, 0))
    
    # 5. 可选：平移地面到 Z=0
    if translate_to_ground:
        # 获取地面内点的质心
        inlier_cloud = pcd.select_by_index(inliers)
        centroid = np.asarray(inlier_cloud.points).mean(axis=0)
        
        # 平移使地面的平均 Z 对齐到 0
        translation = np.array([0, 0, -centroid[2]])
        pcd.translate(translation)
        print(f"  📐 已平移地面到 Z=0 (偏移: {centroid[2]:.4f})")
    
    # 6. 保存
    print(f"  💾 保存扶正后的点云: {output_path}")
    o3d.io.write_point_cloud(output_path, pcd)
    
    print("  ✅ 模型已自动扶正")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="自动扶正 PLY 点云模型 (RANSAC 平面检测 + 旋转对齐)"
    )
    parser.add_argument("input", help="输入 PLY 文件路径")
    parser.add_argument("output", nargs="?", default=None,
                        help="输出 PLY 文件路径 (默认: <input>_aligned.ply)")
    parser.add_argument("--distance_threshold", type=float, default=0.02,
                        help="RANSAC 距离阈值,单位:米 (默认: 0.02)")
    parser.add_argument("--ransac_n", type=int, default=3,
                        help="RANSAC 最少采样点数 (默认: 3)")
    parser.add_argument("--num_iterations", type=int, default=1000,
                        help="RANSAC 迭代次数 (默认: 1000)")
    parser.add_argument("--translate_to_ground", action="store_true",
                        help="将地面平移到 Z=0")
    parser.add_argument("--inplace", action="store_true",
                        help="原地修改 (覆盖输入文件)")
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.inplace:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        # 默认: input_aligned.ply
        import os
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_aligned{ext}"
    
    print(f"=== 自动扶正点云模型 ===")
    print(f"  输入: {args.input}")
    print(f"  输出: {output_path}")
    print()
    
    success = auto_align_model(
        ply_path=args.input,
        output_path=output_path,
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        num_iterations=args.num_iterations,
        translate_to_ground=args.translate_to_ground,
    )
    
    if success:
        print("\n🎉 完成!")
    else:
        print("\n⚠️ 扶正可能未完全生效，请检查输出")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
