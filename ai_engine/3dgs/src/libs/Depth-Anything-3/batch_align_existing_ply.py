#!/home/ltx/my_envs/gs_linux_backup/bin/python
"""
批量扶正已有 PLY 文件 (质心-平面相对位置判定法)
最稳健的室内场景正位算法：通过对比点云质心与平面的相对高度，自动识别地板与天花板。
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

def install_package(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    install_package("plyfile")
    from plyfile import PlyData, PlyElement

try:
    import open3d as o3d
except ImportError:
    install_package("open3d")
    import open3d as o3d

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    install_package("scipy")
    from scipy.spatial.transform import Rotation

def get_rotation_to_z(normal):
    """计算将法线旋转到 Z+ [0,0,1] 的矩阵"""
    n = normal / np.linalg.norm(normal)
    # 强制法线先朝向 z 正半球，方便后续判定
    if n[2] < 0: n = -n
    target = np.array([0, 0, 1])
    axis = np.cross(n, target)
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)
    angle = np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis/np.linalg.norm(axis) * angle)

def align_single_ply_robust(ply_path, output_path, distance_threshold=0.03, num_iterations=10000):
    t0 = time.time()
    print(f"  📂 {ply_path.name}")

    try:
        plydata = PlyData.read(str(ply_path))
    except Exception as e:
        print(f"     ❌ 读取失败: {e}")
        return False

    vertex = plydata['vertex']
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    if len(points) < 1000:
        print("     ⚠️ 点云数量过少")
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 1. RANSAC 寻找主平面 (Top 5 候选)
    cloud = pcd
    candidates = []
    print(f"     🔍 正在分析场景结构 (质心高度判定法)...")
    
    for i in range(5):
        if len(cloud.points) < 100: break
        plane_model, inliers = cloud.segment_plane(distance_threshold, 3, num_iterations)
        [a, b, c, d] = plane_model
        
        # 模拟旋转以便进行纵横比校验
        R_test = get_rotation_to_z(np.array([a, b, c]))
        # 旋转后的平面方程为 z + d = 0, 即平面高度 Z_plane = -d
        # (前提是 [a,b,c] 已被 R 转为 [0,0,1])
        # 我们用 20% 的点云进行逻辑校验以保证速度和稳健性
        sample_idx = np.random.choice(len(points), min(len(points), 50000), replace=False)
        pts_test = points[sample_idx] @ R_test.T
        
        # 计算纵横比得分 (宽/高)
        p_min, p_max = np.min(pts_test, axis=0), np.max(pts_test, axis=0)
        size = p_max - p_min
        aspect_score = (size[0] + size[1]) / (size[2] + 1e-6)
        
        # 判定天花板 vs 地板 (关键逻辑)
        # z + d = 0 -> z_plane = -d
        # 注意：d 的值取决于 R 作用后的常数项。在纯旋转下，d 不变。
        z_plane = -d / np.linalg.norm([a, b, c])
        z_median = np.median(pts_test[:, 2])
        
        is_ceiling = z_median < z_plane  # 大部分点在平面下方 -> 是天花板
        
        # 平面权重：点数越多、越像水平面、纵横比越合理，分越高
        score = aspect_score * np.log10(len(inliers))
        
        candidates.append({
            'model': plane_model,
            'score': score,
            'is_ceiling': is_ceiling,
            'z_plane': z_plane,
            'z_median': z_median
        })
        cloud = cloud.select_by_index(inliers, invert=True)

    if not candidates: return False

    # 2. 选取最佳对齐平面 (最高纵横比评分)
    best = max(candidates, key=lambda x: x['score'])
    print(f"     ✨ 选定面高度: {best['z_plane']:.3f}, 场景中心高度: {best['z_median']:.3f}")

    # 3. 计算最终旋转矩阵
    R = get_rotation_to_z(best['model'][:3])
    
    # 4. 如果判定为天花板，强制执行 180 度翻转
    if best['is_ceiling']:
        print("     ⚠️ 检测到大部分点位于平面下方，判定为天花板，执行 180° 翻转补正...")
        R_flip = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
        R = R_flip @ R

    # 5. 应用变换
    points_rot = points @ R.T
    
    # 平移地面到 Z=0 (使用底部 2% 的分位数作为地面参考点，防止噪点干扰)
    z_floor = np.percentile(points_rot[:, 2], 2)
    points_rot[:, 2] -= z_floor

    # 6. 更新 PLY 数据并保持属性
    vertex['x'], vertex['y'], vertex['z'] = points_rot[:, 0], points_rot[:, 1], points_rot[:, 2]
    
    if 'nx' in vertex.data.dtype.names:
        n = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1)
        # 注意：法线需要应用旋转但不需要平移
        n_rot = n @ R.T
        vertex['nx'], vertex['ny'], vertex['nz'] = n_rot[:, 0], n_rot[:, 1], n_rot[:, 2]
    
    if 'rot_0' in vertex.data.dtype.names:
        # 四元数旋转变换 (w, x, y, z)
        q = np.stack([vertex['rot_1'], vertex['rot_2'], vertex['rot_3'], vertex['rot_0']], axis=-1)
        r_transform = Rotation.from_matrix(R)
        r_old = Rotation.from_quat(q)
        r_new = r_transform * r_old
        q_new = r_new.as_quat()
        vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3'] = q_new[:, 3], q_new[:, 0], q_new[:, 1], q_new[:, 2]

    # 7. 写入文件
    PlyData([vertex], text=False, byte_order='<').write(str(output_path))
    print(f"     ✅ 扶正成功！已保存至 {output_path.name} (耗时 {time.time()-t0:.1f}s)")
    return True

def main():
    parser = argparse.ArgumentParser(description="室内场景智能对齐工具 (质心判定版)")
    parser.add_argument("--input_file", required=True, help="输入的 PLY 文件路径")
    parser.add_argument("--threshold", type=float, default=0.03, help="RANSAC 平面拟合阈值")
    parser.add_argument("--iterations", type=int, default=10000, help="RANSAC 迭代次数")
    args = parser.parse_args()
    
    p = Path(args.input_file)
    if not p.exists():
        print(f"❌ 找不到文件: {args.input_file}")
        return
    
    out = p.parent / f"{p.stem}_aligned{p.suffix}"
    align_single_ply_robust(p, out, args.threshold, args.iterations)

if __name__ == "__main__":
    main()
