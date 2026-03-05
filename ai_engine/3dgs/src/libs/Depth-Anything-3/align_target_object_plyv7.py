#!/home/ltx/my_envs/gs_linux_backup/bin/python
"""
针对复杂伪影场景的智能对齐脚本 V7 (XY紧凑度判定版)
核心逻辑改进：
放弃"比谁大/比谁高"，改为"比谁更聚焦"。
物理原理：
1. 桌面物体(剃须刀)通常是局部、聚焦的，XY 平面占地面积小 (Low Spread)。
2. 桌底伪影通常是弥漫、杂乱的，覆盖范围广，XY 平面占地面积大 (High Spread)。
3. 选择"XY分布半径最小"的那一侧作为正面。
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
    if n[2] < 0: n = -n
    target = np.array([0, 0, 1])
    axis = np.cross(n, target)
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)
    angle = np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis/np.linalg.norm(axis) * angle)

def get_cluster_stats(points, eps, min_points=10):
    """
    DBSCAN 聚类，返回最大连通块的:
    1. XY 分布半径 (Spread) - 衡量"胖瘦"
    2. 点数 (Size)
    """
    if len(points) < min_points:
        return 9999.0, 0  # 如果没点，认为Spread无穷大(不可能被选中)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if len(labels) == 0 or labels.max() == -1:
        return 9999.0, 0
        
    # 找到最大簇
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        return 9999.0, 0
        
    largest_label = unique_labels[np.argmax(counts)]
    max_cluster_size = np.max(counts)
    
    # 提取簇点云
    cluster_indices = np.where(labels == largest_label)[0]
    cluster_pts = points[cluster_indices]
    
    # 计算 XY Spread (标准差的模长)
    # std_x = np.std(cluster_pts[:, 0])
    # std_y = np.std(cluster_pts[:, 1])
    # spread = np.sqrt(std_x**2 + std_y**2)
    
    # 或者用包围盒对角线 (更直观)
    min_bound = np.min(cluster_pts, axis=0)
    max_bound = np.max(cluster_pts, axis=0)
    xy_diagonal = np.linalg.norm(max_bound[:2] - min_bound[:2])
    
    return xy_diagonal, max_cluster_size

def align_compact_ply(ply_path, output_path):
    t0 = time.time()
    print(f"  📂 [XY紧凑度判定模式] 处理文件: {ply_path.name}")

    try:
        plydata = PlyData.read(str(ply_path))
    except Exception as e:
        print(f"     ❌ 读取失败: {e}")
        return False

    vertex = plydata['vertex']
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    # --- 1. 计算场景尺度 ---
    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    scene_size = np.linalg.norm(p_max - p_min)
    
    # 动态参数
    dist_thresh = max(0.001, scene_size * 0.01)
    check_range = scene_size * 0.3  # 看得远一点，包含整个物体
    check_margin = max(0.005, scene_size * 0.01)
    cluster_eps = max(0.002, scene_size * 0.02)
    
    print(f"     📏 场景尺度: {scene_size:.2f}")
    
    has_normals = 'nx' in vertex.data.dtype.names
    normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1) if has_normals else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 2. RANSAC 寻找主平面
    cloud = pcd
    candidates = []
    
    for i in range(3):
        if len(cloud.points) < 1000: break
        
        plane_model, inliers = cloud.segment_plane(dist_thresh, 3, 5000)
        [a, b, c, d] = plane_model
        
        # 旋转对齐
        plane_cloud = cloud.select_by_index(inliers)
        R_test = get_rotation_to_z(np.array([a, b, c]))
        
        # 全局采样
        sample_idx = np.random.choice(len(points), min(len(points), 100000))
        pts_test = points[sample_idx] @ R_test.T
        
        # 平面高度
        plane_cloud_rot = np.asarray(plane_cloud.points) @ R_test.T
        z_plane = np.median(plane_cloud_rot[:, 2])

        # 提取上下
        mask_up = (pts_test[:, 2] > z_plane + check_margin) & (pts_test[:, 2] < z_plane + check_range)
        mask_down = (pts_test[:, 2] < z_plane - check_margin) & (pts_test[:, 2] > z_plane - check_range)
        
        pts_up = pts_test[mask_up]
        pts_down = pts_test[mask_down]
        
        # --- 核心判定：计算 XY Spread ---
        spread_up, size_up = get_cluster_stats(pts_up, eps=cluster_eps)
        spread_down, size_down = get_cluster_stats(pts_down, eps=cluster_eps)
        
        is_upside_down = False
        why = "未知"
        score_val = len(inliers)
        
        # 如果一侧没东西 (Spread=9999)，另一侧赢
        if spread_up > 9000 and spread_down > 9000:
            score_val = 0 # 无效平面
        elif spread_up > 9000:
            is_upside_down = True
            why = "正向无物体"
        elif spread_down > 9000:
            is_upside_down = False
            why = "反向无物体"
        else:
            # 正常比较：谁的 Spread 小，谁就是物体 (Small is Object)
            if spread_up < spread_down:
                is_upside_down = False # 上面更紧凑 -> 正
                ratio = spread_down / (spread_up + 1e-6)
                why = f"正向物体更聚焦 (XY Spread: {spread_up:.2f} vs {spread_down:.2f}, Ratio: {ratio:.1f})"
            else:
                is_upside_down = True # 下面更紧凑 -> 反
                ratio = spread_up / (spread_down + 1e-6)
                why = f"反向物体更聚焦 (XY Spread: {spread_down:.2f} vs {spread_up:.2f}, Ratio: {ratio:.1f})"

        candidates.append({
            'model': plane_model,
            'is_upside_down': is_upside_down,
            'score': score_val,
            'why': why
        })
        cloud = cloud.select_by_index(inliers, invert=True)

    if not candidates:
        print("     ❌ 未找到平面")
        return False
    
    valid = [c for c in candidates if c['score'] > 0]
    if not valid:
        print("     ⚠️ 所有平面两侧均无物体")
        return False
        
    best = max(valid, key=lambda x: x['score'])
    print(f"     ⚖️ 判定结论: {best['why']}")
    
    # 3. 执行旋转
    R = get_rotation_to_z(best['model'][:3])
    
    if best['is_upside_down']:
        print("     🔄 执行 180° 翻转...")
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi) @ R
        
    # 4. 应用变换
    points_rot = points @ R.T
    
    # 5. Z=0 对齐
    z_vals = points_rot[:, 2]
    hist, edges = np.histogram(z_vals, bins=200, range=(np.percentile(z_vals, 5), np.percentile(z_vals, 95)))
    peak_idx = np.argmax(hist)
    z_floor = (edges[peak_idx] + edges[peak_idx+1]) / 2
    
    points_rot[:, 2] -= z_floor
    print(f"     📐 桌面已对齐至 Z=0")
    
    # 6. 保存
    vertex['x'], vertex['y'], vertex['z'] = points_rot[:, 0], points_rot[:, 1], points_rot[:, 2]
    
    if has_normals:
        n_rot = normals @ R.T
        vertex['nx'], vertex['ny'], vertex['nz'] = n_rot[:, 0], n_rot[:, 1], n_rot[:, 2]
        
    if 'rot_0' in vertex.data.dtype.names:
        q = np.stack([vertex['rot_1'], vertex['rot_2'], vertex['rot_3'], vertex['rot_0']], axis=-1)
        r_new = Rotation.from_matrix(R) * Rotation.from_quat(q)
        q_new = r_new.as_quat()
        vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3'] = q_new[:, 3], q_new[:, 0], q_new[:, 1], q_new[:, 2]

    PlyData([vertex], text=False, byte_order='<').write(str(output_path))
    print(f"     ✅ 保存成功: {output_path.name}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()
    
    p = Path(args.input_file)
    if not p.exists(): return
    out = p.parent / f"{p.stem}_compact_aligned{p.suffix}"
    align_compact_ply(p, out)

if __name__ == "__main__":
    main()