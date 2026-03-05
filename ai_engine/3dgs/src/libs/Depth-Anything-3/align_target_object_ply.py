#!/home/ltx/my_envs/gs_linux_backup/bin/python
"""
针对复杂伪影场景的智能对齐脚本 V4 (聚类连通性判定版)
核心改进：引入 DBSCAN 聚类算法。
物理原理：实物（剃须刀）在接触面通常是"单一连通"的大块物体。
         伪影（杂乱椭球）通常是"断裂、分散"的多个小块。
         即使伪影总点数多，但只要它们是分散的，最大连通块就会比实物小。
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

def analyze_connectivity(points, eps=0.01, min_points=10):
    """
    使用 DBSCAN 分析点云连通性
    返回: (最大连通块的点数, 连通块数量, 平均连通块大小)
    """
    if len(points) < min_points:
        return 0, 0, 0
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # DBSCAN 聚类: eps=1cm (同一个物体内的点距离通常小于1cm)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    if len(labels) == 0 or labels.max() == -1:
        return 0, 0, 0
        
    # 统计每个簇的点数 (-1 是噪声)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(counts) == 0:
        return 0, 0, 0
        
    max_cluster_size = np.max(counts)
    num_clusters = len(counts)
    avg_size = np.mean(counts)
    
    return max_cluster_size, num_clusters, avg_size

def align_cluster_ply(ply_path, output_path, distance_threshold=0.03):
    t0 = time.time()
    print(f"  📂 [聚类分析模式] 处理文件: {ply_path.name}")

    try:
        plydata = PlyData.read(str(ply_path))
    except Exception as e:
        print(f"     ❌ 读取失败: {e}")
        return False

    vertex = plydata['vertex']
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    has_normals = 'nx' in vertex.data.dtype.names
    normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1) if has_normals else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 1. RANSAC 寻找主平面
    cloud = pcd
    candidates = []
    print(f"     🔍 正在进行连通性分析 (Cluster Analysis)...")
    
    for i in range(3):
        if len(cloud.points) < 1000: break
        
        plane_model, inliers = cloud.segment_plane(distance_threshold, 3, 5000)
        [a, b, c, d] = plane_model
        
        # 提取并旋转平面
        plane_cloud = cloud.select_by_index(inliers)
        plane_pts = np.asarray(plane_cloud.points)
        R_test = get_rotation_to_z(np.array([a, b, c]))
        plane_pts_rot = plane_pts @ R_test.T
        z_plane = np.median(plane_pts_rot[:, 2])

        # 全局采样测试
        sample_idx = np.random.choice(len(points), min(len(points), 50000))
        pts_test = points[sample_idx] @ R_test.T
        
        # 提取切片 (上下 3cm)
        margin = 0.005
        limit = 0.03 
        
        pts_up = pts_test[(pts_test[:, 2] > z_plane + margin) & (pts_test[:, 2] < z_plane + limit)]
        pts_down = pts_test[(pts_test[:, 2] < z_plane - margin) & (pts_test[:, 2] > z_plane - limit)]
        
        # --- 核心判定：DBSCAN 聚类 ---
        # eps=0.015 (1.5cm) 允许稍微稀疏一点的底座连接在一起
        up_max_size, up_num, _ = analyze_connectivity(pts_up, eps=0.015)
        down_max_size, down_num, _ = analyze_connectivity(pts_down, eps=0.015)
        
        # 评分逻辑：最大连通块越大，越像实物
        is_upside_down = False
        why = "未知"
        
        if up_max_size > down_max_size:
            is_upside_down = False
            ratio = up_max_size / (down_max_size + 1)
            why = f"正向主连通块更大 (Top Cluster: {up_max_size} vs {down_max_size}, Ratio: {ratio:.1f})"
        else:
            is_upside_down = True
            ratio = down_max_size / (up_max_size + 1)
            why = f"反向主连通块更大 (Top Cluster: {down_max_size} vs {up_max_size}, Ratio: {ratio:.1f})"

        candidates.append({
            'model': plane_model,
            'is_upside_down': is_upside_down,
            'score': len(inliers), # 依然优先信任最大的平面是桌面
            'why': why
        })
        cloud = cloud.select_by_index(inliers, invert=True)

    if not candidates:
        print("     ❌ 未找到平面")
        return False
    
    best = max(candidates, key=lambda x: x['score'])
    print(f"     ⚖️ 判定结论: {best['why']}")
    
    # 2. 执行旋转
    R = get_rotation_to_z(best['model'][:3])
    
    if best['is_upside_down']:
        print("     🔄 执行 180° 翻转...")
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi) @ R
        
    # 3. 应用变换
    points_rot = points @ R.T
    
    # 4. Z=0 对齐 (直方图峰值)
    z_vals = points_rot[:, 2]
    hist, edges = np.histogram(z_vals, bins=200, range=(np.percentile(z_vals, 5), np.percentile(z_vals, 95)))
    peak_idx = np.argmax(hist)
    z_floor = (edges[peak_idx] + edges[peak_idx+1]) / 2
    
    points_rot[:, 2] -= z_floor
    print(f"     📐 桌面已对齐至 Z=0")
    
    # 5. 保存
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
    out = p.parent / f"{p.stem}_cluster_aligned{p.suffix}"
    align_cluster_ply(p, out)

if __name__ == "__main__":
    main()