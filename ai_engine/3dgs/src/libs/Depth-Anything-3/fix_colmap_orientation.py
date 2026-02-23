import numpy as np
import open3d as o3d
from pathlib import Path
import argparse
import sys

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, np.argmax(vals)]
    if q[3] < 0: q = -q
    return q[[3, 0, 1, 2]]

def fix_colmap_data(sparse_dir, points_ply_path=None, invert_pose=False):
    sparse_dir = Path(sparse_dir)
    images_txt = sparse_dir / "images.txt"
    points3d_txt = sparse_dir / "points3D.txt"
    
    if not images_txt.exists():
        print("❌ images.txt 不存在")
        return

    print(f"🔧 修正位姿中... (Invert={invert_pose})")
    with open(images_txt, "r") as f:
        lines = f.readlines()
    header = lines[:4]

    # GL (Y up, -Z forward) -> CV (Y down, Z forward)
    flip_yz = np.diag([1, -1, -1])
    
    new_data = []
    centers = []

    for i in range(4, len(lines), 2):
        line = lines[i].strip().split()
        if len(line) < 10: continue
        
        qvec = np.array(list(map(float, line[1:5])))
        tvec = np.array(list(map(float, line[5:8])))
        R = qvec2rotmat(qvec)
        
        if invert_pose:
            # 输入是 C2W (t 是中心)
            C = tvec
            R_c2w = R
        else:
            # 输入是 W2C (已求逆)
            C = -R.T @ tvec
            R_c2w = R.T
            
        # 转换为 CV 坐标系
        C_cv = flip_yz @ C
        R_c2w_cv = flip_yz @ R_c2w @ flip_yz
        
        new_data.append({
            'idx': line[0], 'cam_id': line[8], 'name': line[9],
            'R_c2w': R_c2w_cv, 'C': C_cv
        })
        centers.append(C_cv)

    # 地面对齐
    align_R = np.eye(3)
    pcd = None
    if points_ply_path and Path(points_ply_path).exists():
        pcd = o3d.io.read_point_cloud(str(points_ply_path))
        if not pcd.is_empty():
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) @ flip_yz.T)
            plane_model, _ = pcd.segment_plane(0.05, 3, 2000)
            normal = plane_model[:3]
            normal /= np.linalg.norm(normal)
            target = np.array([0, 0, 1])
            if normal[2] < 0: normal = -normal
            axis = np.cross(normal, target)
            if np.linalg.norm(axis) > 1e-6:
                angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
                align_R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis/np.linalg.norm(axis) * angle)
                print(f"✅ 场景扶正角度: {np.degrees(angle):.1f}°")

    # 写回文件并校验平移
    with open(images_txt, "w") as f:
        f.writelines(header)
        for d in new_data:
            C_final = align_R @ d['C']
            R_final = (align_R @ d['R_c2w']).T
            t_final = -R_final @ C_final
            
            # 关键：防御零位姿，防止 ZeroDivisionError
            if np.linalg.norm(t_final) < 1e-10:
                t_final += 1e-6 # 加入极小偏移
                
            q = rotmat2qvec(R_final)
            f.write(f"{d['idx']} {q[0]} {q[1]} {q[2]} {q[3]} {t_final[0]} {t_final[1]} {t_final[2]} {d['cam_id']} {d['name']}\n\n")

    # points3D 处理
    if points3d_txt.exists():
        with open(points3d_txt, "r") as f: p_lines = f.readlines()
        with open(points3d_txt, "w") as f:
            for l in p_lines:
                if l.startswith("#"): f.write(l); continue
                p = l.strip().split()
                xyz = align_R @ (flip_yz @ np.array(list(map(float, p[1:4]))))
                f.write(f"{p[0]} {xyz[0]} {xyz[1]} {xyz[2]} {' '.join(p[4:])}\n")

    if pcd:
        pcd.rotate(align_R, center=(0,0,0))
        o3d.io.write_point_cloud(str(points_ply_path), pcd)

    # 调试信息
    if new_data:
        print(f"📊 已修复 {len(new_data)} 个相机位姿。示例平移 L2 范数: {np.linalg.norm(centers[0]):.4f}")
    if np.linalg.norm(centers[0]) < 1e-6:
        print("⚠️ 警告: 相机轨迹过于集中在原点，请确认 DA3 输入轨迹是否有效。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_dir", required=True)
    parser.add_argument("--ply_path")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()
    fix_colmap_data(args.sparse_dir, args.ply_path, args.invert)
