"""
DA3 → 3DGS 训练 + 双重对齐 Pipeline (融合方案)

基于 run_da3_to_3dgs_direct.py 的模式重写，融合:
  🅰️ COLMAP model_aligner 平面对齐 (训练前)
  🅱️ Open3D RANSAC 扶正 (训练后，使用 plyfile 保护数据)

用法:
    python run_da3_to_3dgs_aligned.py [选项]
"""

import subprocess
import os
import shutil
import sys
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import struct

# ================= 🔧 路径配置 =================
DA3_DIR = Path("/home/ltx/projects/Depth-Anything-3")
DA3_OUTPUT = DA3_DIR / "output" / "sugar_streaming"
CONDA_PREFIX = "/home/ltx/my_envs/gs_linux_backup"
NS_ENV_BIN = f"{CONDA_PREFIX}/bin"
PYTHON_EXE = f"{NS_ENV_BIN}/python"
NS_TRAIN = f"{NS_ENV_BIN}/ns-train"
NS_EXPORT = f"{NS_ENV_BIN}/ns-export"


# ================= 🛠️ 辅助函数 =================

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ================= 🛠️ 格式转换逻辑 =================

def rotmat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return np.array([qw, qx, qy, qz])


# ================= 🅰️ 坐标修复 =================

def colmap_fix_with_axes(sparse_dir: Path, ply_path: Path = None) -> bool:
    """解决坐标轴翻转(GL->CV)并进行平面对齐"""
    print("\n🔧 [对齐修复] 正在执行 fix_colmap_orientation.py...")
    cmd = [
        "python3", "fix_colmap_orientation.py",
        "--sparse_dir", str(sparse_dir)
        # 不使用 --invert，因为 convert 脚本已处理位姿
    ]
    if ply_path and ply_path.exists():
        cmd.extend(["--ply_path", str(ply_path)])
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"  ⚠️ 修复失败: {e}")
        return False


# ================= 🅰️ COLMAP 对齐 =================

def colmap_align(sparse_dir: Path, aligned_dir: Path, max_error: float = 0.02) -> bool:
    """使用 COLMAP model_aligner 进行平面对齐"""
    print("\n🅰️  [COLMAP 对齐] 使用 model_aligner 进行平面对齐...")
    aligned_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "model_aligner",
        "--input_path", str(sparse_dir),
        "--output_path", str(aligned_dir),
        "--ref_is_gps", "0",
        "--alignment_type", "plane",
        "--alignment_max_error", str(max_error),
    ]

    print(f"  执行: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - t0

        if result.returncode == 0:
            if any((aligned_dir / f).exists() for f in ["cameras.txt", "cameras.bin"]):
                print(f"  ✅ COLMAP 对齐成功 ({elapsed:.1f}s)")
                return True
            print(f"  ⚠️ COLMAP 对齐似乎完成但输出文件缺失")
            return False
        else:
            print(f"  ⚠️ COLMAP 对齐失败 (exit code {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("  ⚠️ COLMAP 对齐超时 (>60s)，跳过")
        return False
    except FileNotFoundError:
        print("  ⚠️ colmap 未安装，跳过 COLMAP 对齐")
        return False


# ================= 🅱️ Open3D 扶正 (Robust) =================

def open3d_align_robust(ply_path: Path, output_path: Path,
                        distance_threshold: float = 0.02,
                        translate_to_ground: bool = False) -> bool:
    """使用 plyfile + Open3D + Scipy 完整保留数据的扶正"""
    print(f"\n🅱️  [Open3D 扶正] Robust Mode (保留 Gaussian 属性)...")

    # 懒加载依赖
    try:
        from plyfile import PlyData
        import open3d as o3d
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("  ⚠️ 缺少依赖，尝试安装 plyfile open3d scipy...")
        install_package("plyfile")
        install_package("open3d")
        install_package("scipy")
        from plyfile import PlyData
        import open3d as o3d
        from scipy.spatial.transform import Rotation

    t0 = time.time()
    
    # 1. 完整读取
    print(f"  📂 读取: {ply_path} ({ply_path.stat().st_size / 1024 / 1024:.1f}MB)")
    try:
        plydata = PlyData.read(str(ply_path))
    except Exception as e:
        print(f"  ❌ 读取 PLY 失败: {e}")
        return False
        
    vertex = plydata['vertex']
    
    # Extract XYZ
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    points = np.stack([x, y, z], axis=-1)
    
    if len(points) < 10:
        print("  ⚠️ 点云太少")
        return False

    # 2. Open3D RANSAC (智能多平面搜索)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 策略：前3次 RANSAC 如果找到 <10度的面直接采用
    iters = 5000
    cloud = pcd
    candidates = []
    best_plane = None
    
    for i in range(3):
        if len(cloud.points) < 100: break
        plane_model, inliers = cloud.segment_plane(distance_threshold, 3, iters)
        [a, b, c, d] = plane_model
        norm = np.linalg.norm(plane_model[:3])
        if norm == 0: continue
        normal_unit = plane_model[:3] / norm
        if normal_unit[2] < 0: normal_unit = -normal_unit
        
        angle = np.degrees(np.arccos(np.clip(normal_unit[2], -1.0, 1.0)))
        candidates.append((plane_model, angle, len(inliers)))
        
        if angle < 10.0:
            best_plane = plane_model
            print(f"     ✨ 发现目标水平面 (角度 {angle:.1f}°)")
            break
        cloud = cloud.select_by_index(inliers, invert=True)

    if best_plane is None:
        best_plane = min(candidates, key=lambda x: x[1])[0]
        print(f"     ⚠️ 未发现完美水平面，选取最接近平面 (角度 {min(candidates, key=lambda x: x[1])[1]:.1f}°)")

    [a, b, c, d] = best_plane
    normal = np.array([a, b, c])
    normal_unit = normal / np.linalg.norm(normal)
    if normal_unit[2] < 0: normal_unit = -normal_unit
    
    # 3. 计算旋转
    target = np.array([0, 0, 1])
    axis = np.cross(normal_unit, target)
    if np.linalg.norm(axis) < 1e-6:
        R = np.eye(3)
    else:
        angle = np.arccos(np.clip(np.dot(normal_unit, target), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis/np.linalg.norm(axis) * angle)
    
    print(f"     🔄 旋转应用中...")
    
    # 4. Apply to XYZ
    points_rotated = np.dot(points, R.T)
    
    if translate_to_ground:
        inlier_points = points_rotated[inliers]
        centroid = np.mean(inlier_points, axis=0)
        points_rotated[:, 2] -= centroid[2]
        print(f"     📐 平移地面到 Z=0")
        
    vertex['x'] = points_rotated[:, 0]
    vertex['y'] = points_rotated[:, 1]
    vertex['z'] = points_rotated[:, 2]
    
    # 5. Apply to Quaternions (if 3DGS)
    if 'rot_0' in vertex.data.dtype.names:
        rot_0, rot_1, rot_2, rot_3 = vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']
        quats = np.stack([rot_1, rot_2, rot_3, rot_0], axis=-1) # (x, y, z, w) for scipy
        
        r_transform = Rotation.from_matrix(R)
        r_old = Rotation.from_quat(quats)
        r_new = r_transform * r_old
        quats_new = r_new.as_quat()
        
        vertex['rot_0'] = quats_new[:, 3] # w
        vertex['rot_1'] = quats_new[:, 0] # x
        vertex['rot_2'] = quats_new[:, 1] # y
        vertex['rot_3'] = quats_new[:, 2] # z
        
    # 6. Apply to Normals
    if 'nx' in vertex.data.dtype.names:
        nx, ny, nz = vertex['nx'], vertex['ny'], vertex['nz']
        normals = np.stack([nx, ny, nz], axis=-1)
        normals_rotated = np.dot(normals, R.T)
        vertex['nx'] = normals_rotated[:, 0]
        vertex['ny'] = normals_rotated[:, 1]
        vertex['nz'] = normals_rotated[:, 2]
        
    # 7. Save
    PlyData([vertex], text=False, byte_order='<').write(str(output_path))
    
    elapsed = time.time() - t0
    out_size = output_path.stat().st_size / 1024 / 1024
    print(f"  💾 保存: {output_path} ({out_size:.1f}MB, {elapsed:.1f}s)")
    return True


# ================= 🚀 主流程 =================

def run_pipeline(args):
    source_dir = Path(args.da3_output)
    ws_root = source_dir / "da3_3dgs_aligned_pipeline"

    print("=" * 60)
    print("  DA3 → 3DGS + 双重对齐 Pipeline (融合方案 v2 Robust)")
    print("=" * 60)
    print(f"  DA3 输出:     {source_dir}")
    print(f"  工作目录:     {ws_root}")
    print(f"  迭代次数:     {args.iterations}")
    print(f"  COLMAP 对齐:  {'跳过' if args.skip_colmap else '启用'}")
    print(f"  Open3D 扶正:  {'跳过' if args.skip_open3d else '启用 (使用 plyfile)'}")
    print()

    total_t0 = time.time()

    # 初始化工作目录
    if ws_root.exists():
        shutil.rmtree(ws_root)
    ws_root.mkdir(parents=True)

    data_dir = ws_root / "data"
    sparse_0 = data_dir / "colmap" / "sparse" / "0"
    dest_imgs = data_dir / "images"
    dest_imgs.mkdir(parents=True)

    # ====== Step 1: 同步图片 ======
    print("🖼️  [Step 1] 同步图片...")
    t0 = time.time()
    count = 0
    for img in sorted((source_dir / "extracted").glob("*.png")):
        shutil.copy2(img, dest_imgs)
        count += 1
    print(f"  ✅ 同步 {count} 张图片 ({time.time()-t0:.1f}s)")

    # ====== Step 2: 格式转换 ======
    print("\n📦 [Step 2] DA3 → COLMAP 格式转换")
    # 此处依然使用原来的转换脚本，但后面我们会用 fix 脚本修正它
    colmap_root = data_dir / "colmap"
    cmd = ["python3", "convert_da3_to_colmap.py", "--base_dir", str(source_dir), "--output_dir", str(colmap_root)]
    subprocess.run(cmd, check=True)

    # ====== Step 3: 核心修复 (解决错位与角度) ======
    pcd_path = source_dir / "pcd" / "combined_pcd.ply"
    colmap_fix_with_axes(sparse_0, pcd_path)

    # ====== Step 4 (可选): COLMAP 平面对齐 ======
    colmap_ok = False
    if not args.skip_colmap:
        aligned_dir = data_dir / "colmap" / "sparse" / "aligned"
        colmap_ok = colmap_align(sparse_0, aligned_dir, args.colmap_error)

        if colmap_ok:
            print("  📐 使用 COLMAP 对齐后的模型替换原始模型")
            for f in aligned_dir.iterdir():
                shutil.copy2(f, sparse_0 / f.name)
    else:
        print("\n🅰️  [COLMAP 对齐] 已跳过")

    # ====== Step 4: 训练 3DGS ======
    print(f"\n🔥 [Step 4] 开始 3DGS 训练 ({args.iterations} 迭代)...")

    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    train_cmd = [
        PYTHON_EXE, NS_TRAIN, "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(ws_root / "outputs"),
        "--experiment-name", "da3_aligned",
        "--pipeline.model.random-init", "False",
        "--max-num-iterations", str(args.iterations),
        "--viewer.quit-on-train-completion", "True",
        "colmap",
        "--orientation-method", "none",
        "--center-method", "poses",
        "--auto-scale-poses", "True"
    ]

    print(f"  执行: {' '.join(train_cmd)}")
    t0 = time.time()
    try:
        subprocess.run(train_cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n  ⚠️ 用户手动停止训练 (Ctrl+C)。尝试继续导出...")
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ 训练非正常退出 (exit code {e.returncode})。尝试继续导出...")

    train_time = time.time() - t0
    print(f"\n  ✅ 训练结束 ({train_time:.0f}s ≈ {train_time/60:.1f}min)")

    # ====== Step 5: 导出 PLY ======
    print("\n📤 [Step 5] 导出 Gaussian Splatting PLY...")
    config_paths = list((ws_root / "outputs/da3_aligned").rglob("config.yml"))
    export_dir = ws_root / "export"
    ply_path = None

    if config_paths:
        latest_config = max(config_paths, key=lambda p: p.stat().st_mtime)
        export_cmd = [
            PYTHON_EXE, NS_EXPORT, "gaussian-splat",
            "--load-config", str(latest_config),
            "--output-dir", str(export_dir)
        ]
        print(f"  执行: {' '.join(export_cmd)}")
        subprocess.run(export_cmd, check=True, env=env)

        ply_candidates = list(export_dir.glob("*.ply"))
        if ply_candidates:
            ply_path = ply_candidates[0]
            ply_size = ply_path.stat().st_size / 1024 / 1024
            print(f"  ✅ 导出成功: {ply_path} ({ply_size:.1f}MB)")
        else:
            print("  ⚠️ 导出目录中未找到 PLY 文件")
    else:
        print("  ⚠️ 未发现 config.yml，无法导出")

    # ====== Step 6 (可选): Open3D Fuzheng (Robust) ======
    open3d_ok = False
    aligned_ply_path = None

    if not args.skip_open3d and ply_path and ply_path.exists():
        aligned_ply_path = ply_path.parent / f"{ply_path.stem}_aligned{ply_path.suffix}"
        
        # 使用 Robust Alignment
        open3d_ok = open3d_align_robust(
            ply_path, aligned_ply_path,
            distance_threshold=args.open3d_threshold,
            translate_to_ground=args.translate_to_ground
        )
    elif args.skip_open3d:
        print("\n🅱️  [Open3D 扶正] 已跳过")

    # ====== 汇总 ======
    total_time = time.time() - total_t0
    print()
    print("=" * 60)
    print(f"  ✨ Pipeline 完成! (总耗时: {total_time:.0f}s)")
    print("=" * 60)
    print(f"  🅰️ COLMAP:  {'✅ 成功' if colmap_ok else '跳过' if args.skip_colmap else '❌ 失败'}")
    print(f"  🅱️ Open3D:  {'✅ 成功' if open3d_ok else '跳过' if args.skip_open3d else '❌ 失败'}")
    print()

    if ply_path and ply_path.exists():
        print("📁 输出文件:")
        print(f"  原始 PLY:  {ply_path} ({ply_path.stat().st_size/1024/1024:.1f}MB)")
        if open3d_ok and aligned_ply_path and aligned_ply_path.exists():
            print(f"  扶正 PLY:  {aligned_ply_path} ({aligned_ply_path.stat().st_size/1024/1024:.1f}MB)")
            print("  (大小应与原始 PLY 几乎一致)")


def main():
    parser = argparse.ArgumentParser(
        description="DA3 → 3DGS + 双重对齐 Pipeline (融合方案 v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--da3_output", type=str,
                        default=str(DA3_OUTPUT),
                        help="DA3 输出目录")
    parser.add_argument("--iterations", type=int, default=15000,
                        help="训练迭代次数")
    parser.add_argument("--colmap_error", type=float, default=0.02,
                        help="COLMAP 对齐最大误差")
    parser.add_argument("--open3d_threshold", type=float, default=0.02,
                        help="Open3D RANSAC 距离阈值")
    parser.add_argument("--translate_to_ground", action="store_true",
                        help="将地面平移到 Z=0")
    parser.add_argument("--skip_colmap", action="store_true",
                        help="跳过 COLMAP 对齐")
    parser.add_argument("--skip_open3d", action="store_true",
                        help="跳过 Open3D 扶正")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
