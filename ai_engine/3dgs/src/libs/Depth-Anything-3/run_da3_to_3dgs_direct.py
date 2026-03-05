import subprocess
import os
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import struct

# ================= 🔧 路径配置 =================
DA3_OUTPUT = Path("/home/ltx/projects/Depth-Anything-3/output/sugar_streaming")
WS_ROOT = DA3_OUTPUT / "da3_3dgs_pipeline"
CONDA_PREFIX = "/home/ltx/my_envs/gs_linux_backup"
NS_ENV_BIN = f"{CONDA_PREFIX}/bin"

# 直接使用环境中的 python 解释器来运行脚本，避免 shebang 导致的 python3.10 找不到错误
PYTHON_EXE = f"{NS_ENV_BIN}/python"
NS_TRAIN = f"{NS_ENV_BIN}/ns-train"
NS_EXPORT = f"{NS_ENV_BIN}/ns-export"

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

def convert_da3_to_colmap(source_dir, target_sparse_dir):
    print("📦 [格式转换] 正在将 DA3 Poses & PCD 转换为 COLMAP 格式...")
    target_sparse_dir.mkdir(parents=True, exist_ok=True)
    
    intrinsics = np.loadtxt(source_dir / "intrinsic.txt")
    poses_c2w = np.loadtxt(source_dir / "camera_poses.txt").reshape(-1, 4, 4)
    img_names = sorted([f.name for f in (source_dir / "extracted").glob("*.png")])
    
    if not img_names:
        raise ValueError(f"在 {source_dir / 'extracted'} 中没有找到图片。")

    with Image.open(source_dir / "extracted" / img_names[0]) as img:
        orig_w, orig_h = img.size
    
    # 1. 写入 cameras.txt
    with open(target_sparse_dir / "cameras.txt", "w") as f:
        # 基于 280x504 的缩放逻辑 (或者根据 intrinsic[0][2]*2 自动推断)
        ref_w = intrinsics[0][2] * 2
        ref_h = intrinsics[0][3] * 2
        scale_x = orig_w / ref_w
        scale_y = orig_h / ref_h
        
        fx, fy, cx, cy = intrinsics[0][0]*scale_x, intrinsics[0][1]*scale_y, intrinsics[0][2]*scale_x, intrinsics[0][3]*scale_y
        f.write(f"1 PINHOLE {orig_w} {orig_h} {fx} {fy} {cx} {cy}\n")

    # 2. 写入 images.txt
    print("📸 [格式转换] 生成 images.txt...")
    with open(target_sparse_dir / "images.txt", "w") as f:
        for i, (pose_c2w, name) in enumerate(zip(poses_c2w, img_names)):
            pose_w2c = np.linalg.inv(pose_c2w)
            R = pose_w2c[:3, :3]
            t = pose_w2c[:3, 3]
            q = rotmat_to_quat(R)
            f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")

    # 3. 写入 points3D.txt
    print("💎 [格式转换] 解析 PLY PCD...")
    pcd_path = source_dir / "pcd" / "combined_pcd.ply"
    
    with open(pcd_path, 'rb') as f:
        header = ""
        while True:
            line = f.readline().decode('ascii')
            header += line
            if line.startswith("end_header"): break
        num_vertices = 0
        for line in header.split('\n'):
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
        data = f.read()
    
    # 解析二进制 PLY (x,y,z 为 float, r,g,b 为 uchar)
    struct_fmt = "fffBBB"
    row_size = len(data) // num_vertices
    
    with open(target_sparse_dir / "points3D.txt", "w") as f_out:
        for i in range(num_vertices):
            offset = i * row_size
            v = struct.unpack_from(struct_fmt, data, offset)
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK
            f_out.write(f"{i+1} {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]} 0\n")
    
    print(f"✅ 转换完成: {len(img_names)} 帧和 {num_vertices} 个点已注册。")

# ================= 🚀 主流程 =================

def run_pipeline():
    # 重新初始化工作目录
    if WS_ROOT.exists(): shutil.rmtree(WS_ROOT)
    WS_ROOT.mkdir(parents=True)
    
    data_dir = WS_ROOT / "data"
    sparse_0 = data_dir / "colmap" / "sparse" / "0"
    dest_imgs = data_dir / "images"
    dest_imgs.mkdir(parents=True)

    print("🖼️ 同步图片中...")
    for img in (DA3_OUTPUT / "extracted").glob("*.png"):
        shutil.copy2(img, dest_imgs)
    
    convert_da3_to_colmap(DA3_OUTPUT, sparse_0)

    print("🔥 [Direct Pipeline] 开始 3DGS 训练...")
    
    # 增加环境变量配置，解决 setuptools/_distutils_hack 导致的 AssertionError
    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    # 使用 PYTHON_EXE 显式调用 ns-train 以绕过 shebang 报错
    cmd = [
        PYTHON_EXE, NS_TRAIN, "splatfacto", 
        "--data", str(data_dir), 
        "--output-dir", str(WS_ROOT / "outputs"),
        "--experiment-name", "da3_direct",
        "--pipeline.model.random-init", "False",
        "--max-num-iterations", "15000",
        "colmap"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    # ================= 📤 自动导出 PLY =================
    print("🔥 [Direct Pipeline] 训练完成，正在导出 Gaussian Splatting PLY...")
    
    # 查找刚才生成的 config.yml
    config_paths = list((WS_ROOT / "outputs/da3_direct").rglob("config.yml"))
    if config_paths:
        # 按修改时间排序，取最新生成的一个
        latest_config = max(config_paths, key=lambda p: p.stat().st_mtime)
        export_dir = WS_ROOT / "export"
        
        export_cmd = [
            PYTHON_EXE, NS_EXPORT, "gaussian-splat",
            "--load-config", str(latest_config),
            "--output-dir", str(export_dir)
        ]
        
        print(f"执行导出命令: {' '.join(export_cmd)}")
        subprocess.run(export_cmd, check=True, env=env)
        print(f"✅ 导出成功！PLY 文件已保存在: {export_dir}")
    else:
        print("⚠️ 未发现训练生成的 config.yml，无法导出 PLY。")

if __name__ == "__main__":
    run_pipeline()
