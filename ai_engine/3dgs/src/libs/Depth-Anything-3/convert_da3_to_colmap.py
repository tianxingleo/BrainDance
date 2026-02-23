#!/usr/bin/env python3
"""
将Depth Anything 3的输出转换为COLMAP标准格式（文本格式）
用于SuGaR重建

输入：
- camera_poses.txt: 相机位姿（c2w格式）
- intrinsic.txt: 相机内参（每帧fx, fy, cx, cy）
- pcd/combined_pcd.ply: 点云文件
- extracted/: 图像目录

输出：
- sparse/0/cameras.txt: COLMAP相机参数
- sparse/0/images.txt: COLMAP图像位姿
- sparse/0/points3D.txt: COLMAP 3D点云
- images/: 图像目录（符号链接）
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import struct
import shutil


def rotmat_to_quat(R):
    """将旋转矩阵转换为四元数"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def read_binary_ply(ply_path):
    """读取二进制PLY点云文件"""
    with open(ply_path, 'rb') as f:
        header = ""
        while True:
            line = f.readline().decode('ascii')
            header += line
            if line.startswith("end_header"):
                break

        # 解析头部以获取顶点数量
        num_vertices = 0
        for line in header.split('\n'):
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])

        # 读取二进制数据
        data = f.read()

    # struct格式：3个浮点数（每个4字节）和3个无符号字符（每个1字节）
    struct_fmt = "fffBBB"
    struct_size = struct.calcsize(struct_fmt)

    # 检查实际数据大小
    actual_struct_size = len(data) // num_vertices

    vertices = []
    colors = []

    for i in range(num_vertices):
        offset = i * actual_struct_size
        v = struct.unpack_from(struct_fmt, data, offset)
        vertices.append(v[:3])
        colors.append(v[3:])

    return np.array(vertices), np.array(colors)


def convert_to_colmap(base_dir, output_dir):
    """
    将Depth Anything 3的输出转换为COLMAP格式

    Args:
        base_dir: DA3输出目录
        output_dir: COLMAP输出目录
    """
    # 输入文件路径
    intrinsic_file = os.path.join(base_dir, "intrinsic.txt")
    pose_file = os.path.join(base_dir, "camera_poses.txt")
    ply_file = os.path.join(base_dir, "pcd/combined_pcd.ply")
    img_dir = os.path.join(base_dir, "extracted")

    # 检查输入文件
    if not os.path.exists(intrinsic_file):
        print(f"❌ 错误: 找不到 {intrinsic_file}")
        return False

    if not os.path.exists(pose_file):
        print(f"❌ 错误: 找不到 {pose_file}")
        return False

    if not os.path.exists(ply_file):
        print(f"❌ 错误: 找不到 {ply_file}")
        return False

    if not os.path.exists(img_dir):
        print(f"❌ 错误: 找不到图像目录 {img_dir}")
        return False

    # 创建输出目录
    sparse_dir = os.path.join(output_dir, "sparse/0")
    os.makedirs(sparse_dir, exist_ok=True)

    print("加载相机数据...")
    print(f"  内参文件: {intrinsic_file}")
    print(f"  位姿文件: {pose_file}")
    print(f"  点云文件: {ply_file}")
    print(f"  图像目录: {img_dir}")

    # 1. 加载数据
    all_intrinsics = np.loadtxt(intrinsic_file)  # N x 4 (fx, fy, cx, cy)
    all_poses_c2w = np.loadtxt(pose_file).reshape(-1, 4, 4)
    img_names = sorted(os.listdir(img_dir))

    num_frames = len(all_poses_c2w)

    if num_frames != len(img_names):
        print(f"⚠️ 警告: 位姿数量 ({num_frames}) 与图像数量 ({len(img_names)}) 不匹配")

    # 检查图像尺寸
    first_img_path = os.path.join(img_dir, img_names[0])
    with Image.open(first_img_path) as img:
        orig_w, orig_h = img.size

    print(f"  图像尺寸: {orig_w}x{orig_h}")
    print(f"  帧数: {num_frames}")

    # 2. 写入cameras.txt
    print("\n写入cameras.txt...")

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

        # 计算缩放因子
        proc_w = all_intrinsics[0][2] * 2
        proc_h = all_intrinsics[0][3] * 2

        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h

        for i in range(num_frames):
            intri = all_intrinsics[i]
            fx, fy, cx, cy = intri[0] * scale_x, intri[1] * scale_y, intri[2] * scale_x, intri[3] * scale_y

            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # MODEL 1 是 PINHOLE: fx, fy, cx, cy
            f.write(f"{i+1} PINHOLE {orig_w} {orig_h} {fx} {fy} {cx} {cy}\n")

    print(f"  ✅ 写入了 {num_frames} 个相机")

    # 3. 写入images.txt
    print("\n写入images.txt...")

    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i in range(num_frames):
            c2w = all_poses_c2w[i]
            w2c = np.linalg.inv(c2w)

            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat_to_quat(R)

            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i+1} {img_names[i]}\n")
            f.write("\n")  # 暂时没有points2D

    print(f"  ✅ 写入了 {num_frames} 个图像位姿")

    # 4. 写入points3D.txt
    print("\n写入points3D.txt...")

    points, colors = read_binary_ply(ply_file)
    print(f"  加载了 {len(points)} 个3D点")

    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for i in range(len(points)):
            p = points[i]
            c = colors[i]
            # ID, X, Y, Z, R, G, B, ERROR, TRACK...
            # 使用dummy error 0 和 no track
            f.write(f"{i+1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n")

    print(f"  ✅ 写入了 {len(points)} 个3D点")

    # 5. 创建图像目录（符号链接或复制）
    print("\n创建images目录...")

    colmap_img_dir = os.path.join(output_dir, "images")
    if os.path.exists(colmap_img_dir):
        if os.path.islink(colmap_img_dir):
            os.unlink(colmap_img_dir)
        else:
            shutil.rmtree(colmap_img_dir)

    # 尝试创建符号链接（使用绝对路径，确保在任何工作目录下都能解析）
    try:
        os.symlink(os.path.abspath(img_dir), colmap_img_dir)
        print(f"  ✅ 创建符号链接: {colmap_img_dir} -> {img_dir}")
    except OSError:
        print("  符号链接失败，复制图像...")
        shutil.copytree(img_dir, colmap_img_dir)
        print(f"  ✅ 复制图像: {colmap_img_dir}")

    print(f"\n✅ COLMAP格式转换完成!")
    print(f"输出目录: {output_dir}")
    print(f"  - cameras.txt: {sparse_dir}/cameras.txt")
    print(f"  - images.txt: {sparse_dir}/images.txt")
    print(f"  - points3D.txt: {sparse_dir}/points3D.txt")
    print(f"  - images: {colmap_img_dir}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将Depth Anything 3的输出转换为COLMAP标准格式（文本格式）"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Depth Anything 3输出目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="COLMAP输出目录"
    )

    args = parser.parse_args()

    success = convert_to_colmap(args.base_dir, args.output_dir)
    sys.exit(0 if success else 1)
