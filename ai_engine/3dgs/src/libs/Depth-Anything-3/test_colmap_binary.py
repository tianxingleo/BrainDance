#!/usr/bin/env python3
"""
测试COLMAP二进制文件是否能被SuGaR正确读取
"""

import sys
import os
import struct
import numpy as np
import collections

# 直接复制COLMAP读取函数，避免依赖SuGaR环境
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

CAMERA_MODEL_IDS = {v: k for k, v in CAMERA_MODELS.items()}
CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12
}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_binary(path_to_model_file):
    """读取二进制相机内参"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODELS[model_id]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_NUM_PARAMS[model_name]
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """读取二进制图像位姿"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = BaseImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_binary(path_to_model_file):
    """读取二进制3D点云"""
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error

    return xyzs, rgbs, errors


def test_colmap_binary(sparse_dir):
    """测试COLMAP二进制文件读取"""
    print(f"测试COLMAP二进制文件读取: {sparse_dir}")
    print("")

    # 文件路径
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points3D_bin = os.path.join(sparse_dir, "points3D.bin")

    # 检查文件是否存在
    for file in [cameras_bin, images_bin, points3D_bin]:
        if not os.path.exists(file):
            print(f"❌ 文件不存在: {file}")
            return False

    # 读取相机内参
    print("1. 读取相机内参...")
    try:
        cameras = read_intrinsics_binary(cameras_bin)
        print(f"   ✅ 读取了 {len(cameras)} 个相机")
        for cam_id in list(cameras.keys())[:3]:
            cam = cameras[cam_id]
            print(f"      相机 {cam_id}: {cam.model} {cam.width}x{cam.height}")
            print(f"         params: {cam.params}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return False

    # 读取图像位姿
    print("\n2. 读取图像位姿...")
    try:
        images = read_extrinsics_binary(images_bin)
        print(f"   ✅ 读取了 {len(images)} 个图像")
        for img_id in list(images.keys())[:3]:
            img = images[img_id]
            print(f"      图像 {img_id}: {img.name}")
            print(f"         qvec: {img.qvec}")
            print(f"         tvec: {img.tvec}")
            print(f"         camera_id: {img.camera_id}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return False

    # 读取3D点云
    print("\n3. 读取3D点云...")
    try:
        xyzs, rgbs, errors = read_points3D_binary(points3D_bin)
        print(f"   ✅ 读取了 {len(xyzs)} 个3D点")
        print(f"      XYZ形状: {xyzs.shape}")
        print(f"      RGB形状: {rgbs.shape}")
        print(f"      Error形状: {errors.shape}")
        print(f"      前3个点:")
        for i in range(min(3, len(xyzs))):
            print(f"         点{i}: XYZ={xyzs[i]}, RGB={rgbs[i]}, Error={errors[i][0]:.4f}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 检查一致性
    print("\n4. 检查数据一致性...")
    if len(cameras) != len(images):
        print(f"   ⚠️ 警告: 相机数量 ({len(cameras)}) 与图像数量 ({len(images)}) 不匹配")
    else:
        print(f"   ✅ 相机数量与图像数量一致: {len(cameras)}")

    print("\n✅ 所有测试通过!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 test_colmap_binary.py <sparse_dir>")
        print("")
        print("示例:")
        print("  python3 test_colmap_binary.py /home/ltx/projects/Depth-Anything-3/output/test_colmap/sparse/0")
        sys.exit(1)

    sparse_dir = sys.argv[1]

    success = test_colmap_binary(sparse_dir)
    sys.exit(0 if success else 1)
