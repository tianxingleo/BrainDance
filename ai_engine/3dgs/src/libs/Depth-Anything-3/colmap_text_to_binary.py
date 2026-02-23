#!/usr/bin/env python3
"""
将COLMAP文本格式转换为二进制格式
用于Depth Anything 3 -> SuGaR pipeline
"""

import os
import sys
import struct
import numpy as np

# Camera model constants
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


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """Pack and write to binary file."""
    if isinstance(data, (list, tuple)):
        bytes_to_write = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes_to_write = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes_to_write)


def write_cameras_binary(cameras, path_to_model_file):
    """将相机信息写入二进制文件"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for camera_id in cameras:
            camera = cameras[camera_id]
            model_id = CAMERA_MODEL_IDS[camera.model]
            camera_properties = [
                camera.id,
                model_id,
                camera.width,
                camera.height
            ]
            write_next_bytes(fid, camera_properties, "iiQQ")
            params = camera.params.tolist()
            write_next_bytes(fid, params, "d" * len(params))


def write_images_binary(images, path_to_model_file):
    """将图像位姿信息写入二进制文件"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for image_id in images:
            image = images[image_id]
            binary_image_properties = [
                image.id,
                image.qvec[0],
                image.qvec[1],
                image.qvec[2],
                image.qvec[3],
                image.tvec[0],
                image.tvec[1],
                image.tvec[2],
                image.camera_id
            ]
            write_next_bytes(fid, binary_image_properties, "idddddddi")
            name_bytes = image.name.encode("utf-8") + b"\x00"
            fid.write(name_bytes)
            write_next_bytes(fid, len(image.xys), "Q")
            for xy, point3D_id in zip(image.xys, image.point3D_ids):
                write_next_bytes(fid, [xy[0], xy[1], point3D_id], "ddq")


def write_points3D_binary(points3D, path_to_model_file):
    """将3D点云写入二进制文件"""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for point3D_id in points3D:
            point3D = points3D[point3D_id]
            binary_point_line_properties = [
                point3D.id,
                point3D.xyz[0],
                point3D.xyz[1],
                point3D.xyz[2],
                point3D.rgb[0],
                point3D.rgb[1],
                point3D.rgb[2],
                point3D.error
            ]
            write_next_bytes(fid, binary_point_line_properties, "QdddBBBd")
            track = point3D.track
            write_next_bytes(fid, len(track), "Q")
            for image_id, point2D_idx in track:
                write_next_bytes(fid, [image_id, point2D_idx], "ii")


def read_cameras_text(path):
    """读取文本格式的相机信息"""
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params
                )
    return cameras


def read_images_text(path):
    """读取文本格式的图像位姿信息"""
    import collections
    BaseImage = collections.namedtuple(
        "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

    images = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = BaseImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids
                )
    return images


def read_points3D_text(path):
    """读取文本格式的3D点云"""
    import collections
    Point3D = collections.namedtuple(
        "Point3D", ["id", "xyz", "rgb", "error", "track"])

    xyzs = []
    rgbs = []
    errors = []
    tracks = []
    point3D_ids = []

    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                track_elems = list(map(int, elems[8:]))
                track = [(track_elems[i], track_elems[i+1]) for i in range(0, len(track_elems), 2)]

                xyzs.append(xyz)
                rgbs.append(rgb)
                errors.append(error)
                tracks.append(track)
                point3D_ids.append(point3D_id)

    points3D = {}
    for i in range(len(point3D_ids)):
        points3D[point3D_ids[i]] = Point3D(
            id=point3D_ids[i],
            xyz=xyzs[i],
            rgb=rgbs[i],
            error=errors[i],
            track=tracks[i]
        )

    return points3D


class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params


def convert_text_to_binary(input_dir, output_dir=None):
    """
    将COLMAP文本格式转换为二进制格式

    Args:
        input_dir: 包含文本格式COLMAP数据的目录
        output_dir: 输出目录（默认为input_dir）
    """
    if output_dir is None:
        output_dir = input_dir

    # 检查输入文件
    cameras_txt = os.path.join(input_dir, "cameras.txt")
    images_txt = os.path.join(input_dir, "images.txt")
    points3D_txt = os.path.join(input_dir, "points3D.txt")

    if not os.path.exists(cameras_txt):
        print(f"❌ 错误: 找不到 {cameras_txt}")
        return False

    if not os.path.exists(images_txt):
        print(f"❌ 错误: 找不到 {images_txt}")
        return False

    if not os.path.exists(points3D_txt):
        print(f"❌ 错误: 找不到 {points3D_txt}")
        return False

    print(f"读取文本格式COLMAP数据...")
    print(f"  相机文件: {cameras_txt}")
    print(f"  图像文件: {images_txt}")
    print(f"  点云文件: {points3D_txt}")

    # 读取文本文件
    cameras = read_cameras_text(cameras_txt)
    images = read_images_text(images_txt)
    points3D = read_points3D_text(points3D_txt)

    print(f"  读取了 {len(cameras)} 个相机")
    print(f"  读取了 {len(images)} 个图像")
    print(f"  读取了 {len(points3D)} 个3D点")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 写入二进制文件
    print(f"\n写入二进制格式COLMAP数据...")

    cameras_bin = os.path.join(output_dir, "cameras.bin")
    images_bin = os.path.join(output_dir, "images.bin")
    points3D_bin = os.path.join(output_dir, "points3D.bin")

    write_cameras_binary(cameras, cameras_bin)
    print(f"  ✅ {cameras_bin}")

    write_images_binary(images, images_bin)
    print(f"  ✅ {images_bin}")

    write_points3D_binary(points3D, points3D_bin)
    print(f"  ✅ {points3D_bin}")

    print(f"\n✅ 转换完成! 输出目录: {output_dir}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python colmap_text_to_binary.py <input_dir> [output_dir]")
        print("")
        print("参数:")
        print("  input_dir  - 包含COLMAP文本格式文件的目录")
        print("  output_dir - 输出目录（默认为input_dir）")
        print("")
        print("示例:")
        print("  python colmap_text_to_binary.py ./sparse/0")
        print("  python colmap_text_to_binary.py ./sparse/0 ./sparse/0")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir

    success = convert_text_to_binary(input_dir, output_dir)
    sys.exit(0 if success else 1)
