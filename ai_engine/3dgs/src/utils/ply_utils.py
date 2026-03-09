# src/utils/ply_utils.py
# 功能：提供PLY点云处理工具函数
# 实现：包含点云后处理和切割算法
# 逻辑：基于统计分位数对点云进行切割，去除背景伪影
# 包含：点云分位数切割函数、PLY文件处理算法
import json
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional, Callable

# --- 依赖检查逻辑 ---
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("⚠️ Warning: 'plyfile' library not found. Point cloud culling will be skipped.")

    
def perform_percentile_culling(ply_path, json_path, output_path, keep_percentile=0.9):
    """
    [点云后处理] 基于统计分位数的暴力切割
    功能：去除 Gaussian Splatting 训练后产生在远处的背景伪影。
    依赖：plyfile 库
    """
    # 检查依赖
    if not HAS_PLYFILE: return False
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    try:
        # 1. 计算场景中心
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        # 2. 读取 PLY 点云数据
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        # 堆叠 x,y,z 坐标
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        
        # 3. 计算所有点到中心的距离
        dists_pts = np.linalg.norm(points - center, axis=1)

        # [算法逻辑] 确定阈值半径
        # 2. ✅ 这里修改：使用传入的参数 keep_percentile
        threshold_radius = np.percentile(dists_pts, keep_percentile * 100)
        
        # 4. 读取不透明度 (Opacity) 并过滤
        # Gaussian Splatting 存储的 opacity 通常经过 sigmoid 激活，需要还原
        # 这里 simplified: 假设 vertex['opacity'] 是 logit
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 联合掩码：(在半径内) AND (不透明度 > 0.05)
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        filtered_vertex = vertex[mask]
        
        # 5. 写入新文件
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True
    except Exception as e:
        print(f"❌ 切割失败: {e}")
        return False


def compress_ply_to_splat(
    ply_path: str,
    splat_path: str,
    opacity_threshold: float = 0.05
) -> str:
    """
    将 3DGS 的 .ply 压缩为移动端可直接加载的 .splat 二进制。
    """
    if not HAS_PLYFILE:
        raise RuntimeError("缺少 plyfile 依赖，无法执行 .ply -> .splat 压缩")

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    fields = set(vertex.data.dtype.names or [])

    required = {
        "x", "y", "z",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(list(required - fields))
    if missing:
        raise RuntimeError(f"PLY 字段不完整，缺失: {', '.join(missing)}")

    opacities = 1.0 / (1.0 + np.exp(-vertex["opacity"]))
    mask = opacities > float(opacity_threshold)
    vertex = vertex[mask]
    opacities = opacities[mask]

    positions = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T.astype(np.float32)
    scales = np.exp(
        np.vstack((vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])).T
    ).astype(np.float32)

    sh_c0 = 0.28209479177387814
    colors = (0.5 + sh_c0 * np.vstack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"])).T) * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    alphas = np.clip(opacities * 255.0, 0, 255).astype(np.uint8)
    rgba = np.hstack((colors, alphas[:, None]))

    rots = np.vstack((vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"])).T.astype(np.float32)
    rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-8)
    rots_uint8 = np.clip((rots * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    dists = np.linalg.norm(positions, axis=1)
    sort_idx = np.argsort(dists)

    output = Path(splat_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for i in sort_idx:
            f.write(positions[i].tobytes())
            f.write(scales[i].tobytes())
            f.write(rgba[i].tobytes())
            f.write(rots_uint8[i].tobytes())

    return str(output)


def convert_ply_to_ksplat(
    ply_path: str,
    ksplat_path: str,
    script_path: str,
    alpha_removal_threshold: int = 1
) -> str:
    """
    调用 GaussianSplats3D 官方 Node 工具，将 .ply 转为 .ksplat。
    """
    if not script_path:
        raise RuntimeError("未提供 KSPLAT 脚本路径 (KSPLAT_SCRIPT_PATH)")

    cmd = [
        "node",
        script_path,
        ply_path,
        ksplat_path,
        str(alpha_removal_threshold),
    ]
    subprocess.run(cmd, check=True)
    return ksplat_path


def compress_model_for_delivery(
    model_path: str,
    output_format: str = "splat",
    opacity_threshold: float = 0.05,
    ksplat_script_path: Optional[str] = None,
    alpha_removal_threshold: int = 1,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    将模型压缩为移动端更友好的格式，失败时抛异常。
    """
    in_path = Path(model_path)
    suffix = in_path.suffix.lower()
    fmt = (output_format or "splat").strip().lower()

    if suffix in (".splat", ".ksplat"):
        return str(in_path)
    if suffix != ".ply":
        raise RuntimeError(f"不支持的输入格式: {in_path.name}")

    if fmt == "ply":
        return str(in_path)

    if fmt == "ksplat":
        out_path = in_path.with_suffix(".ksplat")
        if log_callback:
            log_callback("🗜️ 正在压缩为 .ksplat...")
        return convert_ply_to_ksplat(
            str(in_path),
            str(out_path),
            script_path=ksplat_script_path or "",
            alpha_removal_threshold=alpha_removal_threshold,
        )

    out_path = in_path.with_suffix(".splat")
    if log_callback:
        log_callback("🗜️ 正在压缩为 .splat...")
    return compress_ply_to_splat(
        str(in_path),
        str(out_path),
        opacity_threshold=opacity_threshold,
    )
