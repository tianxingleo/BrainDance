"""
DA3 → DN-Splatter 统一 Pipeline
================================
完整流程:
  1) 将 Depth-Anything-3 的输出 (图片 + Depth NPZ + Poses) 转换为 DN-Splatter 所需格式
     - transforms.json (Nerfstudio格式的相机参数)
     - images/ (原图)
     - depths/ (16-bit PNG 毫米深度图)
     - normals_from_pretrain/ (从深度图生成的法线贴图)
  2) 使用 ns-train dn-splatter 训练
  3) 导出 PLY

用法:
    python run_da3_to_dn_splatter_pipeline.py [--source_dir PATH] [--output_name NAME]

依赖解析:
    - 数据转换步骤: numpy, Pillow, tqdm (当前 python)
    - 训练步骤: nerfstudio + dn_splatter (gs_linux_backup 环境)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= 🔧 默认路径配置 =================
DEFAULT_SOURCE_DIR = Path("/home/ltx/projects/Depth-Anything-3/output/sugar_streaming")
DEFAULT_OUTPUT_NAME = "da3_dn_splatter"

# 环境路径 (DN-Splatter 安装在 gs_linux_backup 环境中)
CONDA_PREFIX = "/home/ltx/my_envs/gs_linux_backup"
NS_ENV_BIN = f"{CONDA_PREFIX}/bin"
NS_PYTHON_EXE = f"{NS_ENV_BIN}/python"
NS_TRAIN = f"{NS_ENV_BIN}/ns-train"
NS_EXPORT = f"{NS_ENV_BIN}/ns-export"

PROJECT_ROOT = Path("/home/ltx/projects/Depth-Anything-3")


# ================= Step 1: 数据格式转换 =================

def depth_to_normal(depth, K):
    """
    从深度图计算表面法线 (纯 numpy 实现)
    depth: float32, (H, W) 以米为单位
    K: (fx, fy, cx, cy)
    Returns: normal_img (H, W, 3) uint8 [0, 255]
    """
    fx, fy, cx, cy = K
    h, w = depth.shape

    # Central difference gradients
    zy, zx = np.gradient(depth)

    # Scale gradients by focal length/depth to account for perspective
    scale_x = fx / (depth + 1e-6)
    scale_y = fy / (depth + 1e-6)

    nx = -zx * scale_x
    ny = -zy * scale_y
    nz = np.ones_like(depth)

    # Normalize
    n = np.sqrt(nx**2 + ny**2 + nz**2)
    n[n == 0] = 1.0

    normal = np.dstack((nx / n, ny / n, nz / n))

    # Map [-1, 1] -> [0, 255]
    normal_img = ((normal + 1) / 2 * 255).astype(np.uint8)
    return normal_img


def load_intrinsics(path):
    """读取内参文件 (单行: fx fy cx cy)"""
    print(f"  📐 加载内参: {path}")
    with open(path, "r") as f:
        lines = f.readlines()
        parts = (
            lines[0].strip().split()
            if "," not in lines[0]
            else lines[0].strip().split(",")
        )
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def load_poses(path):
    """读取相机位姿文件 (每帧16个数字 = 4x4矩阵, 按行排列)"""
    print(f"  📷 加载位姿: {path}")
    poses = []
    with open(path, "r") as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            poses.append(np.array(nums).reshape(4, 4))
    return poses


def convert_da3_to_dn_splatter(source_dir: Path, dataset_dir: Path):
    """
    将 DA3 输出转换为 DN-Splatter 可接受的 Nerfstudio JSON 数据格式
    """
    print("=" * 60)
    print("📦 [Step 1] DA3 → DN-Splatter 数据格式转换")
    print("=" * 60)

    extracted_dir = source_dir / "extracted"
    results_dir = source_dir / "results_output"

    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    if not extracted_dir.exists():
        raise FileNotFoundError(f"提取图片目录不存在: {extracted_dir}")
    if not results_dir.exists():
        raise FileNotFoundError(f"深度结果目录不存在: {results_dir}")

    # 创建输出目录
    out_images = dataset_dir / "images"
    out_depths = dataset_dir / "depths"
    out_normals = dataset_dir / "normals_from_pretrain"

    for d in [out_images, out_depths, out_normals]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 加载元数据
    fx, fy, cx, cy = load_intrinsics(source_dir / "intrinsic.txt")
    poses = load_poses(source_dir / "camera_poses.txt")

    # 2. 匹配文件
    img_files = sorted(list(extracted_dir.glob("*.png")))
    npz_files = sorted(
        list(results_dir.glob("*.npz")), key=lambda x: int(x.stem.split("_")[1])
    )

    if not img_files:
        raise ValueError(f"在 {extracted_dir} 中未找到图片")
    if not npz_files:
        raise ValueError(f"在 {results_dir} 中未找到 NPZ 深度文件")

    num_frames = min(len(img_files), len(npz_files), len(poses))
    if len(img_files) != len(npz_files):
        print(
            f"  ⚠️  图片数量 ({len(img_files)}) 与 NPZ 数量 ({len(npz_files)}) 不匹配，"
            f"使用前 {num_frames} 帧"
        )

    # 3. 检测图片实际分辨率 (PIL returns width, height)
    first_img = Image.open(img_files[0])
    img_w, img_h = first_img.size
    print(f"  📐 图片分辨率: {img_w} x {img_h}")

    # 4. 检测深度图分辨率
    first_depth = np.load(npz_files[0])["depth"]
    depth_h, depth_w = first_depth.shape
    print(f"  📐 深度图分辨率: {depth_w} x {depth_h}")

    # 5. 计算缩放因子 & 调整内参
    # 内参是基于深度图分辨率的，需要缩放到图片分辨率
    if (depth_w, depth_h) != (img_w, img_h):
        scale_x = img_w / depth_w
        scale_y = img_h / depth_h
        print(f"  🔄 分辨率不匹配! 缩放比: x={scale_x:.4f}, y={scale_y:.4f}")
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        print(f"  📐 调整后内参: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, cx={cx_scaled:.2f}, cy={cy_scaled:.2f}")
        need_resize = True
    else:
        fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx, fy, cx, cy
        need_resize = False

    print(f"  🎞️  处理 {num_frames} 帧...")

    # OpenCV → OpenGL 坐标系转换矩阵
    flip_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    frames_data = []

    for i in tqdm(range(num_frames), desc="  转换中"):
        src_img_path = img_files[i]
        src_npz_path = npz_files[i]
        pose = poses[i]

        frame_name = f"frame_{i:05d}"
        dst_name = f"{frame_name}.png"

        # 1) 复制图片
        shutil.copy2(src_img_path, out_images / dst_name)

        # 2) 处理深度图 (NPZ → 16-bit uint16 PNG, 毫米)
        data = np.load(src_npz_path)
        depth_map = data["depth"]  # float32, meters

        # Resize depth to image resolution if needed
        if need_resize:
            depth_pil = Image.fromarray(depth_map)
            depth_pil = depth_pil.resize((img_w, img_h), Image.BILINEAR)
            depth_map = np.array(depth_pil, dtype=np.float32)

        depth_mm = (depth_map * 1000).astype(np.uint16)
        Image.fromarray(depth_mm).save(out_depths / dst_name)

        # 3) 从深度图生成法线图 (使用缩放后的内参)
        normal_map = depth_to_normal(depth_map, (fx_scaled, fy_scaled, cx_scaled, cy_scaled))
        Image.fromarray(normal_map).save(out_normals / dst_name)

        # 4) 处理位姿
        if np.isinf(pose).any() or np.isnan(pose).any():
            print(f"  ⚠️  第 {i} 帧位姿无效，跳过")
            continue

        # DA3 输出为 c2w (camera-to-world), 转换为 OpenGL 坐标系
        c2w_opengl = np.matmul(pose, flip_mat)

        frames_data.append(
            {
                "file_path": f"images/{dst_name}",
                "depth_file_path": f"depths/{dst_name}",
                "normal_file_path": f"normals_from_pretrain/{dst_name}",
                "transform_matrix": c2w_opengl.tolist(),
            }
        )

    # 5) 写入 transforms.json — 使用图片的实际分辨率
    output_json = {
        "fl_x": fx_scaled,
        "fl_y": fy_scaled,
        "cx": cx_scaled,
        "cy": cy_scaled,
        "w": int(img_w),
        "h": int(img_h),
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "camera_model": "OPENCV",
        "frames": frames_data,
    }

    json_path = dataset_dir / "transforms.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"  ✅ 数据转换完成: {len(frames_data)} 帧")
    print(f"     数据集路径: {dataset_dir.resolve()}")
    return True


# ================= Step 2: DN-Splatter 训练 =================

def run_dn_splatter_training(dataset_dir: Path, output_dir: Path, experiment_name: str,
                             max_iterations: int = 30000):
    """
    使用 ns-train 运行 DN-Splatter 训练
    """
    print()
    print("=" * 60)
    print("🔥 [Step 2] DN-Splatter 训练")
    print("=" * 60)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 环境变量
    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    cmd = [
        NS_PYTHON_EXE,
        NS_TRAIN,
        "dn-splatter",
        "--output-dir", str(output_dir),
        "--experiment-name", experiment_name,
        "--max-num-iterations", str(max_iterations),
        # DN-Splatter 特有的深度/法线损失配置
        "--pipeline.model.use-depth-loss", "True",
        "--pipeline.model.depth-lambda", "0.2",
        "--pipeline.model.use-normal-loss", "True",
        "--pipeline.model.normal-lambda", "0.05",
        "--pipeline.model.predict-normals", "True",
        "--pipeline.model.use-normal-tv-loss", "True",
        "--pipeline.model.two-d-gaussians", "True",
        # ===== 密度控制 (RTX 5070 12GB 优化, 目标~10GB显存) =====
        "--pipeline.model.densify-grad-thresh", "0.0004",   # 适中的分裂阈值
        "--pipeline.model.cull-alpha-thresh", "0.005",      # 清理低透明度高斯球
        "--pipeline.model.stop-split-at", "12000",          # 12000步后停止分裂
        "--pipeline.model.max-gs-num", "2000000",           # 高斯球上限 200万
        "--pipeline.model.sh-degree", "3",                  # SH阶数恢复为3, 更好的颜色质量
        # ===== 数据加载加速 =====
        "--pipeline.datamanager.dataloader-num-workers", "4", # 多线程加载数据, 提升GPU利用率
        # ===== Checkpoint 保存 (每5000步保存一次, 防止翻车) =====
        "--steps-per-save", "5000",
        "--save-only-latest-checkpoint", "False",
        # Viewer 设置
        "--viewer.websocket-port", "7007",
        "--viewer.quit-on-train-completion", "True",  # 训练完自动退出, 不阻塞
        "--vis", "viewer+tensorboard",
        # 数据解析器 (normal-nerfstudio 支持 json + depth + normal)
        "normal-nerfstudio",
        "--data", str(dataset_dir),
        # 不使用点云初始化 (DA3 输出中没有 SfM 格式的点云)
        "--load-3D-points", "False",
        "--load-pcd-normals", "False",
    ]

    print(f"  📋 训练命令:\n  {' '.join(cmd[:5])} \\\n    {' '.join(cmd[5:])}")
    print()

    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(PROJECT_ROOT))
        print("  ✅ 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        # Exit code 130 = SIGINT (Ctrl+C). 如果在 "Training Finished" 后按
        # Ctrl+C 退出 viewer, 训练本身是成功的, 应该继续导出 PLY.
        if e.returncode == 130 or e.returncode == -2:
            print("  ⚠️ 训练完成但被 Ctrl+C 中断 (viewer 退出), 继续导出...")
            return True
        print(f"  ❌ 训练失败: {e}")
        return False


# ================= Step 3: 导出 PLY =================

def export_ply(output_dir: Path, experiment_name: str):
    """
    训练完成后导出 Gaussian Splatting PLY 文件
    """
    print()
    print("=" * 60)
    print("📤 [Step 3] 导出 PLY")
    print("=" * 60)

    # 查找最新的 config.yml
    config_paths = list((output_dir / experiment_name).rglob("config.yml"))
    if not config_paths:
        print("  ⚠️  未发现训练生成的 config.yml，无法导出 PLY")
        return False

    latest_config = max(config_paths, key=lambda p: p.stat().st_mtime)
    export_dir = output_dir / "export"

    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    cmd = [
        NS_PYTHON_EXE,
        NS_EXPORT,
        "gaussian-splat",
        "--load-config", str(latest_config),
        "--output-dir", str(export_dir),
    ]

    print(f"  📋 导出命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"  ✅ 导出成功! PLY 文件: {export_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 导出失败: {e}")
        return False


# ================= 🚀 主入口 =================

def main():
    parser = argparse.ArgumentParser(
        description="DA3 → DN-Splatter 统一 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认路径
  python run_da3_to_dn_splatter_pipeline.py

  # 指定源目录和输出名
  python run_da3_to_dn_splatter_pipeline.py \\
    --source-dir /path/to/da3/output \\
    --output-name my_experiment

  # 只跑数据转换 (跳过训练)
  python run_da3_to_dn_splatter_pipeline.py --convert-only

  # 跳过数据转换 (只训练, 假设数据集已准备好)
  python run_da3_to_dn_splatter_pipeline.py --train-only
        """,
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
        help="DA3 输出目录 (包含 extracted/, results_output/, intrinsic.txt, camera_poses.txt)",
    )
    parser.add_argument(
        "--output-name", type=str, default=DEFAULT_OUTPUT_NAME,
        help="实验名称，用于数据集目录和训练输出目录命名",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=30000,
        help="最大训练迭代数 (默认: 30000)",
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="只运行数据格式转换，不训练",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="跳过数据转换，直接训练 (假设数据集已存在)",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="训练完后不导出 PLY",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="清除已有数据集目录后重新转换",
    )

    args = parser.parse_args()

    source_dir = args.source_dir
    dataset_dir = PROJECT_ROOT / f"{args.output_name}_dataset"
    output_dir = PROJECT_ROOT / f"{args.output_name}_output"
    experiment_name = args.output_name

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       DA3 → DN-Splatter 统一 Pipeline                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  源目录:    {str(source_dir)[:45]:45s} ║")
    print(f"║  数据集:    {str(dataset_dir)[:45]:45s} ║")
    print(f"║  输出目录:  {str(output_dir)[:45]:45s} ║")
    print(f"║  最大迭代:  {args.max_iterations:<45d} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    # Step 1: 数据转换
    if not args.train_only:
        if args.clean and dataset_dir.exists():
            print(f"🗑️  清除已有数据集: {dataset_dir}")
            shutil.rmtree(dataset_dir)

        if not convert_da3_to_dn_splatter(source_dir, dataset_dir):
            print("❌ 数据转换失败，中止流水线")
            sys.exit(1)

    if args.convert_only:
        elapsed = time.time() - start_time
        print(f"\n⏱️  数据转换耗时: {elapsed:.1f}s")
        print("✅ 数据转换完成 (--convert-only 模式，不训练)")
        return

    # Step 2: 训练
    time.sleep(1)  # 等待文件系统同步
    if not run_dn_splatter_training(dataset_dir, output_dir, experiment_name,
                                     args.max_iterations):
        print("❌ 训练失败，中止流水线")
        sys.exit(1)

    # Step 3: 导出
    if not args.skip_export:
        export_ply(output_dir, experiment_name)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"🎉 Pipeline 完成! 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
