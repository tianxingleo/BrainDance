"""
DA3 → 3DGS 训练 + COLMAP 自动扶正 Pipeline (方案A)

基于 run_da3_to_3dgs_direct.py 的模式，仅使用 COLMAP model_aligner 进行平面对齐。
无需 Open3D 依赖。

用法:
    python run_da3_to_3dgs_colmap_aligned.py

    # 自定义参数
    python run_da3_to_3dgs_colmap_aligned.py --iterations 30000 --colmap_error 0.05
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


def convert_da3_to_colmap(source_dir: Path, target_sparse_dir: Path):
    """Call external script to convert parameters"""
    print("📦 [Format Conversion] Calling convert_da3_to_colmap.py...")
    
    # Calculate COLMAP root directory (the parent of sparse/0)
    # target_sparse_dir is .../sparse/0
    # external script expects --output_dir which it will populate with sparse/0 inside
    # So we need to pass the parent of 'sparse'. 
    # structure: output_dir/sparse/0
    colmap_output_root = target_sparse_dir.parent.parent
    
    cmd = [
        "python3",
        "convert_da3_to_colmap.py",
        "--base_dir", str(source_dir),
        "--output_dir", str(colmap_output_root)
    ]
    
    print(f"  Execute: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Verify
    if not (target_sparse_dir / "cameras.txt").exists():
        raise FileNotFoundError("Conversion script finished but cameras.txt is missing!")
    print("✅ Conversion completed via external script.")


# ================= 🅰️ 修复与对齐 =================

def colmap_align_fix(sparse_dir: Path, ply_path: Path = None) -> bool:
    """使用自定义脚本修复 C2W->W2C 错位并进行平面对齐"""
    print("\n🔧 [修复与对齐] 正在执行 fix_colmap_orientation.py...")
    
    cmd = [
        "python3",
        "fix_colmap_orientation.py",
        "--sparse_dir", str(sparse_dir),
        # 移除 --invert，因为之前的 convert 已处理过位姿
    ]
    if ply_path and ply_path.exists():
        cmd.extend(["--ply_path", str(ply_path)])

    print(f"  执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"  ⚠️ 修复脚本执行失败: {e}")
        return False


# ================= 🚀 主流程 =================

def run_pipeline(args):
    source_dir = Path(args.da3_output)
    ws_root = source_dir / "da3_3dgs_colmap_aligned_pipeline"

    print("=" * 60)
    print("  DA3 → 3DGS + COLMAP 自动扶正 Pipeline (方案A)")
    print("=" * 60)
    print(f"  DA3 输出:     {source_dir}")
    print(f"  工作目录:     {ws_root}")
    print(f"  迭代次数:     {args.iterations}")
    print(f"  COLMAP 误差:  {args.colmap_error}")
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
    convert_da3_to_colmap(source_dir, sparse_0)

    # ====== Step 3: 坐标系修复与平面对齐 ======
    pcd_path = source_dir / "pcd" / "combined_pcd.ply"
    fix_ok = colmap_align_fix(sparse_0, pcd_path)
    
    if fix_ok:
        print("  📐 坐标系修复与平面对齐已应用")
    else:
        print("  ⚠️ 修复失败，将尝试直接训练 (可能会错位)")

    # ... (前面的代码保持不变)

    # ====== Step 4: 训练 3DGS ======
    print(f"\n🔥 [Step 4] 开始 3DGS 训练 ({args.iterations} 迭代)...")

    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    train_cmd = [
        PYTHON_EXE, NS_TRAIN, "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(ws_root / "outputs"),
        "--experiment-name", "da3_colmap_aligned",
        "--pipeline.model.random-init", "False",
        "--max-num-iterations", str(args.iterations),
        "--viewer.quit-on-train-completion", "True",
        "colmap",
        "--orientation-method", "none", # 关键：使用我们处理好的坐标系
        "--center-method", "poses",      # 基于相机轨迹居中
        "--auto-scale-poses", "True"     # 允许缩放以适应 GPU 负载
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
    config_paths = list((ws_root / "outputs/da3_colmap_aligned").rglob("config.yml"))
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

    # ====== 汇总 ======
    total_time = time.time() - total_t0
    print()
    print("=" * 60)
    print(f"  ✨ Pipeline 完成! (总耗时: {total_time:.0f}s ≈ {total_time/60:.1f}min)")
    print("=" * 60)
    print()
    print(f"📊 修复状态: {'✅ 成功 (相机与点云已合体并扶正)' if fix_ok else '❌ 失败'}")
    print()

    if ply_path and ply_path.exists():
        print("📁 输出文件:")
        print(f"  PLY: {ply_path}")
        print()
        print("👀 查看方法:")
        print("  SuperSplat: https://playcanvas.com/supersplat/editor (拖拽PLY)")
        print()
        if not colmap_ok:
            print("💡 提示: COLMAP 对齐失败，可尝试 Open3D 后处理扶正:")
            print(f"  python batch_align_existing_ply.py --input_file {ply_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DA3 → 3DGS + COLMAP 自动扶正 Pipeline (方案A，无需Open3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认参数
  python run_da3_to_3dgs_colmap_aligned.py

  # 自定义迭代和误差
  python run_da3_to_3dgs_colmap_aligned.py --iterations 30000 --colmap_error 0.05

  # 指定 DA3 输出目录
  python run_da3_to_3dgs_colmap_aligned.py --da3_output /path/to/da3_output
"""
    )
    parser.add_argument("--da3_output", type=str,
                        default=str(DA3_OUTPUT),
                        help="DA3 输出目录 (默认: output/sugar_streaming)")
    parser.add_argument("--iterations", type=int, default=15000,
                        help="训练迭代次数 (默认: 15000)")
    parser.add_argument("--colmap_error", type=float, default=0.02,
                        help="COLMAP 对齐最大误差 (默认: 0.02)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
