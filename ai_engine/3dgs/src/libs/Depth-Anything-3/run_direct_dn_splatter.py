"""
DN-Splatter 直接运行脚本
========================
直接使用已转换好的 dn_splatter_dataset 进行训练
这是一个简化入口, 完整 pipeline 请使用 run_da3_to_dn_splatter_pipeline.py

用法: python run_direct_dn_splatter.py
"""
import subprocess
import os
import sys
from pathlib import Path

# ================= Configuration =================
CONDA_PREFIX = "/home/ltx/my_envs/gs_linux_backup"
NS_ENV_BIN = f"{CONDA_PREFIX}/bin"
NS_PYTHON_EXE = f"{NS_ENV_BIN}/python"
NS_TRAIN = f"{NS_ENV_BIN}/ns-train"

PROJECT_ROOT = Path("/home/ltx/projects/Depth-Anything-3")
DATASET_DIR = PROJECT_ROOT / "dn_splatter_dataset"
OUTPUT_DIR = PROJECT_ROOT / "dn_splatter_output"


def main():
    print("========================================")
    print("   DN-Splatter Direct Training          ")
    print("========================================")

    if not DATASET_DIR.exists():
        print(f"❌ 数据集目录不存在: {DATASET_DIR}")
        print("请先运行数据转换: python run_da3_to_dn_splatter_pipeline.py --convert-only")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    cmd = [
        NS_PYTHON_EXE, NS_TRAIN, "dn-splatter",
        "--output-dir", str(OUTPUT_DIR),
        "--experiment-name", "da3_dn_splatter",
        "--max-num-iterations", "30000",
        "--pipeline.model.use-depth-loss", "True",
        "--pipeline.model.depth-lambda", "0.2",
        "--pipeline.model.use-normal-loss", "True",
        "--pipeline.model.normal-lambda", "0.05",
        "--pipeline.model.predict-normals", "True",
        "--viewer.websocket-port", "7007",
        "--vis", "viewer+tensorboard",
        "normal-nerfstudio",
        "--data", str(DATASET_DIR),
        "--load-3D-points", "False",
        "--load-pcd-normals", "False",
    ]

    print(f"\n🚀 启动训练...\n命令: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(PROJECT_ROOT))
        print("✅ 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
