
import subprocess
import os
import shutil
import time
from pathlib import Path
import json
import numpy as np
from PIL import Image

# ================= 🔧 配置 =================
# 指定路径（根据 run_glomap.py 中的逻辑）
COLMAP_EXE = shutil.which("colmap") or "/usr/local/bin/colmap"
GLOMAP_EXE = shutil.which("glomap") or "/usr/local/bin/glomap"
NS_TRAIN_EXE = "/home/ltx/miniforge3/envs/nerfstudio/bin/ns-train"
NS_EXPORT_EXE = "/home/ltx/miniforge3/envs/nerfstudio/bin/ns-export"

def run_command(cmd, desc):
    print(f"\n🚀 {desc}...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result

def main():
    # 路径设置
    da3_output_root = Path("/home/ltx/projects/Depth-Anything-3/output/sugar_streaming")
    img_dir = da3_output_root / "extracted"
    work_dir = da3_output_root / "glomap_pipeline"
    data_dir = work_dir / "data"
    colmap_dir = data_dir / "colmap"
    sparse_dir = colmap_dir / "sparse"
    database_path = colmap_dir / "database.db"
    
    # 初始化目录
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 同步图片 (Nerfstudio 需要统一的 images 目录)
    dest_images_dir = data_dir / "images"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying images from {img_dir} to {dest_images_dir}...")
    for img in img_dir.glob("*.png"):
        shutil.copy2(img, dest_images_dir / img.name)

    # 2. COLMAP 特征提取
    run_command([
        COLMAP_EXE, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(dest_images_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1"
    ], "COLMAP 特征提取")

    # 3. COLMAP 顺序匹配
    run_command([
        COLMAP_EXE, "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "20"
    ], "COLMAP 顺序匹配")

    # 4. GLOMAP 全局重建 (Mapper)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run_command([
        GLOMAP_EXE, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(dest_images_dir),
        "--output_path", str(sparse_dir)
    ], "GLOMAP 全局重建")

    # 5. 修正目录结构 (GLOMAP 生成在 sparse/0)
    # 确保 sparse/0 存在且包含 bin 文件
    model_0_dir = sparse_dir / "0"
    if not model_0_dir.exists():
        # 如果模型直接在 sparse/ 根下，移动到 0/
        if (sparse_dir / "cameras.bin").exists():
            model_0_dir.mkdir(parents=True, exist_ok=True)
            for f in ["cameras.bin", "images.bin", "points3D.bin"]:
                shutil.move(str(sparse_dir / f), str(model_0_dir / f))

    # 6. 生成 transforms.json (ns-process-data)
    # 使用 nerfstudio 环境下的 ns-process-data
    ns_process_data = str(Path(NS_TRAIN_EXE).parent / "ns-process-data")
    run_command([
        ns_process_data, "images",
        "--data", str(dest_images_dir),
        "--output-dir", str(data_dir),
        "--skip-colmap",
        "--skip-image-processing",
        "--num-downscales", "0"
    ], "生成 transforms.json")

    # 7. Nerfstudio 训练 3DGS (Splatfacto)
    experiment_name = "da3_glomap_3dgs"
    run_command([
        NS_TRAIN_EXE, "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(work_dir / "outputs"),
        "--experiment-name", experiment_name,
        "--pipeline.model.random-init", "False",
        "--max-num-iterations", "10000",
        "--vis", "viewer+tensorboard",
        "--viewer.quit-on-train-completion", "True",
        "colmap",
        "--downscale-factor", "1"
    ], "Nerfstudio 3DGS 训练")

    # 8. 导出 PLY
    # 寻找训练输出的 config.yml
    run_dirs = sorted(list((work_dir / "outputs" / experiment_name / "splatfacto").glob("*")))
    if run_dirs:
        latest_run_config = run_dirs[-1] / "config.yml"
        run_command([
            NS_EXPORT_EXE, "gaussian-splat",
            "--load-config", str(latest_run_config),
            "--output-dir", str(work_dir)
        ], "导出 3DGS 点云")
        print(f"\n🎉 流程运行完毕! 点云已导出至: {work_dir}/point_cloud.ply")
    else:
        print("\n❌ 训练失败或未找到输出，无法导出点云。")

if __name__ == "__main__":
    main()
