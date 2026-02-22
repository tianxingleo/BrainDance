
import subprocess
import sys
import shutil
import os
import time
import datetime
from pathlib import Path
import json
import numpy as np
import logging
import cv2
import re

# ================= 🔧 环境与路径配置 (自适应) =================
# 自动定位工具路径
COLMAP_EXE = shutil.which("colmap") or "/usr/local/bin/colmap"
# 如果没有 glomap，我们退回到 colmap mapper
GLOMAP_EXE = shutil.which("glomap") or "/usr/local/bin/glomap"
NS_ENV_BIN = "/home/ltx/miniforge3/envs/nerfstudio/bin" # nerfstudio 环境路径
NS_TRAIN_EXE = f"{NS_ENV_BIN}/ns-train"
NS_EXPORT_EXE = f"{NS_ENV_BIN}/ns-export"
NS_PROCESS_EXE = f"{NS_ENV_BIN}/ns-process-data"

LINUX_WORK_ROOT = Path("/home/ltx/projects/Depth-Anything-3/output/sugar_streaming/glomap_ws")
MAX_IMAGES = 180 # 限制图片数量以保证速度和一致性
SCENE_RADIUS_SCALE = 1.8 
KEEP_PERCENTILE = 0.9 # 基于分位数的暴力切割保留比例
FORCE_SPHERICAL_CULLING = True

# 检查依赖
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("⚠️ 未安装 plyfile 库，跳过点云裁剪功能。")

# ================= 辅助工具：核心算法 (由 BrainDance 仓库代码移植) =================

def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def smart_filter_blurry_images(image_folder, keep_ratio=0.85, max_images=MAX_IMAGES):
    print(f"\n🧠 [BrainDance] 正在分析图片质量 (混合策略版)...")
    image_dir = Path(image_folder)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])                                                              
    if not images: return
    
    trash_dir = image_dir.parent / "trash_smart"
    trash_dir.mkdir(exist_ok=True)
    img_scores = []

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        grid_h, grid_w = h // 3, w // 3
        max_grid_score = 0
        for r in range(3):
            for c in range(3):
                roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                score = cv2.Laplacian(roi, cv2.CV_64F).var()
                if score > max_grid_score: max_grid_score = score
        img_scores.append((img_path, max_grid_score))

    scores = [s[1] for s in img_scores]
    quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
    
    good_images = []
    for img_path, score in img_scores:
        if score < quality_threshold:
            shutil.move(str(img_path), str(trash_dir / img_path.name))
        else:
            good_images.append(img_path)

    if len(good_images) > max_images:
        indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_images, dtype=int))
        for idx, img_path in enumerate(good_images):
            if idx not in indices_to_keep:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
    print(f"✅ 清洗结束，最终保留 {len(list(image_dir.glob('*')))} 张。")

def analyze_and_calculate_adaptive_collider(json_path):
    print(f"\n🤖 [AI 分析] 解析相机轨迹以计算 Collider...")
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"]
        positions = [np.array(frame["transform_matrix"])[:3, 3] for frame in frames]
        forward_vectors = [np.array(frame["transform_matrix"])[:3, :3] @ np.array([0, 0, -1]) for frame in frames]
        dists_to_origin = [np.linalg.norm(p) for p in positions]
        
        center = np.mean(positions, axis=0)
        vec_to_center = center - positions
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        is_object_mode = ratio > 0.6 or FORCE_SPHERICAL_CULLING
        if is_object_mode:
            avg_dist = np.mean(dists_to_origin)
            min_dist = np.min(dists_to_origin)
            scene_radius = 1.0 * SCENE_RADIUS_SCALE
            calc_near, calc_far = max(0.05, min_dist - scene_radius), avg_dist + scene_radius
            return ["--pipeline.model.enable-collider", "True", "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), "far_plane", str(round(calc_far, 2))], "object"
        return ["--pipeline.model.enable-collider", "True", "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"
    except Exception:
        return ["--pipeline.model.enable-collider", "True"], "unknown"

def perform_percentile_culling(ply_path, json_path, output_path):
    if not HAS_PLYFILE: return False
    print(f"\n✂️ [后处理] 正在对 {KEEP_PERCENTILE*100:.0f}% 分位点进行暴力切割...")
    try:
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        center = np.mean([np.array(f["transform_matrix"])[:3, 3] for f in frames], axis=0)
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        dists_pts = np.linalg.norm(points - center, axis=1)
        threshold_radius = np.percentile(dists_pts, KEEP_PERCENTILE * 100)
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        filtered_vertex = vertex[mask]
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        print(f"✅ 点云已由 {len(points)} 个优化至 {len(filtered_vertex)} 个。")
        return True
    except Exception as e:
        print(f"❌ 切割失败: {e}")
        return False

# ================= 🚀 主流水线 (改编自 BrainDance GLOMAP 脚本) =================

def run_pipeline(img_source_dir, project_name):
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Pipeline] 启动任务: {project_name}")
    
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    
    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 数据准备 (抽帧图片同步)
    temp_extract_dir = work_dir / "temp_extract"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    images_in_source = sorted(list(Path(img_source_dir).glob("*.png")) + list(Path(img_source_dir).glob("*.jpg")))
    for img in images_in_source: shutil.copy2(img, temp_extract_dir / img.name)
    
    smart_filter_blurry_images(temp_extract_dir) # 这里会同时处理质量和数量上限
    
    target_images_dir = data_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    filtered_images = sorted(list(temp_extract_dir.glob("*.[jp][pn]g")))
    for img in filtered_images: shutil.copy2(img, target_images_dir / img.name)
    shutil.rmtree(temp_extract_dir)

    # 2. COLMAP & GLOMAP Reconstruction
    colmap_sparse_root = data_dir / "colmap"
    colmap_sparse_root.mkdir(parents=True, exist_ok=True)
    db_path = colmap_sparse_root / "database.db"
    
    print(f"\n⚡ [1/4] 执行特征提取与匹配...")
    subprocess.run([COLMAP_EXE, "feature_extractor", "--database_path", str(db_path), "--image_path", str(target_images_dir), "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1"], check=True)
    subprocess.run([COLMAP_EXE, "sequential_matcher", "--database_path", str(db_path), "--SequentialMatching.overlap", "20"], check=True)

    print(f"\n⚡ [2/4] 执行稀疏重建 (GLOMAP/Mapper)...")
    sparse_out = colmap_sparse_root / "sparse"
    sparse_out.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(GLOMAP_EXE):
        print("🚀 使用 GLOMAP 核心引擎进行加速重建...")
        subprocess.run([GLOMAP_EXE, "mapper", "--database_path", str(db_path), "--image_path", str(target_images_dir), "--output_path", str(sparse_out)], check=True)
    else:
        print("⚠️ 未发现 GLOMAP，退回到 COLMAP Mapper (由于图片较多可能稍慢)...")
        subprocess.run([COLMAP_EXE, "mapper", "--database_path", str(db_path), "--image_path", str(target_images_dir), "--output_path", str(sparse_out)], check=True)

    # 结构修正
    target_0 = sparse_out / "0"
    if not target_0.exists():
        if (sparse_out / "cameras.bin").exists():
            target_0.mkdir(parents=True, exist_ok=True)
            for f in ["cameras.bin", "images.bin", "points3D.bin"]: shutil.move(str(sparse_out / f), str(target_0 / f))

    # 3. Nerfstudio 处理与训练
    print(f"\n⚡ [3/4] 执行 Splatfacto 训练 (3DGS)...")
    subprocess.run([NS_PROCESS_EXE, "images", "--data", str(target_images_dir), "--output-dir", str(data_dir), "--skip-colmap", "--skip-image-processing", "--num-downscales", "0"], check=True)
    
    transforms_json = data_dir / "transforms.json"
    collider_args, scene_mode = analyze_and_calculate_adaptive_collider(transforms_json)
    
    subprocess.run([NS_TRAIN_EXE, "splatfacto", "--data", str(data_dir), "--output-dir", str(output_dir), "--experiment-name", project_name, "--pipeline.model.random-init", "False", *collider_args, "--max-num-iterations", "10000", "--vis", "viewer+tensorboard", "--viewer.quit-on-train-completion", "True", "colmap", "--downscale-factor", "1"], check=True)

    # 4. 导出与后处理
    print(f"\n⚡ [4/4] 导出 3DGS 模型并修剪背景...")
    exp_path = output_dir / project_name / "splatfacto"
    run_dir = sorted(list(exp_path.glob("*")))[-1]
    
    subprocess.run([NS_EXPORT_EXE, "gaussian-splat", "--load-config", str(run_dir / "config.yml"), "--output-dir", str(work_dir)], check=True)
    
    raw_ply = work_dir / "point_cloud.ply"
    if not raw_ply.exists(): raw_ply = work_dir / "splat.ply"
    
    final_ply = work_dir / f"{project_name}_final.ply"
    if raw_ply.exists():
        perform_percentile_culling(raw_ply, transforms_json, final_ply)
    
    total_time = time.time() - global_start_time
    print(f"\n✨ 任务圆满完成! 总耗时: {format_duration(total_time)}")
    print(f"📂 最终 3DGS 可视化模型: {final_ply.resolve()}")

if __name__ == "__main__":
    # 使用 DA3-Streaming 抽取的图片目录
    source_imgs = "/home/ltx/projects/Depth-Anything-3/output/sugar_streaming/extracted"
    run_pipeline(source_imgs, "da3_braindance_3dgs")
