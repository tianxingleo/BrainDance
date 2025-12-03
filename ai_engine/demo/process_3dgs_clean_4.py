import subprocess
import sys
import shutil
import os
import time
from pathlib import Path
import json
import numpy as np
import logging

# 设置日志级别
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 🔧 用户配置 (暴力裁剪版) =================
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
SCENE_RADIUS_SCALE = 1.8 

# 🔥 强制开启球体切割
FORCE_SPHERICAL_CULLING = True

# 🔥 核心参数：保留百分比 (0.0 ~ 1.0)
# 0.5 表示只保留离中心最近的 50% 的点 (非常狠)
# 0.65 表示保留 65% (推荐，比较平衡)
# 0.9 表示保留 90% (只去极远处的背景)
KEEP_PERCENTILE = 0.6

# 检查依赖
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("❌ 严重警告: 未安装 plyfile 库！无法执行切割。请运行: pip install plyfile")

# ================= 核心算法 1: 训练参数计算 (保持不变) =================
def analyze_and_calculate_adaptive_collider(json_path):
    print(f"\n🤖 [AI 分析] 解析相机轨迹...")
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"]
        if not frames: return [], "unknown"

        positions = []
        forward_vectors = []
        dists_to_origin = [] 
        
        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            positions.append(c2w[:3, 3])
            forward_vectors.append(c2w[:3, :3] @ np.array([0, 0, -1]))
            dists_to_origin.append(np.linalg.norm(c2w[:3, 3]))
            
        positions = np.array(positions)
        forward_vectors = np.array(forward_vectors)
        
        center = np.mean(positions, axis=0)
        vec_to_center = center - positions
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        print(f"    -> 相机聚合度: {ratio:.2f}")

        # 如果强制开启切割，我们在训练时也倾向于使用物体参数
        is_object_mode = ratio > 0.6 or FORCE_SPHERICAL_CULLING

        if is_object_mode:
            avg_dist = np.mean(dists_to_origin)
            min_dist = np.min(dists_to_origin)
            scene_radius = 1.0 * SCENE_RADIUS_SCALE
            calc_near = max(0.05, min_dist - scene_radius)
            calc_far = avg_dist + scene_radius
            
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"

# ================= 核心算法 2: 基于分位数的暴力切割 (New!) =================
def perform_percentile_culling(ply_path, json_path, output_path):
    if not HAS_PLYFILE: 
        print("❌ 缺少 plyfile 库，跳过切割。")
        return False
        
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    print(f"🔥 目标: 只保留离圆心最近的 {KEEP_PERCENTILE*100:.0f}% 点云")

    try:
        # 1. 计算切割中心 (依然使用相机重心，因为它是轨道的圆心，比点云重心更稳)
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        print(f"    -> 切割圆心 (相机重心): {center}")

        # 2. 读取点云
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        
        x, y, z = vertex['x'], vertex['y'], vertex['z']
        points = np.stack([x, y, z], axis=1)
        original_count = len(points)
        
        # 3. 计算所有点到中心的距离
        print("    -> 正在计算所有点的距离分布...")
        dists_pts = np.linalg.norm(points - center, axis=1)
        
        # 4. === 核心逻辑：计算分位数阈值 ===
        # 找到一个距离 D，使得有 KEEP_PERCENTILE 的点距离 < D
        threshold_radius = np.percentile(dists_pts, KEEP_PERCENTILE * 100)
        
        print(f"    -> 统计结果: {KEEP_PERCENTILE*100:.0f}% 的点集中在半径 {threshold_radius:.4f} 以内")
        print(f"    -> 执行切割: 所有大于 {threshold_radius:.4f} 的点将被删除")
        
        # 5. 执行切割
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 逻辑：距离 < 阈值 AND 点比较实
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        
        filtered_vertex = vertex[mask]
        new_count = len(filtered_vertex)
        
        print(f"    -> 原始点数: {original_count}")
        print(f"    -> 剩余点数: {new_count} (删除了 {original_count - new_count} 个背景点)")
        
        # 6. 保存
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True

    except Exception as e:
        print(f"❌ 切割失败详情: {e}")
        return False

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine V13] 启动任务: {project_name}")
    print(f"🔪 切割策略: 保留 {KEEP_PERCENTILE*100}% 最近点云")
    
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 

    # [Step 1] 数据处理
    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 检测到 COLMAP 数据")
    else:
        print(f"🆕 [新任务] 初始化...")
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        shutil.copy(str(video_src), str(work_dir / video_src.name))

        print(f"\n🎥 [1/3] COLMAP 解算")
        extracted_images_dir = data_dir / "images"
        extracted_images_dir.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(["ffmpeg", "-y", "-i", str(work_dir / video_src.name), 
                        "-vf", "scale=1920:-1,fps=4", "-q:v", "2", 
                        str(extracted_images_dir / "frame_%05d.jpg")], check=True) 
        
        subprocess.run(
            ["ns-process-data", "images", "--data", str(extracted_images_dir), "--output-dir", str(data_dir), "--verbose"],
            check=True, env=env
        )

    # [Step 2] 训练
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []
    
    scene_type_detected = "unknown"

    if run_dirs:
        print(f"\n⏩ [训练跳过] 已完成")
        _, scene_type_detected = analyze_and_calculate_adaptive_collider(transforms_file)
    else:
        collider_args, scene_type_detected = analyze_and_calculate_adaptive_collider(transforms_file)
        print(f"\n🧠 [2/3] 开始训练...")
        subprocess.run([
            "ns-train", "splatfacto", "--data", str(data_dir), "--output-dir", str(output_dir), 
            "--experiment-name", project_name, "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005", *collider_args,
            "--max-num-iterations", "15000", "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True", "colmap"
        ], check=True, env=env)

    # [Step 3] 导出
    print(f"\n💾 [3/3] 导出结果")
    if not run_dirs: run_dirs = sorted(list(search_path.glob("*")))
    if not run_dirs: return None
    latest_run = run_dirs[-1]
    
    subprocess.run([
        "ns-export", "gaussian-splat", "--load-config", str(latest_run/"config.yml"), 
        "--output-dir", str(work_dir)
    ], check=True, env=env)
    time.sleep(5) 

    # [Step 3.5] 分位数暴力切割
    raw_ply = work_dir / "point_cloud.ply"
    if not raw_ply.exists(): raw_ply = work_dir / "splat.ply"

    cleaned_ply = work_dir / "point_cloud_cleaned.ply"
    final_ply_to_use = raw_ply

    should_clean = (scene_type_detected == "object") or FORCE_SPHERICAL_CULLING
    
    if should_clean:
        if raw_ply.exists():
            # 使用新的分位数切割函数
            if perform_percentile_culling(raw_ply, transforms_file, cleaned_ply):
                print("✨ 暴力切割成功！")
                final_ply_to_use = cleaned_ply
        else:
            print(f"❌ 警告：未找到 PLY 文件")
    else:
        print(f"ℹ️ 跳过切割")

    # [Step 4] 回传
    print(f"\n📦 [IO 同步] 回传至 Windows...")
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True, parents=True) 
    
    transforms_src = data_dir / "transforms.json"
    final_webgl_poses = target_dir / "webgl_poses.json"
    final_ply_dst = target_dir / f"{project_name}.ply"
    
    if transforms_src.exists():
        try:
            with open(transforms_src, 'r') as f: d = json.load(f)
            frames = [{"file_path": fr["file_path"], "pose_matrix_c2w": np.array(fr["transform_matrix"], dtype=np.float32).tolist()} for fr in d["frames"]]
            with open(final_webgl_poses, 'w') as f: json.dump({"camera_model": d.get("camera_model","OPENCV"), "frames": frames}, f, indent=4)
        except: pass

    if final_ply_to_use and final_ply_to_use.exists():
        try:
            shutil.copy2(str(final_ply_to_use), str(final_ply_dst))
            if transforms_src.exists(): shutil.copy2(str(transforms_src), str(target_dir/"transforms.json"))
            shutil.rmtree(work_dir)
            print(f"✅ 全部完成！: {final_ply_dst}")
            return str(final_ply_dst)
        except Exception as e:
            print(f"❌ 回传失败: {e}")
            return None
    else:
        print("❌ 致命错误：回传失败")
        return None

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])

    if video_file.exists():
        run_pipeline(video_file, "scene_auto_sync")
    else:
        print(f"❌ 找不到视频: {video_file}")