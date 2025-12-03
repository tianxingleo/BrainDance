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

# ================= 🔧 用户配置 (关键修改) =================
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
SCENE_RADIUS_SCALE = 1.8 

# 🔥 强制开启球体切割
FORCE_SPHERICAL_CULLING = True

# 🔥 切割力度 (1.0 = 标准, 1.2 = 宽松, 0.9 = 激进)
# 建议设为 1.0 或 0.9，这样会切掉更多背景。
# 如果设为 1.1，表示保留半径是相机圈的 1.1 倍。
CULLING_MULTIPLIER = 0.3 

# 检查依赖
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("❌ 严重警告: 未安装 plyfile 库！无法执行切割。请运行: pip install plyfile")

# ================= 核心算法 1: 训练参数计算 =================
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

        is_object_mode = ratio > 0.6 or FORCE_SPHERICAL_CULLING

        if is_object_mode:
            scene_type = "object"
            avg_dist = np.mean(dists_to_origin)
            min_dist = np.min(dists_to_origin)
            
            scene_radius = 1.0 * SCENE_RADIUS_SCALE
            calc_near = max(0.05, min_dist - scene_radius)
            calc_far = avg_dist + scene_radius
            
            print(f"    -> 模式: 物体 (Near={calc_near:.2f}, Far={calc_far:.2f})")
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            print(f"    -> 模式: 场景 (保留全景)")
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        print(f"⚠️ 分析失败 ({e})，使用默认参数。")
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"

# ================= 核心算法 2: 导出后球体切割 (升级版) =================
def perform_spherical_culling(ply_path, json_path, output_path):
    if not HAS_PLYFILE: 
        print("❌ 缺少 plyfile 库，跳过切割。")
        return False
        
    print(f"\n✂️ [后处理] 正在执行【抗干扰】球体切割...")

    try:
        # 1. 计算保留半径 (使用分位数抗干扰)
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        
        center = np.mean(cam_pos, axis=0)
        dists = np.linalg.norm(cam_pos - center, axis=1)
        
        # === 关键修改：不再使用 max，而是使用 95% 分位数 ===
        # 这意味着最远的 5% 的相机（可能是飘出去的误差）会被忽略，不会撑大球体
        robust_max_radius = np.percentile(dists, 85)
        
        # 计算最终保留半径
        keep_radius = robust_max_radius * CULLING_MULTIPLIER
        
        print(f"    -> 几何中心: {center}")
        print(f"    -> 抗干扰半径: {robust_max_radius:.2f} (排除离群相机)")
        print(f"    -> 最终切割半径: {keep_radius:.2f} (系数 {CULLING_MULTIPLIER})")

        # 2. 读取点云
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        
        x, y, z = vertex['x'], vertex['y'], vertex['z']
        points = np.stack([x, y, z], axis=1)
        original_count = len(points)
        
        # 3. 执行切割
        dists_pts = np.linalg.norm(points - center, axis=1)
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 逻辑：距离 < 圈子 AND 点比较实
        mask = (dists_pts < keep_radius) & (opacities > 0.05)
        
        filtered_vertex = vertex[mask]
        new_count = len(filtered_vertex)
        
        print(f"    -> 原始点数: {original_count}")
        print(f"    -> 剩余点数: {new_count} (删除了 {original_count - new_count} 个噪点)")
        
        # 4. 保存
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True

    except Exception as e:
        print(f"❌ 切割失败详情: {e}")
        return False

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine V12] 启动任务: {project_name}")
    print(f"🔥 切割力度: {CULLING_MULTIPLIER} (越小切越狠)")
    
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

    # [Step 3.5] 强制物理切割
    raw_ply = work_dir / "point_cloud.ply"
    if not raw_ply.exists(): raw_ply = work_dir / "splat.ply"

    cleaned_ply = work_dir / "point_cloud_cleaned.ply"
    final_ply_to_use = raw_ply

    should_clean = (scene_type_detected == "object") or FORCE_SPHERICAL_CULLING
    
    if should_clean:
        if raw_ply.exists():
            if perform_spherical_culling(raw_ply, transforms_file, cleaned_ply):
                print("✨ 清洗成功！")
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