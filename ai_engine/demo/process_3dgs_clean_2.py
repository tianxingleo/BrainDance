import subprocess
import sys
import shutil
import os
import time
from pathlib import Path
import torch # 引入 torch 用于加载模型
import logging # 引入 logging 用于控制 Nerfstudio 输出
import json # 引入 json 用于读写 transforms 文件
import numpy as np # 引入 numpy 进行矩阵运算

# 设置 Nerfstudio 内部日志级别，避免大量杂项输出干扰
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 配置区域 =================
# Linux 下的临时高速工作区
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
# 场景半径系数：控制裁剪的宽松程度
# 1.8 是一个经验值，能很好地包容多物体桌面，同时切除远处的墙
SCENE_RADIUS_SCALE = 1.8 

# ================= 核心算法：智能场景分析与自适应裁剪 =================
def analyze_and_calculate_adaptive_collider(json_path):
    """
    1. 判断场景类型 (物体 vs 房间)。
    2. 如果是物体，基于相机距离动态计算裁剪范围 (Adaptive Pruning)，保护多主体。
    """
    print(f"\n🤖 [AI 分析] 正在解析空间结构与相机轨迹...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames = data["frames"]
        if not frames: return [], "unknown"

        # --- 第一步：提取几何信息 ---
        positions = []
        forward_vectors = []
        distances_to_origin = [] # Nerfstudio 会将主体中心化到原点
        
        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            pos = c2w[:3, 3]
            positions.append(pos)
            
            # 计算前向向量 (-Z)
            rot = c2w[:3, :3]
            forward = rot @ np.array([0, 0, -1]) 
            forward_vectors.append(forward)
            
            # 计算到原点的距离
            dist = np.linalg.norm(pos)
            distances_to_origin.append(dist)
            
        positions = np.array(positions)
        forward_vectors = np.array(forward_vectors)
        distances_to_origin = np.array(distances_to_origin)
        
        # --- 第二步：判断场景类型 (Inward vs Outward) ---
        center_of_mass = np.mean(positions, axis=0)
        vec_to_center = center_of_mass - positions
        norms = np.linalg.norm(vec_to_center, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0 
        vec_to_center_norm = vec_to_center / norms
        dot_products = np.sum(forward_vectors * vec_to_center_norm, axis=1)
        looking_inward_ratio = np.sum(dot_products > 0) / len(frames)
        
        print(f"    -> 相机聚合度: {looking_inward_ratio:.2f}")

        # --- 第三步：决策与计算 ---
        if looking_inward_ratio > 0.6:
            print("💡 判定结果：【物体/多主体模式 (Inward)】")
            
            # === 核心升级：自适应计算 (Adaptive Calculation) ===
            # 不再使用固定的 2.0/6.0，而是根据实际拍摄距离计算
            
            avg_dist = np.mean(distances_to_origin)
            min_dist = np.min(distances_to_origin)
            
            print(f"    -> 统计数据: 相机平均距离 {avg_dist:.2f}, 最近距离 {min_dist:.2f}")
            
            # 动态计算裁剪面
            # 假设主体在原点，半径约为 1.0 (Nerfstudio 归一化特性)
            # 宽松半径 = 1.0 * 系数
            scene_radius = 1.0 * SCENE_RADIUS_SCALE
            
            # 近平面：最近相机距离 - 场景半径 (防止切掉突出的物体)
            calc_near = max(0.05, min_dist - scene_radius)
            
            # 远平面：平均相机距离 + 场景半径 (切掉背景墙)
            calc_far = avg_dist + scene_radius
            
            print(f"    -> 策略：自适应裁剪。Near={calc_near:.2f}, Far={calc_far:.2f}")
            print(f"       (该策略可保护相邻的多物体，同时去除背景)")
            
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            print("💡 判定结果：【全景/室内模式 (Outward)】")
            print("    -> 策略：放宽裁剪，保留完整环境。")
            
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        print(f"⚠️ 分析失败 ({e})，使用默认参数。")
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine] 启动任务: {project_name}")
    
    # 1. 路径解析
    video_src = Path(video_path).resolve()
    
    # 定义临时工作目录
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    
    # ================= [智能检查] 断点续传逻辑 =================
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 

    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 检测到已存在的 COLMAP 数据: {transforms_file}")
    else:
        print(f"🆕 [新任务] 未找到历史数据，开始初始化工作区...")
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        
        print(f"📂 [IO 优化] 迁移数据...")
        video_dst = work_dir / video_src.name
        shutil.copy(str(video_src), str(video_dst))

        # ================= [Step 1] 数据预处理 =================
        print(f"\n🎥 [1/3] 视频抽帧与位姿解算 (COLMAP)")
        
        extracted_images_dir = data_dir / "images"
        extracted_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 1.1 FFmpeg (1920宽 / 4 FPS)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(video_dst), 
            "-vf", "scale=1920:-1,fps=4", 
            "-q:v", "2", 
            str(extracted_images_dir / "frame_%05d.jpg")
        ]
        subprocess.run(ffmpeg_cmd, check=True) 
        
        # 1.2 Nerfstudio (COLMAP)
        # 添加 --center_method focus 有助于将物体置于原点，配合我们的自适应算法
        cmd_colmap = [
            "ns-process-data", "images",
            "--data", str(extracted_images_dir),
            "--output-dir", str(data_dir),
            "--verbose",
        ]
        
        process_result = subprocess.run(
            cmd_colmap, check=True, env=env, capture_output=True, text=True
        )
        print(process_result.stdout)
        
        if "COLMAP only found poses" in process_result.stdout:
            print("\n🚨🚨🚨 COLMAP 失败：特征点不足。")
            shutil.rmtree(data_dir)
            raise RuntimeError("COLMAP 数据质量不合格。")

        if not transforms_file.exists():
            raise FileNotFoundError("未找到 transforms.json 文件。")


    # ================= [Step 2] 模型训练 =================
    
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []

    if run_dirs:
        print(f"\n⏩ [训练跳过] 检测到已完成的训练结果：{run_dirs[-1].name}")
    else:
        # === 核心修改：调用最新的自适应计算算法 ===
        collider_args, scene_type = analyze_and_calculate_adaptive_collider(transforms_file)
        
        print(f"\n🧠 [2/3] 开始训练 (RTX 5070 加速中)")
        
        cmd_train = [
            "ns-train", "splatfacto",
            "--data", str(data_dir),
            "--output-dir", str(output_dir),
            "--experiment-name", project_name,
            
            "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005",

            # 插入自适应参数
            *collider_args,
            
            "--max-num-iterations", "15000",
            "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True",
            "colmap",
        ]
        subprocess.run(cmd_train, check=True, env=env)

    # ================= [Step 3] 导出结果 =================
    print(f"\n💾 [3/3] 导出结果")
    
    if not run_dirs:
        run_dirs = sorted(list(search_path.glob("*")))

    if not run_dirs:
        print("❌ 错误：训练结果目录为空。")
        return None
        
    latest_run = run_dirs[-1]
    config_path = latest_run / "config.yml"
    
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(work_dir)
    ]
    subprocess.run(cmd_export, check=True, env=env)
    
    print("⏳ 等待文件写入磁盘...")
    time.sleep(5) 

    # ================= [Step 4] 结果回传 =================
    print(f"\n📦 [IO 同步] 回传至 Windows...")
    
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)

    temp_ply_default = work_dir / "point_cloud.ply"
    temp_ply_alt = work_dir / "splat.ply"
    
    if temp_ply_default.exists(): temp_ply = temp_ply_default
    elif temp_ply_alt.exists(): temp_ply = temp_ply_alt
    else: temp_ply = None
        
    transforms_src = data_dir / "transforms.json"
    final_webgl_poses = target_dir / "webgl_poses.json"
    final_ply = target_dir / f"{project_name}.ply"
    final_transforms = target_dir / "transforms.json"
    
    # WebGL 姿态转换
    if transforms_src.exists():
        print("🔄 生成 WebGL 友好姿态文件...")
        try:
            with open(transforms_src, 'r') as f:
                data = json.load(f)
            webgl_frames = []
            for frame in data["frames"]:
                c2w_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
                webgl_frames.append({
                    "file_path": frame["file_path"],
                    "pose_matrix_c2w": c2w_matrix.tolist() 
                })
            webgl_data = {
                "camera_model": data["camera_model"],
                "w": data["w"], "h": data["h"],
                "fl_x": data["fl_x"], "fl_y": data["fl_y"],
                "frames": webgl_frames
            }
            with open(final_webgl_poses, 'w') as f:
                json.dump(webgl_data, f, indent=4)
        except Exception as e:
            print(f"❌ 姿态预处理失败: {e}")

    if temp_ply and temp_ply.exists():
        subprocess.run(f"cp {str(temp_ply)} {str(final_ply)}", check=True, shell=True)
        if transforms_src.exists():
            subprocess.run(f"cp {str(transforms_src)} {str(final_transforms)}", check=True, shell=True)
        
        shutil.rmtree(work_dir)
        print(f"✅ 成功！最终模型: {final_ply}")
        return str(final_ply)
    else:
        print("❌ 导出失败，未找到 PLY 文件。")
        return None

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])

    if video_file.exists():
        run_pipeline(video_file, "scene_auto_sync")
    else:
        print(f"❌ 找不到视频: {video_file}")