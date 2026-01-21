import subprocess
import sys
import shutil
import os
import time
import datetime # 引入时间处理库
from pathlib import Path
import torch 
import logging 
import json 
import numpy as np 
import math

# 设置 Nerfstudio 内部日志级别，避免大量杂项输出干扰
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 配置区域 =================
# Linux 下的临时高速工作区 (训练时的临时文件放这里，速度快 10 倍)
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"

# ================= 辅助工具：时间格式化 =================
def format_duration(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))

# ================= 智能场景分析算法 =================
def analyze_scene_type(json_path):
    """
    分析 transforms.json 中的相机姿态，判断是“向内拍摄(物体)”还是“向外拍摄(场景)”。
    返回建议的 ns-train 参数列表。
    """
    print(f"\n🤖 [AI 分析] 正在读取相机轨迹以判断场景类型...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames = data["frames"]
        if not frames:
            return [], "unknown"

        # 1. 提取所有相机位置
        positions = []
        forward_vectors = []
        
        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            # 位置是第4列前3行
            pos = c2w[:3, 3]
            positions.append(pos)
            
            # 计算前向向量 (Nerfstudio/OpenGL 中，Z轴指向相机后方，所以前向是 -Z)
            rot = c2w[:3, :3]
            forward = rot @ np.array([0, 0, -1]) 
            forward_vectors.append(forward)
            
        positions = np.array(positions)
        forward_vectors = np.array(forward_vectors)
        
        # 2. 计算场景几何中心 (所有相机的中心点)
        center_of_mass = np.mean(positions, axis=0)
        
        # 3. 判断每个相机是否看向中心
        vec_to_center = center_of_mass - positions
        norms = np.linalg.norm(vec_to_center, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0 
        vec_to_center_norm = vec_to_center / norms
        
        # 点积：Forward · ToCenter
        dot_products = np.sum(forward_vectors * vec_to_center_norm, axis=1)
        
        # 4. 统计“看向中心”的相机比例
        looking_inward_ratio = np.sum(dot_products > 0) / len(frames)
        
        print(f"    -> 相机聚合度: {looking_inward_ratio:.2f} (1.0代表完全向内，0.0代表完全向外)")

        # 5. 决策逻辑 (阈值 0.6)
        if looking_inward_ratio > 0.6:
            print("💡 判定结果：【物体扫描模式 (Inward)】")
            print("    -> 策略：相机围着物体转。启用紧凑裁剪(2.0~6.0)，聚焦中心物体，去除背景。")
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "2.0", "far_plane", "6.0"], "object"
        else:
            print("💡 判定结果：【全景/室内模式 (Outward)】")
            print("    -> 策略：相机在内部向外看，或直线扫描。放宽裁剪(0.05~100.0)，保留墙壁和远景。")
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        print(f"⚠️ 分析失败 ({e})，将使用默认保守参数。")
        # 默认保守：不乱切，设大范围
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    # --- 全局计时开始 ---
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Engine] 启动任务: {project_name}")
    print(f"🕒 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 路径解析
    video_src = Path(video_path).resolve()
    
    # 定义临时工作目录
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    
    # ================= [智能检查] 断点续传逻辑 =================
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" # 防止无头模式崩溃

    # ================= [Step 1] 数据预处理 (Manual Split) =================
    step1_start = time.time()
    
    if transforms_file.exists():
        print(f"\n⏩ [Step 1] 检测到已存在的 COLMAP 数据: {transforms_file}，跳过预处理。")
    else:
        print(f"\n🆕 [新任务] 未找到历史数据，开始初始化工作区...")
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        
        print(f"📂 [IO 优化] 正在将数据迁移至 Linux 原生目录加速...")
        video_dst = work_dir / video_src.name
        shutil.copy(str(video_src), str(video_dst))

        print(f"\n🎥 [1/3] 视频抽帧与位姿解算 (COLMAP)")

        # 1.1 手动调用 FFmpeg
        print("    -> 1.1 FFmpeg: 抽帧到 1080P 宽分辨率 (4 FPS) 写入原生目录")

        extracted_images_dir = data_dir / "images"
        extracted_images_dir.mkdir(parents=True, exist_ok=True)

        # FFmpeg 命令
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(video_dst), 
            "-vf", "fps=5",  # <--- 保持原始分辨率，4 FPS
            "-q:v", "2", 
            str(extracted_images_dir / "frame_%05d.jpg")
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 

        # --- 图片数量检查 (Limit to 20000 - 实际上不删除，仅作为保险) ---
        all_images = sorted(list(extracted_images_dir.glob("*.jpg")))
        num_images = len(all_images)
        MAX_IMAGES = 20000

        if num_images > MAX_IMAGES:
            print(f"    ⚠️ 图片数量 ({num_images}) 超过上限 {MAX_IMAGES}，正在进行均匀采样...")
            indices_to_keep = set([int(i * (num_images - 1) / (MAX_IMAGES - 1)) for i in range(MAX_IMAGES)])
            deleted_count = 0
            for idx, img_path in enumerate(all_images):
                if idx not in indices_to_keep:
                    os.remove(img_path) 
                    deleted_count += 1
            print(f"    ✅ 已删除 {deleted_count} 张多余图片，剩余 {MAX_IMAGES} 张用于序列匹配。")
        else:
            print(f"    ✅ 图片数量 ({num_images}) 未超标，无需处理。")
        
        # 1.2 调用 ns-process-data images (COLMAP 解算)
        print("    -> 1.2 Nerfstudio: 调用 COLMAP 进行位姿解算 (模式: Sequential, 实时日志)")

        colmap_data_dir = data_dir 

        cmd_colmap = [
            "ns-process-data", "images",
            "--data", str(extracted_images_dir),
            "--output-dir", str(colmap_data_dir),
            "--verbose",
            # "--matching-method", "sequential"  <--- 已确认使用默认/自动模式
        ]
        
        # --- 使用 Popen 实现“实时直播”日志 ---
        full_log_content = [] 
        
        try:
            with subprocess.Popen(
                cmd_colmap, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env,
                bufsize=1 
            ) as process:
                for line in process.stdout:
                    print(line, end='') 
                    full_log_content.append(line) 
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd_colmap)
        except Exception as e:
            print(f"\n❌ COLMAP 运行出错: {e}")
            raise e

        log_str = "".join(full_log_content)
        
        # 质量检查
        if "COLMAP only found poses" in log_str:
            print("\n🚨🚨🚨 检测到 COLMAP 数据质量极差！自动停止训练。")
            print("❌ 错误原因：视频质量太差或场景反光，只有极少数图片找到了位姿。")
            shutil.rmtree(data_dir)
            raise RuntimeError("COLMAP 数据质量不合格，流程停止。")

        if not transforms_file.exists():
            raise FileNotFoundError("COLMAP 失败，未找到 transforms.json 文件。")
            
    step1_duration = time.time() - step1_start
    print(f"⏱️ [Step 1 完成] 耗时: {format_duration(step1_duration)}")

    # ================= [Step 2] 模型训练 =================
    step2_start = time.time()
    
    # 查找是否有已完成的训练结果
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []

    if run_dirs:
        print(f"\n⏩ [Step 2] 检测到已完成的训练结果：{run_dirs[-1].name}，跳过训练。")
    else:
        # === 调用智能场景分析，获取裁剪参数 ===
        collider_args, scene_type = analyze_scene_type(transforms_file)
        
        print(f"\n🧠 [2/3] 开始训练 (RTX 5070 加速中)")
        
        cmd_train = [
            "ns-train", "splatfacto",
            "--data", str(data_dir),
            "--output-dir", str(output_dir),
            "--experiment-name", project_name,
            
            # --- 强制 COLMAP 初始化参数 ---
            "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005",

            # === 插入：智能分析得出的裁剪参数 ===
            *collider_args,
            
            # --- 训练参数 ---
            "--max-num-iterations", "15000",
            "--vis", "viewer+tensorboard", 
            
            # --- 关键修复：训练完成后自动退出，无需 Ctrl+C ---
            "--viewer.quit-on-train-completion", "True",
            
            # --- Dataparser 子命令 ---
            "colmap",
        ]
        subprocess.run(cmd_train, check=True, env=env)

    step2_duration = time.time() - step2_start
    print(f"⏱️ [Step 2 完成] 耗时: {format_duration(step2_duration)}")

    # ================= [Step 3] 导出结果 =================
    step3_start = time.time()
    print(f"\n💾 [3/3] 导出结果")
    
    if not run_dirs:
        run_dirs = sorted(list(search_path.glob("*")))

    if not run_dirs:
        print("❌ 错误：训练结果目录为空，无法导出。请检查 Step 2 是否成功。")
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

    print(f"✅ 导出成功！文件应已生成于 {work_dir / 'point_cloud.ply'}")
    step3_duration = time.time() - step3_start
    print(f"⏱️ [Step 3 完成] 耗时: {format_duration(step3_duration)}")

    # ================= [Step 4] 结果回传 =================
    print(f"\n📦 [IO 同步] 正在将结果回传至 Windows 项目目录...")
    
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)

    temp_ply_default = work_dir / "point_cloud.ply"
    temp_ply_alt = work_dir / "splat.ply"
    
    if temp_ply_default.exists():
        temp_ply = temp_ply_default
    elif temp_ply_alt.exists():
        temp_ply = temp_ply_alt
    else:
        temp_ply = None
        
    transforms_src = data_dir / "transforms.json"
    final_webgl_poses = target_dir / "webgl_poses.json"
    final_ply = target_dir / f"{project_name}.ply"
    final_transforms = target_dir / "transforms.json"
    
    # --- 姿态预处理逻辑 ---
    if transforms_src.exists():
        print("🔄 正在生成 WebGL 友好姿态文件 (webgl_poses.json)...")
        try:
            with open(transforms_src, 'r') as f:
                data = json.load(f)
            
            webgl_frames = []
            for frame in data["frames"]:
                c2w_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
                # 计算 W2C (虽然这里只存了 C2W，但可以预留逻辑)
                webgl_frames.append({
                    "file_path": frame["file_path"],
                    "pose_matrix_c2w": c2w_matrix.tolist() 
                })
                
            webgl_data = {
                "camera_model": data["camera_model"],
                "w": data["w"],
                "h": data["h"],
                "fl_x": data["fl_x"],
                "fl_y": data["fl_y"],
                "frames": webgl_frames
            }
            
            with open(final_webgl_poses, 'w') as f:
                json.dump(webgl_data, f, indent=4)
            print(f"✅ WebGL 姿态文件已保存至: {final_webgl_poses.resolve()}")
        except Exception as e:
            print(f"❌ 姿态预处理失败: {e}")

    if temp_ply and temp_ply.exists():
        # 复制 PLY 文件
        copy_ply_command_str = f"cp {str(temp_ply)} {str(final_ply)}"
        subprocess.run(copy_ply_command_str, check=True, shell=True)
        
        # 复制 transforms.json 文件
        if transforms_src.exists():
            copy_transforms_cmd_str = f"cp {str(transforms_src)} {str(final_transforms)}"
            subprocess.run(copy_transforms_cmd_str, check=True, shell=True)
        
        # 清理 Linux 临时文件
        shutil.rmtree(work_dir)
        print(f"🧹 清理完成: 已删除工作区 {work_dir}")
        
        # --- 最终时间汇总 ---
        total_time = time.time() - global_start_time
        print(f"\n✅ =============================================")
        print(f"🎉 任务全部完成！安心睡觉吧。")
        print(f"📂 最终模型: {final_ply}")
        print(f"⏱️ 总共耗时: {format_duration(total_time)}")
        print(f"✅ =============================================")
        
        return str(final_ply)
    else:
        print("❌ 导出失败，未找到 PLY 文件 (point_cloud.ply 或 splat.ply)。")
        return None

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])

    if video_file.exists():
        # 请注意：如果您之前已经跑过 Step 1 (COLMAP)，请手动删除 /home/ltx/braindance_workspace/scene_auto_sync/data/transforms.json 文件，以强制重新运行 COLMAP，确保新的参数生效！
        run_pipeline(video_file, "scene_auto_sync")
    else:
        print(f"❌ 找不到视频: {video_file}")