import subprocess
import sys
import shutil
import os
import time
from pathlib import Path
import torch # 引入 torch 用于加载模型
import logging # 引入 logging 用于控制 Nerfstudio 输出

# 设置 Nerfstudio 内部日志级别，避免大量杂项输出干扰
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 配置区域 =================
# Linux 下的临时高速工作区 (训练时的临时文件放这里，速度快 10 倍)
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"

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
    env["QT_QPA_PLATFORM"] = "offscreen" # 防止无头模式崩溃

    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 检测到已存在的 COLMAP 数据: {transforms_file}")
    else:
        # 如果没找到数据，说明是新任务或上次没跑完，重新开始
        print(f"🆕 [新任务] 未找到历史数据，开始初始化工作区...")
        
        # 清理旧的临时文件 (只在需要重新跑 Step 1 时清理)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        
        print(f"📂 [IO 优化] 正在将数据迁移至 Linux 原生目录加速...")
        # 复制视频到 Linux 高速区
        video_dst = work_dir / video_src.name
        shutil.copy(str(video_src), str(video_dst))

        # ================= [Step 1] 数据预处理 (Manual Split) =================
        print(f"\n🎥 [1/3] 视频抽帧与位姿解算 (COLMAP)")
        
        # 1.1 手动调用 FFmpeg (回到低清晰度/低帧率鲁棒性配置)
        print("    -> 1.1 FFmpeg: 抽帧到 1080P 宽分辨率 (1.5 FPS) 写入原生目录")
        
        extracted_images_dir = data_dir / "images"
        extracted_images_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg 命令: 缩放至 1920px 宽 (1080P)，抽取 1.5 帧/秒
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(video_dst), 
            "-vf", "scale=1920:-1,fps=4", # 关键修改：回到 1920 宽 和 1.5 FPS
            "-q:v", "2", 
            str(extracted_images_dir / "frame_%05d.jpg")
        ]
        subprocess.run(ffmpeg_cmd, check=True) 
        
        # 1.2 调用 ns-process-data images (COLMAP 解算)
        print("    -> 1.2 Nerfstudio: 调用 COLMAP 进行位姿解算")
        
        colmap_data_dir = data_dir 
        
        cmd_colmap = [
            "ns-process-data", "images",
            "--data", str(extracted_images_dir),
            "--output-dir", str(colmap_data_dir),
            "--verbose",
        ]
        
        # --- 核心修改：捕获输出，并检查 COLMAP 质量 ---
        process_result = subprocess.run(
            cmd_colmap, 
            check=True, 
            env=env,
            capture_output=True, # 捕获 stdout 和 stderr
            text=True
        )

        # 打印 COLMAP 的完整输出
        print(process_result.stdout)
        print(process_result.stderr)
        
        # 质量检查：如果 COLMAP 仅找到极少数的位姿，则停止
        if "COLMAP only found poses" in process_result.stdout:
            print("\n🚨🚨🚨 检测到 COLMAP 数据质量极差！自动停止训练。")
            print("❌ 错误原因：视频质量太差或场景反光，只有极少数图片找到了位姿。")
            print("➡️ 建议：请重拍视频（降低反光，增加纹理点），然后删除 transforms.json 重新运行。")
            
            # 清理损坏的数据，但保留 workspace 以供调试
            shutil.rmtree(data_dir)
            raise RuntimeError("COLMAP 数据质量不合格，流程停止。")
        # --- 质量检查结束 ---


        # 检查 COLMAP 产物是否存在
        if not transforms_file.exists():
            raise FileNotFoundError("COLMAP 失败，未找到 transforms.json 文件。")


    # ================= [Step 2] 模型训练 =================
    
    # 查找是否有已完成的训练结果 (以避免重复训练)
    search_path = output_dir / project_name / "splatfacto"
    # 获取所有时间戳文件夹
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []

    if run_dirs:
        # 如果找到至少一个运行目录，我们认为训练已完成，跳过
        print(f"\n⏩ [训练跳过] 检测到已完成的训练结果：{run_dirs[-1].name}")
    else:
        # 如果没有找到运行目录，则开始训练
        print(f"\n🧠 [2/3] 开始训练 (RTX 5070 加速中)")
        
        cmd_train = [
            "ns-train", "splatfacto",
            "--data", str(data_dir),
            "--output-dir", str(output_dir),
            "--experiment-name", project_name,
            
            # --- 强制 COLMAP 初始化参数 ---
            "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005",
            
            # --- 训练参数 ---
            "--max-num-iterations", "15000",
            "--vis", "viewer+tensorboard", 
            
            # --- 关键修复：训练完成后自动退出，无需 Ctrl+C ---
            "--viewer.quit-on-train-completion", "True",
            
            # --- Dataparser 子命令 (指定使用 colmap 来解析数据) ---
            "colmap",
        ]
        subprocess.run(cmd_train, check=True, env=env)

    # ================= [Step 3] 导出结果 (使用 CLI，最可靠) =================
    print(f"\n💾 [3/3] 导出结果")
    
    # 确保 run_dirs 包含了最新结果（如果 Step 2 刚跑完）
    if not run_dirs:
        run_dirs = sorted(list(search_path.glob("*")))

    if not run_dirs:
        print("❌ 错误：训练结果目录为空，无法导出。请检查 Step 2 是否成功。")
        return None
        
    latest_run = run_dirs[-1]
    config_path = latest_run / "config.yml"
    
    # 导出命令 (这次用最可靠的 CLI，避免 Python 模块导入错误)
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(work_dir)
    ]
    
    # 只需要运行 CLI 命令，避免 Python 内部复杂调用
    subprocess.run(cmd_export, check=True, env=env)
    
    print("⏳ 等待文件写入磁盘...")
    time.sleep(5) # 强制等待 5 秒，确保大文件写入完成

    print(f"✅ 导出成功！文件应已生成于 {work_dir / 'point_cloud.ply'}")

    # ================= [Step 4] 结果回传 (查找默认文件名并重命名) =================
    print(f"\n📦 [IO 同步] 正在将结果回传至 Windows 项目目录...")
    
    # 查找默认的 PLY 文件名 (Nerfstudio 在某些版本中输出 splat.ply)
    temp_ply_default = work_dir / "point_cloud.ply"
    temp_ply_alt = work_dir / "splat.ply" # 查找另一个可能的默认名 (您的日志显示为 splat.ply)
    
    # 确定哪个文件存在
    if temp_ply_default.exists():
        temp_ply = temp_ply_default
    elif temp_ply_alt.exists():
        temp_ply = temp_ply_alt
    else:
        temp_ply = None

    # 目标路径：脚本所在的目录 (即你的 Windows 项目目录)
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)
    final_ply = target_dir / f"{project_name}.ply"
    
    if temp_ply and temp_ply.exists():
        # 修复权限问题：使用 Bash 字符串执行 cp 命令，确保路径解析正确
        copy_command_str = f"cp {str(temp_ply)} {str(final_ply)}"
        
        # 使用 shell=True 来执行 cp 命令
        subprocess.run(copy_command_str, check=True, shell=True)
        
        print(f"✅ 成功！最终文件已保存至: {final_ply}")
        print(f"📁 您可以在 Windows 资源管理器中打开: {final_ply.resolve()}")
        
        # 清理 Linux 临时文件
        shutil.rmtree(work_dir)
        print(f"🧹 清理完成: 已删除工作区 {work_dir}")
        return str(final_ply)
    else:
        # 如果 CLI 运行成功但文件没找到，可能是命名问题
        print("❌ 导出失败，未找到 PLY 文件 (point_cloud.ply 或 splat.ply)。")
        # 尝试打印目录内容，以供调试
        print(f"🔍 调试信息：{work_dir} 目录内容如下:")
        try:
             print(os.listdir(work_dir))
        except:
             pass
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