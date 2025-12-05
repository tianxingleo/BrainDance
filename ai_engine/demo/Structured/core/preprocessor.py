import shutil
from pathlib import Path
from utils.common import run_command
from config.settings import FFMPEG_WIDTH, FFMPEG_FPS

def process_video(video_src: Path, work_dir: Path, data_dir: Path):
    print(f"\n🎥 [1/3] 视频抽帧与位姿解算 (COLMAP)")
    
    # 1. 迁移视频到工作区
    video_dst = work_dir / video_src.name
    shutil.copy(str(video_src), str(video_dst))
    
    # 2. FFmpeg 抽帧
    extracted_images_dir = data_dir / "images"
    extracted_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"    -> 1.1 FFmpeg: 抽帧到 {FFMPEG_WIDTH}P 宽分辨率 ({FFMPEG_FPS} FPS) 写入原生目录")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(video_dst),
        "-vf", f"scale={FFMPEG_WIDTH}:-1,fps={FFMPEG_FPS}",
        "-q:v", "2",
        str(extracted_images_dir / "frame_%05d.jpg")
    ]
    # ffmpeg 输出太乱，不捕获输出除非报错
    run_command(ffmpeg_cmd, capture_output=False)
    
    # 3. COLMAP 解算
    print("    -> 1.2 Nerfstudio: 调用 COLMAP 进行位姿解算")
    cmd_colmap = [
        "ns-process-data", "images",
        "--data", str(extracted_images_dir),
        "--output-dir", str(data_dir),
        "--verbose",
    ]
    
    # 捕获 COLMAP 输出以检查质量
    result = run_command(cmd_colmap, capture_output=True)
    print(result.stdout)
    
    # 质量检查：如果 COLMAP 仅找到极少数的位姿，则停止
    if "COLMAP only found poses" in result.stdout:
        print("\n🚨🚨🚨 检测到 COLMAP 数据质量极差！自动停止训练。")
        print("❌ 错误原因：视频质量太差或场景反光，只有极少数图片找到了位姿。")
        print("➡️ 建议：请重拍视频（降低反光，增加纹理点），然后删除 transforms.json 重新运行。")
        
        # 清理损坏的数据，但保留 workspace 以供调试
        shutil.rmtree(data_dir)
        raise RuntimeError("COLMAP 数据质量不合格，流程停止。")
        
    if not (data_dir / "transforms.json").exists():
        raise FileNotFoundError("❌ COLMAP 失败，未生成 transforms.json")
