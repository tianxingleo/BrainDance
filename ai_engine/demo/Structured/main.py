import sys
from pathlib import Path
from pipeline import run

if __name__ == "__main__":
    # 默认视频文件 (方便测试)
    default_video = Path("test.mp4")
    
    # 接收命令行参数: python main.py my_video.mp4
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])
    else:
        video_file = default_video

    if video_file.exists():
        # 项目名称默认用视频文件名(不含后缀) + _proj
        project_name = video_file.stem + "_proj"
        run(video_file, project_name)
    else:
        print(f"❌ 错误: 找不到视频文件 {video_file}")
        print(f"👉 用法: python main.py <视频路径>")
