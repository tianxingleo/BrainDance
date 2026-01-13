# main.py
import sys
from pathlib import Path
from src.config import PipelineConfig
from src.core.pipeline import run_pipeline

if __name__ == "__main__":
    # 1. 解析路径
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])
    
    if not video_file.exists():
        print(f"❌ 找不到视频: {video_file}")
        sys.exit(1)

    # 2. 实例化配置
    cfg = PipelineConfig(
        project_name="glomap_test_v1",
        video_path=video_file,
        max_images=100,
        enable_ai=True
    )
    
    # 3. 启动引擎
    try:
        run_pipeline(cfg)
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止任务")
    except Exception as e:
        print(f"\n❌ 发生未捕获异常: {e}")
        raise e