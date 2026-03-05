import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["no_proxy"] = "huggingface.co,hf-mirror.com"
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NO_PROXY"] = "huggingface.co,hf-mirror.com"
# main.py
# 功能：程序入口文件，负责启动不同的运行模式
# 实现：根据命令行参数选择本地调试模式或云端监听模式
# 逻辑：1. 加载环境变量 2. 解析命令行参数 3. 启动相应模式
# 包含：本地模式运行函数、云端模式运行函数、模式选择逻辑
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from src.config import PipelineConfig
from src.core.pipeline import run_pipeline
from src.core.worker import CloudWorker

load_dotenv()

def run_local_mode(video_file: Path):
    """本地单次运行模式 (视频/多图)"""
    if not video_file.exists():
        print(f"❌ 找不到本地文件: {video_file}")
        return

    print(f"💿 启动本地模式: {video_file.name}")
    
    cfg = PipelineConfig(
        project_name="local_test_v1",
        video_path=video_file,
        enable_ai=True
    )
    
    print(f"⚙️  配置加载: Iterations={cfg.training_iterations}, MaxImages={cfg.max_images}")
    run_pipeline(cfg)

def run_cloud_mode():
    """云端监听模式"""
    worker = CloudWorker()
    worker.start()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_local_mode(Path(sys.argv[1]))
    else:
        print("☁️ 启动云端监听模式...")
        run_cloud_mode()