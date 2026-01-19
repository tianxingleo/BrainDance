# main.py
# 功能：程序入口文件，负责启动不同的运行模式
# 实现：根据命令行参数选择本地调试模式或云端监听模式
# 逻辑：1. 加载环境变量 2. 解析命令行参数 3. 启动相应模式
# 包含：本地模式运行函数、云端模式运行函数、模式选择逻辑
import sys
import os
from pathlib import Path
# 引入 dotenv 确保本地运行时也能加载 .env
from dotenv import load_dotenv 

from src.config import PipelineConfig
from src.core.pipeline import run_pipeline
from src.core.worker import CloudWorker

# 加载环境变量 (这行很重要，否则 main.py 读不到 .env)
load_dotenv()

def run_local_mode(video_file: Path):
    """本地单次运行模式"""
    if not video_file.exists():
        print(f"❌ 找不到本地视频: {video_file}")
        return

    print(f"💿 启动本地模式: {video_file.name}")
    
    # 🟢 [修改] 不再硬编码 max_images，直接使用 Config 的默认值 (即 .env 里的值)
    # 当然，你依然可以在这里手动覆盖它，例如: max_images=999
    cfg = PipelineConfig(
        project_name="local_test_v1",
        video_path=video_file,
        enable_ai=True
        # max_images 和 training_iterations 会自动从 .env 读取
    )
    
    # 打印一下参数确认
    print(f"⚙️  配置加载: Iterations={cfg.training_iterations}, MaxImages={cfg.max_images}")
    
    run_pipeline(cfg)

def run_cloud_mode():
    """云端监听模式"""
    # CloudWorker 内部已经处理了 .env 加载，这里直接启动即可
    worker = CloudWorker()
    worker.start()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_local_mode(Path(sys.argv[1]))
    else:
        print("☁️ 未检测到输入文件，默认启动云端监听模式...")
        run_cloud_mode()