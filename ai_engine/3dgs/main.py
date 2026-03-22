import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["no_proxy"] = "huggingface.co,hf-mirror.com"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NO_PROXY"] = "huggingface.co,hf-mirror.com"

# main.py
# 功能：程序入口文件，负责启动不同的运行模式
# 实现：根据命令行参数选择本地调试模式或云端监听模式
# 逻辑：1. 加载环境变量 2. 解析命令行参数 3. 启动相应模式
# 包含：本地模式运行函数、云端模式运行函数、模式选择逻辑
import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import PipelineConfig
from src.core.local_runner import (
    LOCAL_TASK_TYPE_CHOICES,
    SLOW_PIPELINE_CHOICES,
    run_local_mode,
)
from src.core.supervisor import WorkerSupervisor
from src.core.worker import CloudWorker

load_dotenv()


def run_cloud_mode():
    """云端监听模式（Supervisor）"""
    supervisor = WorkerSupervisor()
    supervisor.start()


def run_child_worker_mode():
    """Supervisor 拉起的子 Worker 模式"""
    worker = CloudWorker(PipelineConfig())
    worker.start()


def main():
    parser = argparse.ArgumentParser(description="BrainDance 3DGS local/cloud entry")
    parser.add_argument("input", nargs="?", help="本地输入视频文件路径")
    parser.add_argument(
        "--child-worker",
        action="store_true",
        help="Supervisor 拉起的子 Worker 模式",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="video_dual_chain",
        choices=LOCAL_TASK_TYPE_CHOICES,
        help="本地任务类型，默认 video_dual_chain",
    )
    parser.add_argument(
        "--slow-pipeline",
        type=str,
        default="video_3dgs",
        choices=SLOW_PIPELINE_CHOICES,
        help="仅在 video_dual_chain 下生效",
    )
    parser.add_argument(
        "--best-frame-sample-count",
        type=int,
        default=8,
        help="仅在 video_dual_chain 下生效",
    )
    parser.add_argument(
        "--sam3d-vram-threshold-gb",
        type=float,
        default=25,
        help="仅在 video_dual_chain 下生效",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="local_test_v1",
        help="本地任务 scene_id / 工作目录名",
    )
    args = parser.parse_args()

    if args.child_worker:
        run_child_worker_mode()
        return

    if args.input:
        run_local_mode(
            video_file=Path(args.input),
            task_type=args.task_type,
            slow_pipeline=args.slow_pipeline,
            sample_count=args.best_frame_sample_count,
            sam3d_vram_threshold_gb=args.sam3d_vram_threshold_gb,
            project_name=args.project_name,
        )
        return

    print("☁️ 启动云端监听模式（Supervisor）...")
    run_cloud_mode()


if __name__ == "__main__":
    main()
