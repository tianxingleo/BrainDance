#!/usr/bin/env python3
"""
本地单图 SAM3D 测试脚本

使用方法:
    cd ai_engine/3dgs
    conda activate gs_linux_backup
    python tests/test_local_single_image.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.factory import PipelineFactory
from src.config import PipelineConfig


def main():
    print("=" * 60)
    print("本地单图 SAM3D 测试")
    print("=" * 60)

    config = PipelineConfig()

    # 测试图片路径
    test_image = project_root / "test_data/images/test_image.png"
    if not test_image.exists():
        print(f"❌ 测试图片不存在: {test_image}")
        print("请将测试图片放到: test_data/images/test_image.png")
        return False

    print(f"\n📷 测试图片: {test_image}")

    # 创建工作目录
    work_dir = project_root / "temp_workspace/local_single_image_test"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 创建上下文 (模拟 Worker)
    logs = []

    def log_callback(message):
        logs.append(message)
        print(f"[LOG] {message}")

    context = {
        "task_id": "local_test_single",
        "scene_id": "single_image_test",
        "work_root": str(work_dir),
        "log_callback": log_callback,
    }

    # 获取 Pipeline
    print("\n🔧 获取 Pipeline...")
    pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)
    print(f"✅ Pipeline: {pipeline.__class__.__name__}")

    # 参数
    params = {
        "repo_path": str(config.sam3d_repo_path),
        "model_dir": str(config.sam3d_checkpoint_dir),
    }

    # 执行
    print("\n🔥 执行 Pipeline...")
    try:
        ply_path, metadata = pipeline.run(str(test_image), params)

        print("\n" + "=" * 60)
        print("✅ 测试通过!")
        print("=" * 60)
        print(f"📂 输出: {ply_path}")
        print(f"📊 元数据: {metadata}")
        print(f"📝 日志数: {len(logs)}")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
