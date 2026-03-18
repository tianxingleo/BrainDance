#!/usr/bin/env python3
"""
快速测试脚本：验证 SAM3D Pipeline 是否正常工作

使用方法:
    conda activate gs_linux_backup
    cd ai_engine/3dgs
    python tests/quick_test.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.factory import PipelineFactory
from src.config import PipelineConfig


def quick_test():
    print("🚀 快速测试 SAM3D Pipeline")
    print("=" * 50)

    config = PipelineConfig()

    # 检查依赖
    print(f"\n📁 SAM3D 仓库: {config.sam3d_repo_path}")
    print(f"📁 模型目录: {config.sam3d_checkpoint_dir}")

    if not config.sam3d_repo_path.exists():
        print("❌ SAM3D 仓库不存在!")
        return False

    # 检查测试图片
    test_image = project_root.parent / "demo/SAM3d/test_input.png"
    if not test_image.exists():
        print(f"❌ 测试图片不存在: {test_image}")
        return False
    print(f"📷 测试图片: {test_image}")

    # 创建上下文
    context = {
        "task_id": "quick_test_001",
        "scene_id": "quick_test_scene",
        "work_root": str(project_root / "output/quick_test"),
        "log_callback": lambda msg: print(f"[LOG] {msg}"),
    }

    # 获取 Pipeline
    print("\n🔧 获取 Pipeline...")
    pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)
    print(f"✅ Pipeline: {pipeline.__class__.__name__}")

    # 执行
    print("\n🔥 执行 Pipeline...")
    params = {
        "repo_path": str(config.sam3d_repo_path),
        "model_dir": str(config.sam3d_checkpoint_dir),
    }

    ply_path, metadata = pipeline.run(str(test_image), params)

    print("\n" + "=" * 50)
    print("✅ 测试通过!")
    print(f"📂 输出: {ply_path}")
    print(f"📊 元数据: {metadata}")
    print("=" * 50)

    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
