import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.core.factory import PipelineFactory
from src.config import PipelineConfig

def test_sam3d_pipeline():
    print("🚀 开始集成测试: Single Image SAM3D Pipeline")
    
    config = PipelineConfig()
    
    if not config.sam3d_repo_path.exists():
        print(f"❌ 错误: 找不到 SAM3D 库，请检查 {config.sam3d_repo_path} 是否存在")
        return

    input_image = project_root.parent / "demo/SAM3d/test_input.png"
    if not input_image.exists():
        print(f"⚠️ 警告: 测试图片不存在 {input_image}，跳过执行")
        print("请将测试图片放置到 demo/SAM3d/test_input.png")
        return

    context = {
        "task_id": "test_integration_001",
        "work_dir": str(project_root / "output/test_integration")
    }

    try:
        pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)
        print(f"✅ Pipeline 工厂加载成功: {pipeline}")
    except Exception as e:
        print(f"❌ Pipeline 工厂加载失败: {e}")
        return

    try:
        print("🔥 正在运行 Pipeline (这可能需要几分钟)...")
        params = {
            "model_dir": str(config.sam3d_checkpoint_dir)
        }
        
        output_ply, meta = pipeline.run(str(input_image), params)
        
        print("-" * 30)
        print(f"🎉 测试通过!")
        print(f"📂 输出文件: {output_ply}")
        print(f"ℹ️ 元数据: {meta}")
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ 运行过程中崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sam3d_pipeline()
