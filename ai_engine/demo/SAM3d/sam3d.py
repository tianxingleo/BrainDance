import time
import datetime
from pathlib import Path
from sam3d_engine import SAM3DEngine

# ================= 🔧 配置区域 =================
INPUT_IMAGE_NAME = "input.jpg"  # 脚本会自动优先查找 input.png
LINUX_WORK_ROOT = Path.home() / "sam3d_workspace"
SAM3D_REPO_PATH = Path.home() / "workspace/ai/sam-3d-objects"

def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def run_pipeline():
    global_start_time = time.time()
    
    current_dir = Path(__file__).resolve().parent
    source_png_path = current_dir / "input.png"
    source_jpg_path = current_dir / INPUT_IMAGE_NAME
    
    if source_png_path.exists():
        print(f"✨ 发现 input.png，将优先使用")
        source_img_path = source_png_path
    elif source_jpg_path.exists():
        source_img_path = source_jpg_path
    else:
        print(f"❌ 错误: 找不到输入图片 (input.png 或 {INPUT_IMAGE_NAME})")
        return

    project_name = source_img_path.stem 
    work_dir = LINUX_WORK_ROOT / project_name
    
    print(f"\n🚀 [SAM3D Modular] 启动任务: {source_img_path.name}")
    
    # 初始化引擎
    try:
        engine = SAM3DEngine(str(SAM3D_REPO_PATH))
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        return

    # 执行推理
    try:
        ply_path = engine.run(
            image_path=str(source_img_path),
            output_dir=str(work_dir)
        )
        
        if ply_path and Path(ply_path).exists():
            final_output_path = current_dir / f"{project_name}_3dgs.ply"
            import shutil
            shutil.copy2(ply_path, str(final_output_path))
            print(f"\n🎉 成功！模型已保存: {final_output_path}")
        else:
            print("    ❌ 失败: 未生成 PLY 文件")
            
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n📊 总耗时: {format_duration(time.time() - global_start_time)}")

if __name__ == "__main__":
    run_pipeline()