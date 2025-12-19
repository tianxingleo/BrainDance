import os
import sys
import shutil
import time
import subprocess
from pathlib import Path

# ================= 🔧 配置区域 =================
SHARP_REPO_PATH = Path("/home/ltx/workspace/ai/ml-sharp") 
WINDOWS_SOURCE_DIR = Path(__file__).resolve().parent 
INPUT_IMAGE_NAME = "input.jpg"

# 强制指定使用第 0 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ===============================================

def setup_environment():
    if not SHARP_REPO_PATH.exists():
        print(f"❌ [错误] 找不到仓库: {SHARP_REPO_PATH}")
        sys.exit(1)
    
    if shutil.which("sharp") is None:
        print("\n❌ [关键错误] 未找到 'sharp' 命令！")
        print(f"   请进入 {SHARP_REPO_PATH} 运行: pip install -e .")
        sys.exit(1)

def run_sharp_pipeline():
    # 1. 准备图片
    src_img = WINDOWS_SOURCE_DIR / INPUT_IMAGE_NAME
    if not src_img.exists():
        src_img = WINDOWS_SOURCE_DIR / "input.png"
    
    if not src_img.exists():
        print(f"❌ 找不到图片: {INPUT_IMAGE_NAME} 或 input.png")
        return

    print(f"📸 输入图片: {src_img.name}")

    # 2. 准备输出目录
    project_name = src_img.stem
    output_dir = WINDOWS_SOURCE_DIR / "sharp_outputs" / project_name
    
    # 清理旧结果，防止混淆
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 构造命令 (已移除 --gpu，改用环境变量控制)
    cmd = [
        "sharp", "predict",
        "-i", str(src_img.resolve()),
        "-o", str(output_dir.resolve())
    ]

    print(f"\n🚀 [开始推理] 正在调用 SHARP...")
    print(f"   执行命令: {' '.join(cmd)}")

    try:
        # 运行推理 (capture_output=False 让它直接打印进度条到屏幕)
        subprocess.run(cmd, check=True, cwd=str(SHARP_REPO_PATH))
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败，退出码: {e.returncode}")
        print("   如果是因为显存不足，请尝试调小图片尺寸。")
        return

    # 4. 结果回传
    ply_files = list(output_dir.glob("*.ply"))
    if ply_files:
        final_ply = WINDOWS_SOURCE_DIR / f"{project_name}_sharp.ply"
        shutil.copy2(str(ply_files[0]), str(final_ply))
        print(f"\n🎉 [成功] 3DGS 模型已生成: {final_ply}")
    else:
        print(f"\n⚠️ 警告: 命令运行成功，但未找到 .ply 文件。")
        print(f"   请手动检查输出目录: {output_dir}")

if __name__ == "__main__":
    setup_environment()
    run_sharp_pipeline()