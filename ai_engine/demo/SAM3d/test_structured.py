from pathlib import Path
from sam3d_engine.core import SAM3DEngine

# 配置路径
REPO_PATH = Path.home() / "workspace/ai/sam-3d-objects"
# 假设模型文件就在 demo 目录下
MODEL_DIR = Path(__file__).parent.parent 

def main():
    # 实例化引擎 (自动加载 MaskGenerator)
    engine = SAM3DEngine(repo_path=str(REPO_PATH), model_dir=str(MODEL_DIR))
    
    # 运行 (不传 mask_path，会自动触发 smart mask)
    # 注意：如果当前目录下有 input.jpg 才会运行成功
    image_path = "./input.jpg"
    if not Path(image_path).exists():
        # 尝试 input.png
        if Path("./input.png").exists():
            image_path = "./input.png"
        else:
            print(f"❌ 错误: 找不到输入图片 {image_path}")
            return

    engine.run(
        image_path=image_path,
        output_dir="./output"
    )

if __name__ == "__main__":
    main()
