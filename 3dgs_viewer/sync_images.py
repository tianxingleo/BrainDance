# sync_images.py
# 功能：从现有的训练数据中提取图片并同步到 WebGL 查看器目录
# 实现：读取现有 transforms.json 并拷贝文件
import json
import shutil
import os
from pathlib import Path

def sync_images():
    # 路径配置
    base_dir = Path(__file__).parent
    viewer_public_models = base_dir / "my-3dgs-viewer/public/models"
    webgl_images_dir = viewer_public_models / "images"
    
    # 查找最近的输出目录
    outputs_dir = base_dir / "outputs"
    if not outputs_dir.exists():
        print("❌ 找不到 outputs 目录")
        return

    # 找到所有的 transforms.json
    all_transforms = list(outputs_dir.glob("**/transforms.json"))
    if not all_transforms:
        print("❌ 未在 outputs 下找到任何 transforms.json")
        return

    # 按修改时间排序，取最新的
    latest_transforms = max(all_transforms, key=os.path.getmtime)
    print(f"🔍 发现最新训练数据: {latest_transforms}")

    # 确定原始图片目录
    # 通常在 outputs/scene_xxx/nerfstudio/data/ 
    data_dir = latest_transforms.parent
    
    # 创建目标目录
    webgl_images_dir.mkdir(exist_ok=True, parents=True)

    try:
        with open(latest_transforms, 'r') as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        count = 0
        for frame in frames:
            file_path = frame.get("file_path")
            if not file_path: continue
            
            src_img = data_dir / file_path
            dst_img = webgl_images_dir / Path(file_path).name
            
            if src_img.exists():
                shutil.copy2(str(src_img), str(dst_img))
                count += 1
        
        print(f"✅ 成功同步 {count} 张图片至 {webgl_images_dir}")
        print(f"✨ 现在刷新网页即可看到预览图！")
        
    except Exception as e:
        print(f"❌ 同步失败: {e}")

if __name__ == "__main__":
    sync_images()
