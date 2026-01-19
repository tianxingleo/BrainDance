import numpy as np
from pathlib import Path

def generate_cpu_config(original_config_path: Path):
    """从原始 yaml 生成一份强制 device: cpu 的临时配置"""
    cpu_config_path = original_config_path.parent / "cpu_pipeline.yaml"
    
    print(f"📝 [Config Hack] 正在创建 CPU 初始化配置: {cpu_config_path}")
    if not original_config_path.exists():
        raise FileNotFoundError(f"Config not found: {original_config_path}")
        
    with open(original_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace("device: cuda", "device: cpu")
    new_content = new_content.replace('device: "cuda"', 'device: "cpu"')
    
    with open(cpu_config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"    ✅ 已生成 CPU 配置")
    return cpu_config_path

def auto_generate_fallback_mask(image_np):
    """
    如果用户没有提供 Mask，使用简单的亮度阈值生成 Mask。
    (从原 sam3d.py 移植)
    """
    intensity = image_np.mean(axis=2)
    is_white_bg = intensity > 240
    is_black_bg = intensity < 15
    
    white_pixel_count = np.sum(is_white_bg)
    black_pixel_count = np.sum(is_black_bg)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    
    if white_pixel_count > total_pixels * 0.1:
        print("    🎨 检测到浅色背景，正在自动抠图...")
        return np.where(is_white_bg, 0, 255).astype(np.uint8)
    elif black_pixel_count > total_pixels * 0.1:
        print("    🎨 检测到深色背景，正在自动抠图...")
        return np.where(is_black_bg, 0, 255).astype(np.uint8)
    else:
        print("    ⚠️ 背景颜色不明确，使用全图 Mask (可能会生成方块)")
        return np.ones(image_np.shape[:2], dtype=np.uint8) * 255
