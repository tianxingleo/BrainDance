from ultralytics import SAM
import torch

# 指向您存放 sam3.pt 的路径
model_path = "/home/ltx/workspace/ai/sam3/sam3.pt"

try:
    print(f"正在加载 SAM 3: {model_path}")
    model = SAM(model_path)
    print("✅ 模型加载成功！依赖库正常。")
except Exception as e:
    print(f"❌ 加载失败: {e}")