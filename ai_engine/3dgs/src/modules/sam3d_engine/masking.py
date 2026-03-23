import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# 尝试引入 AI 库，如果环境没有装也不报错，自动降级
try:
    from ultralytics import SAM, YOLOWorld
    HAS_AI = True
except ImportError:
    HAS_AI = False

class MaskGenerator:
    def __init__(self, model_dir: str = "./models"):
        """
        初始化抠图引擎
        :param model_dir: 存放 YOLO 和 SAM 权重的目录
        """
        self.model_dir = Path(model_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_mask(self, image_path: str, method: str = "smart") -> np.ndarray:
        """
        获取 Mask 的主入口
        :param image_path: 图片路径
        :param method: 'smart' (AI模型) 或 'simple' (亮度阈值)
        :return: (H, W) 的 uint8 numpy 数组 (0=背景, 255=前景)
        """
        image_path = Path(image_path)
        
        # 1. 优先检查图片本身是否自带 Alpha 通道 (png)
        img_pil = Image.open(image_path).convert("RGBA")
        if np.array(img_pil)[:, :, 3].min() < 255:
            print("    🎭 [Mask] 检测到 PNG 透明通道，直接使用")
            return np.array(img_pil)[:, :, 3]

        # 2. 根据策略生成
        if method == "smart" and HAS_AI:
            return self._smart_mask_gen(image_path)
        else:
            if method == "smart" and not HAS_AI:
                print("    ⚠️ [Mask] 未安装 ultralytics，降级为 simple 模式")
            return self._simple_mask_gen(img_pil)

    def _smart_mask_gen(self, image_path: Path) -> np.ndarray:
        """
        [Smart] 使用 YOLOv8-World + SAM 2.1 进行智能抠图
        """
        print(f"    🧠 [Mask] 正在运行 AI 智能抠图...")
        
        yolo_path = self.model_dir / "yolov8s-worldv2.pt"
        sam_path = self.model_dir / "sam2.1_l.pt"

        if not yolo_path.exists() or not sam_path.exists():
            print(f"    ⚠️ [Mask] 模型文件缺失 ({yolo_path.name} 或 {sam_path.name})，降级为 simple")
            # 递归调用 simple
            return self._simple_mask_gen(Image.open(image_path))

        try:
            # Step 1: YOLO 检测中心物体
            # 临时屏蔽 yolov8 的啰嗦日志
            det_model = YOLOWorld(str(yolo_path))
            det_results = det_model.predict(
                str(image_path), 
                conf=0.1, 
                classes=None, # 检测所有类别，依靠 prompt
                verbose=False
            )
            # 设置通用的中心物体提示词
            det_model.set_classes(["central object", "main item"])
            
            # 获取最中心的框
            if len(det_results[0].boxes) > 0:
                bboxes = det_results[0].boxes.xyxy.cpu()
                best_box = self._pick_center_box(bboxes, det_results[0].orig_shape)
            else:
                # 没检测到，给个中心点的默认框
                h, w = det_results[0].orig_shape[:2]
                m = 20 # 边距
                best_box = torch.tensor([[w/2-m, h/2-m, w/2+m, h/2+m]])

            # Step 2: SAM 分割
            sam_model = SAM(str(sam_path))
            sam_results = sam_model(str(image_path), bboxes=best_box, verbose=False)
            
            if sam_results[0].masks is not None:
                # 合并所有 mask
                mask = np.any(sam_results[0].masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255
            else:
                print("    ⚠️ [Mask] SAM 未生成 Mask，降级处理")
                return self._simple_mask_gen(Image.open(image_path))
            
            print("    ✅ [Mask] AI 抠图完成")
            return mask

        except Exception as e:
            print(f"    ❌ [Mask] AI 引擎出错: {e}，降级处理")
            return self._simple_mask_gen(Image.open(image_path))

    def _simple_mask_gen(self, img_pil: Image) -> np.ndarray:
        """
        [Simple] 基于亮度阈值的快速抠图 (原 sam3d.py 逻辑)
        """
        print("    🎨 [Mask] 运行 Simple 规则抠图...")
        image_np = np.array(img_pil.convert("RGB"))
        
        intensity = image_np.mean(axis=2)
        is_white = intensity > 240
        is_black = intensity < 15
        
        total = image_np.size / 3
        if np.sum(is_white) > total * 0.1:
            return np.where(is_white, 0, 255).astype(np.uint8)
        elif np.sum(is_black) > total * 0.1:
            return np.where(is_black, 0, 255).astype(np.uint8)
        else:
            # 全白 mask (保留全图)
            return np.ones(image_np.shape[:2], dtype=np.uint8) * 255

    def _pick_center_box(self, bboxes, img_shape):
        """辅助：选离画面中心最近的框"""
        img_h, img_w = img_shape[:2]
        center = torch.tensor([img_w / 2.0, img_h / 2.0])
        min_dist = float('inf')
        best_idx = 0
        
        for i, box in enumerate(bboxes):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            dist = torch.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return bboxes[best_idx].unsqueeze(0)
