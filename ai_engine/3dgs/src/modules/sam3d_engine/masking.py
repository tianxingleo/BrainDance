import cv2
import numpy as np
import torch
import os
from pathlib import Path
from PIL import Image
from typing import Optional, Union

# 关闭 Ultralytics 运行时自动安装依赖，避免线上环境抖动导致失败。
# 需要在导入 ultralytics 前设置，才能让其 AUTOINSTALL 常量生效。
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

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

    def _find_weight(self, filename: str) -> Optional[Path]:
        """在多候选目录中查找权重文件，避免因目录差异导致误判缺失。"""
        candidates = [
            self.model_dir,
            self.model_dir / "segmentation",
            # ai_engine/models/segmentation
            Path(__file__).resolve().parents[4] / "models" / "segmentation",
            # 仓库根目录下的 models 与 models/segmentation
            Path(__file__).resolve().parents[5] / "models",
            Path(__file__).resolve().parents[5] / "models" / "segmentation",
        ]
        for directory in candidates:
            path = directory / filename
            if path.exists():
                return path
        return None

    def _is_lfs_pointer(self, path: Optional[Path]) -> bool:
        """检测文件是否是 Git LFS 指针而非真实权重。"""
        if path is None or (not path.exists()) or path.stat().st_size > 1024:
            return False
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:200]
            return head.startswith("version https://git-lfs.github.com/spec/v1")
        except Exception:
            return False

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
        
        yolo_env = Path(os.getenv("YOLO_WORLD_MODEL_PATH")) if os.getenv("YOLO_WORLD_MODEL_PATH") else None
        sam_env = Path(os.getenv("SAM2_MODEL_PATH")) if os.getenv("SAM2_MODEL_PATH") else None

        yolo_path = yolo_env if yolo_env and yolo_env.exists() else self._find_weight("yolov8s-worldv2.pt")
        sam_path = sam_env if sam_env and sam_env.exists() else self._find_weight("sam2.1_l.pt")
        if sam_path is None:
            # 兼容轻量权重命名
            sam_path = self._find_weight("sam2.1_b.pt")

        if yolo_path is None or sam_path is None:
            print("    ⚠️ [Mask] 模型文件缺失 (yolov8s-worldv2.pt 或 sam2.1_l.pt/sam2.1_b.pt)，降级为 simple")
            # 递归调用 simple
            return self._simple_mask_gen(Image.open(image_path))

        # LFS 指针会触发 `invalid load key, 'v'`，此处改用模型名让 ultralytics 自动下载真实权重
        yolo_target: Union[str, Path] = yolo_path
        sam_target: Union[str, Path] = sam_path
        if self._is_lfs_pointer(yolo_path):
            print(f"    ⚠️ [Mask] 检测到 YOLO 权重是 Git LFS 指针: {yolo_path}，改为自动下载")
            yolo_target = "yolov8s-worldv2.pt"
        if self._is_lfs_pointer(sam_path):
            print(f"    ⚠️ [Mask] 检测到 SAM 权重是 Git LFS 指针: {sam_path}，改为自动下载")
            # 若本地命中的是 b 权重，优先尝试 b；否则用 l
            sam_target = "sam2.1_b.pt" if sam_path.name == "sam2.1_b.pt" else "sam2.1_l.pt"

        print(f"    📦 [Mask] 权重加载源: YOLO={yolo_target} | SAM={sam_target}")

        # Step 1: YOLO 检测中心物体（失败时回退中心框，不中断智能抠图）
        best_box = self._detect_box_with_yolo(image_path=image_path, yolo_target=yolo_target)
        if best_box is None:
            best_box = self._default_center_box(image_path)

        # Step 2: SAM 分割（仅当 SAM 失败时降级 simple）
        try:
            sam_model = SAM(str(sam_target))
            sam_results = sam_model(str(image_path), bboxes=best_box, verbose=False)

            if sam_results[0].masks is not None:
                # 合并所有 mask
                mask = np.any(sam_results[0].masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255
                print("    ✅ [Mask] AI 抠图完成")
                return mask

            print("    ⚠️ [Mask] SAM 未生成 Mask，降级处理")
            return self._simple_mask_gen(Image.open(image_path))
        except Exception as e:
            print(f"    ❌ [Mask] SAM 引擎出错: {e}，降级处理")
            if "invalid load key" in str(e):
                print("    💡 [Mask] 提示: 当前 .pt 很可能是 Git LFS 指针文件。请安装 git-lfs 并拉取真实权重，或让 Ultralytics 联网自动下载。")
            return self._simple_mask_gen(Image.open(image_path))

    def _detect_box_with_yolo(self, image_path: Path, yolo_target: Union[str, Path]) -> Optional[torch.Tensor]:
        """尝试用 YOLOWorld 找到目标框；失败时返回 None。"""
        try:
            det_model = YOLOWorld(str(yolo_target))
            det_results = det_model.predict(
                str(image_path),
                conf=0.1,
                classes=None,
                verbose=False,
            )
            # 默认禁用 open-vocab，避免触发 CLIP 额外网络依赖。
            if os.getenv("MASK_USE_OPEN_VOCAB", "0") == "1":
                try:
                    det_model.set_classes(["central object", "main item"])
                except Exception as e:
                    print(f"    ⚠️ [Mask] set_classes 失败，继续使用默认类别: {e}")
            if det_results and len(det_results[0].boxes) > 0:
                bboxes = det_results[0].boxes.xyxy.cpu()
                return self._pick_center_box(bboxes, det_results[0].orig_shape)
            return None
        except Exception as e:
            print(f"    ⚠️ [Mask] YOLOWorld 不可用，回退中心框: {e}")
            return None

    def _default_center_box(self, image_path: Path) -> torch.Tensor:
        """生成中心默认框，供 SAM 在无检测结果时使用。"""
        with Image.open(image_path) as img:
            w, h = img.size
        m = 20
        return torch.tensor([[w / 2 - m, h / 2 - m, w / 2 + m, h / 2 + m]])

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
