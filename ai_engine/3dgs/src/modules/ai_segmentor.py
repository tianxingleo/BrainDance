# [业务类] 存放 AISegmentor + get_central_object_prompt
import os
import os
import shutil
import json
import cv2
import numpy as np
import torch
from pathlib import Path

# --- 1. 软依赖引入 (AI 库) ---
try:
    import dashscope
    from dashscope import MultiModalConversation
    from ultralytics import SAM, YOLOWorld
    HAS_AI = True
except ImportError:
    HAS_AI = False
    print("⚠️ [Module Warning] 'dashscope' or 'ultralytics' not found. AI features will be disabled.")

# --- 2. 项目引用 ---
from src.config import PipelineConfig
# 关键：引入清洗 Mask 的算法
from src.utils.cv_algorithms import clean_and_verify_mask


def get_central_object_prompt(images_dir: Path, sample_count=3):
    """
    [Step 1.1] 使用 Qwen-VL-Plus 多图分析，提取中心物体的文本描述
    
    参数:
        images_dir (Path): 图片文件夹路径
        sample_count (int): 采样图片数量，默认3张，节省 Token 并加快速度
    
    返回:
        prompt_text (str): 大模型生成的物体描述提示词
    """
    # 获取 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY，无法调用大模型。")
        return None

    print(f"\n🧠 [AI 分析] 正在调用 Qwen-VL-Plus 分析场景...")
    
    # [Python 进阶] 使用 glob 获取所有 jpg/png 图片，并排序确保顺序一致
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files: return None
    
    # [算法逻辑] 均匀采样：使用 numpy 的 linspace 在图片序列中均匀抽取 sample_count 张图
    # 这样能覆盖物体的不同角度，比只取前三张更稳健
    indices = np.linspace(0, len(image_files) - 1, sample_count, dtype=int)
    sampled_imgs = [image_files[i] for i in indices]
    
    # 构建多模态消息体 (Dashscope SDK 要求的格式)
    content = [{"image": str(img_path)} for img_path in sampled_imgs]
    content.append({
        "text": (
            "这些是一个视频的抽帧图片。请分析画面中心始终存在的、最主要的一个物体是什么。"
            "请输出一个适合用于物体检测模型的英文名词短语（Prompt）。"
            "⚠️ 关键策略：请优先描述【视觉特征】（颜色、材质、形状），而不是【功能名称】。"
            "越简单、越'土'的词，检测模型越容易识别。"
            "例如："
            " - 不要说 'electric shaver' (电动剃须刀)，请说 'gray metal object' 或 'device'。"
            " - 不要说 'portable charger' (充电宝)，请说 'white rectangular box'。"
            "要求：严格只输出这个英文短语，不要包含任何标点符号、解释。"
        )
    })
    
    # 封装用户消息
    messages = [{"role": "user", "content": content}]

    try:
        # 调用阿里云 Qwen-VL-Plus 模型
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus', 
            messages=messages
        )
        
        # 解析返回结果
        if response.status_code == 200:
            # 提取文本内容
            prompt_text = response.output.choices[0].message.content[0]["text"].strip()
            # [数据清洗] 去掉可能存在的标点符号，防止干扰 YOLO
            prompt_text = prompt_text.replace(".", "").replace('"', "").replace("'", "")
            # \033[92m 是 ANSI 转义码，用于在控制台输出绿色文字
            print(f"    🤖 Qwen 认为中心物体是: [ \033[92m{prompt_text}\033[0m ]")
            return prompt_text
        else:
            print(f"❌ Qwen 调用失败: {response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"❌ API 连接异常: {e}")
        return None


class AISegmentor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.images_dir = cfg.images_dir
        self.masks_dir = cfg.masks_dir

    def run(self):
        """执行 AI 分割总流水线 (对应原 run_ai_segmentation_pipeline)"""
        if not HAS_AI or not self.cfg.enable_ai:
            print("⏩ 跳过 AI 分割 (未启用或缺少依赖)")
            return False
            
        if not self.cfg.transforms_file.exists():
            print("⚠️ transforms.json 不存在，无法进行 AI 分割")
            return False

        print(f"\n✂️ [AI 分割] 正在初始化...")
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # 1. 获取提示词
        text_prompt = self._get_prompt()
        
        # 2. 加载模型
        try:
            # 自动迁移模型文件逻辑
            self._ensure_model_exists("yolov8s-worldv2.pt")
            self._ensure_model_exists("sam2.1_l.pt")
            
            yolo_path = self.cfg.work_root / "yolov8s-worldv2.pt"
            sam_path = self.cfg.work_root / "sam2.1_l.pt"
            
            print("    -> 正在加载 AI 模型...")
            det_model = YOLOWorld(str(yolo_path))
            det_model.set_classes([text_prompt])
            sam_model = SAM(str(sam_path))
        except Exception as e:
            print(f"❌ AI 模型加载失败: {e}")
            return False

        # 3. 读取元数据
        with open(self.cfg.transforms_file, 'r') as f: meta = json.load(f)
        frames_map = {Path(f["file_path"]).name: f for f in meta["frames"]}
        
        image_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        valid_frames_list = []
        deleted_count = 0
        
        print(f"    -> 开始处理 {len(image_files)} 张图片...")

        # 4. 循环处理
        for i, img_path in enumerate(image_files):
            try:
                # YOLO 检测
                det_results = det_model.predict(img_path, conf=0.05, verbose=False)
                bboxes = det_results[0].boxes.xyxy.cpu()
                
                # 筛选中心框 (逻辑与之前相同，这里简化展示)
                if len(bboxes) > 1:
                    bboxes = self._pick_center_box(bboxes, det_results[0].orig_shape)
                
                # SAM 分割
                if len(bboxes) == 0:
                    # 中心点模式
                    h, w = det_results[0].orig_shape[:2]
                    cx, cy, margin = w / 2, h / 2, 5
                    bboxes = [[cx-margin, cy-margin, cx+margin, cy+margin]]
                
                sam_results = sam_model(img_path, bboxes=bboxes, verbose=False)
                
                # 合并 Mask
                if sam_results[0].masks is not None:
                    final_mask = np.any(sam_results[0].masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255
                else:
                    final_mask = np.zeros(det_results[0].orig_shape[:2], dtype=np.uint8)

                # 清洗 Mask (调用内部方法)
                is_good, cleaned_mask, reason = self._clean_and_verify_mask(final_mask)
                
                if is_good:
                    final_name = self._save_transparent_png(img_path, cleaned_mask)
                    if img_path.name in frames_map:
                        frame_data = frames_map[img_path.name]
                        frame_data["file_path"] = f"images/{final_name}"
                        valid_frames_list.append(frame_data)
                else:
                    print(f"       🗑️ [剔除] {img_path.name}: {reason}")
                    img_path.unlink()
                    deleted_count += 1

            except Exception as e:
                print(f"       ❌ 错误 {img_path.name}: {e}")
                continue
            
            if i % 10 == 0: print(f"       进度: {i}/{len(image_files)}...", end="\r")

        # 5. 更新 json
        if valid_frames_list:
            meta["frames"] = valid_frames_list
            with open(self.cfg.transforms_file, 'w') as f: json.dump(meta, f, indent=4)
            print(f"\n    ✅ AI 处理完成，剩余可用: {len(valid_frames_list)}")
            return True
        else:
            print("\n❌ 错误：所有图片都被剔除了")
            return False

    def _get_prompt(self):
        """原 get_central_object_prompt 的封装"""
        # 这里你可以调用之前定义的全局函数 get_central_object_prompt(self.images_dir)
        # 或者把那段代码搬进来。为了省事，建议直接调用现有的全局函数：
        try:
            prompt = get_central_object_prompt(self.images_dir)
            return prompt if prompt else "central object"
        except:
            return "central object"

    def _ensure_model_exists(self, model_name):
        target = self.cfg.work_root / model_name
        local = Path(__file__).parent / model_name
        if not target.exists() and local.exists():
            shutil.copy2(str(local), str(target))

    def _pick_center_box(self, bboxes, img_shape):
        """筛选最中心的框"""
        import torch
        img_h, img_w = img_shape[:2]
        screen_center = torch.tensor([img_w / 2.0, img_h / 2.0])
        min_dist = float('inf')
        best_idx = 0
        for idx, box in enumerate(bboxes):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            dist = torch.sqrt((cx - screen_center[0])**2 + (cy - screen_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        return bboxes[best_idx].unsqueeze(0)

    def _clean_and_verify_mask(self, mask):
        """原 clean_and_verify_mask 的封装"""
        # 直接调用之前的全局函数即可
        return clean_and_verify_mask(mask)

    def _save_transparent_png(self, img_path, mask):
        """合成并保存 PNG"""
        img = cv2.imread(str(img_path))
        mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        alpha = mask_blurred.astype(np.float32) / 255.0
        img_float = img.astype(np.float32)
        b, g, r = cv2.split(img_float)
        img_bgra = cv2.merge([
            (b * alpha).astype(np.uint8),
            (g * alpha).astype(np.uint8),
            (r * alpha).astype(np.uint8),
            mask_blurred
        ])
        new_path = img_path.with_suffix('.png')
        cv2.imwrite(str(new_path), img_bgra)
        if img_path.suffix.lower() == '.jpg':
            try: img_path.unlink()
            except: pass
        return new_path.name
        