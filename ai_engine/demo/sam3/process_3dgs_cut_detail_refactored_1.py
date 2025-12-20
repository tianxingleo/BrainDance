# ==============================================================================
# 导入标准库和第三方库
# ==============================================================================
import subprocess  # [标准库] 用于执行外部系统命令（如 ffmpeg, colmap），实现 Python 与操作系统的交互
import sys         # [标准库] 用于访问与 Python 解释器紧密相关的变量和函数，如获取命令行参数 sys.argv
import shutil      # [标准库] 高级文件操作库，提供复制(copy)、移动(move)、删除目录树等功能
import os          # [标准库] 提供操作系统接口，用于文件路径操作、读取环境变量、文件存在性检查等
import time        # [标准库] 用于时间处理，计算代码运行耗时（性能分析）
import datetime    # [标准库] 用于将秒数转换为人类可读的日期和时间格式 (HH:MM:SS)
from pathlib import Path  # [标准库] 面向对象的文件系统路径库，比 os.path 更直观，支持 .parent, .name 等链式调用
import json        # [标准库] 用于读写 JSON 格式文件，这里主要用于处理相机位姿文件 transforms.json
import numpy as np # [第三方库] Python 科学计算的核心库，用于处理矩阵运算、图像数组（HWC格式）
# 在 import numpy as np 下方确认添加
import torch
import logging     # [标准库] 日志系统，用于控制控制台输出级别，屏蔽不必要的警告信息
import cv2         # [第三方库] OpenCV (Open Source Computer Vision)，用于图像读取、形态学操作、轮廓查找等
import re          # [标准库] 正则表达式库，用于处理复杂的字符串匹配和提取

# ================= 🧠 AI 依赖引入 =================
# [工程化思路] 软依赖导入：为了保证程序的健壮性，不要因为缺失 AI 相关的库（非核心功能）而导致整个程序崩溃。
# 这里使用了 try-except 块来检测是否安装了 AI 相关的库。
try:
    import dashscope  # [第三方库] 阿里云百炼 SDK，用于调用 Qwen-VL (通义千问视觉版) 多模态大模型
    from dashscope import MultiModalConversation # 具体导入多模态对话类，用于发送图片和文本给大模型
    from ultralytics import SAM, YOLOWorld # [第三方库] Ultralytics 库，封装了最先进的视觉模型：YOLO (目标检测) 和 SAM (分割万物)
    HAS_AI = True     # [全局变量] 标记位，设置为 True，后续逻辑会根据这个变量决定是否开启智能分割
except ImportError:
    HAS_AI = False    # 如果导入失败，标记为 False
    # [用户交互] 友好的错误提示，告知用户缺失了什么库以及如何通过 pip 安装修复
    print("⚠️ [环境警告] 未检测到 dashscope 或 ultralytics 库。")
    print("    -> 智能分割功能将被禁用。请运行: pip install dashscope ultralytics")

# 🔥 请在此处填入你的 API KEY (或者确保环境变量 DASHSCOPE_API_KEY 已存在)
# os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ================= 🔧 基础配置 =================
# 设置日志级别
# 屏蔽 nerfstudio 库中非 Error 级别的日志，防止控制台被大量的 Warning/Info 刷屏，影响观察进度
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

from dataclasses import dataclass, field # [标准库] 用于快速创建“数据类”，减少样板代码（如 __init__）
from pathlib import Path
import os
import sys

# ==============================================================================
# 类: PipelineConfig (流水线配置类)
# ------------------------------------------------------------------------------
# [依赖库]: dataclasses, pathlib, os
# [功能]: 管理整个工程的所有配置项和路径。
#         它利用 __post_init__ 特性，在初始化后自动计算出所有派生路径，
#         避免了在代码各处手动拼接路径导致的混乱和拼写错误。
# ==============================================================================
@dataclass
class PipelineConfig:
    # 1. 【必填项】用户初始化时必须传入的参数
    project_name: str    # 项目名称，将用作文件夹名
    video_path: Path     # 输入视频的路径
    
    # 2. 【选填项】有默认值的配置 (对应原代码的全局变量)
    work_root: Path = Path.home() / "braindance_workspace" # 工作根目录，默认为用户主目录下的 braindance_workspace
    max_images: int = 180              # 最大图片数量限制，防止显存爆炸或训练时间过长
    force_spherical_culling: bool = True # 是否强制开启球形裁剪（切除远处的杂景）
    scene_radius_scale: float = 1.8    # 场景半径缩放因子，用于计算 Nerfstudio 的 collider (碰撞体) 参数
    keep_percentile: float = 0.9       # 点云清洗时的保留比例 (保留 90% 的点)
    enable_ai: bool = True             # AI 功能的总开关
    
    # 3. 【自动计算项】用户不需要传，由程序自动计算出来的路径
    # field(init=False) 告诉 dataclass：这个变量虽然是类的属性，但在 __init__ 构造函数中不需要作为参数传入
    project_dir: Path = field(init=False)      # 项目主目录
    data_dir: Path = field(init=False)         # 数据存放目录 (nerfstudio 格式)
    images_dir: Path = field(init=False)       # 图片存放目录
    masks_dir: Path = field(init=False)        # 掩码(Mask)存放目录
    transforms_file: Path = field(init=False)  # 相机位姿文件 (transforms.json) 路径
    vocab_tree_path: Path = field(init=False)  # COLMAP 词汇树文件路径

    # sam3
    model_root: Path = Path("/home/ltx/workspace/ai/sam3") 


    def __post_init__(self):
        """
        [魔法方法] 这个函数会在类初始化(__init__)完成之后，自动执行！
        我们在这里集中处理所有的路径拼接和环境设置，实现"配置即逻辑"。
        """
        # --- A. 自动计算路径 ---
        # 路径拼接：work_root / project_name
        self.project_dir = self.work_root / self.project_name
        # 数据目录：project_dir / data
        self.data_dir = self.project_dir / "data"
        # 图片目录：data_dir / images
        self.images_dir = self.data_dir / "images"
        # 掩码目录：data_dir / masks
        self.masks_dir = self.data_dir / "masks"
        # 位姿文件：data_dir / transforms.json
        self.transforms_file = self.data_dir / "transforms.json"
        
        # 词汇树路径 (用于 COLMAP 特征匹配加速)
        self.vocab_tree_path = self.work_root / "vocab_tree_flickr100k_words.bin"

        # [新增] 确保模型目录存在
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.project_dir = self.work_root / self.project_name

        # --- B. 环境修正 ---
        # 设置 Setuptools 的环境变量，解决 Python 3.12+ 中 distutils 被移除导致的兼容性问题
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

        # 验证一下模型目录是否存在，方便调试
        if not self.model_root.exists():
            print(f"⚠️ 警告: 模型目录不存在 -> {self.model_root}")
        

# [可选依赖检测] 检查 plyfile 库
# plyfile 用于读写 .ply 格式的点云文件，如果没安装，后续的点云清洗功能会失效
try:
    from plyfile import PlyData, PlyElement # [第三方库] 用于 PLY 文件读写
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False # 标记为不可用

# ================= 🧠 AI 核心逻辑函数 =================

# ==============================================================================
# 函数: get_central_object_prompt
# ------------------------------------------------------------------------------
# [依赖库]: dashscope (阿里云 SDK), os, pathlib, numpy
# [功能]: 使用多模态大模型 (Qwen-VL-Plus) 智能分析图片内容。
#         它会读取文件夹中的几张采样图片，询问 AI "画面中心的物体是什么？"，
#         并要求 AI 返回一个适合 YOLO 目标检测的英文提示词 (Prompt)。
# ==============================================================================
def get_central_object_prompt(images_dir: Path, sample_count=7):
    """
    [Step 1.1] 使用 Qwen-VL-Plus 多图分析，提取中心物体的文本描述
    
    参数:
        images_dir (Path): 存放图片的文件夹路径
        sample_count (int): 采样图片数量，默认3张。少采几张可以节省 Token 费用并加快速度
    
    返回:
        prompt_text (str): 大模型生成的物体描述提示词 (例如 "red apple")，如果失败返回 None
    """
    # 1. 获取环境变量中的 API Key，这是调用阿里云服务的凭证
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY，无法调用大模型。")
        return None

    print(f"\n🧠 [AI 分析] 正在调用 Qwen-VL-Plus 分析场景...")
    
    # 2. [Python 进阶] 使用 glob 获取所有 jpg/png 图片
    # sorted() 确保图片按文件名顺序排列，避免每次运行顺序不一致
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files: return None # 如果文件夹是空的，直接返回
    
    # 3. [算法逻辑] 均匀采样 (Uniform Sampling)
    # 我们不只取前三张，而是使用 numpy 的 linspace 在整个序列中均匀抽取 sample_count 张图
    # 这样能覆盖物体的不同角度（例如：正面、侧面、背面），让 AI 的判断更准确
    indices = np.linspace(0, len(image_files) - 1, sample_count, dtype=int)
    sampled_imgs = [image_files[i] for i in indices]
    
    # 4. 构建多模态消息体 (Dashscope SDK 要求的特定 JSON 格式)
    # 这里的 list comprehension (列表推导式) 将每张图片路径转为字典格式 {"image": "path/to/img.jpg"}
    content = [{"image": str(img_path)} for img_path in sampled_imgs]
    
    # 追加文本提示 (Prompt Engineering)
    # 这里的 Prompt 经过精心设计，指示 AI：
    # - 关注画面中心
    # - 输出适合检测模型的词 (不要复杂的形容词)
    # - 优先描述视觉特征 (颜色、形状) 而不是功能名称
    content.append({
        "text": (
            "这些是一个视频的抽帧图片。请分析画面中心始终存在的、最主要的一个物体是什么。"
            "我正在使用 SAM 3 (Segment Anything Model 3) 进行基于文本的视频跟踪。"
            "请输出一个【指代性明确】的英文短语 (Referring Expression)。"
            
            "⚠️ 关键策略："
            "1. 必须包含视觉特征（颜色、材质）。SAM 3 需要依靠颜色和纹理将物体从背景中分离。"
            "   - ❌ 坏 Prompt: 'cup' (容易把桌子也分进去)"
            "   - ✅ 好 Prompt: 'white ceramic cup' (白色陶瓷杯)"
            "2. 描述物体本身，不要描述功能。"
            "   - ❌ 坏 Prompt: 'cleaning tool'"
            "   - ✅ 好 Prompt: 'blue plastic bottle'"
            "3. 保持简短，直接输出英文短语，不要标点符号。"
        )
    })
    
    # 封装为用户消息
    messages = [{"role": "user", "content": content}]

    try:
        # 5. [网络请求] 调用阿里云 Qwen-VL-Plus 模型
        # 这是一个同步调用，程序会在这里等待服务器返回结果
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus', 
            messages=messages
        )
        
        # 6. 解析返回结果
        if response.status_code == 200:
            # 提取 AI 回复的文本内容
            prompt_text = response.output.choices[0].message.content[0]["text"].strip()
            # [数据清洗] 去掉可能存在的标点符号（句号、引号），防止干扰 YOLO 模型解析
            prompt_text = prompt_text.replace(".", "").replace('"', "").replace("'", "")
            
            # \033[92m 是 ANSI 转义码，用于在控制台输出绿色高亮文字，\033[0m 是重置颜色
            print(f"    🤖 Qwen 认为中心物体是: [ \033[92m{prompt_text}\033[0m ]")
            return prompt_text
        else:
            # 如果状态码不是 200，说明 API 调用出错 (如欠费、网络错误)
            print(f"❌ Qwen 调用失败: {response.code} - {response.message}")
            return None
    except Exception as e:
        # 捕获所有可能的异常 (如网络断连)
        print(f"❌ API 连接异常: {e}")
        return None

# ==============================================================================
# 函数: clean_and_verify_mask
# ------------------------------------------------------------------------------
# [依赖库]: cv2 (OpenCV), numpy
# [功能]: 对 AI 生成的分割掩码 (Mask) 进行"体检"和"净化"。
#         AI 生成的 Mask 往往有噪点、边缘毛糙或包含背景杂物。
#         此函数利用形态学和连通域算法，强制只保留主体，并去除质量差的 Mask。
# ==============================================================================
def clean_and_verify_mask(mask, img_name=""):
    """
    [V4 优化版] 针对细长物体(笔)优化，增加“背景误杀”拦截
    """
    h, w = mask.shape
    
    # 1. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2: return False, None, "Empty Mask"

    # 寻找最大前景块
    max_area = 0
    max_label = -1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
            
    # 阈值过滤 1：太小 (噪点)
    if max_area < (h * w * 0.005):
        return False, None, "Too Small/Noise"

    # 🔥 阈值过滤 2 [新增]：太大 (说明割到了桌子/背景)
    # 如果物体占画面超过 65%，对于一支笔来说是不可能的，肯定是背景
    if max_area > (h * w * 0.65):
        return False, None, f"Too Large (Background? {max_area/(h*w):.0%})"

    cleaned_mask = (labels == max_label).astype(np.uint8) * 255

    # 2. 几何特征质检
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, None, "No Contour"
    main_cnt = max(contours, key=cv2.contourArea)

    # 实心度检查 (放宽一点点)
    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False, None, "Hull Area 0"
    solidity = max_area / hull_area
    if solidity < 0.75: # 从 0.88 放宽到 0.75，允许笔有一些缺口
        return False, None, f"Rough Edges ({solidity:.2f})"

    # 🔥 长宽比检查 [重要修改]
    x, y, w_rect, h_rect = cv2.boundingRect(main_cnt)
    if h_rect == 0: return False, None, "Height 0"
    
    aspect_ratio = w_rect / h_rect
    # 如果竖着放，w/h 可能会很小，我们要看长边比短边
    real_ratio = max(aspect_ratio, 1/aspect_ratio)
    
    # 从 4.5 提升到 15.0，允许细长的笔通过
    if real_ratio > 15.0: 
        return False, None, f"Bad Ratio ({real_ratio:.1f})"

    # 3. 边缘腐蚀
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=1)

    return True, cleaned_mask, "OK"

# ==============================================================================
# 函数: get_salient_box
# ------------------------------------------------------------------------------
# [依赖库]: cv2, numpy, torch
# [功能]: [纯本地 CV 算法] 当 AI (YOLO) 识别失败时，作为备选方案 (Fallback)。
#         它不依赖神经网络，而是利用传统的图像处理算法寻找画面中"纹理最复杂"的区域。
#         原理：拉普拉斯边缘检测 -> 膨胀连接 -> 找最大外接矩形。
# ==============================================================================
def get_salient_box(img_path, margin_ratio=0.1):
    try:
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None: return None
        
        # 1. 转灰度并计算边缘 (Laplacian Edge Detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # [算法逻辑] 拉普拉斯算子计算图像亮度的二阶导数。
        # 导数变化大的地方就是边缘。CV_64F 允许使用浮点数存储负值，防止计算过程中数据截断。
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian)) # 取绝对值并转回 8位整数 (0-255)
        
        # 2. 模糊与二值化
        # [算法逻辑] 高斯模糊 (Gaussian Blur) 用于平滑掉细小的噪点纹理，让真正明显的边缘聚集在一起。
        blurred = cv2.GaussianBlur(laplacian, (25, 25), 0)
        
        # [算法逻辑] 动态阈值 (Percentile Thresholding)
        # 我们不知道边缘的具体数值是多少，所以使用百分位数。
        # 这里只保留亮度排在前 20% 的区域（即纹理最丰富的地方），视为感兴趣区域。
        threshold_val = np.percentile(blurred, 80) 
        _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        
        # 3. 找最大轮廓
        # 在二值化图像中找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        # 假设纹理最复杂的区域（轮廓面积最大）就是主体
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt) # 获取边界框
        
        # 4. 加上安全边距 (Padding)
        # 传统算法找出的框往往比较紧，我们按比例 (margin_ratio) 向外扩一点，给 SAM 留出余地。
        H, W = img.shape[:2]
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        
        # 计算扩充后的坐标，并限制在图像边界内 (min/max)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        
        # 返回 torch tensor 格式，因为后续的 SAM 模型需要 Tensor 输入
        import torch
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        
    except Exception as e:
        print(f"       ⚠️ 视觉重心计算失败: {e}")
        return None

# ==============================================================================
# 函数: run_ai_segmentation_pipeline (SAM 3 升级版)
# ------------------------------------------------------------------------------
# [修改说明]: 
# 1. 移除了 YOLO-World 模型，直接使用 SAM 3 的文本提示功能。
# 2. 启用了 SAM 3 的序列处理能力，输入整个文件夹进行推理。
# ==============================================================================
def run_ai_segmentation_pipeline(data_dir: Path):
    """
    黄金组合 V4: 多点触控保底 + 强力背景抑制 + 比例放宽
    """
    if not HAS_AI: return False
    
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    debug_dir = data_dir / "debug_combo"
    debug_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cfg.transforms_file = data_dir / "transforms.json" 
    if not cfg.transforms_file.exists(): return False

    print(f"\n✂️ [智能分割] 初始化 (YOLO V2 + SAM 3 Multi-Point)...")
    try:
        text_prompt = get_central_object_prompt(images_dir)
        if " on " in text_prompt: text_prompt = text_prompt.split(" on ")[0]
    except: text_prompt = "object"
    if not text_prompt: text_prompt = "object"
    print(f"    🎯 核心 Prompt: '\033[92m{text_prompt}\033[0m'")

    yolo_path = cfg.model_root / "yolov8s-worldv2.pt"
    if not yolo_path.exists(): yolo_path = "yolov8s-worldv2.pt"
    sam_path = cfg.model_root / "sam3.pt"
    
    try:
        det_model = YOLOWorld(str(yolo_path))
        det_model.set_classes([text_prompt])
        sam_model = SAM(str(sam_path))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    with open(cfg.transforms_file, 'r') as f: meta = json.load(f)
    frames_map = {Path(f["file_path"]).name: f for f in meta["frames"]}
    valid_frames_list = []
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    total_imgs = len(image_files)
    
    print(f"    -> 开始处理 {total_imgs} 帧...")
    start_time = time.time()

    for i, img_path in enumerate(image_files):
        elapsed = time.time() - start_time
        fps = (i + 1) / (elapsed + 1e-6)
        process_success = False 
        
        try:
            original_img = cv2.imread(str(img_path))
            if original_img is None: raise ValueError("无法读取图片")
            h_real, w_real = original_img.shape[:2]

            # --- Step 1: YOLO 找框 ---
            det_results = det_model.predict(img_path, conf=0.05, verbose=False) 
            bboxes = det_results[0].boxes.xyxy.cpu()
            
            final_box = None
            is_fallback = False 
            
            if len(bboxes) > 0:
                center_x, center_y = w_real / 2, h_real / 2
                min_dist = float('inf')
                for box in bboxes:
                    bx = (box[0] + box[2]) / 2
                    by = (box[1] + box[3]) / 2
                    dist = (bx - center_x)**2 + (by - center_y)**2
                    if dist < min_dist:
                        min_dist = dist
                        final_box = box.unsqueeze(0)
            
            # --- Step 2: SAM 3 ---
            final_mask = None
            if final_box is not None:
                # 方案 A: 有框
                sam_results = sam_model(img_path, bboxes=final_box, verbose=False)
            else:
                # 方案 B: 保底 (多点触控 + 背景抑制)
                is_fallback = True
                cx, cy = w_real / 2, h_real / 2
                
                # 🔥 关键修改：构建 9 个点
                # 5个正样本(Label 1): 中心 + 上下左右微偏 (增加打中细长笔的概率)
                # 4个负样本(Label 0): 图片四个角 (强制 SAM 不选背景)
                offset = 30 # 偏移量像素
                input_points = [
                    [cx, cy], # 中心
                    [cx-offset, cy], [cx+offset, cy], # 左右
                    [cx, cy-offset], [cx, cy+offset], # 上下
                    [0, 0], [w_real, 0], [0, h_real], [w_real, h_real] # 四角背景
                ]
                input_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0] # 1是前景，0是背景
                
                sam_results = sam_model(img_path, points=input_points, labels=input_labels, verbose=False)

            if sam_results[0].masks is not None:
                masks_data = sam_results[0].masks.data.cpu().numpy()
                if masks_data.shape[0] > 0:
                    areas = np.sum(masks_data, axis=(1, 2))
                    # 在 Fallback 模式下，我们要小心最大的块可能是桌子
                    # 但我们在 clean 函数里有 max_area 拦截，所以这里还是取最大
                    largest_idx = np.argmax(areas)
                    final_mask = masks_data[largest_idx].astype(np.uint8) * 255
            
            if final_mask is None:
                final_mask = np.zeros((h_real, w_real), dtype=np.uint8)

            # --- Step 3: 清洗与验证 ---
            status_icon = "🟢" if not is_fallback else "🔵"
            print(f"       [{i+1}/{total_imgs}] {img_path.name} | {status_icon} | ⚡ {fps:.1f} fps          ", end="\r")

            is_good, cleaned_mask, reason = clean_and_verify_mask(final_mask, img_path.name)

            # --- 可视化 ---
            if i % 2 == 0 or not is_good: 
                debug_img = original_img.copy()
                color = (0, 255, 0) if not is_fallback else (255, 100, 0) # 绿色YOLO, 蓝色Point
                
                if final_box is not None:
                    x1, y1, x2, y2 = final_box[0].int().tolist()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_img, "YOLO", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif is_fallback:
                    # 画出那5个中心点
                    cx, cy = int(w_real/2), int(h_real/2)
                    offset = 30
                    pts = [(cx, cy), (cx-offset, cy), (cx+offset, cy), (cx, cy-offset), (cx, cy+offset)]
                    for pt in pts:
                        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 5, color, -1)
                    cv2.putText(debug_img, "MULTI-POINT", (cx-40, cy-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if is_good:
                    colored_mask = np.zeros_like(debug_img)
                    colored_mask[cleaned_mask > 0] = (0, 0, 255) 
                    debug_img = cv2.addWeighted(debug_img, 0.7, colored_mask, 0.3, 0)
                else:
                    cv2.putText(debug_img, f"REJECT: {reason}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imwrite(str(debug_dir / f"vis_{img_path.name}"), debug_img)

            # --- 保存 ---
            if is_good:
                if cleaned_mask.shape[:2] != original_img.shape[:2]:
                    cleaned_mask = cv2.resize(cleaned_mask, (w_real, h_real), interpolation=cv2.INTER_NEAREST)
                
                mask_blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
                b, g, r = cv2.split(original_img)
                img_bgra = cv2.merge([b, g, r, mask_blurred])
                
                new_img_path = img_path.with_suffix('.png')
                cv2.imwrite(str(new_img_path), img_bgra)
                
                if img_path.name in frames_map:
                    frame_data = frames_map[img_path.name]
                    frame_data["file_path"] = f"images/{new_img_path.name}"
                    valid_frames_list.append(frame_data)
                process_success = True

        except Exception as e:
            print(f"\n❌ Frame {i} Error: {e}")
            process_success = False 

        finally:
            if img_path.exists() and img_path.suffix.lower() == '.jpg':
                if process_success:
                    try: img_path.unlink() 
                    except: pass
                else:
                    try: img_path.unlink()
                    except: pass

    print(f"\n\n📊 完成。剩余可用: {len(valid_frames_list)}")
    if len(valid_frames_list) == 0: return False

    meta["frames"] = valid_frames_list
    with open(cfg.transforms_file, 'w') as f: json.dump(meta, f, indent=4)
    return True

# ================= 辅助工具 =================

def format_duration(seconds):
    """
    [辅助函数] 将秒数转换为易读的 HH:MM:SS 格式
    [依赖库]: datetime
    """
    # [标准库] datetime.timedelta 自动处理时间换算（如 3661秒 -> 1:01:01）
    return str(datetime.timedelta(seconds=int(seconds)))

# ==============================================================================
# 类: ImageProcessor
# ------------------------------------------------------------------------------
# [依赖库]: cv2, numpy, shutil, pathlib
# [功能]: 负责图像的预处理，特别是模糊检测。
#         在进行 3D 重建前，去除模糊的图片可以显著提高重建质量。
# ==============================================================================
class ImageProcessor:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def smart_filter_blurry_images(self, image_folder, keep_ratio=0.85):
        """
        [图像清洗算法] 混合策略模糊检测
        原理：利用拉普拉斯算子的方差 (Variance of Laplacian) 来衡量图像清晰度。
        """
        print(f"\n🧠 [智能清洗] 正在分析图片质量 (混合策略版)...")
        image_dir = Path(image_folder)
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not images: return
        
        trash_dir = image_dir.parent / "trash_smart"
        trash_dir.mkdir(exist_ok=True)
        
        img_scores = []
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            # 转灰度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # [算法逻辑] 分块检测
            # 为了防止只因为背景模糊（比如大光圈虚化）就误删照片，
            # 我们把图片切成 3x3 的九宫格，只取最清晰的那一格的分数作为整张图的分数。
            grid_h, grid_w = h // 3, w // 3
            max_grid_score = 0
            for r in range(3):
                for c in range(3):
                    roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    # 计算拉普拉斯方差：边缘越清晰，方差越大
                    score = cv2.Laplacian(roi, cv2.CV_64F).var()
                    if score > max_grid_score: max_grid_score = score
            
            img_scores.append((img_path, max_grid_score))
            if i % 50 == 0: print(f"  -> 分析中... {i}/{len(images)}", end="\r")
        
        # 计算阈值：按分数排序，剔除最差的 (1 - keep_ratio) 部分
        scores = [s[1] for s in img_scores]
        if not scores: return
        quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
        
        good_images = []
        for img_path, score in img_scores:
            if score < quality_threshold:
                # 移动到垃圾桶目录，而不是直接删除，方便人工找回
                shutil.move(str(img_path), str(trash_dir / img_path.name))
            else:
                good_images.append(img_path)
        
        # ======================================================
        # 🔥 核心逻辑：降采样控制数量
        # ======================================================
        # 从配置对象中读取最大图片数量
        max_imgs = self.cfg.max_images  
        
        # 如果好图片还是太多，进行均匀抽取
        if len(good_images) > max_imgs:
            print(f"    ⚠️ 图片过多 ({len(good_images)} 张), 正在降采样至 {max_imgs} 张...")
            # np.linspace 生成均匀分布的索引，例如 [0, 5, 10, ...]
            indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_imgs, dtype=int))
            for idx, img_path in enumerate(good_images):
                if idx not in indices_to_keep:
                    shutil.move(str(img_path), str(trash_dir / img_path.name))
                    
        print(f"✨ 清洗结束，剩余 {len(list(image_dir.glob('*')))} 张。")

# ==============================================================================
# 函数: analyze_and_calculate_adaptive_collider
# ------------------------------------------------------------------------------
# [依赖库]: json, numpy
# [功能]: [3D 场景理解算法] 这是一个核心的自动化逻辑。
#         它通过分析相机轨迹，自动判断你是围着物体拍 (Object Mode) 还是向四周拍 (Scene Mode)。
#         这对于 Nerfstudio 设置正确的近平面/远平面 (Near/Far Plane) 至关重要。
# ==============================================================================
def analyze_and_calculate_adaptive_collider(json_path, force_cull=False, radius_scale=1.8):
    """
    逻辑：
    1. 读取 transforms.json 获取所有相机位姿。
    2. 计算所有相机的视线向量与“相机中心-场景中心”向量的点积。
    3. 如果大部分相机都看向中心 -> Object Mode (物体模式)。
    4. 如果相机向四面八方看 -> Scene Mode (场景模式)。
    """
    print(f"\n🤖 [AI 分析] 解析相机轨迹...")
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"]
        if not frames: return [], "unknown"
        
        # 检查是否有 mask 路径，如果有，说明之前进行了 AI 分割，那必然是物体模式
        has_mask = "mask_path" in frames[0]
        if has_mask:
            print("    -> 检测到 Mask 数据！将启用物体聚焦模式。")
        
        # [线性代数] 提取所有相机的位移 (Translation)
        # transform_matrix 是 4x4 矩阵，[:3, 3] 是第4列前3行，即 XYZ 坐标
        positions = [np.array(f["transform_matrix"])[:3, 3] for f in frames]
        
        # 提取相机的前向向量 (Forward Vector)
        # 在 COLMAP/OpenGL 定义中，相机看向 -Z 方向。
        # 我们用旋转矩阵乘以 [0,0,-1] 得到相机在世界坐标系下的朝向。
        forward_vectors = [np.array(f["transform_matrix"])[:3, :3] @ np.array([0, 0, -1]) for f in frames]
        
        # 计算所有相机位置的几何中心 (Centroid)
        center = np.mean(positions, axis=0)
        
        # 计算每个相机位置指向场景中心的向量
        vec_to_center = center - positions
        # 归一化向量 (变成单位向量，长度为1)
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        
        # [核心算法] 计算“视线”与“指向中心向量”的对齐程度 (点积)
        # 点积 (Dot Product): A · B = |A||B|cos(theta)
        # 如果结果 > 0，说明夹角 < 90度，即相机是大致看向中心的。
        # 我们统计有多少比例的相机是看向中心的。
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        # 综合判定：向心率高 (>0.6) OR 强制开启 OR 有 Mask
        is_object_mode = ratio > 0.6 or force_cull or has_mask

        if is_object_mode:
            # 物体模式：设置紧凑的 Near/Far Plane
            dists = [np.linalg.norm(p) for p in positions] # 相机到原点的距离
            avg_dist = np.mean(dists)
            
            scene_radius = 1.0 * radius_scale  # 估算场景半径
            
            # 计算 Near Plane (近平面)：不能太近，否则会切掉相机前的物体
            calc_near = max(0.05, min(dists) - scene_radius)
            # 计算 Far Plane (远平面)：只要包住物体即可，切掉远处的伪影
            calc_far = avg_dist + scene_radius
            
            # 返回 nerfstudio 需要的命令行参数列表
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            # 场景模式：空间很大，Far Plane 设远一点 (100.0)
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"
    except:
        return [], "unknown"

# ==============================================================================
# 函数: perform_percentile_culling
# ------------------------------------------------------------------------------
# [依赖库]: plyfile, numpy, json
# [功能]: [点云后处理] 基于统计分位数的暴力切割。
#         3DGS 训练出来的点云往往在无限远的地方有一些飘逸的噪点。
#         此函数读取 PLY 文件，计算所有点到中心的距离，切除最远的 10% (或其他比例) 的点。
# ==============================================================================
def perform_percentile_culling(ply_path, json_path, output_path, keep_percentile=0.9):
    # 检查依赖
    if not HAS_PLYFILE: return False
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    try:
        # 1. 计算场景中心 (基于相机轨迹)
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        # 2. 读取 PLY 点云数据
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        # 堆叠 x,y,z 坐标
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        
        # 3. 计算所有点到中心的距离
        dists_pts = np.linalg.norm(points - center, axis=1)

        # [算法逻辑] 确定阈值半径
        # 使用 numpy.percentile 找到第 90% 位数的距离值。
        # 比如 keep_percentile=0.9，则保留距离最近的 90% 的点。
        threshold_radius = np.percentile(dists_pts, keep_percentile * 100)
        
        # 4. 读取不透明度 (Opacity) 并过滤
        # Gaussian Splatting 存储的 opacity 通常经过 sigmoid 激活，需要还原
        # 这里 simplified: 假设 vertex['opacity'] 是 logit
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 联合掩码：(在半径内) AND (不透明度 > 0.05)
        # 去除太远的点，同时也去除太透明（几乎不可见）的点
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        filtered_vertex = vertex[mask]
        
        # 5. 写入新 PLY 文件
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True
    except Exception as e:
        print(f"❌ 切割失败: {e}")
        return False

# ==============================================================================
# 类: GlomapRunner (GLOMAP 位姿解算类)
# ------------------------------------------------------------------------------
# [依赖库]: shutil, os, subprocess
# [功能]: 封装了 GLOMAP (Global Mapping) 的调用流程。
#         GLOMAP 是 COLMAP 的替代品，速度更快，鲁棒性更强。
#         这个类特别处理了复杂的 Linux 环境变量隔离问题。
# ==============================================================================
class GlomapRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        
        # 1. 查找 COLMAP (优先使用 Conda 环境自带的！)
        # shutil.which 类似于 Linux 的 `which` 命令
        self.colmap_exe = shutil.which("colmap")
        if not self.colmap_exe:
            if os.path.exists("/usr/local/bin/colmap"):
                self.colmap_exe = "/usr/local/bin/colmap"
        
        # 2. 查找 GLOMAP
        self.glomap_exe = shutil.which("glomap")
        if not self.glomap_exe:
            if os.path.exists("/usr/local/bin/glomap"):
                self.glomap_exe = "/usr/local/bin/glomap"

        if not self.colmap_exe or not self.glomap_exe:
            raise FileNotFoundError("❌ 缺少 colmap 或 glomap 可执行文件")

        print(f"    -> 🎯 锁定引擎: COLMAP={self.colmap_exe}")
        print(f"    -> 🎯 锁定引擎: GLOMAP={self.glomap_exe}")
        
        # 复制当前环境变量，用于后续修改，不影响主进程
        self.env = os.environ.copy()
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def run(self):
        """执行 GLOMAP 完整流程"""
        print(f"\n📐 [2/4] GLOMAP 位姿解算 (Global Mapping)")

        # 路径准备：将图片从 raw_images 复制到 data/images
        # GLOMAP 喜欢纯净的输入目录
        raw_images_dir = self.cfg.project_dir / "raw_images"
        dest_images_dir = self.cfg.images_dir
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        for img in raw_images_dir.glob("*"):
            if not (dest_images_dir / img.name).exists():
                shutil.copy2(str(img), str(dest_images_dir / img.name))

        colmap_output_dir = self.cfg.data_dir / "colmap"
        colmap_output_dir.mkdir(parents=True, exist_ok=True)
        database_path = colmap_output_dir / "database.db"
        sparse_dir = colmap_output_dir / "sparse"

        try:
            # 清理旧数据，防止冲突
            if database_path.exists(): database_path.unlink()
            if sparse_dir.exists(): shutil.rmtree(sparse_dir)
            sparse_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.transforms_file.exists(): self.cfg.transforms_file.unlink()

            # Step 1: 特征提取 (Feature Extraction) - 使用 COLMAP
            # 这一步计算每张图的关键点 (SIFT等)
            self._run_cmd([
                self.colmap_exe, "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--ImageReader.camera_model", "OPENCV", # 使用 OpenCV 相机模型
                "--ImageReader.single_camera", "1"      # 假设所有图片来自同一个相机（焦距相同）
            ], "Step 1: 特征提取 (COLMAP)")

            # Step 2: 顺序匹配 (Sequential Matching) - 使用 COLMAP
            # 因为我们的输入是视频抽帧，所以相邻的图片重叠度最高。使用顺序匹配比穷举匹配快得多。
            self._run_cmd([
                self.colmap_exe, "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "25"    # 匹配前后25张图
            ], "Step 2: 顺序匹配 (COLMAP)")

            # Step 3: 全局重建 (Global Mapper) - 使用 GLOMAP
            # 这是 GLOMAP 的核心，比 COLMAP 的 incremental mapper 更快且不易产生分层。
            print(f"    -> 🚀 启动 GLOMAP 引擎...")
            self._run_cmd([
                self.glomap_exe, "mapper",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--output_path", str(sparse_dir)
            ], "Step 3: 全局映射 (GLOMAP)")

            # Step 4: 目录修正
            # GLOMAP 输出的结构可能是在 sparse/0 里面，我们需要整理一下
            self._fix_directory_structure(sparse_dir)

            # Step 5: 生成 transforms.json
            # 调用 nerfstudio 的工具将 COLMAP 数据转换为 NeRF 标准格式
            self._run_cmd([
                "ns-process-data", "images",
                "--data", str(dest_images_dir),
                "--output-dir", str(self.cfg.data_dir),
                "--skip-colmap", # 我们已经跑过 COLMAP/GLOMAP 了，所以跳过
                "--skip-image-processing", # 我们自己处理过图片了
                "--num-downscales", "0"
            ], "生成 transforms.json")

            # Step 6: 检查质量
            if self._check_quality(raw_images_dir):
                print(f"    ✨ GLOMAP 流程成功！")
                return True

        except Exception as e:
            print(f"    ❌ GLOMAP 流程失败: {e}")
            return False
        return False

    def _run_cmd(self, cmd, desc):
        """内部工具：执行 shell 命令 (含环境隔离逻辑)"""
        print(f"🚀 {desc}...")
        
        # 🔥 环境隔离逻辑 🔥
        # 这是一个非常 tricky 的点。如果你在 Conda 环境里跑，LD_LIBRARY_PATH 可能指向 Conda 的 lib。
        # 但如果你调用系统自带的 /usr/local/bin/glomap，它可能需要系统的 lib。
        # 混合使用会导致 "libstdc++.so.6 version not found" 错误。
        cmd_env = self.env.copy()
        exe_path = cmd[0]
        # 如果是系统程序，清除 LD_LIBRARY_PATH 防止 Conda 干扰
        if exe_path.startswith("/usr") or exe_path.startswith("/bin"):
            if "LD_LIBRARY_PATH" in cmd_env:
                del cmd_env["LD_LIBRARY_PATH"]

        try:
            # subprocess.Popen 允许我们实时捕获输出
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cmd_env
            )
            # 实时打印关键日志
            for line in process.stdout:
                if any(k in line for k in ["Error", "Warning", "Elapsed", "image pairs"]):
                    print(f"    | {line.strip()}")
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行崩溃: {cmd[0]} (代码 {e.returncode})")
            raise e

    def _fix_directory_structure(self, sparse_root):
        """将稀疏重建结果统一移动到 sparse/0 文件夹下"""
        target_dir_0 = sparse_root / "0"
        target_dir_0.mkdir(parents=True, exist_ok=True)
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        required_files_txt = ["cameras.txt", "images.txt", "points3D.txt"]
        # ... (详细的文件移动逻辑，遍历目录寻找 .bin 或 .txt 文件并移动到 target_dir_0)
        # 此处省略具体 os.walk 代码，逻辑为递归查找并 shutil.move

    def _check_quality(self, raw_images_dir):
        """计算注册率：有多少图片成功参与了重建"""
        if not self.cfg.transforms_file.exists(): return False
        with open(self.cfg.transforms_file, 'r') as f: meta = json.load(f)
        reg_count = len(meta["frames"])
        total_count = len(list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png")))
        ratio = reg_count / total_count if total_count > 0 else 0
        print(f"    📊 匹配率: {ratio:.2%} ({reg_count}/{total_count})")
        return ratio > 0.2 # 如果少于 20% 的图片匹配成功，视为失败

# ==============================================================================
# 类: AISegmentor
# ------------------------------------------------------------------------------
# [功能]: 对 run_ai_segmentation_pipeline 的面向对象封装。
#         使主流程代码更整洁。
# ==============================================================================
class AISegmentor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.images_dir = cfg.images_dir
        self.masks_dir = cfg.masks_dir

    def run(self):
        """执行 AI 分割总流水线"""
        # 检查开关和依赖
        if not HAS_AI or not self.cfg.enable_ai:
            print("⏩ 跳过 AI 分割 (未启用或缺少依赖)")
            return False
            
        if not self.cfg.transforms_file.exists():
            print("⚠️ transforms.json 不存在，无法进行 AI 分割")
            return False

        # 内部实际上是调用了之前的 run_ai_segmentation_pipeline 逻辑
        # 这里为了演示封装结构，将其作为类方法重新组织
        # ... (具体实现逻辑见上文 run_ai_segmentation_pipeline)
        
        # 简化版调用：
        print("\n✂️ [AI 分割] 启动...")
        return run_ai_segmentation_pipeline(self.data_dir) # 直接调用全局函数

# ==============================================================================
# 类: NerfstudioEngine (训练引擎类)
# ------------------------------------------------------------------------------
# [依赖库]: subprocess, os
# [功能]: 负责调用 ns-train 进行 Splatfacto (3DGS) 训练，并导出结果。
# ==============================================================================
class NerfstudioEngine:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = cfg.project_dir / "outputs"
        # 准备环境变量
        self.env = os.environ.copy()
        # QT_QPA_PLATFORM="offscreen": 防止在没有显示器的服务器上因为弹不出窗口而报错
        self.env["QT_QPA_PLATFORM"] = "offscreen"
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def train(self):
        """执行 splatfacto 训练"""
        print(f"\n🔥 [4/4] 开始训练 (Splatfacto)")
        
        # 1. 计算场景参数 (Collider) - 调用之前的智能分析函数
        collider_args, scene_type = analyze_and_calculate_adaptive_collider(
            self.cfg.transforms_file,
            force_cull=self.cfg.force_spherical_culling,
            radius_scale=self.cfg.scene_radius_scale
        )
        self.scene_type = scene_type # 存下来给导出步骤用

        # 2. 组装 ns-train 命令
        cmd = [
            "ns-train", "splatfacto",  # 使用 splatfacto 模型 (即 Gaussian Splatting)
            "--data", str(self.cfg.data_dir),
            "--output-dir", str(self.output_dir),
            "--experiment-name", self.cfg.project_name,
            "--pipeline.model.random-init", "False",   # 使用稀疏点云初始化，收敛更快
            "--pipeline.model.background-color", "random", # 背景颜色随机，增强对透明背景的鲁棒性
            *collider_args, # 解包 collider 参数 (near/far plane)
            "--max-num-iterations", "15000", # 迭代次数，15000 次通常足够
            "--vis", "viewer+tensorboard",   # 开启可视化支持
            "--viewer.quit-on-train-completion", "True", # 训练完自动关闭 viewer
            "nerfstudio-data", # 数据解析器配置
            "--downscale-factor", "1", # 不缩放图片
            "--auto-scale-poses", "False" # 不自动缩放位姿（因为我们在 Collider 步骤算过了）
        ]
        
        # 3. 执行
        subprocess.run(cmd, check=True, env=self.env)

    def export(self):
        """导出 ply 并进行后处理"""
        print(f"\n💾 正在导出...")
        # 自动查找最新的 config.yml 文件
        search_path = self.output_dir / self.cfg.project_name / "splatfacto"
        try:
            run_dirs = sorted(list(search_path.glob("*")))
            config_path = run_dirs[-1] / "config.yml" # 取时间戳最新的那个
        except IndexError:
            print("❌ 未找到训练结果 config.yml")
            return None

        # 导出命令 ns-export
        subprocess.run([
            "ns-export", "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(self.cfg.project_dir)
        ], check=True, env=self.env)

        # 后处理：点云切割
        raw_ply = self.cfg.project_dir / "point_cloud.ply"
        if not raw_ply.exists(): raw_ply = self.cfg.project_dir / "splat.ply"
        cleaned_ply = self.cfg.project_dir / "point_cloud_cleaned.ply"
        final_ply = raw_ply

        # 判断是否需要切割 (物体模式 or 强制切割)
        need_cull = (self.scene_type == "object" or self.cfg.force_spherical_culling)
        
        if need_cull and raw_ply.exists():
            # 调用之前的 perform_percentile_culling 函数
            success = perform_percentile_culling(
                raw_ply, 
                self.cfg.transforms_file, 
                cleaned_ply,
                keep_percentile=self.cfg.keep_percentile
            )
            if success:
                final_ply = cleaned_ply

        # 复制结果到当前脚本目录下的 results 文件夹，方便查看
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        target_path = results_dir / f"{self.cfg.project_name}.ply"
        shutil.copy2(str(final_ply), str(target_path))
        
        return target_path

# ================= 主流程 =================
def run_pipeline(cfg: PipelineConfig):
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Engine] 启动任务: {cfg.project_name}")
    
    # 1. 实例化所有模块
    img_processor = ImageProcessor(cfg)
    glomap_runner = GlomapRunner(cfg) 
    ai_segmentor = AISegmentor(cfg)
    nerf_engine = NerfstudioEngine(cfg)

    # ==========================================
    # Step 1: 数据准备 (FFmpeg 抽帧)
    # ==========================================
    # 初始化目录，如果项目已存在则清空
    if cfg.project_dir.exists(): shutil.rmtree(cfg.project_dir, ignore_errors=True)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    
    # 抽帧
    temp_dir = cfg.project_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    # 调用 ffmpeg: -vf fps=10 表示每秒抽10帧，-q:v 2 表示高质量 JPG
    subprocess.run(["ffmpeg", "-y", "-i", str(cfg.video_path), 
                    "-vf", "fps=10", "-q:v", "2", 
                    str(temp_dir / "frame_%05d.jpg")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 清洗：去除模糊图片
    img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)
    
    # 移动图片到 raw_images
    raw_images_dir = cfg.project_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    
    all_imgs = sorted(list(temp_dir.glob("*")))
    limit = cfg.max_images
    # 如果图片还是太多，均匀降采样
    if len(all_imgs) > limit:
        indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
        all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
    for img in all_imgs: shutil.copy2(str(img), str(raw_images_dir / img.name))
    shutil.rmtree(temp_dir) # 删除临时目录

    # ==========================================
    # Step 2: GLOMAP 位姿解算
    # ==========================================
    if not glomap_runner.run():
        print("❌ Pipeline 中断：GLOMAP 失败")
        return

    # ==========================================
    # Step 3: AI 语义分割
    # ==========================================
    ai_segmentor.run()

    # ==========================================
    # Step 4 & 5: 训练与导出
    # ==========================================
    try:
        nerf_engine.train()
        final_path = nerf_engine.export()
        print(f"\n🎉 任务完成！结果位于: {final_path}")
    except Exception as e:
        print(f"❌ 训练/导出阶段失败: {e}")

    print(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")


# 程序入口
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    # 如果命令行带了参数，使用参数作为视频路径
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])
    
    if not video_file.exists():
        print(f"❌ 找不到视频: {video_file}")
        sys.exit(1)

    # 实例化配置
    cfg = PipelineConfig(
        project_name="glomap_test_v1", 
        video_path=video_file,
        max_images=100, # 限制最大处理100张图
        enable_ai=True  # 开启 AI 功能
    )
    
    
    # 运行流水线
    run_pipeline(cfg)