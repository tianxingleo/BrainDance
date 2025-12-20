# ==============================================================================
# 导入标准库和第三方库
# ==============================================================================
import subprocess  # [标准库] 用于执行外部系统命令（如 ffmpeg, colmap），实现 Python 与操作系统的交互
import sys         # [标准库] 用于访问与 Python 解释器紧密相关的变量和函数，如 sys.argv, sys.path
import shutil      # [标准库] 高级文件操作库，用于复制、移动、删除文件和目录
import os          # [标准库] 提供操作系统接口，用于路径操作、环境变量获取等
import time        # [标准库] 用于时间处理，计算代码运行耗时
import datetime    # [标准库] 用于处理日期和时间格式
from pathlib import Path  # [Python 进阶] 面向对象的文件系统路径库，比 os.path 更优雅、易用
import json        # [标准库] 用于读写 JSON 格式文件（如 transforms.json 相机参数文件）
import numpy as np # [第三方库] 科学计算库，用于处理矩阵、图像数组（HWC格式）
import logging     # [标准库] 日志系统，用于控制控制台输出级别
import cv2         # [第三方库] OpenCV，用于图像处理（读取、写入、形态学操作、轮廓查找等）
import re          # [标准库] 正则表达式，虽在开头引入但前350行暂未显式用到复杂的正则

# ================= 🧠 AI 依赖引入 =================
# [工程化思路] 软依赖导入：不要因为缺失非核心功能的库而导致整个程序崩溃。
# 这里使用了 try-except 块来检测是否安装了 AI 相关的库。
try:
    import dashscope  # [第三方库] 阿里云百炼 SDK，用于调用 Qwen-VL 多模态大模型
    from dashscope import MultiModalConversation # 具体导入多模态对话类
    from ultralytics import SAM, YOLOWorld # [第三方库] Ultralytics 库，封装了 YOLO（目标检测）和 SAM（分割万物）
    HAS_AI = True     # [变量] 标记位，用于后续逻辑判断是否启用 AI 功能
except ImportError:
    HAS_AI = False
    # [用户交互] 友好的错误提示，告知用户缺失了什么以及如何修复
    print("⚠️ [环境警告] 未检测到 dashscope 或 ultralytics 库。")
    print("    -> 智能分割功能将被禁用。请运行: pip install dashscope ultralytics")

# 🔥 请在此处填入你的 API KEY (或者确保环境变量 DASHSCOPE_API_KEY 已存在)
# os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ================= 🔧 基础配置 =================
# [工程化思路] 环境变量优先级管理
# 很多服务器上系统自带的 colmap 版本较老，这里强制将用户编译的高版本路径提到 PATH 环境变量的最前面
# sys_path = "/usr/local/bin" # [变量] 指定高优先级二进制文件路径
# current_path = os.environ.get("PATH", "") # 获取当前 PATH
# # 检查 sys_path 是否已经在 PATH 的首位
# if sys_path not in current_path.split(os.pathsep)[0]:
#     print(f"⚡ [环境修正] 强制设置 PATH 优先级: {sys_path} -> Priority High")
#     # 拼接新的 PATH，将 sys_path 放在最前面
#     os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"

# 设置日志级别
# 屏蔽 nerfstudio 库中非 Error 级别的日志，防止控制台被刷屏
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 






# # 工作区配置
# # [Python 进阶] 使用 Path.home() 获取用户主目录，实现跨平台（Windows/Linux）兼容
# LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
# # 🔥 新增：词汇树文件路径，COLMAP 进行特征匹配时需要的预训练数据
# VOCAB_TREE_PATH = LINUX_WORK_ROOT / "vocab_tree_flickr100k_words.bin" 
# SCENE_RADIUS_SCALE = 1.8  # [变量] 场景半径缩放因子，用于计算相机近平面/远平面
# MAX_IMAGES = 180          # [变量] 限制最大处理图片数量，防止显存爆炸或训练时间过长

# # 切割配置
# FORCE_SPHERICAL_CULLING = True # [变量] 是否强制使用球形裁剪（保留中心物体，切除周围杂景）
# KEEP_PERCENTILE = 0.9          # [变量] 保留 90% 的点云密度，去除离群点
from dataclasses import dataclass, field
from pathlib import Path
import os
import sys

@dataclass
class PipelineConfig:
    # 1. 【必填项】用户初始化时必须给我的
    project_name: str
    video_path: Path
    
    # 2. 【选填项】有默认值的配置 (对应你原代码的全局变量)
    work_root: Path = Path.home() / "braindance_workspace"
    max_images: int = 180
    force_spherical_culling: bool = True 
    scene_radius_scale: float = 1.8
    keep_percentile: float = 0.9
    enable_ai: bool = True  # 新增控制开关
    
    # 3. 【自动计算项】用户不用传，我自己算出来的路径
    # field(init=False) 的意思是：这个变量存在，但在初始化(__init__)时不需要作为参数传入
    project_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    images_dir: Path = field(init=False)
    masks_dir: Path = field(init=False)
    transforms_file: Path = field(init=False)
    vocab_tree_path: Path = field(init=False)

    def __post_init__(self):
        """
        这个函数会在类初始化完成之后，自动执行！
        我们在这里集中处理所有的路径拼接和环境设置。
        """
        # --- A. 自动计算路径 (再也不用在主函数里写一遍了) ---
        self.project_dir = self.work_root / self.project_name
        self.data_dir = self.project_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.transforms_file = self.data_dir / "transforms.json"
        
        # 词汇树路径 (对应原代码 VOCAB_TREE_PATH)
        self.vocab_tree_path = self.work_root / "vocab_tree_flickr100k_words.bin"

        # --- B. 环境修正 (对应原代码的 PATH 设置逻辑) ---
        # 把设置环境变量的逻辑搬到这里，保证 config 一加载，环境就是对的
        # sys_path = "/usr/local/bin"
        # current_path = os.environ.get("PATH", "")
        # if sys_path not in current_path.split(os.pathsep)[0]:
        #     print(f"⚡ [Config] 自动优化 PATH 优先级: {sys_path}")
        #     os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"
            
        # 设置 Setuptools 修复 (对应原代码 env["SETUPTOOLS_USE_DISTUTILS"])
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

# 检查 plyfile 库
# plyfile 用于读写 .ply 点云文件
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

# ================= 🧠 AI 核心逻辑函数 =================

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

def clean_and_verify_mask(mask, img_name=""):
    """
    [净化版] Mask 后处理核心算法
    功能：
    1. 强制清洗：只保留画面中最大的连通块 (去除孤立噪点)。
    2. 严格质检：清洗后如果形状依然毛糙(粘连阴影)，则剔除。
    3. 边缘腐蚀：向内收缩 Mask，去除边缘杂色。
    
    参数:
        mask (numpy array): 单通道二值图像 (0或255)
        img_name (str): 用于日志输出的文件名
        
    返回:
        tuple: (是否合格 bool, 清洗后的干净Mask, 原因 str)
    """
    h, w = mask.shape # 获取图像高宽
    
    # --- 1. 连通域分析 & 强制清洗 (Cleaning) ---
    # [算法逻辑] 连通组件分析 (Connected Components)
    # 这里的 connectivity=8 表示判断像素相连时考虑周围8个方向
    # stats 包含每个连通块的 [左上角x, 左上角y, 宽, 高, 面积]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # num_labels 至少为 2 (背景0 + 至少一个前景块)，如果小于2说明全是黑的
    if num_labels < 2: 
        return False, None, "空蒙版"

    # [算法逻辑] 寻找最大前景块
    # 遍历所有标签（从1开始，跳过0背景），找到面积最大的那个
    max_area = 0
    max_label = -1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
            
    # [工程化思路] 阈值过滤：如果最大的块占比不到全图的 0.5%，通常是噪点
    if max_area < (h * w * 0.005):
        return False, None, "主体过小，疑似噪点"

    # 🔥 核心操作：重构 Mask
    # 只保留 label 等于 max_label 的像素，其余置为 0。
    # 这步操作能完美去除周围的飞溅噪点。
    cleaned_mask = (labels == max_label).astype(np.uint8) * 255

    # --- 2. 对清洗后的 Mask 进行“体检” (Verification) ---
    
    # [算法逻辑] 轮廓提取
    # RETR_EXTERNAL 只取最外层轮廓
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, None, "清洗后无轮廓"
    
    # 取最大轮廓
    main_cnt = max(contours, key=cv2.contourArea)
    
    # [算法逻辑] 实心度 (Solidity) 计算
    # 凸包 (Convex Hull) 像是用橡皮筋包住物体的形状。
    # 实心度 = 轮廓面积 / 凸包面积。
    # 正常物体实心度高 (~0.95)，如果有粘连阴影，轮廓会很不规则，实心度会降低。
    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False, None, "凸包面积为0"
    
    solidity = max_area / hull_area
    
    # 阈值设定：0.88 (经验值，低于此值通常意味着边缘非常毛糙或有粘连)
    if solidity < 0.88:
        return False, None, f"边缘严重毛糙/粘连阴影 (实心度 {solidity:.2f})"

    # [算法逻辑] 长宽比检查 (Aspect Ratio)
    # 防止把长条形的桌子缝隙、墙角线当成物体
    x, y, w_rect, h_rect = cv2.boundingRect(main_cnt)
    aspect_ratio = w_rect / h_rect
    if aspect_ratio > 4.5: # 允许一定程度的长条，但超过 4.5 倍就太夸张了
        return False, None, f"形状异常 (长宽比 {aspect_ratio:.1f})"

    # 🔥 新增：边缘腐蚀 (Erosion)
    # [算法逻辑] 腐蚀操作
    # 卷积核 kernel 在图像上滑动，只有核内全为 255 时才保留中心点。
    # 效果是让白色区域向内收缩，切掉物体边缘可能存在的“光晕”或背景杂色。
    kernel_size = 3  # 3x3 的核，大约收缩 1 像素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=1)
    
    return True, cleaned_mask, "合格"

def get_salient_box(img_path, margin_ratio=0.1):
    """
    [纯本地 CV 算法] 当 AI 失败时，使用传统视觉算法计算'视觉显著区域'。
    原理：利用拉普拉斯算子找边缘 -> 膨胀连接 -> 找最大外接矩形
    
    参数:
        img_path: 图片路径
        margin_ratio: 结果框的扩边比例 (padding)
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None: return None
        
        # 1. 转灰度并计算边缘 (Laplacian)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # [算法逻辑] 拉普拉斯算子计算二阶导数，对边缘极其敏感
        # CV_64F 允许负值，防止截断
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian)) # 取绝对值转回 uint8
        
        # 2. 模糊与二值化
        # [算法逻辑] 高斯模糊用于平滑纹理，让零散的边缘聚集
        blurred = cv2.GaussianBlur(laplacian, (25, 25), 0)
        # [算法逻辑] 动态阈值：只保留亮度前 20% 的区域（即纹理最丰富的地方）
        threshold_val = np.percentile(blurred, 80) 
        _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        
        # 3. 找最大轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        # 假设纹理最复杂的区域就是主体
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)
        
        # 4. 加上安全边距 (Padding)
        H, W = img.shape[:2]
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        
        # 限制坐标不超出图像边界
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        
        # 返回 torch tensor 格式，适配 YOLO/SAM 的输入要求
        import torch
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        
    except Exception as e:
        print(f"       ⚠️ 视觉重心计算失败: {e}")
        return None

def run_ai_segmentation_pipeline(data_dir: Path):
    """
    [Step 1.2] 执行 AI 分割总流水线
    逻辑流程：
    1. 尝试用 Qwen 分析物体 -> 得到 Prompt
    2. 加载 YOLO 和 SAM 模型
    3. 遍历每一张图：
        a. YOLO 根据 Prompt 找框
        b. 如果有多个框，选最中心的
        c. 如果没框，用 SAM 中心点模式
        d. SAM 生成 Mask
        e. clean_and_verify_mask 清洗 Mask
        f. 如果合格 -> 生成透明 PNG，更新 transforms.json
        g. 如果不合格 -> 删除图片
    """
    if not HAS_AI: return False # 如果缺少库，直接跳过
    
    # 路径定义
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    cfg.transforms_file = data_dir / "transforms.json" # COLMAP 生成的相机位姿文件

    if not cfg.transforms_file.exists():
        print("⚠️ 未找到 transforms.json，无法进行 Mask 处理。")
        return False

    # ================= 核心修改逻辑开始 =================
    print(f"\n✂️ [AI 分割] 正在初始化...")

    # --- 第一层：尝试调用大模型获取精准 Prompt ---
    text_prompt = None
    try:
        # 调用之前定义的函数
        text_prompt = get_central_object_prompt(images_dir)
    except Exception as e:
        print(f"    ⚠️ 大模型调用出错: {e}")

    # --- 第二层：如果大模型失败，使用通用 Prompt ---
    if not text_prompt:
        # [工程化思路] 降级策略 (Fallback)
        # 如果大模型挂了，或者 API 没钱了，使用通用词 "central object"
        text_prompt = "central object; single object"
        print(f"    ⚠️ 未能获取精准描述，降级使用通用 Prompt: '{text_prompt}'")
    else:
        print(f"    🎯 获取到精准 Prompt: '\033[92m{text_prompt}\033[0m'")

    masks_dir.mkdir(parents=True, exist_ok=True) # 创建 Mask 目录
    # ================= 核心修改逻辑结束 =================

    # 2. 加载模型 (推荐用 Large)
    print("    -> 正在加载 SAM 2 Large 模型...")
    
    # 🔥 自动迁移 AI 模型文件
    # [工程化思路] 自动部署：脚本运行时检查当前目录是否有模型文件，如果有则复制到工作区
    model_files = ["yolov8s-worldv2.pt", "sam2.1_l.pt"]
    for model_name in model_files:
        target_model_path = cfg.work_root / model_name
        local_model_path = Path(__file__).parent / model_name # 脚本所在目录
        
        if not target_model_path.exists():
            if local_model_path.exists():
                print(f"    📦 检测到本地模型 {model_name}，正在迁移至工作区...")
                shutil.copy2(str(local_model_path), str(target_model_path))
            else:
                # 如果都没有，Ultralytics 库会在调用时自动下载
                print(f"    ⚠️ 未在脚本目录找到 {model_name}，将尝试自动下载...")

    try:
        # 定义模型路径
        yolo_path = cfg.work_root / "yolov8s-worldv2.pt"
        sam_path = cfg.work_root / "sam2.1_l.pt"
        
        # YOLO-World: 开放词汇检测模型，能“听懂”文字并找到框
        det_model = YOLOWorld(str(yolo_path) if yolo_path.exists() else "yolov8s-worldv2.pt") 
        # 设置 YOLO 需要寻找的类别
        det_model.set_classes([text_prompt])
        
        # SAM 2: 分割模型，根据框（Box Prompt）或点（Point Prompt）抠图
        sam_model = SAM(str(sam_path) if sam_path.exists() else "sam2.1_l.pt") 
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    # 3. 读取 transforms.json
    # [Python 进阶] 读取 JSON 到字典
    with open(cfg.transforms_file, 'r') as f:
        meta = json.load(f)
    
    # [Python 进阶] 建立哈希映射 (HashMap / Dict)
    # 将文件名映射到帧数据对象，后续查找复杂度为 O(1)，避免遍历列表
    # Path(f["file_path"]).name 提取如 "frame_0001.jpg"
    frames_map = {Path(f["file_path"]).name: f for f in meta["frames"]}
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    total_imgs = len(image_files)
    
    valid_frames_list = [] # 存放清洗合格后的帧数据
    deleted_count = 0
    
    print(f"    -> 开始处理 {total_imgs} 张图片...")

    # [Python 进阶] enumerate 用于同时获取索引 i 和元素 img_path
    for i, img_path in enumerate(image_files):
        # --- A. 检测与分割 ---
        try:
            # 1. YOLO 检测
            # conf=0.05: 只要置信度大于 5% 就认为可能有东西
            det_results = det_model.predict(img_path, conf=0.05, verbose=False)
            
            # ============================================================
            # 🕵️‍♂️ [DEBUG 模式] 调试可视化代码块
            # ============================================================
            debug_dir = data_dir / "debug_yolo_visuals"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            num_boxes = len(det_results[0].boxes)
            
            if num_boxes > 0:
                plotted_img = det_results[0].plot() # YOLO 自带画图功能
                debug_path = debug_dir / f"debug_{img_path.name}"
                cv2.imwrite(str(debug_path), plotted_img)
                
                if i < 3: # 只打印前3张，防止刷屏
                    print(f"\n    👀 [DEBUG] {img_path.name}: 找到了 {num_boxes} 个目标")
            # ============================================================

            # 获取检测框坐标 (xyxy格式: xmin, ymin, xmax, ymax)
            bboxes = det_results[0].boxes.xyxy.cpu() 

            # ============================================================
            # 🔥 核心修改：多目标筛选 (只取最中间的一个)
            # ============================================================
            if len(bboxes) > 1:
                import torch
                # 获取原图尺寸
                img_h, img_w = det_results[0].orig_shape[:2]
                # 计算屏幕中心坐标
                screen_center = torch.tensor([img_w / 2.0, img_h / 2.0])
                
                min_dist = float('inf') # 初始化最小距离为无穷大
                best_idx = 0
                
                # 遍历每个框，计算其中心点到屏幕中心的欧氏距离
                for idx, box in enumerate(bboxes):
                    box_center_x = (box[0] + box[2]) / 2.0
                    box_center_y = (box[1] + box[3]) / 2.0
                    
                    dist = torch.sqrt((box_center_x - screen_center[0])**2 + (box_center_y - screen_center[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                
                # 更新 bboxes，只保留最中心的一个
                # unsqueeze(0) 用于保持维度为 [1, 4]，而不是变成 [4]
                bboxes = bboxes[best_idx].unsqueeze(0) 

            # ============================================================
            # 🔥 核心修改：从“死框”改为“智能中心点扩散”
            # ============================================================
            
            use_point_prompt = False
            
            # 如果 YOLO 没找到任何东西
            if len(bboxes) == 0:
                print(f"       ⚠️ YOLO 未识别到物体，切换为 [SAM 中心点模式]")
                h, w = det_results[0].orig_shape[:2]
                
                # [算法逻辑] 盲猜中心：假设物体在画面正中央
                # 给 SAM 一个中心点提示，让它尝试向外扩散分割
                use_point_prompt = True
            
            # 3. 执行 SAM 分割
            if use_point_prompt:
                # 构造一个位于中心的极小框，模拟点击效果
                cx, cy = w / 2, h / 2
                margin = 5 
                bboxes = torch.tensor([[cx-margin, cy-margin, cx+margin, cy+margin]], device=det_model.device)
                
                # 调用 SAM
                sam_results = sam_model(img_path, bboxes=bboxes, verbose=False)
            else:
                # 使用 YOLO 确定的框调用 SAM
                sam_results = sam_model(img_path, bboxes=bboxes, verbose=False)
            
            # 获取 SAM 结果
            if sam_results[0].masks is not None:
                # masks.data 是 [N, H, W] 的 tensor
                all_masks = sam_results[0].masks.data.cpu().numpy()
                # [Python 进阶] np.any(axis=0) 将所有检测到的 mask 合并（逻辑或）
                final_mask = np.any(all_masks, axis=0).astype(np.uint8) * 255
            else:
                final_mask = np.zeros(det_results[0].orig_shape[:2], dtype=np.uint8)

            # -------------------------------------------------
            # 🔥 核心修改：调用清洗函数进行质检 🔥
            # -------------------------------------------------
            # 这里调用了前面定义的 clean_and_verify_mask
            is_good, cleaned_mask, reason = clean_and_verify_mask(final_mask, img_path.name)

            if is_good:
                # ✅ 合格逻辑
                original_img = cv2.imread(str(img_path))
                if original_img is not None:
                    # [算法逻辑] 羽化 (Feathering)
                    # 对 Mask 进行高斯模糊，使边缘半透明，避免合成时出现锯齿
                    mask_blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
                    
                    # 归一化 Alpha 通道 (0.0 - 1.0)
                    alpha_channel = mask_blurred.astype(np.float32) / 255.0
                    img_float = original_img.astype(np.float32)
                    
                    # [算法逻辑] 预乘 Alpha (Premultiplied Alpha)
                    # 标准的图形学操作：RGB = RGB * Alpha
                    # 这样背景区域就会变成纯黑 (0,0,0)
                    b, g, r = cv2.split(img_float)
                    b = b * alpha_channel
                    g = g * alpha_channel
                    r = r * alpha_channel
                    
                    # 合并通道生成 BGRA 图片
                    img_bgra = cv2.merge([
                        b.astype(np.uint8), 
                        g.astype(np.uint8), 
                        r.astype(np.uint8), 
                        mask_blurred # Alpha 通道
                    ])
                    
                    # 保存为 PNG (JPG 不支持透明通道)
                    new_img_path = img_path.with_suffix('.png')
                    cv2.imwrite(str(new_img_path), img_bgra)
                    
                    # 删除旧的 JPG 文件以节省空间和避免混淆
                    if img_path.suffix.lower() == '.jpg':
                        try: img_path.unlink()
                        except: pass
                        
                    final_img_path_name = new_img_path.name
                else:
                    final_img_path_name = img_path.name

                # 保存 Mask 供调试或训练使用
                cv2.imwrite(str(masks_dir / f"{img_path.stem}.png"), cleaned_mask)

                # 更新 JSON 元数据
                if img_path.name in frames_map:
                    frame_data = frames_map[img_path.name]
                    # 修改文件路径指向新的 PNG
                    frame_data["file_path"] = f"images/{final_img_path_name}" 
                    valid_frames_list.append(frame_data)

            else:
                # ❌ 不合格逻辑
                print(f"       🗑️ [剔除] {img_path.name}: {reason}")
                # [工程化思路] 物理删除质量差的数据，防止进入训练流程污染模型
                img_path.unlink() 
                deleted_count += 1
                # valid_frames_list 中不添加该帧，相当于在 transforms.json 中也删除了

        except Exception as e:
            # 异常处理：单个图片处理失败不影响整体流程
            print(f"       ❌ 处理出错 {img_path.name}: {e}")
            continue

        # 进度条打印
        if i % 10 == 0:
            # end="\r" 实现单行刷新
            print(f"       进度: {i}/{total_imgs} (已剔除 {deleted_count} 张)...", end="\r")

    # 4. 结算与更新
    print(f"\n\n📊 筛选报告:")
    print(f"   - 原始总数: {total_imgs}")
    print(f"   - 剔除数量: {deleted_count} ({deleted_count/total_imgs:.1%})")
    print(f"   - 剩余可用: {len(valid_frames_list)}")

    if len(valid_frames_list) == 0:
        print("❌ 错误：所有图片都被剔除了！请检查提示词或拍摄质量。")
        return False

    # 5. 重写 transforms.json
    # [关键步骤] 用清洗后的 valid_frames_list 覆盖原始数据
    # 这样 Nerfstudio 训练时就只会读取到干净、带有透明通道的图片
    meta["frames"] = valid_frames_list
    with open(cfg.transforms_file, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"    ✅ transforms.json 已更新，数据集已清洗完毕。")
    return True

# ================= 辅助工具 =================

def format_duration(seconds):

    """
    [辅助函数] 将秒数转换为易读的 HH:MM:SS 格式
    """
    # [标准库] datetime.timedelta 自动处理时间换算（如 3661秒 -> 1:01:01）
    return str(datetime.timedelta(seconds=int(seconds)))

class ImageProcessor:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def smart_filter_blurry_images(self, image_folder, keep_ratio=0.85):
        """
        [图像清洗算法] 混合策略模糊检测
        """
        print(f"\n🧠 [智能清洗] 正在分析图片质量 (混合策略版)...")
        image_dir = Path(image_folder)
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not images: return
        
        trash_dir = image_dir.parent / "trash_smart"
        trash_dir.mkdir(exist_ok=True)
        
        # --- 这里的代码保持不变，直到 good_images 计算完毕 ---
        img_scores = []
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            grid_h, grid_w = h // 3, w // 3
            max_grid_score = 0
            for r in range(3):
                for c in range(3):
                    roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    score = cv2.Laplacian(roi, cv2.CV_64F).var()
                    if score > max_grid_score: max_grid_score = score
            img_scores.append((img_path, max_grid_score))
            if i % 50 == 0: print(f"  -> 分析中... {i}/{len(images)}", end="\r")
        
        scores = [s[1] for s in img_scores]
        if not scores: return
        quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
        
        good_images = []
        for img_path, score in img_scores:
            if score < quality_threshold:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
            else:
                good_images.append(img_path)
        
        # ======================================================
        # 🔥 核心修改在这里 🔥
        # ======================================================
        # 从配置对象中读取最大图片数量
        max_imgs = self.cfg.max_images  
        
        # 使用 max_imgs 替代原来的 max_images
        if len(good_images) > max_imgs:
            print(f"    ⚠️ 图片过多 ({len(good_images)} 张), 正在降采样至 {max_imgs} 张...")
            # np.linspace 生成均匀分布的索引
            indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_imgs, dtype=int))
            for idx, img_path in enumerate(good_images):
                if idx not in indices_to_keep:
                    shutil.move(str(img_path), str(trash_dir / img_path.name))
                    
        print(f"✨ 清洗结束，剩余 {len(list(image_dir.glob('*')))} 张。")

def analyze_and_calculate_adaptive_collider(json_path, force_cull=False, radius_scale=1.8):
    """
    [3D 场景理解算法] 解析相机轨迹，自动判断场景类型并计算包围盒 (Collider)
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
        
        has_mask = "mask_path" in frames[0]
        if has_mask:
            print("    -> 检测到 Mask 数据！将启用物体聚焦模式。")
        
        # [线性代数] 提取所有相机的位移 (Translation)
        # transform_matrix 是 4x4 矩阵，[:3, 3] 是 XYZ 坐标
        positions = [np.array(f["transform_matrix"])[:3, 3] for f in frames]
        
        # 提取相机的前向向量 (Forward Vector)
        # 在 OpenCV/Colmap 定义中，+Z 轴通常是相机看向的方向，或者 -Z，需根据具体坐标系判定
        # 这里假设 -Z 是前方 (NeRF 常用约定)
        forward_vectors = [np.array(f["transform_matrix"])[:3, :3] @ np.array([0, 0, -1]) for f in frames]
        
        # 计算所有相机位置的几何中心
        center = np.mean(positions, axis=0)
        
        # 计算每个相机位置指向场景中心的向量
        vec_to_center = center - positions
        # 归一化向量 (除以模长)
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        
        # [核心算法] 计算“视线”与“指向中心向量”的对齐程度
        # 点积 > 0 表示方向基本一致（夹角小于90度）
        # 如果 ratio > 0.6，说明超过 60% 的相机都看向中心区域
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        # 综合判定：向心率高 OR 强制开启球形裁剪 OR 有 Mask
        is_object_mode = ratio > 0.6 or force_cull or has_mask

        if is_object_mode:
            # 物体模式：设置紧凑的 Near/Far Plane
            dists = [np.linalg.norm(p) for p in positions] # 相机到原点的距离
            avg_dist = np.mean(dists)
            
            scene_radius = 1.0 * radius_scale  # 场景半径
            
            # 计算 Near Plane (近平面)：不能太近，否则会切掉相机前的物体
            calc_near = max(0.05, min(dists) - scene_radius)
            # 计算 Far Plane (远平面)：只要包住物体即可
            calc_far = avg_dist + scene_radius
            
            # 返回 nerfstudio 需要的训练参数
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            # 场景模式：空间很大，Far Plane 设远一点
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"
    except:
        return [], "unknown"

def perform_percentile_culling(ply_path, json_path, output_path, keep_percentile=0.9):
    """
    [点云后处理] 基于统计分位数的暴力切割
    功能：去除 Gaussian Splatting 训练后产生在远处的背景伪影。
    依赖：plyfile 库
    """
    # 检查依赖
    if not HAS_PLYFILE: return False
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    try:
        # 1. 计算场景中心
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
        # 2. ✅ 这里修改：使用传入的参数 keep_percentile
        threshold_radius = np.percentile(dists_pts, keep_percentile * 100)
        
        # 4. 读取不透明度 (Opacity) 并过滤
        # Gaussian Splatting 存储的 opacity 通常经过 sigmoid 激活，需要还原
        # 这里 simplified: 假设 vertex['opacity'] 是 logit
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 联合掩码：(在半径内) AND (不透明度 > 0.05)
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        filtered_vertex = vertex[mask]
        
        # 5. 写入新文件
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True
    except Exception as e:
        print(f"❌ 切割失败: {e}")
        return False


# ==============================================================================
# 模块化：GLOMAP 位姿解算类 (环境隔离增强版)
# ==============================================================================
class GlomapRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        
        # 1. 查找 COLMAP (优先使用 Conda 环境自带的！)
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
        
        self.env = os.environ.copy()
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def run(self):
        """执行 GLOMAP 完整流程"""
        print(f"\n📐 [2/4] GLOMAP 位姿解算 (Global Mapping)")

        # 路径准备
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
            # 清理
            if database_path.exists(): database_path.unlink()
            if sparse_dir.exists(): shutil.rmtree(sparse_dir)
            sparse_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.transforms_file.exists(): self.cfg.transforms_file.unlink()

            # Step 1: 特征提取
            self._run_cmd([
                self.colmap_exe, "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--ImageReader.camera_model", "OPENCV",
                "--ImageReader.single_camera", "1"
            ], "Step 1: 特征提取 (COLMAP)")

            # Step 2: 顺序匹配
            self._run_cmd([
                self.colmap_exe, "sequential_matcher",
                "--database_path", str(database_path),
                "--SequentialMatching.overlap", "25"
            ], "Step 2: 顺序匹配 (COLMAP)")

            # Step 3: 全局重建
            print(f"    -> 🚀 启动 GLOMAP 引擎...")
            self._run_cmd([
                self.glomap_exe, "mapper",
                "--database_path", str(database_path),
                "--image_path", str(raw_images_dir),
                "--output_path", str(sparse_dir)
            ], "Step 3: 全局映射 (GLOMAP)")

            # Step 4: 目录修正
            self._fix_directory_structure(sparse_dir)

            # Step 5: 生成 json
            self._run_cmd([
                "ns-process-data", "images",
                "--data", str(dest_images_dir),
                "--output-dir", str(self.cfg.data_dir),
                "--skip-colmap",
                "--skip-image-processing",
                "--num-downscales", "0"
            ], "生成 transforms.json")

            # Step 6: 检查
            if self._check_quality(raw_images_dir):
                print(f"    ✨ GLOMAP 流程成功！")
                return True

        except Exception as e:
            print(f"    ❌ GLOMAP 流程失败: {e}")
            return False
        return False

    def _run_cmd(self, cmd, desc):
        """内部工具：执行命令 (含环境隔离逻辑)"""
        print(f"🚀 {desc}...")
        
        # 🔥 环境隔离逻辑 🔥
        cmd_env = self.env.copy()
        exe_path = cmd[0]
        # 如果是系统程序 (/usr/local/bin/glomap)，清除 LD_LIBRARY_PATH 防止 Conda 干扰
        if exe_path.startswith("/usr") or exe_path.startswith("/bin"):
            if "LD_LIBRARY_PATH" in cmd_env:
                del cmd_env["LD_LIBRARY_PATH"]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cmd_env
            )
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
        target_dir_0 = sparse_root / "0"
        target_dir_0.mkdir(parents=True, exist_ok=True)
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        required_files_txt = ["cameras.txt", "images.txt", "points3D.txt"]
        model_found = False
        for root, dirs, files in os.walk(sparse_root):
            if all(f in files for f in required_files):
                src = Path(root)
                if src != target_dir_0:
                    for f in required_files:
                        if (target_dir_0/f).exists(): (target_dir_0/f).unlink()
                        shutil.move(str(src/f), str(target_dir_0/f))
                model_found = True
                break
            if all(f in files for f in required_files_txt):
                src = Path(root)
                if src != target_dir_0:
                    for f in required_files_txt:
                        if (target_dir_0/f).exists(): (target_dir_0/f).unlink()
                        shutil.move(str(src/f), str(target_dir_0/f))
                model_found = True
                break
        if not model_found: raise RuntimeError("GLOMAP 未生成有效的稀疏模型文件！")

    def _check_quality(self, raw_images_dir):
        if not self.cfg.transforms_file.exists(): return False
        with open(self.cfg.transforms_file, 'r') as f: meta = json.load(f)
        reg_count = len(meta["frames"])
        total_count = len(list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png")))
        ratio = reg_count / total_count if total_count > 0 else 0
        print(f"    📊 匹配率: {ratio:.2%} ({reg_count}/{total_count})")
        return ratio > 0.2

# ==============================================================================
# 模块化：AI 语义分割类
# ==============================================================================
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
        

# ==============================================================================
# 模块化：Nerfstudio 训练引擎类
# ==============================================================================
class NerfstudioEngine:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = cfg.project_dir / "outputs"
        # 准备环境变量
        self.env = os.environ.copy()
        self.env["QT_QPA_PLATFORM"] = "offscreen"
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def train(self):
        """执行 splatfacto 训练"""
        print(f"\n🔥 [4/4] 开始训练 (Splatfacto)")
        
        # 1. 计算场景参数 (Collider) - 直接调用之前的全局函数
        collider_args, scene_type = analyze_and_calculate_adaptive_collider(
            self.cfg.transforms_file,
            force_cull=self.cfg.force_spherical_culling,
            radius_scale=self.cfg.scene_radius_scale
        )
        self.scene_type = scene_type # 存下来给导出步骤用

        # 2. 组装命令
        cmd = [
            "ns-train", "splatfacto",
            "--data", str(self.cfg.data_dir),
            "--output-dir", str(self.output_dir),
            "--experiment-name", self.cfg.project_name,
            "--pipeline.model.random-init", "False",
            "--pipeline.model.background-color", "random",
            "--pipeline.model.cull-alpha-thresh", "0.05",
            "--pipeline.model.stop-split-at", "10000",
            *collider_args,
            "--max-num-iterations", "15000",
            "--vis", "viewer+tensorboard",
            "--viewer.quit-on-train-completion", "True",
            "nerfstudio-data",
            "--downscale-factor", "1",
            "--auto-scale-poses", "False"
        ]
        
        # 3. 执行
        subprocess.run(cmd, check=True, env=self.env)

    def export(self):
        """导出 ply 并进行后处理"""
        print(f"\n💾 正在导出...")
        # 找到最新的 config.yml
        search_path = self.output_dir / self.cfg.project_name / "splatfacto"
        try:
            run_dirs = sorted(list(search_path.glob("*")))
            config_path = run_dirs[-1] / "config.yml"
        except IndexError:
            print("❌ 未找到训练结果 config.yml")
            return None

        # 导出命令
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
            # 调用之前的全局函数
            success = perform_percentile_culling(
                raw_ply, 
                self.cfg.transforms_file, 
                cleaned_ply,
                keep_percentile=self.cfg.keep_percentile
            )
            if success:
                final_ply = cleaned_ply

        # 复制结果到 results 目录
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
    # colmap_runner = ColmapRunner(cfg)
    glomap_runner = GlomapRunner(cfg) 
    ai_segmentor = AISegmentor(cfg)
    nerf_engine = NerfstudioEngine(cfg)

    # ==========================================
    # Step 1: 数据准备
    # ==========================================
    # 初始化目录
    if cfg.project_dir.exists(): shutil.rmtree(cfg.project_dir, ignore_errors=True)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(cfg.video_path), str(cfg.project_dir / cfg.video_path.name))
    
    # 抽帧 (这里逻辑简单，直接写这里也行，或者封装进 ImageProcessor)
    temp_dir = cfg.project_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(cfg.project_dir / cfg.video_path.name), 
                    "-vf", "fps=10", "-q:v", "2", 
                    str(temp_dir / "frame_%05d.jpg")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 清洗
    img_processor.smart_filter_blurry_images(temp_dir, keep_ratio=0.85)
    
    # 移动图片到 raw_images
    raw_images_dir = cfg.project_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    # 简单的移动逻辑保留在这里，或者也可以移入 ImageProcessor
    all_imgs = sorted(list(temp_dir.glob("*")))
    limit = cfg.max_images
    if len(all_imgs) > limit:
        indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
        all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
    for img in all_imgs: shutil.copy2(str(img), str(raw_images_dir / img.name))
    shutil.rmtree(temp_dir)

    # ==========================================
    # Step 2: GOLMAP
    # ==========================================
    if not glomap_runner.run():
        print("❌ Pipeline 中断：GLOMAP 失败")
        return

    # ==========================================
    # Step 3: AI
    # ==========================================
    # 这里会自动判断是否开启，内部已处理异常
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
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])
    
    if not video_file.exists():
        print(f"❌ 找不到视频: {video_file}")
        sys.exit(1)

    # 实例化配置
    cfg = PipelineConfig(
        project_name="glomap_test_v1", # 改个名字
        video_path=video_file,
        max_images=100,
        enable_ai=True
    )
    
    # 运行流水线
    run_pipeline(cfg)