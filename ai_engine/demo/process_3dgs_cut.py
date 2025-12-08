import subprocess
import sys
import shutil
import os
import time
import datetime
from pathlib import Path
import json
import numpy as np
import logging
import cv2
import re

# ================= 🧠 AI 依赖引入 =================
try:
    import dashscope
    from dashscope import MultiModalConversation
    from ultralytics import SAM, YOLOWorld
    HAS_AI = True
except ImportError:
    HAS_AI = False
    print("⚠️ [环境警告] 未检测到 dashscope 或 ultralytics 库。")
    print("    -> 智能分割功能将被禁用。请运行: pip install dashscope ultralytics")

# 🔥 请在此处填入你的 API KEY (或者确保环境变量 DASHSCOPE_API_KEY 已存在)
# os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ================= 🔧 基础配置 =================
# 🔥【绝杀】强制将编译好的系统级 colmap 路径提到最前面
sys_path = "/usr/local/bin"
current_path = os.environ.get("PATH", "")
if sys_path not in current_path.split(os.pathsep)[0]:
    print(f"⚡ [环境修正] 强制设置 PATH 优先级: {sys_path} -> Priority High")
    os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"

# 设置日志级别
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# 工作区配置
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
# 🔥 新增：词汇树文件路径 (请确保你下载了它！)
VOCAB_TREE_PATH = LINUX_WORK_ROOT / "vocab_tree_flickr100k_words.bin" 
SCENE_RADIUS_SCALE = 1.8 
MAX_IMAGES = 180 

# 切割配置
FORCE_SPHERICAL_CULLING = True
KEEP_PERCENTILE = 0.9

# 检查 plyfile
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

# ================= 🧠 AI 核心逻辑函数 =================

def get_central_object_prompt(images_dir: Path, sample_count=3):
    """
    [Step 1.1] 使用 Qwen-VL-Plus 多图分析，提取中心物体的文本描述
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY，无法调用大模型。")
        return None

    print(f"\n🧠 [AI 分析] 正在调用 Qwen-VL-Plus 分析场景...")
    
    # 随机采样 3 张图片
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files: return None
    
    indices = np.linspace(0, len(image_files) - 1, sample_count, dtype=int)
    sampled_imgs = [image_files[i] for i in indices]
    
    # 构建多模态消息
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

    messages = [{"role": "user", "content": content}]

    try:
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus', 
            messages=messages
        )
        
        if response.status_code == 200:
            prompt_text = response.output.choices[0].message.content[0]["text"].strip()
            # 简单的清洗，去掉可能的标点
            prompt_text = prompt_text.replace(".", "").replace('"', "").replace("'", "")
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
    [净化版] 
    1. 强制清洗：只保留画面中最大的连通块 (去除孤立噪点)。
    2. 严格质检：清洗后如果形状依然毛糙(粘连阴影)，则剔除。
    3. 返回：(是否合格, 清洗后的干净Mask, 原因)
    """
    h, w = mask.shape
    
    # --- 1. 连通域分析 & 强制清洗 (Cleaning) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 如果全黑，直接扔
    if num_labels < 2: 
        return False, None, "空蒙版"

    # 找出最大的前景块 (忽略 index 0 的背景)
    max_area = 0
    max_label = -1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
            
    # 如果最大的块也太小 (比如只占屏幕 0.5%)，那是垃圾
    if max_area < (h * w * 0.005):
        return False, None, "主体过小，疑似噪点"

    # 🔥 核心操作：重构 Mask，只保留最大的那一块
    # Frame 103 的顶部噪点和 Frame 13 的左下角碎点在这里会被直接抹除
    cleaned_mask = (labels == max_label).astype(np.uint8) * 255

    # --- 2. 对清洗后的 Mask 进行“体检” (Verification) ---
    
    # 提取轮廓
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, None, "清洗后无轮廓"
    
    main_cnt = max(contours, key=cv2.contourArea)
    
    # [检查 A] 实心度 (Solidity)
    # 针对 Frame 21 底部那种粘连的锯齿状阴影。
    # 正常的剃须刀是圆润的，Solidity 应该接近 0.95 以上。
    # 如果底部粘了一滩烂泥一样的阴影，Solidity 会掉到 0.85 以下。
    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False, None, "凸包面积为0"
    
    solidity = max_area / hull_area
    
    # 阈值设定：0.88 (非常严格，只允许极其轻微的边缘不平整)
    if solidity < 0.88:
        return False, None, f"边缘严重毛糙/粘连阴影 (实心度 {solidity:.2f})"

    # [检查 B] 极其夸张的长宽比 (防止把桌子缝隙当成物体)
    x, y, w_rect, h_rect = cv2.boundingRect(main_cnt)
    aspect_ratio = w_rect / h_rect
    if aspect_ratio > 4.5: # 放宽了之前的标准，但太离谱的长条还是要杀
        return False, None, f"形状异常 (长宽比 {aspect_ratio:.1f})"

    # 注意：这里完全移除了“边界溢出”检查，碰到边界也能过。

    # 🔥 新增：边缘腐蚀 (Erosion)
    # 这一步是为了切掉物体边缘沾染的桌面反光和那一圈淡淡的阴影
    kernel_size = 3  # 腐蚀力度，3x3 约等于缩减 1-2 个像素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=1)
    
    return True, cleaned_mask, "合格"

def get_salient_box(img_path, margin_ratio=0.1):
    """
    [纯本地 CV 算法] 计算画面的'视觉显著区域'，以此作为 SAM 的提示框。
    原理：利用拉普拉斯算子找边缘 -> 膨胀连接 -> 找最大外接矩形
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None: return None
        
        # 1. 转灰度并计算边缘 (Laplacian)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算梯度/边缘，越是物体边缘越亮
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 2. 模糊与二值化 (把零散的边缘连成块)
        # 高斯模糊让纹理聚集
        blurred = cv2.GaussianBlur(laplacian, (25, 25), 0)
        # 阈值处理：只保留最'强烈'的纹理区域 (取前20%亮的区域)
        threshold_val = np.percentile(blurred, 80) 
        _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        
        # 3. 找最大轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        # 找到面积最大的轮廓（通常就是主体）
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)
        
        # 4. 加上一点安全边距 (Padding)，防止框太紧
        H, W = img.shape[:2]
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        
        # 返回符合 YOLO/SAM 格式的 tensor
        import torch
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        
    except Exception as e:
        print(f"       ⚠️ 视觉重心计算失败: {e}")
        return None

def run_ai_segmentation_pipeline(data_dir: Path):
    """
    [Step 1.2] 执行 AI 分割
    逻辑：Qwen分析 -> 失败则用通用词 -> YOLO识别 -> 失败则强制中心框 -> SAM分割
    """
    if not HAS_AI: return False
    
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    transforms_file = data_dir / "transforms.json"

    if not transforms_file.exists():
        print("⚠️ 未找到 transforms.json，无法进行 Mask 处理。")
        return False

    # ================= 核心修改逻辑开始 =================
    print(f"\n✂️ [AI 分割] 正在初始化...")

    # --- 第一层：尝试调用大模型获取精准 Prompt ---
    text_prompt = None
    try:
        # 尝试调用你写的那个函数
        text_prompt = get_central_object_prompt(images_dir)
    except Exception as e:
        print(f"    ⚠️ 大模型调用出错: {e}")

    # --- 第二层：如果大模型失败，使用通用 Prompt ---
    if not text_prompt:
        # 使用一个非常通用的词，让 YOLO-World 去找画面里最显著的东西
        # "salient object" (显著物体) 或 "central object" (中心物体) 效果通常不错
        text_prompt = "central object; single object"
        print(f"    ⚠️ 未能获取精准描述，降级使用通用 Prompt: '{text_prompt}'")
    else:
        print(f"    🎯 获取到精准 Prompt: '\033[92m{text_prompt}\033[0m'")

    masks_dir.mkdir(parents=True, exist_ok=True)
    # ================= 核心修改逻辑结束 =================

    # 2. 加载模型 (推荐用 Large)
    print("    -> 正在加载 SAM 2 Large 模型...")
    
    # 🔥 自动迁移 AI 模型文件
    model_files = ["yolov8s-worldv2.pt", "sam2.1_l.pt"]
    for model_name in model_files:
        target_model_path = LINUX_WORK_ROOT / model_name
        local_model_path = Path(__file__).parent / model_name
        
        if not target_model_path.exists():
            if local_model_path.exists():
                print(f"    📦 检测到本地模型 {model_name}，正在迁移至工作区...")
                shutil.copy2(str(local_model_path), str(target_model_path))
            else:
                print(f"    ⚠️ 未在脚本目录找到 {model_name}，将尝试自动下载...")

    try:
        # 使用绝对路径加载（如果存在），否则使用默认名称（触发下载）
        yolo_path = LINUX_WORK_ROOT / "yolov8s-worldv2.pt"
        sam_path = LINUX_WORK_ROOT / "sam2.1_l.pt"
        
        # YOLO-World: 听懂文字，找框
        det_model = YOLOWorld(str(yolo_path) if yolo_path.exists() else "yolov8s-worldv2.pt") 
        det_model.set_classes([text_prompt])
        
        # SAM 2: 根据框，抠图
        # 注意：使用 sam2.1_l.pt (Large版本) 精度更高，速度较慢
        sam_model = SAM(str(sam_path) if sam_path.exists() else "sam2.1_l.pt") 
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    # 3. 读取 transforms.json (用于最后过滤)
    with open(transforms_file, 'r') as f:
        meta = json.load(f)
    
    # 建立文件名到帧数据的映射，方便后续删除
    # 注意：file_path 可能是 "images/frame_001.jpg"，我们只取文件名匹配
    frames_map = {Path(f["file_path"]).name: f for f in meta["frames"]}
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    total_imgs = len(image_files)
    
    valid_frames_list = [] # 存放合格的帧数据
    deleted_count = 0
    
    print(f"    -> 开始处理 {total_imgs} 张图片...")

    for i, img_path in enumerate(image_files):
        # --- A. 检测与分割 (同前) ---
        try:
            # 1. YOLO 检测
            det_results = det_model.predict(img_path, conf=0.05, verbose=False)
            
            # ============================================================
            # 🕵️‍♂️ [DEBUG 模式] 看看 YOLO 到底看到了什么？
            # ============================================================
            
            # 1. 准备调试目录 (只会创建一次)
            debug_dir = data_dir / "debug_yolo_visuals"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 检查检测结果
            num_boxes = len(det_results[0].boxes)
            
            if num_boxes > 0:
                # 获取画了框的图片 (numpy 数组)
                plotted_img = det_results[0].plot()
                
                # 保存到 debug 目录，文件名加个前缀方便找
                debug_path = debug_dir / f"debug_{img_path.name}"
                cv2.imwrite(str(debug_path), plotted_img)
                
                # 在控制台打印坐标信息 (只打印前 3 张图，避免刷屏)
                if i < 3: 
                    print(f"\n    👀 [DEBUG] {img_path.name}: 找到了 {num_boxes} 个目标")
                    box_coords = det_results[0].boxes.xyxy.cpu().numpy()[0] # 取第一个框
                    conf_score = det_results[0].boxes.conf.cpu().numpy()[0]
                    print(f"       -> 位置: {box_coords} (置信度: {conf_score:.2f})")
                    print(f"       -> 调试图已保存: {debug_path}")
            else:
                if i < 3:
                    print(f"\n    🙈 [DEBUG] {img_path.name}: YOLO 没找到任何东西 (0 boxes)")
            
            # ============================================================

            bboxes = det_results[0].boxes.xyxy.cpu() 

            # ============================================================
            # 🔥 核心修改：从“死框”改为“智能中心点扩散”
            # ============================================================
            
            # 标记是否使用点提示
            use_point_prompt = False
            
            # 如果 YOLO 没找到框，或者框太离谱
            if len(bboxes) == 0:
                print(f"       ⚠️ YOLO 未识别到物体，切换为 [SAM 中心点模式]")
                h, w = det_results[0].orig_shape[:2]
                import torch
                
                # 策略：给 SAM 一个中心点 (x, y)，让它自己去“泛洪填充”
                # points 格式: [[x, y]]
                input_points = [[w / 2, h / 2]]
                # labels 格式: [1] (1表示前景点，0表示背景点)
                input_labels = [1]
                
                use_point_prompt = True
            
            # 3. 执行 SAM 分割
            if use_point_prompt:
                # 方式 A: 使用点提示 (Point Prompt)
                # 注意：Ultralytics 的 SAM 接口调用方式可能略有不同，
                # 如果是官方 SAM，通常是 predict(points=..., labels=...)
                # 在 Ultralytics 封装中，我们通常把点转成微小的框，或者直接传参
                
                # 为了兼容性最强，我们这里用一个“极小框”模拟“点”
                # 这样 SAM 会认为这是一个非常确定的中心区域
                cx, cy = w / 2, h / 2
                margin = 5 # 5像素的中心区域
                bboxes = torch.tensor([[cx-margin, cy-margin, cx+margin, cy+margin]], device=det_model.device)
                
                # 调用 SAM (Ultralytics 会把这个小框当做提示)
                sam_results = sam_model(img_path, bboxes=bboxes, verbose=False)
            else:
                # 方式 B: 使用 YOLO 的框 (Box Prompt)
                sam_results = sam_model(img_path, bboxes=bboxes, verbose=False)
            
            if sam_results[0].masks is not None:
                all_masks = sam_results[0].masks.data.cpu().numpy()
                final_mask = np.any(all_masks, axis=0).astype(np.uint8) * 255
            else:
                final_mask = np.zeros(det_results[0].orig_shape[:2], dtype=np.uint8)

            # -------------------------------------------------
            # 🔥 核心修改：使用净化函数 🔥
            # -------------------------------------------------
            # 注意：这里接收 3 个返回值 (是否合格, 新Mask, 原因)
            is_good, cleaned_mask, reason = clean_and_verify_mask(final_mask, img_path.name)

            if is_good:
                # ✅ 合格：使用清洗后的 Mask (cleaned_mask) 进行处理
                
                # 1. 涂黑操作 -> 改为生成 RGBA (PNG) 图片
                original_img = cv2.imread(str(img_path))
                if original_img is not None:
                    # 羽化边缘 (减少硬切伪影)
                    mask_blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
                    
                    # 确保 alpha_channel 是 float32
                    alpha_channel = mask_blurred.astype(np.float32) / 255.0
                    
                    # 转换原图为 float32 以便计算
                    img_float = original_img.astype(np.float32)
                    
                    # 预乘 Alpha (Premultiplied Alpha)
                    b, g, r = cv2.split(img_float)
                    b = b * alpha_channel
                    g = g * alpha_channel
                    r = r * alpha_channel
                    
                    # 🔥 修复点：在 merge 之前，强制所有通道转回 uint8
                    # 这样 b, g, r, a 全部都是 uint8 类型，OpenCV 就不会报错了
                    img_bgra = cv2.merge([
                        b.astype(np.uint8), 
                        g.astype(np.uint8), 
                        r.astype(np.uint8), 
                        mask_blurred # 已经是 uint8，直接用
                    ])
                    
                    # 保存为 PNG (必须用 PNG 存透明通道)
                    new_img_path = img_path.with_suffix('.png')
                    cv2.imwrite(str(new_img_path), img_bgra)
                    
                    # 如果原图是 jpg，删掉它，避免重复
                    if img_path.suffix.lower() == '.jpg':
                        try: img_path.unlink()
                        except: pass
                        
                    final_img_path_name = new_img_path.name
                else:
                    final_img_path_name = img_path.name

                # 2. 保存 Mask (一定要保存清洗后的！)
                cv2.imwrite(str(masks_dir / f"{img_path.stem}.png"), cleaned_mask)

                # 3. 加入合格列表
                # 记得在这里更新 json 里的文件名 (后缀变成了 .png)
                if img_path.name in frames_map:
                    frame_data = frames_map[img_path.name]
                    frame_data["file_path"] = f"images/{final_img_path_name}" 
                    frame_data["mask_path"] = f"masks/{img_path.stem}.png"
                    valid_frames_list.append(frame_data)

            else:
                # ❌ 不合格：物理删除
                print(f"       🗑️ [剔除] {img_path.name}: {reason}")
                img_path.unlink() # 删除图片文件
                deleted_count += 1
                # 注意：这里不把它加入 valid_frames_list，它自然就从 transforms.json 里消失了

        except Exception as e:
            print(f"       ❌ 处理出错 {img_path.name}: {e}")
            # 出错也视为不合格，不加入列表
            continue

        if i % 10 == 0:
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
    # 只保留合格的帧，这样 Nerfstudio 就只会训练这些“纯净”的黑背景图
    meta["frames"] = valid_frames_list
    with open(transforms_file, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"    ✅ transforms.json 已更新，数据集已清洗完毕。")
    return True

# ================= 辅助工具 =================
def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def smart_filter_blurry_images(image_folder, keep_ratio=0.85, max_images=MAX_IMAGES):
    # (保持原有的清洗逻辑不变)
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
    
    if len(good_images) > max_images:
        indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_images, dtype=int))
        for idx, img_path in enumerate(good_images):
            if idx not in indices_to_keep:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
    print(f"✨ 清洗结束，剩余 {len(list(image_dir.glob('*')))} 张。")

def analyze_and_calculate_adaptive_collider(json_path):
    # (保持原有逻辑，但如果检测到 Mask，可以更加激进)
    print(f"\n🤖 [AI 分析] 解析相机轨迹...")
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"]
        if not frames: return [], "unknown"
        
        # 简单判定：是否有 mask_path
        has_mask = "mask_path" in frames[0]
        if has_mask:
            print("    -> 检测到 Mask 数据！将启用物体聚焦模式。")
        
        # (原有的轨迹分析逻辑...)
        positions = [np.array(f["transform_matrix"])[:3, 3] for f in frames]
        forward_vectors = [np.array(f["transform_matrix"])[:3, :3] @ np.array([0, 0, -1]) for f in frames]
        center = np.mean(positions, axis=0)
        vec_to_center = center - positions
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        # 如果有 Mask，或者相机向内看，都认为是物体模式
        is_object_mode = ratio > 0.6 or FORCE_SPHERICAL_CULLING or has_mask

        if is_object_mode:
            dists = [np.linalg.norm(p) for p in positions]
            avg_dist = np.mean(dists)
            scene_radius = 1.0 * SCENE_RADIUS_SCALE
            calc_near = max(0.05, min(dists) - scene_radius)
            calc_far = avg_dist + scene_radius
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"
    except:
        return [], "unknown"

def perform_percentile_culling(ply_path, json_path, output_path):
    # (保持原有逻辑不变)
    if not HAS_PLYFILE: return False
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    try:
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        
        dists_pts = np.linalg.norm(points - center, axis=1)
        threshold_radius = np.percentile(dists_pts, KEEP_PERCENTILE * 100)
        
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        filtered_vertex = vertex[mask]
        
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True
    except Exception as e:
        print(f"❌ 切割失败: {e}")
        return False

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Engine AI-Enhanced] 启动任务: {project_name}")
    print(f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 
    # 【新增】修复 distutils 报错的关键环境变量
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib" 

    # [Step 1] 数据处理
    step1_start = time.time()
    
    # ... (目录初始化逻辑保持不变)
    if work_dir.exists(): shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(video_src), str(work_dir / video_src.name))

    print(f"\n🎥 [1/4] 数据准备与清洗")
    temp_dir = work_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    extracted_images_dir = work_dir / "raw_images"
    extracted_images_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg 抽帧
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(work_dir / video_src.name), 
                        "-vf", "fps=10", "-q:v", "2", 
                        str(temp_dir / "frame_%05d.jpg")], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
    except: pass
    
    # 清洗
    smart_filter_blurry_images(temp_dir, keep_ratio=0.85)
    
    # 迁移
    all_candidates = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")))
    final_images_list = []
    if len(all_candidates) > MAX_IMAGES:
        indices = np.linspace(0, len(all_candidates) - 1, MAX_IMAGES, dtype=int)
        indices = sorted(list(set(indices)))
        for idx in indices: final_images_list.append(all_candidates[idx])
    else:
        final_images_list = all_candidates

    for img_path in final_images_list:
        shutil.copy2(str(img_path), str(extracted_images_dir / img_path.name))
    shutil.rmtree(temp_dir)

    # COLMAP 流程 (增强版 - 包含自动修正)
    print(f"\n📐 [2/4] COLMAP 位姿解算 (增强版)")
    colmap_output_dir = data_dir / "colmap"
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    database_path = colmap_output_dir / "database.db"
    
    # 查找 colmap
    system_colmap_exe = shutil.which("colmap") or "/usr/local/bin/colmap"

    full_log_content = []

    def run_colmap_step(cmd, description):
        print(f"\n🚀 {description}...")
        try:
            # 使用 Popen 实时打印输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            
            # 实时读取输出
            for line in process.stdout:
                full_log_content.append(line)
                # 过滤掉过于频繁的进度输出，保留关键信息
                # 扩充关键词，确保 Mapper 阶段能看到 Registering 和 Bundle adjustment 等信息
                if any(k in line for k in ["Iteration", "Error", "Loading", "Elapsed", "Registering", "Image #", "Bundle adjustment", "Retriangulation", "Filtering"]):
                    print(f"    [COLMAP] {line.strip()}")
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
        except Exception as e:
            print(f"❌ {description} 失败: {e}")
            raise e

    # ==========================================
    # 🔥 核心修改：闭环重试机制 (包含质量检测) 🔥
    # ==========================================
    MAX_RETRIES = 3
    colmap_success = False

    # 提前同步图片 (只需要做一次)
    dest_images_dir = data_dir / "images"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    for img in extracted_images_dir.glob("*"): 
        shutil.copy2(str(img), str(dest_images_dir / img.name))

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n🔄 [COLMAP] 正在执行第 {attempt} / {MAX_RETRIES} 次尝试...")
        
        # --- 1. 每次重试前，强制清理环境 ---
        if attempt > 1:
            print("    🧹 [重试准备] 正在清理旧数据...")
            if database_path.exists(): 
                try: database_path.unlink()
                except: pass
            
            sparse_dir = colmap_output_dir / "sparse"
            if sparse_dir.exists(): 
                try: shutil.rmtree(sparse_dir)
                except: pass
            sparse_dir.mkdir(parents=True, exist_ok=True)
            
            # 删除可能存在的旧 transforms.json，防止误读
            if transforms_file.exists():
                transforms_file.unlink()

        try:
            # --- 2. 运行 COLMAP 核心流程 ---
            
            # Step 1: 特征提取
            run_colmap_step([
                system_colmap_exe, "feature_extractor", 
                "--database_path", str(database_path), 
                "--image_path", str(extracted_images_dir), 
                "--ImageReader.camera_model", "OPENCV", 
                "--ImageReader.single_camera", "1"
            ], "Step 1: 特征提取")

            # Step 2: 词汇树匹配 (Vocab Tree)
            print("    -> 🌳 词汇树匹配 (Vocab Tree Matcher)...")
            local_vocab_path = Path(__file__).parent / "vocab_tree_flickr100k_words.bin"
            if not VOCAB_TREE_PATH.exists():
                if local_vocab_path.exists():
                    shutil.copy2(str(local_vocab_path), str(VOCAB_TREE_PATH))
                else:
                    raise FileNotFoundError(f"Missing vocab tree: {VOCAB_TREE_PATH}")

            run_colmap_step([
                system_colmap_exe, "vocab_tree_matcher", 
                "--database_path", str(database_path),
                "--VocabTreeMatching.vocab_tree_path", str(VOCAB_TREE_PATH),
                "--VocabTreeMatching.match_list_path", "" 
            ], "Step 2: 词汇树匹配")

            # Step 3: 稀疏重建
            sparse_dir = colmap_output_dir / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            run_colmap_step([
                system_colmap_exe, "mapper", 
                "--database_path", str(database_path), 
                "--image_path", str(extracted_images_dir), 
                "--output_path", str(sparse_dir)
            ], "Step 3: 稀疏重建")

            # --- 3. 立即执行 Auto-Fix (目录修正) ---
            # (必须在循环内做，因为每次 mapper 可能会乱生成目录)
            print("    🔧 正在检查模型结构...")
            sparse_root = colmap_output_dir / "sparse"
            target_dir_0 = sparse_root / "0"
            target_dir_0.mkdir(parents=True, exist_ok=True)
            
            required_files = ["cameras.bin", "images.bin", "points3D.bin"]
            model_found = False
            
            # 扫描并归位
            if all((target_dir_0 / f).exists() for f in required_files):
                model_found = True
            else:
                for root, dirs, files in os.walk(sparse_root):
                    if all(f in files for f in required_files):
                        source_dir = Path(root)
                        if source_dir != target_dir_0:
                            for f in required_files:
                                if (target_dir_0/f).exists(): (target_dir_0/f).unlink()
                                shutil.move(str(source_dir/f), str(target_dir_0/f))
                        model_found = True
                        break
            
            if not model_found:
                raise RuntimeError("COLMAP 未生成有效的稀疏模型文件！")

            # --- 4. 立即生成 transforms.json 以检测质量 ---
            print("    -> 正在生成数据以进行质量检测...")
            run_colmap_step([
                "ns-process-data", "images", 
                "--data", str(dest_images_dir), 
                "--output-dir", str(data_dir), 
                "--skip-colmap", 
                "--skip-image-processing", 
                "--num-downscales", "0"
            ], "生成 transforms.json")

            # --- 5. 🔥 关键：质量判决 (Quality Gate) 🔥 ---
            if not transforms_file.exists():
                raise RuntimeError("transforms.json 生成失败")

            with open(transforms_file, 'r') as f:
                meta = json.load(f)
            
            registered_count = len(meta["frames"])
            total_count = len(list(extracted_images_dir.glob("*.jpg")) + list(extracted_images_dir.glob("*.png")))
            
            match_ratio = registered_count / total_count if total_count > 0 else 0
            print(f"    📊 本次匹配率: {match_ratio:.2%} ({registered_count}/{total_count})")

            if match_ratio < 0.35: # 阈值 35%
                print(f"    ⚠️ 匹配率过低，判定为失败！准备重试...")
                # 主动抛出异常，触发 except 块，进入下一次循环
                raise RuntimeError(f"Low match ratio: {match_ratio:.2%}")
            
            # 如果走到这里，说明一切正常
            print(f"    ✨ 质量达标！COLMAP 在第 {attempt} 次尝试中成功！")
            colmap_success = True
            break # 跳出重试循环

        except Exception as e:
            print(f"    ❌ 第 {attempt} 次尝试失败: {e}")
            if attempt < MAX_RETRIES:
                print("    ⏳ 3秒后进行下一次重试...")
                time.sleep(3)
            else:
                print("    🛑 已耗尽所有重试机会。")

    if not colmap_success:
        print("❌ COLMAP 最终失败 (质量不达标)，任务终止。")
        return None

    # ================= 🔥 AI 介入点 (新增) =================
    if HAS_AI:
        print(f"\n🧠 [3/4] AI 智能分割介入 (Qwen + YOLO + SAM)")
        ai_success = run_ai_segmentation_pipeline(data_dir)
        if ai_success:
            print("✨ AI 分割流程完成，Mask 已注入！")
        else:
            print("⚠️ AI 分割流程遇到问题，将使用原始图像训练。")
    else:
        print("\n⏩ 跳过 AI 分割 (未满足依赖)")
    # ======================================================

    step1_duration = time.time() - step1_start
    print(f"⏱️ [预处理完成] 耗时: {format_duration(step1_duration)}")

    # [Step 2] 训练
    step2_start = time.time()
    print(f"\n🔥 [4/4] 开始训练 (Splatfacto)")
    
    collider_args, scene_type = analyze_and_calculate_adaptive_collider(transforms_file)
    
    # 构建训练命令
    train_cmd = [
        "ns-train", "splatfacto", 
        "--data", str(data_dir), 
        "--output-dir", str(output_dir), 
        "--experiment-name", project_name, 
        "--pipeline.model.random-init", "False", 
        
        # 🔥 新增参数 1: 告诉 Nerfstudio 背景是透明的，不要把黑色渲染出来
        "--pipeline.model.background-color", "random", 
        
        # 🔥 新增参数 2: 提高不透明度阈值，让那层薄薄的黑色烟雾直接消失
        "--pipeline.model.cull-alpha-thresh", "0.05", # 默认是 0.005，改大到 0.05

        # 🔥 新增：提高分裂门槛 (默认 0.0002 -> 0.0008)
        "--pipeline.model.densify-grad-thresh", "0.0008",
        # 🔥 新增：提前停止分裂 (默认 15000 -> 10000)
        "--pipeline.model.stop-split-at", "10000",
        # 🔥 新增：缩短热身期 (默认 500 -> 500)
        "--pipeline.model.warmup-length", "500",
        *collider_args,
        "--max-num-iterations", "15000", 
        "--vis", "viewer+tensorboard", 
        "--viewer.quit-on-train-completion", "True", 
        "nerfstudio-data", 
        "--downscale-factor", "1",
        "--orientation-method", "none", 
        "--center-method", "none",
        "--auto-scale-poses", "False"
    ]
    
    subprocess.run(train_cmd, check=True, env=env)
    step2_duration = time.time() - step2_start

    # [Step 3] 导出
    step3_start = time.time()
    print(f"\n💾 正在导出...")
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*")))
    latest_run = run_dirs[-1]
    
    subprocess.run([
        "ns-export", "gaussian-splat", 
        "--load-config", str(latest_run/"config.yml"), 
        "--output-dir", str(work_dir)
    ], check=True, env=env)
    
    # 暴力切割
    raw_ply = work_dir / "point_cloud.ply"
    if not raw_ply.exists(): raw_ply = work_dir / "splat.ply"
    cleaned_ply = work_dir / "point_cloud_cleaned.ply"
    final_ply = raw_ply
    
    if (scene_type == "object" or FORCE_SPHERICAL_CULLING) and raw_ply.exists():
        if perform_percentile_culling(raw_ply, transforms_file, cleaned_ply):
            final_ply = cleaned_ply
    
    step3_duration = time.time() - step3_start

    # [Step 4] 回传
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)
    shutil.copy2(str(final_ply), str(target_dir / f"{project_name}.ply"))
    
    total_duration = time.time() - global_start_time
    print(f"\n🎉 全部完成！模型已保存至: {target_dir / f'{project_name}.ply'}")
    print(f"📊 耗时统计:")
    print(f"   - 预处理 (COLMAP + AI): {format_duration(step1_duration)}")
    print(f"   - 训练 (Splatfacto):    {format_duration(step2_duration)}")
    print(f"   - 导出与后处理:         {format_duration(step3_duration)}")
    print(f"   - 总耗时:               {format_duration(total_duration)}")
    
    return str(target_dir / f"{project_name}.ply")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])

    if video_file.exists():
        run_pipeline(video_file, "scene_ai_test")
    else:
        print(f"❌ 找不到视频: {video_file}")