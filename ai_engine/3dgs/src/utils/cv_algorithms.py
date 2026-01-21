# src/utils/cv_algorithms.py
# 功能：提供计算机视觉算法工具函数
# 实现：包含图像处理和分析的算法函数
# 逻辑：提供可复用的计算机视觉算法
# 包含：Mask清洗验证函数、显著区域计算函数
import cv2
import numpy as np
import torch

# [工具函数] 存放 clean_and_verify_mask, get_salient_box

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
