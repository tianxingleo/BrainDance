# [工具函数] 存放 analyze_and_calculate_adaptive_collider
import json
import numpy as np
from pathlib import Path

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
