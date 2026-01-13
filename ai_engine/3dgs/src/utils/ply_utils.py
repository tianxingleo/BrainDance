# [工具函数] 存放 perform_percentile_culling
import json
import numpy as np
from pathlib import Path

# --- 依赖检查逻辑 ---
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("⚠️ Warning: 'plyfile' library not found. Point cloud culling will be skipped.")

    
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
