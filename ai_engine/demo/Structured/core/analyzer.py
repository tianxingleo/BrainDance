import json
import numpy as np

def analyze_scene_type(json_path):
    """
    分析 transforms.json 判断是物体(Object)还是场景(Scene)。
    返回 (collider_args, scene_type)
    """
    print(f"\n🤖 [AI 分析] 正在读取相机轨迹以判断场景类型...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        if not frames:
            return [], "unknown"

        positions = []
        forward_vectors = []
        
        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            positions.append(c2w[:3, 3])
            # Forward = R * [0, 0, -1] (Nerfstudio/OpenGL 坐标系)
            forward_vectors.append(c2w[:3, :3] @ np.array([0, 0, -1]))
            
        positions = np.array(positions)
        forward_vectors = np.array(forward_vectors)
        
        # 计算重心
        center_of_mass = np.mean(positions, axis=0)
        
        # 向量：相机 -> 中心
        vec_to_center = center_of_mass - positions
        norms = np.linalg.norm(vec_to_center, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0 # 防止除零
        
        # 点积判断视线夹角
        dot_products = np.sum(forward_vectors * (vec_to_center / norms), axis=1)
        
        # 统计“看向中心”的比例
        looking_inward_ratio = np.sum(dot_products > 0) / len(frames)
        
        print(f"    -> 相机聚合度: {looking_inward_ratio:.2f} (1.0=完全向内, 0.0=完全向外)")

        if looking_inward_ratio > 0.6:
            print("💡 判定结果：【物体扫描模式 (Inward)】")
            print("    -> 策略：相机围着物体转。启用紧凑裁剪(2.0~6.0)，聚焦中心物体，去除背景。")
            
            # 物体模式参数
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "2.0", "far_plane", "6.0"], "object"
        else:
            print("💡 判定结果：【全景/室内模式 (Outward)】")
            print("    -> 策略：相机在内部向外看，或直线扫描。放宽裁剪(0.05~100.0)，保留墙壁和远景。")
            
            # 室内/全景模式参数
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        print(f"⚠️ 分析失败 ({e})，使用默认参数。")
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"
