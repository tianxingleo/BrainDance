import numpy as np

def convert_pose_to_webgl(c2w_matrix):
    """
    将 Nerfstudio (OpenGL) 姿态转换为 WebGL 友好格式 (列表形式)。
    通常 WebGL 需要 W2C 或者特定的坐标系转换，这里暂时保持 C2W 原始数据，
    前端加载时可根据需求求逆。
    """
    # 确保是 float32 并转为 list
    return np.array(c2w_matrix, dtype=np.float32).tolist()
