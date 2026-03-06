import json
import numpy as np

def export_aligned_poses(transforms_path, dataparser_path, output_path):
    print(f"Reading transforms from {transforms_path}")
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
        
    print(f"Reading dataparser transforms from {dataparser_path}")
    with open(dataparser_path, 'r') as f:
        dp_data = json.load(f)
        
    # 获取 Nerfstudio 的全局居中与缩放参数
    scale = dp_data['scale']
    dp_transform = np.array(dp_data['transform']) # 3x4
    global_transform = np.eye(4)
    global_transform[:3, :4] = dp_transform
    
    print(f"Using Scale: {scale}")
    print(f"Using Transform:\n{global_transform}")

    webgl_poses = []
    
    for frame in transforms['frames']:
        c2w = np.array(frame['transform_matrix']) # 4x4
        
        # 1. 应用缩放 (极其重要：平移向量必须缩放)
        c2w[:3, 3] *= scale
        
        # 2. 应用全局旋转与平移
        c2w = global_transform @ c2w
        
        # 3. OpenCV -> WebGL (OpenGL) 坐标系转换 (翻转Y和Z)
        # Nerfstudio/OpenCV: +X Right, +Y Down, +Z Forward
        # OpenGL/Three.js:   +X Right, +Y Up,   +Z Back
        # Y_gl = -Y_cv
        # Z_gl = -Z_cv
        # This is equivalent to multiplying by diagonal(1, -1, -1, 1) on the right
        # c2w = c2w @ np.diag([1, -1, -1, 1])
        # Or flipping columns directly:
        c2w[:, 1] *= -1
        c2w[:, 2] *= -1
        
        # Three.js 默认按列存储矩阵 (Column-major)
        # But wait, Three.js Matrix4.fromArray() expects Column-Major IF you pass a flat array derived from elements property.
        # But standard JSON export usually writes Row-Major list of lists.
        # Let's see what the frontend expects. 
        # Frontend code:
        # const targetMatrix = new THREE.Matrix4();
        # if (Array.isArray(poseData.matrix[0])) { targetMatrix.set(..., ...); } // Accepts Row-Major (n11, n12, n13, n14...)
        # else { targetMatrix.fromArray(poseData.matrix); } // Accepts Column-Major
        
        # flatten('F') means Column-Major. flatten() means Row-Major.
        # The provided user code uses `c2w.T.flatten().tolist()`.
        # c2w.T switches rows and cols. flatten() then reads Row-Major of the Transpose.
        # This effectively produces Column-Major sequence of the original matrix.
        # So we should output Column-Major flat list.
        matrix_threejs = c2w.T.flatten().tolist()
        
        webgl_poses.append({
            'id': frame['file_path'].split('/')[-1],
            'matrix': matrix_threejs,
            # 'fl_y': transforms.get('fl_y'), # Fix: fl_y is top level for some, frame level for others.
            # Usually top level in Nerfstudio transforms.json
            'image_url': f"/models/images/{frame['file_path'].split('/')[-1]}"
        })
    
    # Add metadata
    output_data = {
        'frames': webgl_poses,
        'fl_y': transforms.get('fl_y'),
        'fl_x': transforms.get('fl_x'),
        'w': transforms.get('w'),
        'h': transforms.get('h'),
        'camera_model': transforms.get('camera_model')
    }
        
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"成功导出 {len(webgl_poses)} 个完美对齐的位姿到 {output_path}")

if __name__ == "__main__":
    # Hardcoded paths based on workspace context
    transforms_path = "/home/ltx/projects/BrainDance/ai_engine/demo/data_cache/test_scene_01/transforms.json"
    dataparser_path = "/home/ltx/projects/BrainDance/ai_engine/demo/outputs/test_scene_01/splatfacto/2025-12-03_023653/dataparser_transforms.json"
    output_path = "/home/ltx/projects/BrainDance/ai_engine/demo/webgl/my-3dgs-viewer/public/models/webgl_poses.json"
    
    export_aligned_poses(transforms_path, dataparser_path, output_path)
