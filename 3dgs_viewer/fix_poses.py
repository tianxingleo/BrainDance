import json
import numpy as np
import os
import shutil
from pathlib import Path
import re

work_dir = Path('/home/ltx/braindance_workspace/scene_auto_sync')
data_dir = work_dir / 'data'
public_dir = Path('/home/jiangbeihu/ltx/projects/BrainDance/3dgs_viewer/my-3dgs-viewer/public/models')
final_webgl_poses = public_dir / 'webgl_poses.json'
webgl_images_dir = public_dir / 'images'
webgl_images_dir.mkdir(exist_ok=True, parents=True)

# 1. READ ORIGINAL TRANSFORMS FOR METADATA
with open(data_dir / 'transforms.json', 'r') as f:
    origData = json.load(f)
fl_x = origData.get("fl_x")
fl_y = origData.get("fl_y")
w = origData.get("w")
h = origData.get("h")
camera_model = origData.get("camera_model")

# 2. READ NS-EXPORT CAMERAS
cameras_json_path = work_dir / 'cameras_export' / 'transforms_train.json'
with open(cameras_json_path, 'r') as f:
    frames_list = json.load(f)

webgl_poses = []

for frame in frames_list:
    # It is a 3x4 matrix
    c2w_3x4 = np.array(frame['transform'])
    # Pad to 4x4
    c2w = np.eye(4)
    c2w[:3, :4] = c2w_3x4
    
    # Transpose and flatten for Three.js
    c2w_threejs = c2w.T.flatten().tolist()
    
    file_path = frame.get('file_path')
    img_name = Path(file_path).name
    
    src_img = data_dir / 'images' / img_name
    if not src_img.exists():
        src_img = Path(file_path)
        
    if src_img.exists():
        shutil.copy2(str(src_img), str(webgl_images_dir / img_name))
    
    webgl_poses.append({
        'id': img_name,
        'fl_y': fl_y,
        'h': h,
        'matrix': c2w_threejs,
        'image_url': f'/models/images/{img_name}'
    })

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s['id'])]
webgl_poses.sort(key=natural_sort_key)

output_data = {
    'w': w,
    'h': h,
    'fl_x': fl_x,
    'fl_y': fl_y,
    'camera_model': camera_model,
    'frames': webgl_poses
}
    
with open(final_webgl_poses, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"✅ Reparsed {len(webgl_poses)} poses directly!")
