

import os
import json
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ================= Configuration =================
SOURCE_DIR = Path("/home/ltx/projects/Depth-Anything-3/output/sugar_streaming")
OUTPUT_DIR = Path("dn_splatter_dataset")
EXTRACTED_DIR = SOURCE_DIR / "extracted"
RESULTS_DIR = SOURCE_DIR / "results_output"

# ================= Helpers =================

def load_intrinsics(path):
    """Refactored to handle single line intrinsic file."""
    print(f"Loading intrinsics from {path}")
    with open(path, 'r') as f:
        lines = f.readlines()
        # Parse first line
        parts = lines[0].strip().split() if ',' not in lines[0] else lines[0].strip().split(',')
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

def load_poses(path):
    """Load camera poses from text file."""
    print(f"Loading poses from {path}")
    poses = []
    with open(path, 'r') as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            poses.append(np.array(nums).reshape(4, 4))
    return poses

def depth_to_normal(depth, K):
    """
    Calculate surface normals from depth map using pure numpy.
    depth: float32, (H, W) in meters
    K: (fx, fy, cx, cy)
    Returns: normal_img (H, W, 3) uint8 [0, 255]
    """
    fx, fy, cx, cy = K
    h, w = depth.shape

    # Gradients using numpy (central difference)
    # np.gradient returns [dy, dx] for 2D array
    zy, zx = np.gradient(depth)
    
    # Scale gradients by focal length/depth to account for perspective
    scale_x = fx / (depth + 1e-6)
    scale_y = fy / (depth + 1e-6)
    
    nx = -zx * scale_x
    ny = -zy * scale_y
    nz = np.ones_like(depth)
    
    # Normalize
    n = np.sqrt(nx**2 + ny**2 + nz**2)
    # Avoid division by zero
    n[n == 0] = 1.0
    
    normal = np.dstack((nx/n, ny/n, nz/n))
    
    # Map to [0, 255]
    # [-1, 1] -> [0, 255]
    normal_img = ((normal + 1) / 2 * 255).astype(np.uint8)
    return normal_img

def process():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    # Create output directories
    OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
    OUTPUT_DEPTH_DIR = OUTPUT_DIR / "depths"
    OUTPUT_NORMAL_DIR = OUTPUT_DIR / "normals_from_pretrain"
    OUTPUT_JSON_PATH = OUTPUT_DIR / "transforms.json"

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Metadata
    fx, fy, cx, cy = load_intrinsics(SOURCE_DIR / "intrinsic.txt")
    poses = load_poses(SOURCE_DIR / "camera_poses.txt")
    
    # 2. Match Files
    # Extracted images (1-based index usually: frame_000001.png)
    img_files = sorted(list(EXTRACTED_DIR.glob("*.png")))
    # Result npz files (0-based index usually: frame_0.npz)
    npz_files = sorted(list(RESULTS_DIR.glob("*.npz")), key=lambda x: int(x.stem.split('_')[1]))
    
    if len(img_files) != len(npz_files):
        print(f"Warning: Number of images ({len(img_files)}) and npz files ({len(npz_files)}) mismatch.")
        # We will truncate to the minimum
    
    num_frames = min(len(img_files), len(npz_files), len(poses))
    print(f"Processing {num_frames} frames...")

    frames_data = []
    
    # Transformation from OpenCV to OpenGL coordinate system
    # OpenCV: Right, Down, Forward
    # OpenGL: Right, Up, Back
    # Rotate 180 degrees around X axis
    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for i in tqdm(range(num_frames)):
        src_img_path = img_files[i]
        src_npz_path = npz_files[i]
        pose = poses[i]
        
        # Define output filenames
        frame_name = f"frame_{i:05d}"
        dst_img_name = f"{frame_name}.png"
        dst_depth_name = f"{frame_name}.png"
        dst_normal_name = f"{frame_name}.png"
        
        # 1. Copy Image
        shutil.copy2(src_img_path, OUTPUT_IMAGE_DIR / dst_img_name)
        
        # 2. Process Depth (Load npz)
        data = np.load(src_npz_path)
        depth_map = data['depth'] # float32 meters
        
        # Save as 16-bit uint16 (millimeters)
        depth_mm = (depth_map * 1000).astype(np.uint16)
        Image.fromarray(depth_mm).save(OUTPUT_DEPTH_DIR / dst_depth_name)
        
        # 3. Generate Normal
        normal_map = depth_to_normal(depth_map, (fx, fy, cx, cy))
        Image.fromarray(normal_map).save(OUTPUT_NORMAL_DIR / dst_normal_name)
        
        # 4. Process Pose
        # Check if pose contains any NaNs or Infs
        if np.isinf(pose).any() or np.isnan(pose).any():
            print(f"Warning: Invalid pose at frame {i}, skipping.")
            continue
            
        # Convert to OpenGL
        # Assuming input is c2w. If it was w2c, we would inv first.
        # DA3 output usually follows standard VSLAM/SfM: c2w.
        c2w_opengl = np.matmul(pose, flip_mat)
        
        # Append to list
        # DN-Splatter expects specific folder names relative to data dir
        frames_data.append({
            "file_path": f"images/{dst_img_name}",
            "depth_file_path": f"depths/{dst_depth_name}",
            # Use 'normals_from_pretrain' here to match what we created
            "normal_file_path": f"normals_from_pretrain/{dst_normal_name}", 
            "transform_matrix": c2w_opengl.tolist()
        })

    # 5. Write transforms.json
    # Assume image size from the first image
    h, w = depth_mm.shape
    
    output_json = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "camera_model": "OPENCV",
        "frames": frames_data
    }
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output_json, f, indent=4)
        
    print(f"Done! Dataset ready at {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    process()
