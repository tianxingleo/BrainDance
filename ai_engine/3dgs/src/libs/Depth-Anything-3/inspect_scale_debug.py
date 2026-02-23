
import numpy as np
import glob
import os

def inspect_data(streaming_dir):
    print(f"Inspecting {streaming_dir}...")
    
    # 1. Inspect Poses
    poses_file = os.path.join(streaming_dir, "camera_poses.txt")
    if not os.path.exists(poses_file):
        print("❌ Poses file not found!")
        return
        
    poses = np.loadtxt(poses_file).reshape(-1, 4, 4)
    trans = poses[:, :3, 3]
    
    # Calculate pose statistics
    mean_t = np.mean(trans, axis=0)
    std_t = np.std(trans, axis=0)
    min_t = np.min(trans, axis=0)
    max_t = np.max(trans, axis=0)
    
    # Calculate trajectory length (cumulative distance)
    dists = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    traj_len = np.sum(dists)
    avg_step = np.mean(dists)
    
    print("\n=== Pose Statistics (World Units) ===")
    print(f"Num Frames: {len(poses)}")
    print(f"Translation Mean: {mean_t}")
    print(f"Translation Std:  {std_t}")
    print(f"Bounds (Min): {min_t}")
    print(f"Bounds (Max): {max_t}")
    print(f"Trajectory Length: {traj_len:.4f}")
    print(f"Avg Step Size:     {avg_step:.4f}")
    
    # 2. Inspect Depth
    npz_files = sorted(glob.glob(os.path.join(streaming_dir, "results_output", "frame_*.npz")))
    if not npz_files:
        print("❌ No NPZ files found!")
        return

    print(f"\n=== Depth Statistics (Sampled {min(5, len(npz_files))} frames) ===")
    
    depth_means = []
    depth_medians = []
    depth_maxs = []
    
    for i in np.linspace(0, len(npz_files)-1, 5, dtype=int):
        data = np.load(npz_files[i])
        d = data['depth']
        conf = data['conf']
        
        # Filter valid depth
        valid_mask = (d > 0) & (conf > 0.5)
        if valid_mask.sum() == 0:
            print(f"Frame {i}: No valid depth!")
            continue
            
        valid_d = d[valid_mask]
        
        d_mean = np.mean(valid_d)
        d_med = np.median(valid_d)
        d_max = np.max(valid_d)
        
        depth_means.append(d_mean)
        depth_medians.append(d_med)
        depth_maxs.append(d_max)
        
        print(f"Frame {i}: Mean={d_mean:.4f}, Median={d_med:.4f}, Max={d_max:.4f}, ValidPixel%={valid_mask.mean()*100:.1f}%")

    avg_depth_mean = np.mean(depth_means)
    
    print(f"\n=== Comparison ===")
    print(f"Avg Scene Depth: {avg_depth_mean:.4f}")
    print(f"Trajectory bounds range: {np.linalg.norm(max_t - min_t):.4f}")
    
    ratio = avg_depth_mean / (traj_len + 1e-6)
    print(f"Depth / Trajectory Length Ratio: {ratio:.4f}")
    
    if ratio > 100:
        print("⚠️  WARNING: Depth is much larger than camera movement. Scenes might look like a sphere around origin.")
    elif ratio < 0.01:
        print("⚠️  WARNING: Depth is much smaller than camera movement. Scenes might look flat.")
    else:
        print("✅ Scale seems reasonable.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='output/sugar_streaming')
    args = parser.parse_args()
    inspect_data(args.dir)
