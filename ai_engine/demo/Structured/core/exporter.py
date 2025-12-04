import shutil
import json
import time
from pathlib import Path
from utils.common import run_command
from utils.math_utils import convert_pose_to_webgl

def export_results(project_name, work_dir, output_dir, data_dir, target_dir_root):
    print(f"\n💾 [3/3] 导出结果")
    
    # 查找最新的训练结果目录
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []
    
    if not run_dirs:
        raise FileNotFoundError("❌ 未找到训练结果目录，无法导出。")
        
    latest_run = run_dirs[-1]
    config_path = latest_run / "config.yml"
    
    print(f"    -> 加载配置: {config_path.name}")
    
    # 调用 ns-export 导出 Splat
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(work_dir)
    ]
    run_command(cmd_export)
    
    print("    -> 等待文件写入...")
    time.sleep(5) 
    
    # 同步回项目目录
    sync_to_local(project_name, work_dir, data_dir, target_dir_root)

def sync_to_local(project_name, work_dir, data_dir, target_dir_root):
    target_dir = target_dir_root / "results"
    target_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📦 [IO 同步] 正在将结果保存至: {target_dir}")
    
    # 1. 生成 WebGL JSON
    transforms_src = data_dir / "transforms.json"
    generate_webgl_json(transforms_src, target_dir / "webgl_poses.json")
    
    # 2. 复制 PLY
    ply_src = work_dir / "point_cloud.ply"
    # 兼容部分版本输出为 splat.ply 的情况
    if not ply_src.exists():
        ply_src = work_dir / "splat.ply"
        
    if ply_src.exists():
        final_ply_path = target_dir / f"{project_name}.ply"
        shutil.copy(str(ply_src), str(final_ply_path))
        
        # 同时也复制原始 transforms.json 备用
        if transforms_src.exists():
            shutil.copy(str(transforms_src), str(target_dir / "transforms.json"))
            
        print(f"✅ 导出成功！")
        print(f"    - 模型: {final_ply_path}")
        print(f"    - 姿态: {target_dir / 'webgl_poses.json'}")
    else:
        print("❌ 错误：未找到导出的 PLY 文件。")

def generate_webgl_json(src_json, dst_json):
    if not src_json.exists(): 
        return
        
    try:
        with open(src_json, 'r') as f:
            data = json.load(f)
        
        webgl_frames = []
        for frame in data["frames"]:
            webgl_frames.append({
                "file_path": frame["file_path"],
                "pose_matrix_c2w": convert_pose_to_webgl(frame["transform_matrix"])
            })
            
        # 复制除了 frames 以外的元数据 (camera_model, w, h 等)
        webgl_data = {k: v for k, v in data.items() if k != "frames"}
        webgl_data["frames"] = webgl_frames
        
        with open(dst_json, 'w') as f:
            json.dump(webgl_data, f, indent=4)
        print("    -> WebGL 姿态文件已生成")
        
    except Exception as e:
        print(f"❌ WebGL 姿态生成失败: {e}")
