import subprocess
import sys
import shutil
import os
import time
import base64
import json
import logging
import numpy as np
import cv2
from pathlib import Path

# ================= 🔧 用户配置区域 =================
OPENAI_API_KEY = ""  # 如果有 Key，填入后开启 GPT-4o 语义分析
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
MODEL_YOLO = 'yolov8x-worldv2.pt'
MODEL_SAM = 'sam_l.pt'

# ================= 📦 库导入 =================
logging.getLogger('nerfstudio').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

try:
    from plyfile import PlyData, PlyElement
    has_plyfile = True
except ImportError:
    has_plyfile = False
    print("⚠️ 未安装 plyfile 库，将跳过 PLY 后处理清洗。建议: pip install plyfile")

try:
    from openai import OpenAI
    has_openai = True if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-") else False
except: has_openai = False

try:
    from ultralytics import YOLOWorld, SAM
    has_ultralytics = True
except: has_ultralytics = False

# ================= 🧹 新增：基于相机的 PLY 清洗算法 =================

def clean_ply_based_on_cameras(ply_path, json_path, output_path):
    """
    读取 PLY 和相机参数，删除位于相机轨迹包围盒之外的噪点。
    """
    if not has_plyfile: return False
    
    print(f"\n🧹 [后处理] 正在基于相机轨迹清洗点云噪声...")
    
    # 1. 读取相机数据
    try:
        with open(json_path, 'r') as f:
            frames = json.load(f)["frames"]
        
        positions = []
        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            positions.append(c2w[:3, 3])
        
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        
        # 计算所有相机到中心的距离
        dists = np.linalg.norm(positions - center, axis=1)
        max_cam_radius = np.max(dists)
        avg_cam_radius = np.mean(dists)
        
        print(f"    -> 相机群统计: 中心={center[:3]}, 平均半径={avg_cam_radius:.2f}, 最大半径={max_cam_radius:.2f}")

    except Exception as e:
        print(f"⚠️ 读取相机数据失败: {e}")
        return False

    # 2. 读取 PLY 文件
    try:
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        
        # 提取点的位置 (x, y, z)
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.stack([x, y, z], axis=1)
        
        # 提取不透明度 (opacity) - 3DGS 存储的是 logit(opacity)
        # 通常 opacity < 0.05 也是不可见的噪点
        opacities = 1 / (1 + np.exp(-vertex['opacity'])) # sigmoid
        
        original_count = len(points)
        
        # --- 核心逻辑：定义保留区域 ---
        # 策略：保留 [中心点] 到 [最大相机半径 * 1.2] 范围内的点
        # 1.2 是一个安全系数，防止切掉边缘
        
        # 计算每个点到中心的距离
        point_dists = np.linalg.norm(points - center, axis=1)
        
        # 判定条件 1: 距离过滤 (只保留相机包围圈内的点 + 一点余量)
        # 注意：这主要适用于“物体模式”。如果是场景模式，这个逻辑会被跳过。
        is_object_mode = True # 默认假设物体模式，如果你有前面的 scene_type 变量更好
        
        # 这里我们用简单的启发式：如果相机聚合度高（物体），就切；否则放宽
        # 为了安全，我们用 max_cam_radius * 1.5 作为界限。
        # 任何比相机还要远 1.5 倍的点，通常都是背景漂浮物。
        radius_mask = point_dists < (max_cam_radius * 1.5)
        
        # 判定条件 2: 透明度过滤 (删除极其稀薄的点)
        opacity_mask = opacities > 0.02 
        
        # 合并掩码
        final_mask = radius_mask & opacity_mask
        
        # 应用过滤
        filtered_vertex = vertex[final_mask]
        new_count = len(filtered_vertex)
        
        print(f"    -> 原始点数: {original_count}")
        print(f"    -> 剩余点数: {new_count} (删除了 {original_count - new_count} 个噪点)")
        
        # 3. 保存新的 PLY
        ply_element = PlyElement.describe(filtered_vertex, 'vertex')
        PlyData([ply_element]).write(str(output_path))
        print(f"✅ 清洗完成！已保存至: {output_path}")
        return True

    except Exception as e:
        print(f"❌ PLY 清洗失败: {e}")
        return False

# ================= 🧠 AI & 几何分析模块 (保留) =================

class SmartMasker:
    def __init__(self):
        if not has_ultralytics: return
        print(f"\n🦾 [AI 引擎] 加载 YOLO+SAM (RTX 5070)...")
        self.detector = YOLOWorld(MODEL_YOLO)
        self.segmentor = SAM(MODEL_SAM)

    def generate_mask(self, image_path, prompt):
        try:
            self.detector.set_classes([prompt])
            det = self.detector.predict(image_path, conf=0.15, verbose=False)
            if len(det[0].boxes) == 0: return None
            sam = self.segmentor(image_path, bboxes=det[0].boxes.xyxy, verbose=False)
            if len(sam[0].masks) == 0: return None
            
            final = np.zeros(sam[0].orig_shape[:2], dtype=np.uint8)
            for m in sam[0].masks.data.cpu().numpy():
                m_u8 = (m * 255).astype(np.uint8)
                if m_u8.shape != final.shape: m_u8 = cv2.resize(m_u8, (final.shape[1], final.shape[0]))
                final = cv2.bitwise_or(final, m_u8)
            return final
        except: return None

def analyze_scene_geometry(json_path):
    try:
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        if not frames: return "scene", False, "none"
        pos = [np.array(f["transform_matrix"])[:3, 3] for f in frames]
        fwds = [np.array(f["transform_matrix"])[:3, :3] @ np.array([0,0,-1]) for f in frames]
        center = np.mean(pos, axis=0)
        vecs = center - np.array(pos)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6)
        ratio = np.sum(np.sum(np.array(fwds) * vecs, axis=1) > 0) / len(frames)
        
        # 返回：类型, 是否需抠图, 主体名, 是否启用激进后处理
        if ratio > 0.6: return "object", True, "object", True
        else: return "scene", False, "none", False
    except: return "scene", False, "none", False

def analyze_with_gpt4o(images_dir):
    if not has_openai: return None
    print(f"\n🧠 [GPT-4o] 分析中...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    imgs = sorted(list(images_dir.glob("*.jpg")))
    if not imgs: return None
    samples = imgs[::max(1, len(imgs)//6)][:6]
    
    content = [{"type": "text", "text": "Analyze for 3D Gaussian Splatting."}]
    for p in samples:
        with open(p, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}","detail":"low"}})
    
    content.append({"type": "text", "text": """Return JSON: {"type": "object"|"scene", "subject": string, "masking_needed": bool}"""})
    try:
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":content}], response_format={"type":"json_object"})
        ret = json.loads(res.choices[0].message.content)
        # GPT 不直接决定后处理力度，我们在后面逻辑判断
        return ret
    except: return None

# ================= 🚀 主流程 =================

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine 6.0] 启动任务: {project_name}")
    
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 

    # [Step 1] 数据处理
    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 已有 COLMAP 数据")
    else:
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        shutil.copy(str(video_src), str(work_dir / video_src.name))

        print(f"\n🎥 [1/3] COLMAP 解算")
        img_dir = data_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(work_dir / video_src.name), "-vf", "scale=1920:-1,fps=4", "-q:v", "2", str(img_dir / "frame_%05d.jpg")], check=True)
        
        res = subprocess.run(["ns-process-data", "images", "--data", str(img_dir), "--output-dir", str(data_dir), "--verbose"], check=True, env=env, capture_output=True, text=True)
        if "COLMAP only found poses" in res.stdout: raise RuntimeError("COLMAP 失败")

    # [Step 2] 智能训练
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []

    # 标记变量：是否启用激进的后处理清洗
    enable_aggressive_cleaning = False 

    if run_dirs:
        print(f"\n⏩ [训练跳过] 已完成")
        # 如果跳过训练，我们需要重新判断一下类型来决定是否清洗
        type_res, _, _, enable_aggressive_cleaning = analyze_scene_geometry(transforms_file)
    else:
        img_dir = data_dir / "images"
        gpt_res = analyze_with_gpt4o(img_dir)
        
        if gpt_res:
            stype, need_mask, subj = gpt_res["type"], gpt_res["masking_needed"], gpt_res["subject"]
            enable_aggressive_cleaning = (stype == "object") # 物体模式启用清洗
        else:
            stype, need_mask, subj, enable_aggressive_cleaning = analyze_scene_geometry(transforms_file)

        collider_params = ["near_plane", "0.05", "far_plane", "100.0"] if stype == "scene" else ["near_plane", "2.0", "far_plane", "6.0"]
        
        if stype == "object" and need_mask and has_ultralytics:
            print(f"💡 [AI] 物体模式 ('{subj}') -> 启动 SAM 抠图...")
            masks_dir = data_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            if not any(masks_dir.iterdir()):
                masker = SmartMasker()
                imgs = sorted(list(img_dir.glob("*.jpg")))
                for i, p in enumerate(imgs):
                    m = masker.generate_mask(str(p), subj)
                    if m is not None: cv2.imwrite(str(masks_dir / (p.stem+".png")), m)
                    print(f"    生成蒙版 {i+1}/{len(imgs)}", end="\r")
                print("")
                with open(transforms_file,'r') as f: d=json.load(f)
                for fr in d["frames"]:
                    mp = masks_dir/(Path(fr["file_path"]).stem+".png")
                    if mp.exists(): fr["mask_path"] = f"masks/{mp.name}"
                with open(transforms_file,'w') as f: json.dump(d,f,indent=4)

        print(f"\n🧠 [2/3] 开始训练...")
        subprocess.run([
            "ns-train", "splatfacto", "--data", str(data_dir), "--output-dir", str(output_dir), 
            "--experiment-name", project_name, "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005", 
            "--pipeline.model.enable-collider", "True", "--pipeline.model.collider-params", *collider_params,
            "--max-num-iterations", "15000", "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True", "colmap"
        ], check=True, env=env)

    # [Step 3] 导出
    print(f"\n💾 [3/3] 导出结果")
    if not run_dirs: run_dirs = sorted(list(search_path.glob("*")))
    if not run_dirs: return None
    subprocess.run(["ns-export", "gaussian-splat", "--load-config", str(run_dirs[-1]/"config.yml"), "--output-dir", str(work_dir)], check=True, env=env)
    time.sleep(5)

    # [Step 3.5] 基于相机轨迹的后处理清洗 (New!)
    raw_ply = work_dir / "point_cloud.ply"
    cleaned_ply = work_dir / "point_cloud_cleaned.ply"
    
    final_ply_to_copy = raw_ply # 默认用原版
    
    # 只有在物体模式(且 has_plyfile)下才执行清洗，防止把房间切碎
    if enable_aggressive_cleaning and has_plyfile and raw_ply.exists():
        if clean_ply_based_on_cameras(raw_ply, transforms_file, cleaned_ply):
            final_ply_to_copy = cleaned_ply # 清洗成功，替换为清洗版

    # [Step 4] 回传
    print(f"\n📦 [IO 同步] 回传结果...")
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)
    
    # WebGL Pose
    try:
        with open(transforms_file,'r') as f: d=json.load(f)
        frames = [{"file_path": fr["file_path"], "pose_matrix_c2w": fr["transform_matrix"]} for fr in d["frames"]]
        with open(target_dir/"webgl_poses.json",'w') as f: json.dump({"camera_model": d.get("camera_model","OPENCV"), "frames": frames}, f, indent=4)
    except: pass

    final_dst = target_dir / f"{project_name}.ply"
    if final_ply_to_copy.exists():
        shutil.copy(str(final_ply_to_copy), str(final_dst))
        shutil.rmtree(work_dir)
        print(f"✅ 完成！模型已保存至: {final_dst}")
        return str(final_dst)
    
    return None

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])
    if video_file.exists(): run_pipeline(video_file, "scene_auto_sync")
    else: print(f"❌ 找不到视频")