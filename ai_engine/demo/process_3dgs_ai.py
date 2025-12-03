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
# 1. OpenAI API Key (用于 GPT-4o 语义分析)
# 如果留空，将自动降级为使用“几何算法”进行分析，无需联网
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

# 2. Linux 工作区路径
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"

# 3. 硬件配置 (您有 5070，直接拉满)
# YOLO-World 模型: yolov8x-worldv2.pt (最强版)
# SAM 模型: sam_l.pt (Large版，精度最高)
MODEL_YOLO = 'yolov8x-worldv2.pt'
MODEL_SAM = 'sam_l.pt'

# ================= 📦 库导入与初始化 =================
logging.getLogger('nerfstudio').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

try:
    from openai import OpenAI
    has_openai = True if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-") else False
except ImportError:
    has_openai = False
    print("⚠️ 未安装 openai 库，将使用本地几何分析模式。")

try:
    from ultralytics import YOLOWorld, SAM
    has_ultralytics = True
except ImportError:
    has_ultralytics = False
    print("⚠️ 未安装 ultralytics 库，将跳过本地 AI 抠图。")

# ================= 🧠 核心 AI 模块 =================

class SmartMasker:
    """
    本地 AI 引擎：结合 YOLO-World (听觉/视觉) + SAM (触觉/分割)
    """
    def __init__(self):
        if not has_ultralytics: return
        print(f"\n🦾 [本地 AI 引擎] 正在加载大模型 (RTX 5070 算力全开)...")
        print(f"    -> 加载 {MODEL_YOLO} (语义识别)...")
        self.detector = YOLOWorld(MODEL_YOLO)
        print(f"    -> 加载 {MODEL_SAM} (像素分割)...")
        self.segmentor = SAM(MODEL_SAM)
        print("✅ AI 引擎就绪。")

    def generate_mask(self, image_path, prompt):
        """输入图片和提示词，返回蒙版"""
        try:
            # 1. YOLO-World: 寻找物体框
            self.detector.set_classes([prompt])
            det_results = self.detector.predict(image_path, conf=0.15, verbose=False)
            
            if len(det_results[0].boxes) == 0:
                return None # 没找到物体
            
            # 2. SAM: 根据框生成蒙版
            bboxes = det_results[0].boxes.xyxy
            sam_results = self.segmentor(image_path, bboxes=bboxes, verbose=False)
            
            if len(sam_results[0].masks) == 0:
                return None

            # 3. 合并蒙版
            final_mask = np.zeros(sam_results[0].orig_shape[:2], dtype=np.uint8)
            masks = sam_results[0].masks.data.cpu().numpy()
            
            for mask in masks:
                mask_uint8 = (mask * 255).astype(np.uint8)
                if mask_uint8.shape != final_mask.shape:
                     mask_uint8 = cv2.resize(mask_uint8, (final_mask.shape[1], final_mask.shape[0]))
                final_mask = cv2.bitwise_or(final_mask, mask_uint8)
                
            return final_mask
        except Exception as e:
            print(f"❌ 分割错误: {e}")
            return None

def analyze_with_gpt4o(images_dir):
    """
    使用 GPT-4o 多模态大模型分析场景
    """
    if not has_openai: return None

    print(f"\n🧠 [GPT-4o] 正在上传关键帧进行语义理解...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 均匀抽取 6 张图
    all_imgs = sorted(list(images_dir.glob("*.jpg")))
    if not all_imgs: return None
    step = max(1, len(all_imgs) // 6)
    sampled_imgs = all_imgs[::step][:6]

    content = [{"type": "text", "text": "Analyzing frames for 3D Gaussian Splatting training."}]
    
    for img_path in sampled_imgs:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
        })

    prompt = """
    Analyze these images and output a JSON with:
    1. "type": "object" (if focusing on specific items like a toy, shoe, person) or "scene" (room, street, large area).
    2. "subject": If "object", give a short English specific description for detection (e.g. "red nike shoes", "anime figure"). If "scene", use "none".
    3. "masking_needed": Boolean. True if it's an object with messy background. False if it's a scene OR object with white/clean background.
    """
    content.append({"type": "text", "text": prompt})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        print(f"🤖 GPT 洞察: 类型=[{result['type']}] | 主体=[{result['subject']}] | 需抠图=[{result['masking_needed']}]")
        return result
    except Exception as e:
        print(f"⚠️ GPT 分析失败: {e}")
        return None

def analyze_scene_geometry(json_path):
    """
    (备用方案) 几何分析算法，当 GPT 不可用时使用
    """
    print(f"📐 [几何分析] GPT 未启用，正在计算相机轨迹聚合度...")
    try:
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        if not frames: return "scene", False, "none"

        positions, forwards = [], []
        for f in frames:
            c2w = np.array(f["transform_matrix"])
            positions.append(c2w[:3, 3])
            forwards.append(c2w[:3, :3] @ np.array([0, 0, -1]))
            
        center = np.mean(positions, axis=0)
        vecs = center - np.array(positions)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6)
        
        # 计算视线聚合度
        ratio = np.sum(np.sum(np.array(forwards) * vecs, axis=1) > 0) / len(frames)
        
        if ratio > 0.6:
            print(f"    -> 聚合度 {ratio:.2f} (>0.6)，判定为物体模式。")
            return "object", True, "object" # 几何模式下默认开启抠图，通用提示词
        else:
            print(f"    -> 聚合度 {ratio:.2f} (<0.6)，判定为场景模式。")
            return "scene", False, "none"
    except:
        return "scene", False, "none"


# ================= 🚀 主流程 =================

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine 5.0] 启动任务: {project_name}")
    print(f"⚡ 硬件加速: Intel 14600KF + RTX 5070 | AI 核心: GPT-4o + YOLO-World + SAM")
    
    # 1. 路径解析
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 

    # ================= [Step 1] 数据处理 (标准流程) =================
    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 检测到 COLMAP 数据")
    else:
        print(f"🆕 [新任务] 初始化工作区...")
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        shutil.copy(str(video_src), str(work_dir / video_src.name))

        print(f"\n🎥 [1/3] 视频处理 (COLMAP)")
        img_dir = data_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # 保持原有 1920 / 4fps 设置
        subprocess.run([
            "ffmpeg", "-y", "-i", str(work_dir / video_src.name), 
            "-vf", "scale=1920:-1,fps=4", "-q:v", "2", 
            str(img_dir / "frame_%05d.jpg")
        ], check=True) 
        
        # COLMAP
        res = subprocess.run([
            "ns-process-data", "images",
            "--data", str(img_dir),
            "--output-dir", str(data_dir),
            "--verbose"
        ], check=True, env=env, capture_output=True, text=True)
        print(res.stdout)
        if "COLMAP only found poses" in res.stdout: raise RuntimeError("COLMAP 失败")

    # ================= [Step 2] 智能分析与训练 =================
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []

    if run_dirs:
        print(f"\n⏩ [训练跳过] 已完成")
    else:
        img_dir = data_dir / "images"
        
        # --- A. 智能决策阶段 ---
        gpt_result = analyze_with_gpt4o(img_dir)
        
        if gpt_result:
            # 使用 GPT 的结果
            scene_type = gpt_result["type"]
            need_mask = gpt_result["masking_needed"]
            subject_prompt = gpt_result["subject"]
        else:
            # 回退到几何分析
            scene_type, need_mask, subject_prompt = analyze_scene_geometry(transforms_file)

        # --- B. 决策执行阶段 ---
        collider_params = []
        
        if scene_type == "scene":
            print("💡 策略：【场景模式】 -> 禁用抠图，宽松裁剪。")
            collider_params = ["near_plane", "0.05", "far_plane", "100.0"]
            
        else: # object
            print(f"💡 策略：【物体模式】 -> 主体: '{subject_prompt}'")
            collider_params = ["near_plane", "2.0", "far_plane", "6.0"]
            
            # --- C. 执行 YOLO+SAM 抠图 ---
            if need_mask and has_ultralytics:
                print(f"✂️ [AI 执行] 启动 YOLO-World + SAM 语义抠图...")
                masks_dir = data_dir / "masks"
                masks_dir.mkdir(exist_ok=True)
                
                # 检查是否已处理过
                if not any(masks_dir.iterdir()):
                    masker = SmartMasker()
                    all_imgs = sorted(list(img_dir.glob("*.jpg")))
                    processed_count = 0
                    
                    for img_p in all_imgs:
                        mask = masker.generate_mask(str(img_p), subject_prompt)
                        if mask is not None:
                            # 存为 png
                            cv2.imwrite(str(masks_dir / (img_p.stem + ".png")), mask)
                            processed_count += 1
                        print(f"    处理: {img_p.name} ... {'✅' if mask is not None else '⚠️'}", end="\r")
                    
                    print(f"\n✅ 蒙版生成完成：{processed_count}/{len(all_imgs)}")
                    
                    # 更新 transforms.json
                    with open(transforms_file, 'r') as f: meta = json.load(f)
                    for frame in meta["frames"]:
                        msk_p = masks_dir / (Path(frame["file_path"]).stem + ".png")
                        if msk_p.exists():
                            frame["mask_path"] = f"masks/{msk_p.name}"
                    with open(transforms_file, 'w') as f: json.dump(meta, f, indent=4)
                else:
                    print("⏩ 检测到现有蒙版，跳过生成。")

        # --- D. 开始训练 ---
        print(f"\n🧠 [2/3] 开始训练...")
        cmd_train = [
            "ns-train", "splatfacto",
            "--data", str(data_dir),
            "--output-dir", str(output_dir),
            "--experiment-name", project_name,
            "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005",
            # 插入智能参数
            "--pipeline.model.enable-collider", "True",
            "--pipeline.model.collider-params", *collider_params,
            
            "--max-num-iterations", "15000",
            "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True",
            "colmap",
        ]
        subprocess.run(cmd_train, check=True, env=env)

    # ================= [Step 3] 导出结果 (保持原功能) =================
    print(f"\n💾 [3/3] 导出结果")
    if not run_dirs: run_dirs = sorted(list(search_path.glob("*")))
    if not run_dirs: return None
        
    latest_run = run_dirs[-1]
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(latest_run / "config.yml"),
        "--output-dir", str(work_dir)
    ]
    subprocess.run(cmd_export, check=True, env=env)
    time.sleep(5)

    # ================= [Step 4] 回传与姿态处理 (保持原功能) =================
    print(f"\n📦 [IO 同步] 回传至 Windows...")
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)

    temp_ply = work_dir / "point_cloud.ply"
    if not temp_ply.exists(): temp_ply = work_dir / "splat.ply"
    
    # WebGL 姿态生成 (保留)
    final_webgl_poses = target_dir / "webgl_poses.json"
    if (data_dir / "transforms.json").exists():
        try:
            with open(data_dir / "transforms.json", 'r') as f: data = json.load(f)
            webgl_frames = []
            for frame in data["frames"]:
                c2w = np.array(frame["transform_matrix"], dtype=np.float32)
                # 计算 W2C
                w2c = np.linalg.inv(c2w) 
                webgl_frames.append({
                    "file_path": frame["file_path"],
                    "pose_matrix_c2w": c2w.tolist() # 保留 C2W
                })
            with open(final_webgl_poses, 'w') as f:
                json.dump({"camera_model": data.get("camera_model","OPENCV"), 
                           "frames": webgl_frames}, f, indent=4)
            print(f"✅ WebGL 姿态已保存: {final_webgl_poses}")
        except Exception as e: print(f"❌ 姿态处理失败: {e}")

    final_ply = target_dir / f"{project_name}.ply"
    if temp_ply and temp_ply.exists():
        shutil.copy(str(temp_ply), str(final_ply))
        if (data_dir / "transforms.json").exists():
            shutil.copy(str(data_dir / "transforms.json"), str(target_dir / "transforms.json"))
        shutil.rmtree(work_dir)
        print(f"✅ 成功！模型已保存至: {final_ply}")
        return str(final_ply)
    
    return None

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])

    if video_file.exists():
        # 如果你想重新触发 GPT 分析，请手动删除 transforms.json 或 masks 目录
        run_pipeline(video_file, "scene_auto_sync")
    else:
        print(f"❌ 找不到视频: {video_file}")