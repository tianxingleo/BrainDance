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

# 🔥【绝杀】强制将编译好的系统级 colmap 路径提到最前面
# 这样系统找 colmap 时，第一个看到的就是 /usr/local/bin 里的那个好版本
sys_path = "/usr/local/bin"
current_path = os.environ.get("PATH", "")

if sys_path not in current_path.split(os.pathsep)[0]: # 如果不在第一位
    print(f"⚡ [环境修正] 强制设置 PATH 优先级: {sys_path} -> Priority High")
    os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"

# 验证一下
colmap_loc = shutil.which("colmap")
print(f"🧐 [自检] 当前脚本使用的 COLMAP 路径: {colmap_loc}")

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
MAX_IMAGES = 200 # 🔥 全局最大图片数量限制

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
        
        # 1. 定义两个隔离区域
        # temp_dir: 存放 ffmpeg 原始产物
        temp_dir = work_dir / "temp_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        img_dir = data_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. FFmpeg 抽帧 (输出到临时目录)
        print(f"    -> 正在抽帧到临时目录...")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(work_dir / video_src.name), 
            "-vf", "scale=1920:-1,fps=4", "-q:v", "2", 
            str(temp_dir / "frame_%05d.jpg")
        ], check=True) 
        
        # 3. 【关键步骤】数量限制与迁移 (从 temp -> images)
        print("    -> 正在执行【数量限制与迁移】...")
        
        # 读取所有图片
        all_candidates = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")))
        total_candidates = len(all_candidates)
        
        final_images_list = []
        
        if total_candidates > MAX_IMAGES:
            print(f"    ⚠️ 图片过多 ({total_candidates}), 正在均匀选取 {MAX_IMAGES} 张...")
            # 均匀采样索引
            indices = np.linspace(0, total_candidates - 1, MAX_IMAGES, dtype=int)
            # 使用集合去重
            indices = sorted(list(set(indices)))
            
            for idx in indices:
                final_images_list.append(all_candidates[idx])
        else:
            print(f"    ✅ 图片数量 ({total_candidates}) 未超标，全部保留。")
            final_images_list = all_candidates

        # 执行复制
        for img_path in final_images_list:
            shutil.copy2(str(img_path), str(img_dir / img_path.name))
            
        print(f"    ✅ 已将 {len(final_images_list)} 张干净图片移入 COLMAP 专用目录。")
        print(f"    🧹 正在清理临时文件...")
        shutil.rmtree(temp_dir) 
        
        # =========================================================
        # 🚀 COLMAP 启动 (手动挡 + 强制修正)
        # =========================================================
        print(f"    ✅ 准备启动 COLMAP (Linux GPU 模式)...")
        
        colmap_output_dir = data_dir / "colmap"
        colmap_output_dir.mkdir(parents=True, exist_ok=True)
        database_path = colmap_output_dir / "database.db"
        
        # 绝对路径调用
        system_colmap_exe = "/usr/local/bin/colmap" 
        if not os.path.exists(system_colmap_exe):
            found_path = shutil.which("colmap")
            if found_path and "conda" not in found_path:
                system_colmap_exe = found_path
                print(f"    ⚠️ 警告: /usr/local/bin/colmap 不存在，尝试使用: {system_colmap_exe}")

        def run_colmap_step(cmd, step_desc):
            print(f"\n⚡ {step_desc}...")
            try:
                with subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    env=env,
                    bufsize=1 
                ) as process:
                    for line in process.stdout:
                        print(line, end='') 
                    process.wait()
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, cmd)
            except Exception as e:
                print(f"\n❌ {step_desc} 执行异常: {e}")
                raise e

        # 1. Feature Extractor
        run_colmap_step([
            system_colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(img_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1"
        ], "[1/4] GPU 特征提取")

        # 2. Sequential Matcher
        run_colmap_step([
            system_colmap_exe, "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", "25" 
        ], "[2/4] GPU 顺序匹配")

        # 3. Mapper
        sparse_output_dir = colmap_output_dir / "sparse" / "0"
        sparse_output_dir.mkdir(parents=True, exist_ok=True)
        
        run_colmap_step([
            system_colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(img_dir),
            "--output_path", str(sparse_output_dir)
        ], "[3/4] 稀疏重建 (Mapper)")

        print(f"✅ COLMAP 计算完成！正在检查并修正目录结构...")

        # =========================================================
        # 🔧 [3.5] 目录结构强力修正 (Auto-Fixer)
        # =========================================================
        colmap_root = colmap_output_dir
        sparse_root = colmap_root / "sparse"
        target_dir_0 = sparse_root / "0"
        target_dir_0.mkdir(parents=True, exist_ok=True)

        required_files_bin = ["cameras.bin", "images.bin", "points3D.bin"]
        required_files_txt = ["cameras.txt", "images.txt", "points3D.txt"]
        
        model_found = False

        # 1. 检查是不是已经在 sparse/0
        if all((target_dir_0 / f).exists() for f in required_files_bin):
            model_found = True
        elif all((target_dir_0 / f).exists() for f in required_files_txt):
            model_found = True
            
        # 2. 检查是不是在 sparse 根目录 -> 搬运
        if not model_found:
            if all((sparse_root / f).exists() for f in required_files_bin):
                print("    🔧 检测到 BIN 模型在 sparse 根目录，正在归位...")
                for f in required_files_bin:
                    shutil.move(str(sparse_root / f), str(target_dir_0 / f))
                model_found = True
            elif all((sparse_root / f).exists() for f in required_files_txt):
                print("    🔧 检测到 TXT 模型在 sparse 根目录，正在归位...")
                for f in required_files_txt:
                    shutil.move(str(sparse_root / f), str(target_dir_0 / f))
                model_found = True

        # 3. 检查是不是在子目录 -> 搬运
        if not model_found:
            for root, dirs, files in os.walk(sparse_root):
                if all(f in files for f in required_files_bin):
                    src_path = Path(root)
                    if src_path == target_dir_0: continue
                    print(f"    🔧 在子目录 {src_path} 找到 BIN 模型，正在归位...")
                    for f in required_files_bin:
                        shutil.move(str(src_path / f), str(target_dir_0 / f))
                    model_found = True
                    break
                if all(f in files for f in required_files_txt):
                    src_path = Path(root)
                    if src_path == target_dir_0: continue
                    print(f"    🔧 在子目录 {src_path} 找到 TXT 模型，正在归位...")
                    for f in required_files_txt:
                        shutil.move(str(src_path / f), str(target_dir_0 / f))
                    model_found = True
                    break

        if not model_found:
            raise FileNotFoundError("COLMAP Mapper failed to generate valid model files.")

        # 4. 生成 transforms.json (跳过 COLMAP)
        print(f"\n🔄 [4/4] 生成 transforms.json (Nerfstudio)...")
        res = subprocess.run([
            "ns-process-data", "images",
            "--data", str(img_dir),
            "--output-dir", str(data_dir),
            "--skip-colmap", 
            "--verbose"
        ], check=True, env=env, capture_output=True, text=True)
        print(res.stdout)

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