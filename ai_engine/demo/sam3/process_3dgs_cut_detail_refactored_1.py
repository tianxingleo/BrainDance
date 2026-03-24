# ==============================================================================
# 导入标准库和第三方库
# ==============================================================================
import subprocess
import sys
import shutil
import os
import time
import datetime
from pathlib import Path
import json
import numpy as np
import torch
import logging
import cv2
import re

# ================= 🧠 AI 依赖引入 =================
try:
    import dashscope
    from dashscope import MultiModalConversation
    from ultralytics import SAM, YOLOWorld
    HAS_AI = True
except ImportError:
    HAS_AI = False
    print("⚠️ [环境警告] 未检测到 dashscope 或 ultralytics 库。")

# ================= 🔧 基础配置 =================
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    project_name: str
    video_path: Path
    
    work_root: Path = Path.home() / "braindance_workspace"
    max_images: int = 180
    force_spherical_culling: bool = True 
    scene_radius_scale: float = 1.8
    keep_percentile: float = 0.9
    enable_ai: bool = True
    
    project_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    images_dir: Path = field(init=False)
    masks_dir: Path = field(init=False)
    transforms_file: Path = field(init=False)
    vocab_tree_path: Path = field(init=False)

    # SAM3 模型路径配置
    model_root: Path = Path("/home/ltx/workspace/ai/sam3") 

    def __post_init__(self):
        self.project_dir = self.work_root / self.project_name
        self.data_dir = self.project_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.transforms_file = self.data_dir / "transforms.json"
        self.vocab_tree_path = self.work_root / "vocab_tree_flickr100k_words.bin"

        self.model_root.mkdir(parents=True, exist_ok=True)
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

# 检查 plyfile
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

# ================= 🧠 AI 核心逻辑函数 =================

def get_central_object_prompt(images_dir: Path, sample_count=7):
    """使用 Qwen3.5-Plus 提取 Prompt"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY")
        return None

    print(f"\n🧠 [AI 分析] 正在调用 Qwen3.5-Plus 分析场景...")
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files: return None
    
    indices = np.linspace(0, len(image_files) - 1, sample_count, dtype=int)
    sampled_imgs = [image_files[i] for i in indices]
    
    content = [{"image": str(img_path)} for img_path in sampled_imgs]
    content.append({
        "text": (
            "这些是一个视频的抽帧图片。请分析画面中心始终存在的、最主要的一个物体是什么。"
            "我正在使用 SAM 3 (Segment Anything Model 3) 进行基于文本的视频跟踪。"
            "请输出一个【指代性明确】的英文短语 (Referring Expression)。"
            "⚠️ 关键策略："
            "1. 必须包含视觉特征（颜色、材质）。"
            "2. 描述物体本身，不要描述功能。"
            "3. 保持简短，直接输出英文短语，不要标点符号。"
        )
    })
    
    messages = [{"role": "user", "content": content}]

    try:
        response = dashscope.MultiModalConversation.call(model='qwen-plus', messages=messages)
        if response.status_code == 200:
            prompt_text = response.output.choices[0].message.content[0]["text"].strip()
            prompt_text = prompt_text.replace(".", "").replace('"', "").replace("'", "")
            print(f"    🤖 Qwen 认为中心物体是: [ \033[92m{prompt_text}\033[0m ]")
            return prompt_text
        else:
            print(f"❌ Qwen 调用失败: {response.code}")
            return None
    except Exception as e:
        print(f"❌ API 连接异常: {e}")
        return None

def clean_and_verify_mask(mask, img_name=""):
    """
    [经典版] 腐蚀 (Erosion) 模式
    """
    h, w = mask.shape
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2: return False, None, "Empty Mask"

    max_area = 0
    max_label = -1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
            
    if max_area < (h * w * 0.005): return False, None, "Too Small/Noise"
    if max_area > (h * w * 0.90): return False, None, f"Too Large ({max_area/(h*w):.0%})"

    cleaned_mask = (labels == max_label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, None, "No Contour"
    main_cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return False, None, "Hull Area 0"
    
    # 🔥 核心：使用腐蚀 (Erosion) 而不是膨胀
    kernel_size = 3 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=1)

    return True, cleaned_mask, "OK"

def run_ai_segmentation_pipeline(data_dir: Path):
    """
    SAM 3 + Premultiplied Alpha
    """
    if not HAS_AI: return False
    
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    debug_dir = data_dir / "debug_combo"
    debug_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cfg.transforms_file = data_dir / "transforms.json" 
    if not cfg.transforms_file.exists(): return False

    print(f"\n✂️ [智能分割] 初始化 (YOLO-World + SAM 3 Multi-Point)...")
    try:
        text_prompt = get_central_object_prompt(images_dir)
        if " on " in text_prompt: text_prompt = text_prompt.split(" on ")[0]
    except: text_prompt = "object"
    if not text_prompt: text_prompt = "object"
    print(f"    🎯 核心 Prompt: '\033[92m{text_prompt}\033[0m'")

    yolo_path = cfg.model_root / "yolov8s-worldv2.pt"
    sam_path = cfg.model_root / "sam3.pt"
    
    try:
        det_model = YOLOWorld(str(yolo_path) if yolo_path.exists() else "yolov8s-worldv2.pt")
        det_model.set_classes([text_prompt])
        sam_model = SAM(str(sam_path))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

    with open(cfg.transforms_file, 'r') as f: meta = json.load(f)
    frames_map = {Path(f["file_path"]).name: f for f in meta["frames"]}
    valid_frames_list = []
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    total_imgs = len(image_files)
    
    print(f"    -> 开始处理 {total_imgs} 帧...")
    start_time = time.time()

    for i, img_path in enumerate(image_files):
        elapsed = time.time() - start_time
        fps = (i + 1) / (elapsed + 1e-6)
        process_success = False 
        
        try:
            original_img = cv2.imread(str(img_path))
            if original_img is None: raise ValueError("无法读取图片")
            h_real, w_real = original_img.shape[:2]

            # --- Step 1: YOLO ---
            det_results = det_model.predict(img_path, conf=0.05, verbose=False) 
            bboxes = det_results[0].boxes.xyxy.cpu()
            
            final_box = None
            is_fallback = False 
            
            if len(bboxes) > 0:
                center_x, center_y = w_real / 2, h_real / 2
                min_dist = float('inf')
                for box in bboxes:
                    bx = (box[0] + box[2]) / 2
                    by = (box[1] + box[3]) / 2
                    dist = (bx - center_x)**2 + (by - center_y)**2
                    if dist < min_dist:
                        min_dist = dist
                        final_box = box.unsqueeze(0)
            
            # --- Step 2: SAM 3 ---
            final_mask = None
            if final_box is not None:
                sam_results = sam_model(img_path, bboxes=final_box, verbose=False)
            else:
                is_fallback = True
                h_img, w_img = det_results[0].orig_shape[:2]
                cx, cy = w_img / 2, h_img / 2
                margin = 5  
                fallback_box = torch.tensor([[cx-margin, cy-margin, cx+margin, cy+margin]], device=det_model.device)
                sam_results = sam_model(img_path, bboxes=fallback_box, verbose=False)

            if sam_results[0].masks is not None:
                masks_data = sam_results[0].masks.data.cpu().numpy()
                if masks_data.shape[0] > 0:
                    areas = np.sum(masks_data, axis=(1, 2))
                    largest_idx = np.argmax(areas)
                    final_mask = masks_data[largest_idx].astype(np.uint8) * 255
            
            if final_mask is None:
                final_mask = np.zeros((h_real, w_real), dtype=np.uint8)

            # --- Step 3: 清洗 ---
            status_icon = "🟢" if not is_fallback else "🔵"
            print(f"       [{i+1}/{total_imgs}] {img_path.name} | {status_icon} | ⚡ {fps:.1f} fps          ", end="\r")

            is_good, cleaned_mask, reason = clean_and_verify_mask(final_mask, img_path.name)

            if is_good:
                if cleaned_mask.shape[:2] != original_img.shape[:2]:
                    cleaned_mask = cv2.resize(cleaned_mask, (w_real, h_real), interpolation=cv2.INTER_NEAREST)
                
                # 🔥 Premultiplied Alpha + Feathering 🔥
                mask_blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
                alpha = mask_blurred.astype(np.float32) / 255.0
                img_float = original_img.astype(np.float32)
                
                b, g, r = cv2.split(img_float)
                b = b * alpha
                g = g * alpha
                r = r * alpha
                
                img_bgra = cv2.merge([
                    b.astype(np.uint8),
                    g.astype(np.uint8),
                    r.astype(np.uint8),
                    mask_blurred
                ])
                
                new_img_path = img_path.with_suffix('.png')
                cv2.imwrite(str(new_img_path), img_bgra)
                cv2.imwrite(str(masks_dir / f"{img_path.stem}.png"), cleaned_mask)
                
                if img_path.name in frames_map:
                    frame_data = frames_map[img_path.name]
                    frame_data["file_path"] = f"images/{new_img_path.name}"
                    valid_frames_list.append(frame_data)
                process_success = True

        except Exception as e:
            print(f"\n❌ Frame {i} Error: {e}")
            process_success = False 

        finally:
            if img_path.exists() and img_path.suffix.lower() == '.jpg':
                if process_success:
                    try: img_path.unlink() 
                    except: pass
                else:
                    try: img_path.unlink()
                    except: pass

    print(f"\n\n📊 完成。剩余可用: {len(valid_frames_list)}")
    if len(valid_frames_list) == 0: return False

    meta["frames"] = valid_frames_list
    with open(cfg.transforms_file, 'w') as f: json.dump(meta, f, indent=4)
    return True

# ================= 辅助工具 =================

def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

class ImageProcessor:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def smart_filter_blurry_images(self, image_folder, keep_ratio=0.85):
        print(f"\n🧠 [智能清洗] 正在分析图片质量...")
        image_dir = Path(image_folder)
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not images: return
        
        trash_dir = image_dir.parent / "trash_smart"
        trash_dir.mkdir(exist_ok=True)
        
        img_scores = []
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            grid_h, grid_w = h // 3, w // 3
            max_grid_score = 0
            for r in range(3):
                for c in range(3):
                    roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                    score = cv2.Laplacian(roi, cv2.CV_64F).var()
                    if score > max_grid_score: max_grid_score = score
            img_scores.append((img_path, max_grid_score))
            if i % 50 == 0: print(f"  -> 分析中... {i}/{len(images)}", end="\r")
        
        scores = [s[1] for s in img_scores]
        if not scores: return
        quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
        
        good_images = []
        for img_path, score in img_scores:
            if score < quality_threshold:
                shutil.move(str(img_path), str(trash_dir / img_path.name))
            else:
                good_images.append(img_path)
        
        max_imgs = self.cfg.max_images  
        if len(good_images) > max_imgs:
            print(f"    ⚠️ 图片过多 ({len(good_images)} 张), 正在降采样至 {max_imgs} 张...")
            indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_imgs, dtype=int))
            for idx, img_path in enumerate(good_images):
                if idx not in indices_to_keep:
                    shutil.move(str(img_path), str(trash_dir / img_path.name))
                    
        print(f"✨ 清洗结束，剩余 {len(list(image_dir.glob('*')))} 张。")

def analyze_and_calculate_adaptive_collider(json_path, force_cull=False, radius_scale=1.8):
    """
    场景理解算法
    """
    print(f"\n🤖 [AI 分析] 解析相机轨迹...")
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"]
        if not frames: return [], "unknown"

        has_mask = "mask_path" in frames[0]
        positions = [np.array(f["transform_matrix"])[:3, 3] for f in frames]
        forward_vectors = [np.array(f["transform_matrix"])[:3, :3] @ np.array([0, 0, -1]) for f in frames]
        center = np.mean(positions, axis=0)
        vec_to_center = center - positions
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        is_object_mode = ratio > 0.6 or force_cull or has_mask

        if is_object_mode:
            dists = [np.linalg.norm(p) for p in positions]
            avg_dist = np.mean(dists)
            scene_radius = 1.0 * radius_scale
            # 保护笔尖：确保近平面足够小
            calc_near = max(0.01, min(dists) - scene_radius) 
            calc_far = avg_dist + scene_radius
            
            print(f"    -> 物体模式: Near={calc_near:.2f}, Far={calc_far:.2f}")
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"
    except Exception as e:
        print(f"    ⚠️ 分析失败: {e}")
        return [], "unknown"

def perform_percentile_culling(ply_path, json_path, output_path, keep_percentile=0.9):
    if not HAS_PLYFILE: return False
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    try:
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        
        dists_pts = np.linalg.norm(points - center, axis=1)
        threshold_radius = np.percentile(dists_pts, keep_percentile * 100)
        
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        filtered_vertex = vertex[mask]
        
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True
    except Exception as e:
        print(f"❌ 切割失败: {e}")
        return False

# ==============================================================================
# 类: ColmapRunner (回归经典：使用标准 COLMAP 而不是 GLOMAP)
# ------------------------------------------------------------------------------
# 原因：COLMAP 的 Incremental Mapper 对细微物体（如笔尖）的重建能力远强于 GLOMAP
# ==============================================================================
class ColmapRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.colmap_exe = shutil.which("colmap") or "/usr/local/bin/colmap"
        if not os.path.exists(self.colmap_exe):
            raise FileNotFoundError("❌ 缺少 colmap 可执行文件")
        
        print(f"    -> 🎯 锁定引擎: COLMAP={self.colmap_exe} (回归高精度模式)")
        self.env = os.environ.copy()
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    def run(self):
        print(f"\n📐 [2/4] COLMAP 位姿解算 (High Precision)")
        raw_images_dir = self.cfg.project_dir / "raw_images"
        dest_images_dir = self.cfg.images_dir
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        for img in raw_images_dir.glob("*"):
            if not (dest_images_dir / img.name).exists():
                shutil.copy2(str(img), str(dest_images_dir / img.name))

        colmap_output_dir = self.cfg.data_dir / "colmap"
        colmap_output_dir.mkdir(parents=True, exist_ok=True)
        database_path = colmap_output_dir / "database.db"
        sparse_dir = colmap_output_dir / "sparse"

        try:
            if database_path.exists(): database_path.unlink()
            if sparse_dir.exists(): shutil.rmtree(sparse_dir)
            sparse_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.transforms_file.exists(): self.cfg.transforms_file.unlink()

            # Step 1: 特征提取
            self._run_cmd([self.colmap_exe, "feature_extractor", "--database_path", str(database_path), "--image_path", str(raw_images_dir), "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera", "1"], "Step 1: 特征提取")
            
            # Step 2: 顺序匹配
            self._run_cmd([self.colmap_exe, "sequential_matcher", "--database_path", str(database_path), "--SequentialMatching.overlap", "25"], "Step 2: 顺序匹配")
            
            # Step 3: 增量映射 (Incremental Mapper) - 这就是找回笔尖的关键！
            sparse_0 = sparse_dir / "0"
            sparse_0.mkdir(parents=True, exist_ok=True)
            self._run_cmd([self.colmap_exe, "mapper", "--database_path", str(database_path), "--image_path", str(raw_images_dir), "--output_path", str(sparse_dir)], "Step 3: 增量映射 (COLMAP)")

            # Step 4: 转 json
            # COLMAP 输出通常在 sparse/0 中，nerfstudio 能自动识别
            self._run_cmd(["ns-process-data", "images", "--data", str(dest_images_dir), "--output-dir", str(self.cfg.data_dir), "--skip-colmap", "--skip-image-processing", "--num-downscales", "0"], "生成 transforms.json")

            return self._check_quality(raw_images_dir)
        except Exception as e:
            print(f"❌ COLMAP 流程失败: {e}")
            return False

    def _run_cmd(self, cmd, desc):
        print(f"🚀 {desc}...")
        cmd_env = self.env.copy()
        if cmd[0].startswith("/usr") or cmd[0].startswith("/bin"):
            if "LD_LIBRARY_PATH" in cmd_env: del cmd_env["LD_LIBRARY_PATH"]
        subprocess.run(cmd, check=True, env=cmd_env, stdout=subprocess.DEVNULL) # 简化输出

    def _check_quality(self, raw_images_dir):
        if not self.cfg.transforms_file.exists(): return False
        with open(self.cfg.transforms_file, 'r') as f: meta = json.load(f)
        ratio = len(meta["frames"]) / len(list(raw_images_dir.glob("*")))
        print(f"    📊 匹配率: {ratio:.2%}")
        return ratio > 0.2

class AISegmentor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
    def run(self):
        return run_ai_segmentation_pipeline(self.cfg.data_dir)

class NerfstudioEngine:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = cfg.project_dir / "outputs"
        self.env = os.environ.copy()
        self.env["QT_QPA_PLATFORM"] = "offscreen"
        self.env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
        self.scene_type = "object" 

    def train(self):
        print(f"\n🔥 [4/4] 开始训练 (Splatfacto)")
        
        collider_args, scene_type = analyze_and_calculate_adaptive_collider(
            self.cfg.transforms_file,
            force_cull=self.cfg.force_spherical_culling,
            radius_scale=self.cfg.scene_radius_scale
        )
        self.scene_type = scene_type 

        cmd = [
            "ns-train", "splatfacto",
            "--data", str(self.cfg.data_dir),
            "--output-dir", str(self.output_dir),
            "--experiment-name", self.cfg.project_name,
            
            # 🔥 关键配置：关闭 random-init，依靠 COLMAP 的高精度点云
            "--pipeline.model.random-init", "False", 
            
            "--pipeline.model.background-color", "random",
            "--pipeline.model.cull-alpha-thresh", "0.05",
            "--pipeline.model.stop-split-at", "25000",
            *collider_args,
            "--max-num-iterations", "15000",
            "--vis", "viewer+tensorboard",
            "--viewer.quit-on-train-completion", "True",
            "nerfstudio-data",
            "--downscale-factor", "1",
            "--auto-scale-poses", "False"
        ]
        
        subprocess.run(cmd, check=True, env=self.env)

    def export(self):
        print(f"\n💾 [导出] 正在转换模型格式...")
        base_dir = self.output_dir / self.cfg.project_name / "splatfacto"
        try:
            runs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
            config_path = runs[-1] / "config.yml"
        except IndexError:
            print("❌ 未找到训练记录")
            return None

        subprocess.run([
            "ns-export", "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(self.cfg.project_dir)
        ], check=True, env=self.env)

        raw_ply = self.cfg.project_dir / "splat.ply"
        if not raw_ply.exists():
            raw_ply = self.cfg.project_dir / "point_cloud.ply"
            
        cleaned_ply = self.cfg.project_dir / f"{self.cfg.project_name}_clean.ply"
        final_ply = raw_ply

        need_cull = (self.scene_type == "object" or self.cfg.force_spherical_culling)
        
        if need_cull and raw_ply.exists():
            print(f"    -> 检测到物体模式，执行点云清洗...")
            success = perform_percentile_culling(
                raw_ply, 
                self.cfg.transforms_file, 
                cleaned_ply,
                keep_percentile=self.cfg.keep_percentile
            )
            if success:
                final_ply = cleaned_ply

        script_dir = Path(__file__).parent
        results_dir = script_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        target_path = results_dir / f"{self.cfg.project_name}.ply"
        
        if final_ply.exists():
            shutil.copy2(str(final_ply), str(target_path))
            print(f"    📦 已复制结果到: {target_path}")
            return target_path
        else:
            print("❌ 导出失败，找不到 PLY 文件")
            return None

# ================= 主流程 =================
def run_pipeline(cfg: PipelineConfig):
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Engine] 启动任务: {cfg.project_name}")
    
    img_processor = ImageProcessor(cfg)
    colmap_runner = ColmapRunner(cfg) # 使用 COLMAP 
    ai_segmentor = AISegmentor(cfg)
    nerf_engine = NerfstudioEngine(cfg)

    # Step 1: 准备
    if cfg.project_dir.exists(): shutil.rmtree(cfg.project_dir, ignore_errors=True)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = cfg.project_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(cfg.video_path), "-vf", "fps=10", "-q:v", "2", str(temp_dir / "frame_%05d.jpg")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    img_processor.smart_filter_blurry_images(temp_dir)
    
    raw_images_dir = cfg.project_dir / "raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    all_imgs = sorted(list(temp_dir.glob("*")))
    limit = cfg.max_images
    if len(all_imgs) > limit:
        indices = np.linspace(0, len(all_imgs)-1, limit, dtype=int)
        all_imgs = [all_imgs[i] for i in sorted(list(set(indices)))]
    for img in all_imgs: shutil.copy2(str(img), str(raw_images_dir / img.name))
    shutil.rmtree(temp_dir)

    # Step 2: COLMAP (慢但准)
    if not colmap_runner.run(): return

    # Step 3: AI (生成干净的透明图)
    ai_segmentor.run()
    
    try:
        # 强制清理缓存
        output_cache = cfg.project_dir / "outputs"
        if output_cache.exists(): shutil.rmtree(output_cache)
        
        nerf_engine.train()
        final_path = nerf_engine.export()
        if final_path:
            print(f"\n🎉 任务完成！结果位于: \033[92m{final_path}\033[0m")
        else:
            print("\n❌ 任务完成但导出失败")
    except Exception as e:
        print(f"❌ 失败: {e}")

    print(f"⏱️ 总耗时: {format_duration(time.time() - global_start_time)}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])
    
    if not video_file.exists():
        print(f"❌ 找不到视频: {video_file}")
        sys.exit(1)

    cfg = PipelineConfig(
        project_name="process_3dgs_final", # 改个名防冲突
        video_path=video_file,
        max_images=100, 
        enable_ai=True
    )
    
    run_pipeline(cfg)