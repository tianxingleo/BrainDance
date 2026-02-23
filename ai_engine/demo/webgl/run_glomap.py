import subprocess
import sys
import shutil
import os
import time
import datetime # 引入时间处理库
from pathlib import Path
import json
import numpy as np
import logging
import cv2 # 引入OpenCV库
import re # 引入正则库用于日志分析

# 自动查找可用的 COLMAP，可在 Conda env 下使用
colmap_loc = shutil.which("colmap")
print(f"🧐 [自检] 当前使用的 COLMAP: {colmap_loc}")
# 确保它指向 /home/ltx/miniconda3/envs/vggt/bin/colmap

# 设置日志级别
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 🔧 用户配置 (暴力裁剪版) =================
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
SCENE_RADIUS_SCALE = 1.8 
MAX_IMAGES = 350 # 🔥 全局最大图片数量限制

# ================= 辅助工具：时间格式化 =================
def format_duration(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))

# ================= 辅助工具：模糊图片过滤 =================
def smart_filter_blurry_images(image_folder, keep_ratio=0.85, max_images=MAX_IMAGES):
    """
    升级版清洗脚本：混合策略 (Hybrid Strategy)
    
    目标：既要画质好，又要视角全。
    
    流程：
    1. [质量清洗]：先无条件剔除最差的 15% (keep_ratio)，干掉绝对的废片。
    2. [均匀采样]：如果剩下的好图数量依然 > max_images，则按时间轴均匀抽样，
       确保视频的每一段都有图保留，防止某个视角被“团灭”。
    """
    print(f"\n🧠 [智能清洗] 正在分析图片质量 (混合策略版)...")
    
    image_dir = Path(image_folder)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not images:
        print("❌ 没找到图片")
        return

    trash_dir = image_dir.parent / "trash_smart"
    trash_dir.mkdir(exist_ok=True)

    img_scores = []

    # --- 第一步：计算分数 ---
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 九宫格评分
        grid_h, grid_w = h // 3, w // 3
        max_grid_score = 0
        for r in range(3):
            for c in range(3):
                roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                score = cv2.Laplacian(roi, cv2.CV_64F).var()
                if score > max_grid_score:
                    max_grid_score = score
        
        img_scores.append((img_path, max_grid_score))
        if i % 20 == 0:
            print(f"  -> 分析中... {img_path.name}: 局部最高分 {max_grid_score:.1f}")

    # --- 第二步：质量清洗 (剔除废片) ---
    scores = [s[1] for s in img_scores]
    if not scores: return

    num_total = len(scores)
    # 无论如何，先剔除最差的 (1-keep_ratio)
    quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
    
    print(f"\n📊 统计结果:")
    print(f"   - 图片总数: {num_total}")
    print(f"   - 质量阈值 (Bottom {(1-keep_ratio)*100:.0f}%): {quality_threshold:.2f}")

    good_images = [] # 暂存合格的图片 (路径, 分数)
    removed_count_quality = 0

    for img_path, score in img_scores:
        if score < quality_threshold:
            # 质量太差，直接扔垃圾桶
            # print(f"  ❌ [废片剔除] {img_path.name} ({score:.1f})")
            shutil.move(str(img_path), str(trash_dir / img_path.name))
            removed_count_quality += 1
        else:
            good_images.append(img_path)

    print(f"   -> 第一轮清洗完成: 剔除 {removed_count_quality} 张废片，剩余 {len(good_images)} 张合格图片。")

    # --- 第三步：数量控制 (均匀采样) ---
    removed_count_quantity = 0
    
    if len(good_images) > max_images:
        print(f"   ⚠️ 合格图片 ({len(good_images)}) 仍超过上限 ({max_images})")
        print(f"   -> 执行【均匀采样】以保证视角覆盖...")
        
        # 生成保留索引：在 0 到 len-1 之间均匀取 max_images 个点
        # 例如：[0, 2, 4, 6...]
        indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_images, dtype=int))
        
        for idx, img_path in enumerate(good_images):
            if idx not in indices_to_keep:
                # 虽然质量合格，但为了数量限制不得不删
                # print(f"  ✂️ [均匀采样] {img_path.name} (保留名额不足)")
                shutil.move(str(img_path), str(trash_dir / img_path.name))
                removed_count_quantity += 1
    else:
        print(f"   ✅ 合格图片数量 ({len(good_images)}) 未超标，全部保留。")

    total_removed = removed_count_quality + removed_count_quantity
    final_count = num_total - total_removed
    print(f"✨ 清洗结束: 共移除 {total_removed} 张 (废片 {removed_count_quality} + 采样 {removed_count_quantity})，最终保留 {final_count} 张。")

# 🔥 强制开启球体切割
FORCE_SPHERICAL_CULLING = True

# 🔥 核心参数：保留百分比 (0.0 ~ 1.0)
# 0.5 表示只保留离中心最近的 50% 的点 (非常狠)
# 0.65 表示保留 65% (推荐，比较平衡)
# 0.9 表示保留 90% (只去极远处的背景)
KEEP_PERCENTILE = 0.9

# 检查依赖
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("❌ 严重警告: 未安装 plyfile 库！无法执行切割。请运行: pip install plyfile")

# ================= 核心算法 1: 训练参数计算 (保持不变) =================
def analyze_and_calculate_adaptive_collider(json_path):
    print(f"\n🤖 [AI 分析] 解析相机轨迹...")
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"]
        if not frames: return [], "unknown"

        positions = []
        forward_vectors = []
        dists_to_origin = [] 
        
        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            positions.append(c2w[:3, 3])
            forward_vectors.append(c2w[:3, :3] @ np.array([0, 0, -1]))
            dists_to_origin.append(np.linalg.norm(c2w[:3, 3]))
            
        positions = np.array(positions)
        forward_vectors = np.array(forward_vectors)
        
        center = np.mean(positions, axis=0)
        vec_to_center = center - positions
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        print(f"    -> 相机聚合度: {ratio:.2f}")

        # 如果强制开启切割，我们在训练时也倾向于使用物体参数
        is_object_mode = ratio > 0.6 or FORCE_SPHERICAL_CULLING

        if is_object_mode:
            avg_dist = np.mean(dists_to_origin)
            min_dist = np.min(dists_to_origin)
            scene_radius = 1.0 * SCENE_RADIUS_SCALE
            calc_near = max(0.05, min_dist - scene_radius)
            calc_far = avg_dist + scene_radius
            
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"

# ================= 核心算法 2: 基于分位数的暴力切割 (New!) =================
def perform_percentile_culling(ply_path, json_path, output_path):
    if not HAS_PLYFILE: 
        print("❌ 缺少 plyfile 库，跳过切割。")
        return False
        
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    print(f"🔥 目标: 只保留离圆心最近的 {KEEP_PERCENTILE*100:.0f}% 点云")

    try:
        # 1. 计算切割中心 (依然使用相机重心，因为它是轨道的圆心，比点云重心更稳)
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        print(f"    -> 切割圆心 (相机重心): {center}")

        # 2. 读取点云
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']
        
        x, y, z = vertex['x'], vertex['y'], vertex['z']
        points = np.stack([x, y, z], axis=1)
        original_count = len(points)
        
        # 3. 计算所有点到中心的距离
        print("    -> 正在计算所有点的距离分布...")
        dists_pts = np.linalg.norm(points - center, axis=1)
        
        # 4. === 核心逻辑：计算分位数阈值 ===
        # 找到一个距离 D，使得有 KEEP_PERCENTILE 的点距离 < D
        threshold_radius = np.percentile(dists_pts, KEEP_PERCENTILE * 100)
        
        print(f"    -> 统计结果: {KEEP_PERCENTILE*100:.0f}% 的点集中在半径 {threshold_radius:.4f} 以内")
        print(f"    -> 执行切割: 所有大于 {threshold_radius:.4f} 的点将被删除")
        
        # 5. 执行切割
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 逻辑：距离 < 阈值 AND 点比较实
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        
        filtered_vertex = vertex[mask]
        new_count = len(filtered_vertex)
        
        print(f"    -> 原始点数: {original_count}")
        print(f"    -> 剩余点数: {new_count} (删除了 {original_count - new_count} 个背景点)")
        
        # 6. 保存
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True

    except Exception as e:
        print(f"❌ 切割失败详情: {e}")
        return False

# ================= 主流程 =================

def run_pipeline(video_path, project_name):
    # --- 全局计时开始 ---
    global_start_time = time.time()
    print(f"\n🚀 [BrainDance Engine V13] 启动任务: {project_name}")
    print(f"🕒 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔪 切割策略: 保留 {KEEP_PERCENTILE*100}% 最近点云")
    
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    # 强制解决 setuptools/distutils 冲突
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    # [Step 1] 数据处理
    step1_start = time.time()
    
    print(f"🆕 [强制重置] 正在初始化工作环境...")
    if work_dir.exists(): 
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            print(f"⚠️ 警告: 旧目录清理失败 (可能被占用): {e}")
    
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(video_src), str(work_dir / video_src.name))

    print(f"\n🎥 [1/3] 数据准备 (沙盒隔离模式)")
    
    # 1. 定义两个隔离区域
    # temp_dir: 存放 ffmpeg 原始产物，可能包含几百张图
    temp_dir = work_dir / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # target_dir: 最终送给 COLMAP 的干净目录 (只放 200 张)
    extracted_images_dir = work_dir / "raw_images"
    extracted_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. FFmpeg 抽帧 (输出到临时目录 temp_dir)
    print(f"    -> 正在抽帧到临时目录...")
    cap = cv2.VideoCapture(str(work_dir / video_src.name))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    
    vf_param = "fps=4"
    if width > 1920:
        vf_param = "scale=1920:-1,fps=4"
        
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(work_dir / video_src.name), 
                        "-vf", vf_param, "-q:v", "2", 
                        str(temp_dir / "frame_%05d.jpg")], check=False) 
    except Exception as e:
        print(f"    ⚠️ FFmpeg 结束: {e}")
    
    # 3. 在临时目录进行清洗
    smart_filter_blurry_images(temp_dir, keep_ratio=0.85)
    
    # 4. 【关键步骤】白名单复制 (从 temp -> raw_images)
    print("    -> 正在执行【数量限制与迁移】...")
    
    # 读取所有合格图片
    all_candidates = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")))
    total_candidates = len(all_candidates)
    # MAX_IMAGES = 200 # Already global
    
    final_images_list = []
    
    if total_candidates > MAX_IMAGES:
        print(f"    ⚠️ 图片过多 ({total_candidates}), 正在均匀选取 {MAX_IMAGES} 张...")
        # 均匀采样索引
        indices = np.linspace(0, total_candidates - 1, MAX_IMAGES, dtype=int)
        # 使用集合去重 (防止极端情况)
        indices = sorted(list(set(indices)))
        
        for idx in indices:
            final_images_list.append(all_candidates[idx])
    else:
        print(f"    ✅ 图片数量 ({total_candidates}) 未超标，全部保留。")
        final_images_list = all_candidates

    # 执行复制：只把选中的放入 COLMAP 目录
    for img_path in final_images_list:
        shutil.copy2(str(img_path), str(extracted_images_dir / img_path.name))
        
    print(f"    ✅ 已将 {len(final_images_list)} 张干净图片移入 COLMAP 专用目录。")
    print(f"    🧹 正在清理临时文件...")
    shutil.rmtree(temp_dir) # 删掉脏区，防止混淆

    # =========================================================
    # 🚀 COLMAP 启动
    # =========================================================
    
    print(f"    ✅ 准备启动 COLMAP (Linux GPU 模式)...")
    
    # 数据库路径
    colmap_output_dir = data_dir / "colmap"
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    database_path = colmap_output_dir / "database.db"
    
    # ================= 修正开始 =================
    # 优先查找系统路径中的 colmap (Conda 环境优先)
    system_colmap_exe = shutil.which("colmap")
    
    # 如果找不到，再尝试硬编码路径
    if not system_colmap_exe:
        if os.path.exists("/usr/local/bin/colmap"):
            system_colmap_exe = "/usr/local/bin/colmap"
        else:
            raise FileNotFoundError("❌ 无法找到 colmap 可执行文件，请确保已安装！")
    
    print(f"🎯 [执行引擎] 锁定 COLMAP: {system_colmap_exe}")
    # ================= 修正结束 =================

    full_log_content = []

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
                    full_log_content.append(line)
                
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
        except Exception as e:
            print(f"\n❌ {step_desc} 执行异常: {e}")
            raise e

    # 3. 手动运行 Feature Extractor (特征提取)
    # 注意：移除 --SiftExtraction.use_gpu 和 --SiftExtraction.num_threads，因为部分 COLMAP 版本不识别这些参数
    # 如果编译了 CUDA，COLMAP 默认会自动使用 GPU；线程数也会自动管理
    run_colmap_step([
        system_colmap_exe, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1"
    ], "[1/4] GPU 特征提取")

    # 4. 手动运行 Sequential Matcher (顺序匹配)
    run_colmap_step([
        system_colmap_exe, "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "25" 
    ], "[2/4] GPU 顺序匹配")

    # 4.5 运行 GLOMAP Mapper (全局重建)
    # 注意: GLOMAP 会在输出路径下自动创建 '0' 子目录，无需手动创建
    glomap_output_dir = colmap_output_dir / "sparse"
    glomap_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试查找 GLOMAP 可执行文件
    system_glomap_exe = shutil.which("glomap")
    if not system_glomap_exe:
        system_glomap_exe = "/usr/local/bin/glomap"
    
    print(f"🚀 切换至 GLOMAP 引擎: {system_glomap_exe}")

    run_colmap_step([
        system_glomap_exe, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--output_path", str(glomap_output_dir)
    ], "[3/4] GLOMAP 全局重建")

    print(f"✅ COLMAP 计算完成！正在检查并修正目录结构...")

    # =========================================================
    # 🔧 [3.5] 目录结构强力修正 (Auto-Fixer)
    # 目标：无论 COLMAP 把模型生成在哪里，都强行移动到 {data}/colmap/sparse/0
    # =========================================================
    
    colmap_root = colmap_output_dir  # .../data/colmap
    sparse_root = colmap_root / "sparse"
    target_dir_0 = sparse_root / "0"
    target_dir_0.mkdir(parents=True, exist_ok=True)

    required_files_bin = ["cameras.bin", "images.bin", "points3D.bin"]
    required_files_txt = ["cameras.txt", "images.txt", "points3D.txt"]
    
    model_found = False

    # 1. 检查是不是已经在 sparse/0 (完美情况)
    if all((target_dir_0 / f).exists() for f in required_files_bin):
        print("    ✅ 模型文件 (BIN) 位置正确。")
        model_found = True
    elif all((target_dir_0 / f).exists() for f in required_files_txt):
        print("    ✅ 模型文件 (TXT) 位置正确。")
        model_found = True
        
    # 2. 检查是不是在 sparse 根目录 (常见情况) -> 搬运
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

    # 3. 检查是不是在子目录 (例如 sparse/1 或 sparse/0/0) -> 搬运
    if not model_found:
        # 递归搜索所有子目录
        for root, dirs, files in os.walk(sparse_root):
            # 检查当前目录是否有 bin 模型
            if all(f in files for f in required_files_bin):
                src_path = Path(root)
                if src_path == target_dir_0: continue # 跳过自己
                print(f"    🔧 在子目录 {src_path} 找到 BIN 模型，正在归位...")
                for f in required_files_bin:
                    shutil.move(str(src_path / f), str(target_dir_0 / f))
                model_found = True
                break
            # 检查当前目录是否有 txt 模型
            if all(f in files for f in required_files_txt):
                src_path = Path(root)
                if src_path == target_dir_0: continue
                print(f"    🔧 在子目录 {src_path} 找到 TXT 模型，正在归位...")
                for f in required_files_txt:
                    shutil.move(str(src_path / f), str(target_dir_0 / f))
                model_found = True
                break

    if not model_found:
        print("❌ [严重错误] 在 sparse 目录下找不到完整的 COLMAP 模型文件！")
        print("    -> 可能原因：Mapper 失败，未能重建出场景。")
        # 这里可以选择抛出异常，或者让它继续跑看看日志
        raise FileNotFoundError("COLMAP Mapper failed to generate valid model files.")

    # [3.6] 提前同步图片 (为了让 ns-process-data 能找到)
    print(f"    -> 正在同步图片: raw_images -> data/images ...")
    dest_images_dir = data_dir / "images"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    
    valid_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        valid_images.extend(list(extracted_images_dir.glob(ext)))
        
    for img_path in valid_images:
        shutil.copy2(str(img_path), str(dest_images_dir / img_path.name))
    print(f"    ✅ 已同步 {len(valid_images)} 张图片。")

    print(f"✅ 数据准备就绪！正在生成 transforms.json (用于后续切割)...")

    # 5. 运行 ns-process-data (生成 transforms.json)
    # 修正：--data 指向 data/images，--output-dir 指向 data
    # 这样它会在 data/colmap 找模型，在 data/images 找图片
    run_colmap_step([
        "ns-process-data", "images", 
            "--data", str(dest_images_dir), 
            "--output-dir", str(data_dir), 
            "--verbose", 
            "--skip-colmap", 
            "--skip-image-processing", 
            "--num-downscales", "0"
    ], "[4/4] 生成 transforms.json")
    # --- 质量检测逻辑 ---
    full_log = "".join(full_log_content)
    
    # 1. 检测 "No convergence"
    if "Termination : No convergence" in full_log:
        print("\n❌ [严重错误] COLMAP 无法收敛 (No convergence)！")
        
        # 用户要求：输出百分比而不是看不懂的 px 误差
        # 尝试提取匹配率
        match_pct = re.search(r"COLMAP only found poses for (\d+\.?\d*)% of the images", full_log)
        if match_pct:
            print(f"    -> 成功注册图片比例: {match_pct.group(1)}% (质量过低)")
        else:
            # 备选方案：从日志中抓取注册数量并手动计算
            # COLMAP 日志通常包含 "Registered images ... X"
            reg_match = re.findall(r"Registered images.*?(\d+)", full_log)
            if reg_match:
                # 取最后一个匹配到的数量（因为可能有多次迭代）
                registered_count = int(reg_match[-1])
                ratio = (registered_count / num_images) * 100 if num_images > 0 else 0
                print(f"    -> 成功注册图片: {registered_count}/{num_images} ({ratio:.2f}%)")
            
        print("🛑 任务已终止，因为生成的稀疏点云质量无法满足训练要求。")
        
        # 清理 Linux 临时文件
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print(f"🧹 清理完成: 已删除工作区 {work_dir}")
        return None

    # 2. 检测匹配率过低
    # 示例日志: COLMAP only found poses for 10.00% of the images. This is low.
    match = re.search(r"COLMAP only found poses for (\d+\.?\d*)% of the images", full_log)
    if match:
        matched_percentage = float(match.group(1))
        print(f"\n📊 COLMAP 匹配率检测: {matched_percentage:.2f}%")
        
        if matched_percentage < 35.0:
            print(f"❌ [质量警告] 匹配率过低 (< 35%)！")
            print("    -> 这意味着大部分图片无法被定位，生成的 3D 场景将严重残缺。")
            print("🛑 任务已终止。建议：增加图片数量、保证图片清晰度或增加重叠率。")
            
            # 清理 Linux 临时文件
            if work_dir.exists():
                shutil.rmtree(work_dir)
                print(f"🧹 清理完成: 已删除工作区 {work_dir}")
            return None

    step1_duration = time.time() - step1_start
    print(f"⏱️ [Step 1 完成] 耗时: {format_duration(step1_duration)}")

    # [Step 2] 训练
    step2_start = time.time()
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []
    
    scene_type_detected = "unknown"

    if run_dirs:
        print(f"\n⏩ [训练跳过] 已完成")
        _, scene_type_detected = analyze_and_calculate_adaptive_collider(transforms_file)
    else:
        collider_args, scene_type_detected = analyze_and_calculate_adaptive_collider(transforms_file)
        print(f"\n🧠 [2/3] 开始训练...")
        
        subprocess.run([
            "ns-train", "splatfacto", 
            "--data", str(data_dir), 
            "--output-dir", str(output_dir), 
            "--experiment-name", project_name, 
            "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005", 
            *collider_args,
            "--max-num-iterations", "5000", 
            "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True", 
            
            # 👇 子命令：指定使用 colmap 数据解析器
            "colmap", 
            
            # 👇 参数修正：只需写短名，并且必须放在 "colmap" 后面
            "--downscale-factor", "1"
        ], check=True, env=env)

    step2_duration = time.time() - step2_start
    print(f"⏱️ [Step 2 完成] 耗时: {format_duration(step2_duration)}")

    # [Step 3] 导出
    step3_start = time.time()
    print(f"\n💾 [3/3] 导出结果")
    if not run_dirs: run_dirs = sorted(list(search_path.glob("*")))
    if not run_dirs: return None
    latest_run = run_dirs[-1]
    
    subprocess.run([
        "ns-export", "gaussian-splat", "--load-config", str(latest_run/"config.yml"), 
        "--output-dir", str(work_dir)
    ], check=True, env=env)
    
    # 导出相机
    cameras_export_dir = work_dir / "cameras_export"
    cameras_export_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "ns-export", "cameras", "--load-config", str(latest_run/"config.yml"), 
            "--output-dir", str(cameras_export_dir)
        ], check=True, env=env)
    except Exception as e:
        print(f"⚠️ 无法使用 ns-export cameras 导出相机, 可能是版本不支持: {e}")
        
    time.sleep(5) 

    # [Step 3.5] 分位数暴力切割
    raw_ply = work_dir / "point_cloud.ply"
    if not raw_ply.exists(): raw_ply = work_dir / "splat.ply"

    cleaned_ply = work_dir / "point_cloud_cleaned.ply"
    final_ply_to_use = raw_ply

    should_clean = (scene_type_detected == "object") or FORCE_SPHERICAL_CULLING
    
    if should_clean:
        if raw_ply.exists():
            # 使用新的分位数切割函数
            if perform_percentile_culling(raw_ply, transforms_file, cleaned_ply):
                print("✨ 暴力切割成功！")
                final_ply_to_use = cleaned_ply
        else:
            print(f"❌ 警告：未找到 PLY 文件")
    else:
        print(f"ℹ️ 跳过切割")

    step3_duration = time.time() - step3_start
    print(f"⏱️ [Step 3 完成] 耗时: {format_duration(step3_duration)}")

    # [Step 4] 回传
    print(f"\n📦 [IO 同步] 回传至 WebGL 模型目录...")
    # 硬编码输出路径到网页公共资源目录
    public_dir = Path(__file__).parent / "my-3dgs-viewer/public/models"
    public_dir.mkdir(exist_ok=True, parents=True) 

    # 姿态回传
    final_webgl_poses = public_dir / "webgl_poses.json"
    final_ply_dst = public_dir / f"{project_name}.ply"
    
    # 创建图片存放目录
    webgl_images_dir = public_dir / "images"
    webgl_images_dir.mkdir(exist_ok=True, parents=True)

    transforms_src = data_dir / "transforms.json"
    dataparser_src = latest_run / "dataparser_transforms.json"
    
    # === 新姿态生成逻辑：直接读取 ns-export cameras 的绝对正确相机位姿 ===
    cameras_json_path = work_dir / "cameras_export" / "transforms_train.json"
    if transforms_src.exists() and cameras_json_path.exists():
        print("🔄 正在读取 ns-export cameras 导出的对齐相机数据...")
        try:
            # 1. 提取元数据 (FOV等)
            with open(transforms_src, 'r') as f:
                orig_data = json.load(f)
            fl_x = orig_data.get("fl_x")
            fl_y = orig_data.get("fl_y")
            w = orig_data.get("w")
            h = orig_data.get("h")
            camera_model = orig_data.get("camera_model")
            
            # 2. 提取对齐后的矩阵
            with open(cameras_json_path, 'r') as f:
                frames_list = json.load(f)
                
            webgl_poses = []

            for frame in frames_list:
                # ns-export cameras 输出为 3x4
                c2w_3x4 = np.array(frame['transform'])
                # 补成 4x4
                c2w = np.eye(4)
                c2w[:3, :4] = c2w_3x4
                
                # 注意：Three.js 的 Matrix4.fromArray 默认接受列优先 (Column-major) 数组
                # 所以这里必须用 .T 转置后再 flatten！
                c2w_threejs = c2w.T.flatten().tolist()
                
                # resolving image path
                file_path = frame.get('file_path')
                img_name = Path(file_path).name
                
                # copy image
                src_img = data_dir / "images" / img_name
                if not src_img.exists():
                    src_img = Path(file_path)
                    
                if src_img.exists():
                    shutil.copy2(str(src_img), str(webgl_images_dir / img_name))
                else:
                    print(f"⚠️ 无法复制参考图，找不到文件: {src_img}")
                
                webgl_poses.append({
                    "id": img_name,
                    "fl_y": fl_y,
                    "h": h,
                    "matrix": c2w_threejs,
                    "image_url": f"/models/images/{img_name}"
                })
            
            # 对 webgl_poses 根据 id 自然排序
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', s['id'])]
            webgl_poses.sort(key=natural_sort_key)
            
            # 导出带参数的 JSON
            output_data = {
                "w": w,
                "h": h,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "camera_model": camera_model,
                "frames": webgl_poses
            }
                
            with open(final_webgl_poses, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"✅ WebGL 姿态文件与 {len(webgl_poses)} 张参考图已准备就绪 (基于 ns-export cameras)。")
        except Exception as e:
            print(f"❌ 姿态预处理失败: {e}")
    else:
        print("❌ 未找到 transforms_train.json 或 transforms.json，请检查 ns-export cameras 是否执行成功。")
    if final_ply_to_use and final_ply_to_use.exists():
        try:
            # 1. 复制最终 PLY (可能是裁剪过的)
            shutil.copy2(str(final_ply_to_use), str(final_ply_dst))
            
            # --- 最终时间汇总 ---
            total_time = time.time() - global_start_time
            print(f"\n✅ =============================================")
            print(f"🎉 任务全部完成！安心睡觉吧。")
            print(f"📂 最终模型: {final_ply_dst}")
            print(f"⏱️ 总共耗时: {format_duration(total_time)}")
            print(f"✅ =============================================")
            
            return str(final_ply_dst)
        except Exception as e:
            print(f"❌ 回传失败: {e}")
            return None
    else:
        print("❌ 导出失败，未找到 PLY 文件 (point_cloud.ply 或 splat.ply)。")
        return None

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])

    if video_file.exists():
        run_pipeline(video_file, "scene_auto_sync")
    else:
        print(f"❌ 找不到视频: {video_file}")