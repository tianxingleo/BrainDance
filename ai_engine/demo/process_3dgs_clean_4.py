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

# 设置日志级别
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 🔧 用户配置 (暴力裁剪版) =================
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
SCENE_RADIUS_SCALE = 1.8 

# ================= 辅助工具：时间格式化 =================
def format_duration(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))

# ================= 辅助工具：模糊图片过滤 =================
def smart_filter_blurry_images(image_folder, keep_ratio=0.85, max_images=200):
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
KEEP_PERCENTILE = 0.6

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

    # [Step 1] 数据处理
    step1_start = time.time()
    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 检测到 COLMAP 数据")
    else:
        print(f"🆕 [新任务] 初始化...")
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        shutil.copy(str(video_src), str(work_dir / video_src.name))

        print(f"\n🎥 [1/3] COLMAP 解算")
        # 修改：将原始图片提取到 raw_images，与 data/images 隔离，防止 ns-process-data 混淆
        extracted_images_dir = work_dir / "raw_images"
        extracted_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查视频宽度
        cap = cv2.VideoCapture(str(work_dir / video_src.name))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        vf_param = "fps=4"
        if width > 1920:
            vf_param = "scale=1920:-1,fps=4"
            print(f"    -> 视频宽度 {width} > 1920，将缩小至 1920P")
        else:
            print(f"    -> 视频宽度 {width} <= 1920，保持原始分辨率")

        # --- FFMPEG 执行 (增加容错) ---
        print(f"    -> 正在抽帧...")
        try:
            # 即使 ffmpeg 报错(如文件尾部IO错误)，只要生成了图片也是可以接受的
            subprocess.run(["ffmpeg", "-y", "-i", str(work_dir / video_src.name), 
                            "-vf", vf_param, "-q:v", "2", 
                            str(extracted_images_dir / "frame_%05d.jpg")], check=False) 
        except Exception as e:
            print(f"    ⚠️ FFmpeg 返回异常 (通常可忽略): {e}")
        
        # 🔥【关键修复】强制等待文件系统同步 (解决 WSL 延迟问题)
        print("    -> 等待文件系统索引更新...")
        time.sleep(3) 

        # --- 图片质量检查 ---
        smart_filter_blurry_images(extracted_images_dir, keep_ratio=0.85)
        
        # 🔥【关键修复】重试机制：确保读到图片
        all_images = []
        for retry in range(3):
            all_images = sorted(list(extracted_images_dir.glob("*.jpg")))
            num_images = len(all_images)
            if num_images > 0:
                break
            print(f"    ⚠️ 未找到图片，正在重试 ({retry+1}/3)...")
            time.sleep(2)

        print(f"    -> 当前图片总数: {num_images}") # 调试打印

        # --- 图片数量检查 (Limit to 200) ---
        MAX_IMAGES = 200
        if num_images > MAX_IMAGES:
            print(f"    ⚠️ 图片数量 ({num_images}) 超过上限 {MAX_IMAGES}，正在进行均匀采样...")
            # 计算保留索引
            indices_to_keep = set([int(i * (num_images - 1) / (MAX_IMAGES - 1)) for i in range(MAX_IMAGES)])
            deleted_count = 0
            for idx, img_path in enumerate(all_images):
                if idx not in indices_to_keep:
                    try:
                        os.remove(img_path) 
                        deleted_count += 1
                    except OSError:
                        pass # 忽略删除失败
            print(f"    ✅ 已删除 {deleted_count} 张多余图片，剩余 {MAX_IMAGES} 张用于序列匹配。")
            
            # 再次确认
            final_count = len(list(extracted_images_dir.glob("*.jpg")))
            print(f"    -> 最终确认数量: {final_count}")
        else:
            print(f"    ✅ 图片数量 ({num_images}) 未超标，无需处理。")

        # --- 运行 COLMAP (带质量监控) ---
        print(f"    -> 正在运行 COLMAP (ns-process-data)...")
        # 使用 capture_output=True 捕获日志用于分析
        process = subprocess.run(
            ["ns-process-data", "images", "--data", str(extracted_images_dir), "--output-dir", str(data_dir), "--verbose"],
            env=env, capture_output=True, text=True
        )
        
        # 实时打印日志 (因为捕获后屏幕不显示了，这里手动打出来)
        if process.stdout: print(process.stdout)
        if process.stderr: print(process.stderr)

        if process.returncode != 0:
            print(f"❌ COLMAP 运行出错 (Return Code: {process.returncode})")
            raise subprocess.CalledProcessError(process.returncode, process.args)

        # --- 质量检测逻辑 ---
        full_log = (process.stdout or "") + (process.stderr or "")
        
        # 1. 检测 "No convergence"
        if "Termination : No convergence" in full_log:
            print("\n❌ [严重错误] COLMAP 无法收敛 (No convergence)！")
            
            # 尝试提取 Final cost
            cost_match = re.search(r"Final cost : (\d+\.?\d*)", full_log)
            if cost_match:
                print(f"    -> 最终代价 (Final Cost): {cost_match.group(1)} [px] (通常意味着误差极大)")
                
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
            "ns-train", "splatfacto", "--data", str(data_dir), "--output-dir", str(output_dir), 
            "--experiment-name", project_name, "--pipeline.model.random-init", "False", 
            "--pipeline.model.cull-alpha-thresh", "0.005", *collider_args,
            "--max-num-iterations", "15000", "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True", "colmap"
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
    print(f"\n📦 [IO 同步] 回传至 Windows...")
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True, parents=True) 
    
    transforms_src = data_dir / "transforms.json"
    final_webgl_poses = target_dir / "webgl_poses.json"
    final_ply_dst = target_dir / f"{project_name}.ply"
    final_transforms = target_dir / "transforms.json"
    
    # --- 姿态预处理逻辑 (来自 process_3dgs.py) ---
    if transforms_src.exists():
        print("🔄 正在生成 WebGL 友好姿态文件 (webgl_poses.json)...")
        try:
            with open(transforms_src, 'r') as f:
                data = json.load(f)
            
            webgl_frames = []
            for frame in data["frames"]:
                c2w_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
                # 计算 W2C (虽然这里只存了 C2W，但可以预留逻辑)
                webgl_frames.append({
                    "file_path": frame["file_path"],
                    "pose_matrix_c2w": c2w_matrix.tolist() 
                })
                
            webgl_data = {
                "camera_model": data.get("camera_model", "OPENCV"),
                "w": data.get("w", 0),
                "h": data.get("h", 0),
                "fl_x": data.get("fl_x", 0),
                "fl_y": data.get("fl_y", 0),
                "frames": webgl_frames
            }
            
            with open(final_webgl_poses, 'w') as f:
                json.dump(webgl_data, f, indent=4)
            print(f"✅ WebGL 姿态文件已保存至: {final_webgl_poses.resolve()}")
        except Exception as e:
            print(f"❌ 姿态预处理失败: {e}")

    if final_ply_to_use and final_ply_to_use.exists():
        try:
            # 复制 PLY 文件
            shutil.copy2(str(final_ply_to_use), str(final_ply_dst))
            
            # 复制 transforms.json 文件
            if transforms_src.exists():
                shutil.copy2(str(transforms_src), str(final_transforms))
            
            # 清理 Linux 临时文件
            shutil.rmtree(work_dir)
            print(f"🧹 清理完成: 已删除工作区 {work_dir}")
            
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