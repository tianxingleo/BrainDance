import sys
import os

# 🔥【关键】强制添加 vggt 库的路径
# 你的 vggt 库所在的真实路径
vggt_lib_path = "/home/ltx/workspace/ai/vggt"

if vggt_lib_path not in sys.path:
    print(f"⚡ [环境修正] 添加 VGGT 库路径: {vggt_lib_path}")
    sys.path.insert(0, vggt_lib_path)

# 🔥【关键】确保当前目录不在 sys.path 的首位，防止误引用
# (可选，但推荐)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir in sys.path:
        sys.path.remove(current_dir)
        sys.path.append(current_dir) # 把它放到最后
except:
    pass

import subprocess
import shutil
import time
import datetime # 引入时间处理库
from pathlib import Path
import json
import numpy as np
import logging
import cv2 # 引入OpenCV库
import re # 引入正则库用于日志分析

# --- 🔥 新增 VGGT 导入 ---
import torch
import torch.nn.functional as F
# 假设你已经 pip install -e . 安装了 vggt，或者将 vggt 文件夹放在了同一目录
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track, batch_np_matrix_to_pycolmap
# 如果需要 BA (Bundle Adjustment)，还需要引入 track 相关库，但为了速度建议先仅使用前馈

import os

# 🔥【绝杀】强制将编译好的系统级 colmap 路径提到最前面
# 这样系统找 colmap 时，第一个看到的就是 /usr/local/bin 里的那个好版本
sys_path = "/usr/local/bin"
current_path = os.environ.get("PATH", "")

if sys_path not in current_path.split(os.pathsep)[0]: # 如果不在第一位
    print(f"⚡ [环境修正] 强制设置 PATH 优先级: {sys_path} -> Priority High")
    os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"

# 验证一下
import shutil
colmap_loc = shutil.which("colmap")
print(f"🧐 [自检] 当前脚本使用的 COLMAP 路径: {colmap_loc}")

# 设置日志级别
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 🔧 用户配置 (暴力裁剪版) =================
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
SCENE_RADIUS_SCALE = 1.8 
MAX_IMAGES =25 # 🔥 全局最大图片数量限制 (VGGT 显存优化)

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

# ================= VGGT 核心处理函数 =================
def run_vggt_pipeline(image_dir, output_sparse_dir, use_ba=False):
    """
    使用 VGGT 替代 COLMAP 进行稀疏重建
    """
    print(f"🚀 [VGGT] 正在启动神经网络 SfM...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 1. 加载模型 (会自动下载权重)
    model = VGGT()
    # 如果无法连接 HuggingFace，请手动下载模型并修改此处路径
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    try:
        state_dict = torch.hub.load_state_dict_from_url(_URL)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"    ⚠️ 自动下载模型失败: {e}")
        print("    -> 请确保网络通畅或手动下载模型。")
        raise e

    model.eval()
    model = model.to(device)
    print("    -> VGGT 模型加载完成")

    # 2. 加载图片
    image_paths = sorted(list(Path(image_dir).glob("*")))
    if not image_paths:
        raise ValueError("VGGT 输入目录为空")
        
    # VGGT 默认推理分辨率 518，加载分辨率设为 1024 (保持细节)
    vggt_res = 336
    load_res = 1024 
    
    print(f"    -> 正在预处理 {len(image_paths)} 张图片...")
    # images_tensor: (B, 3, H, W), coords: (B, 6) [x1, y1, x2, y2, w, h]
    images_tensor, original_coords = load_and_preprocess_images_square(image_paths, target_size=load_res)
    images_tensor = images_tensor.to(device)
    original_coords = original_coords.to(device)

    # 3. 运行 VGGT 推理
    print("    -> 正在执行前向推理 (这可能需要几秒钟)...")
    images_input = F.interpolate(images_tensor, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # 添加 batch 维度
            aggregated_tokens_list, ps_idx = model.aggregator(images_input[None])
            
            # 预测相机
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_input.shape[-2:])
            
            # 预测深度
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_input[None], ps_idx)
    
    del aggregated_tokens_list
    torch.cuda.empty_cache()

    # 移除 batch 维度并转为 numpy
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    # 3. 反投影生成 3D 点云
    print("    -> 生成稀疏点云...")
    # 直接传入 numpy 数组，不需要转 tensor，也不需要再调 .numpy()
    points_3d = unproject_depth_map_to_point_map(
        depth_map, 
        extrinsic, 
        intrinsic
    )

    # 4. 转换为 COLMAP 格式
    print("    -> 正在转换为 COLMAP 格式...")
    
    # 准备点云颜色和坐标网格
    points_rgb = F.interpolate(images_tensor, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    
    num_frames, height, width, _ = points_3d.shape
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    
    # 🔥 [修改] 动态点云过滤策略 (防止点云为空)
    # 目标：保留置信度最高的 10万个点 (至少保留一些点)
    TARGET_POINTS = 100000
    
    # 将置信度图展平
    conf_flat = depth_conf.reshape(-1)
    
    # 如果总像素点少于目标数，则只过滤极低信心的点
    if conf_flat.shape[0] <= TARGET_POINTS:
        conf_threshold = 0.1
    else:
        # 找到第 K 大的置信度值作为阈值
        # 使用 np.partition 快速找到 Top-K 的分界线
        # 我们取倒数第 TARGET_POINTS 个位置的值
        k_idx = conf_flat.shape[0] - TARGET_POINTS
        conf_threshold = np.partition(conf_flat, k_idx)[k_idx]
        
        # 确保阈值至少是 0.1 (过滤掉纯噪声)
        conf_threshold = max(float(conf_threshold), 0.1)

    print(f"    -> 动态调整置信度阈值: {conf_threshold:.4f} (保留 Top {TARGET_POINTS} 点)")
    
    # 生成掩码
    conf_mask = depth_conf >= conf_threshold
    
    points_3d_filtered = points_3d[conf_mask]
    points_xyf_filtered = points_xyf[conf_mask]
    points_rgb_filtered = points_rgb[conf_mask]
    
    # 生成 PyCOLMAP 重建对象
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d_filtered,
        points_xyf_filtered,
        points_rgb_filtered,
        extrinsic,
        intrinsic,
        image_size=np.array([vggt_res, vggt_res]),
        camera_type="PINHOLE"
    )
    
    # 5. 修正相机参数 (Rescale back to original resolution)
    # 这一步非常重要，因为 VGGT 是在缩放后的方形图上推理的
    base_image_names = [p.name for p in image_paths]
    
    # 引用 demo_colmap.py 中的 rename_colmap_recons_and_rescale_camera 逻辑
    # 这里为了简洁直接嵌入逻辑
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = base_image_names[pyimageid - 1]
        
        # 获取原始尺寸信息
        # original_coords: [x1, y1, x2, y2, width, height]
        real_w = original_coords[pyimageid - 1, 4].item()
        real_h = original_coords[pyimageid - 1, 5].item()
        max_dim = max(real_w, real_h)
        
        # 计算缩放比例: 从 vggt_res (518) 还原到 load_res (1024) 再还原到原始尺寸
        # 注意：load_and_preprocess_images_square 做了两件事：padding square 和 resize
        # VGGT 输出的是基于 vggt_res 的参数
        
        # 修正逻辑：
        # VGGT output (518) -> Load Res (1024) -> Original
        scale_vggt_to_load = load_res / vggt_res
        
        # load_res 是对原图做 padding 后 resize 得到的
        # scale_original_to_load = load_res / max(original_w, original_h)
        scale_load_to_original = max(real_w, real_h) / load_res
        
        total_scale = scale_vggt_to_load * scale_load_to_original
        
        # 缩放内参 (focal, cx, cy)
        pycamera.params *= total_scale
        
        # 修正主点 (Principal Point) 偏移
        # 原始预处理中可能有 padding (left, top)
        # padding 在 load_res 尺度下是：
        padding_left_load = original_coords[pyimageid - 1, 0].item()
        padding_top_load = original_coords[pyimageid - 1, 1].item()
        
        # 我们需要在还原后的尺度上减去这个 padding 带来的偏移吗？
        # demo_colmap.py 的逻辑是：
        # pred_params[-2:] = real_image_size / 2 (强制设为中心)
        # 这是一个简化假设，假设主点在中心。VGGT 训练时通常使用了中心裁剪。
        pycamera.params[2] = real_w / 2.0
        pycamera.params[3] = real_h / 2.0
        
        pycamera.width = int(real_w)
        pycamera.height = int(real_h)
        
        # 如果有点的 2D 观测，也需要 shift，但 batch_np_matrix_to_pycolmap_wo_track 
        # 生成的 reconstruction 点的 2D 坐标是基于 518 分辨率的
        # Nerfstudio 训练时会重新投影，或者我们可以忽略 sparse model 里的 2D points 位置，只用相机姿态
    
    # 6. 保存
    output_sparse_dir = Path(output_sparse_dir)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_sparse_dir))
    print(f"    -> 结果已保存至: {output_sparse_dir}")

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
    # ...
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 
    
    # 🔥 [新增] 强制修复 setuptools/distutils 冲突
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    # ...

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
    
    # 绝对路径调用
    system_colmap_exe = "/usr/local/bin/colmap" 
    
    # 双重保险：检查文件是否存在
    if not os.path.exists(system_colmap_exe):
        # shutil 已在文件头部导入，直接使用
        found_path = shutil.which("colmap")
        if found_path and "conda" not in found_path:
            system_colmap_exe = found_path
            print(f"    ⚠️ 警告: /usr/local/bin/colmap 不存在，尝试使用: {system_colmap_exe}")
        else:
            pass

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
    # [已移除] VGGT 不需要 COLMAP 特征提取
    # run_colmap_step([ ... ], "[1/4] GPU 特征提取")

    # 4. 手动运行 Sequential Matcher (顺序匹配)
    # [已移除] VGGT 不需要 COLMAP 匹配
    # run_colmap_step([ ... ], "[2/4] GPU 顺序匹配")

    # 4.5 手动运行 Mapper (稀疏重建)
    # [已移除] VGGT 不需要 COLMAP Mapper
    # run_colmap_step([ ... ], "[3/4] 稀疏重建 (Mapper)")

    # 修正路径定义
    colmap_output_dir = data_dir / "colmap"
    sparse_output_dir = colmap_output_dir / "sparse" / "0"
    
    # 🔥🔥🔥 替换开始: 使用 VGGT 替代 COLMAP 🔥🔥🔥
    print(f"\n⚡ [1/4] 使用 VGGT 替代 COLMAP 进行稀疏重建...")
    try:
        # 这里的 extracted_images_dir 是你之前清洗好的图片目录
        run_vggt_pipeline(
            image_dir=extracted_images_dir,
            output_sparse_dir=sparse_output_dir,
            use_ba=False # 初始版本建议 False，速度最快。如果需要更高精度可改为 True (需额外代码)
        )
    except Exception as e:
        print(f"❌ VGGT 运行失败: {e}")
        return None
    # 🔥🔥🔥 替换结束 🔥🔥🔥

    print(f"✅ VGGT 计算完成！")

    # =========================================================
    # 🔧 [3.5] 目录结构强力修正 (Auto-Fixer)
    # 由于 VGGT 直接输出到了正确位置，大部分修正逻辑可以跳过
    # =========================================================
    
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
            "--max-num-iterations", "15000", 
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
            # 1. 复制最终 PLY (可能是裁剪过的)
            shutil.copy2(str(final_ply_to_use), str(final_ply_dst))
            
            # 2. 额外回传原始未裁剪模型 (用于对比或备份)
            final_raw_ply_dst = target_dir / f"{project_name}_raw.ply"
            if raw_ply.exists():
                shutil.copy2(str(raw_ply), str(final_raw_ply_dst))
                print(f"    -> 原始模型已备份: {final_raw_ply_dst.name}")
            
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