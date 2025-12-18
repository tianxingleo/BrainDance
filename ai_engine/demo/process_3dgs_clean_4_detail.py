import subprocess # 引入子进程管理库：用于在Python脚本中执行外部系统命令（如 ffmpeg, colmap, ns-train 等 CLI 工具）
import sys # 引入系统相关库：用于获取命令行参数 (sys.argv) 和管理 Python 路径
import shutil # 引入高级文件操作库：用于复制文件 (copy), 移动文件 (move), 删除目录树 (rmtree) 以及查找可执行文件路径 (which)
import os # 引入操作系统接口库：用于环境变量设置, 路径拼接, 文件状态检测等
import time # 引入时间库：用于计算程序运行耗时 (time.time()) 和 线程休眠 (time.sleep)
import datetime # 引入日期时间库：用于将秒数格式化为人类可读的时间格式 (HH:MM:SS)
from pathlib import Path # 引入面向对象的文件路径处理库：比 os.path 更直观，用于跨平台路径操作
import json # 引入JSON处理库：用于读取和写入相机姿态文件 (transforms.json)
import numpy as np # 引入数值计算库：用于矩阵运算、向量计算、统计分析（如计算分位数、均值等），是科学计算的核心
import logging # 引入日志库：用于控制第三方库（如 nerfstudio）的日志输出级别
import cv2 # 引入OpenCV库 (计算机视觉库)：用于读取图片、图像灰度化、计算拉普拉斯梯度（模糊检测）、视频抽帧等
import re # 引入正则表达式库：用于从 COLMAP 的日志文本中提取关键信息（如匹配率百分比）

import os # (重复导入，无影响)
# 设置环境变量，强制 setuptools 使用标准库的 distutils。
# 这是为了解决高版本 Python (3.10+) 中 setuptools 和 distutils 的兼容性警告或报错问题。
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import torch # 引入 PyTorch 深度学习框架：Nerfstudio 的底层引擎
# 设置矩阵乘法的精度为 'high' (相当于开启 TF32 - TensorFloat-32)。
# 功能：在 NVIDIA Ampere 架构及以后的显卡上，能显著提升训练速度，同时保持足够的精度。
torch.set_float32_matmul_precision('high') 

# 🔥【绝杀】强制将编译好的系统级 colmap 路径提到最前面
# 背景：Conda 环境中常自带一个阉割版或旧版的 colmap，会导致功能缺失。
# 逻辑：强制将系统默认路径 (/usr/local/bin) 插入到 PATH 环境变量的最前面。
sys_path = "/usr/local/bin" # 定义系统级二进制文件目录
current_path = os.environ.get("PATH", "") # 获取当前的环境变量 PATH

# 判断 sys_path 是否已经在 PATH 的第一个位置 (用 os.pathsep 分割，Linux下是冒号)
if sys_path not in current_path.split(os.pathsep)[0]: 
    print(f"⚡ [环境修正] 强制设置 PATH 优先级: {sys_path} -> Priority High")
    # 将 sys_path 拼接到最前面，覆盖掉 Conda 或其他环境中的同名工具
    os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"

# 验证一下 colmap 的路径
import shutil # (重复导入，无影响)
colmap_loc = shutil.which("colmap") # 查找当前环境下 'colmap' 命令的具体路径
print(f"🧐 [自检] 当前脚本使用的 COLMAP 路径: {colmap_loc}") # 打印路径供用户核对

# 设置日志级别
# 功能：屏蔽 Nerfstudio 的 INFO/WARNING 级别日志，只显示 ERROR，让控制台输出更清爽。
logging.getLogger('nerfstudio').setLevel(logging.ERROR) 

# ================= 🔧 用户配置 (暴力裁剪版) =================
# 定义工作根目录：在当前用户的主目录下创建一个名为 "braindance_workspace" 的文件夹
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"
# 场景半径缩放比例：用于 adaptive_collider 计算，决定训练时的近平面和远平面范围。1.8 表示在计算出的物体半径基础上扩大 1.8 倍。
SCENE_RADIUS_SCALE = 1.8 
# 🔥 全局最大图片数量限制：为了防止显存爆炸或训练时间过长，限制送入 COLMAP 的图片不超过 200 张。
MAX_IMAGES = 200 

# ================= 辅助工具：时间格式化 =================
def format_duration(seconds):
    """
    功能：将浮点数秒数转换为 HH:MM:SS 格式的字符串。
    参数 seconds: 耗时秒数 (float)。
    """
    return str(datetime.timedelta(seconds=int(seconds)))

# ================= 辅助工具：模糊图片过滤 =================
def smart_filter_blurry_images(image_folder, keep_ratio=0.85, max_images=MAX_IMAGES):
    """
    升级版清洗脚本：混合策略 (Hybrid Strategy)
    该函数实现了一套复杂的图片筛选逻辑，包含质量评估和均匀采样。
    
    参数:
        image_folder: 图片所在的文件夹路径。
        keep_ratio: 质量保留比例，0.85 表示剔除最差的 15%。
        max_images: 最终保留的最大图片数量。
    """
    print(f"\n🧠 [智能清洗] 正在分析图片质量 (混合策略版)...")
    
    image_dir = Path(image_folder) # 将路径字符串转换为 Path 对象
    # 获取目录下所有 jpg, jpeg, png 后缀的文件，并排序
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not images: # 如果列表为空
        print("❌ 没找到图片")
        return

    # 创建一个存放废弃图片的目录 "trash_smart"，位于图片目录的上级目录中
    trash_dir = image_dir.parent / "trash_smart"
    trash_dir.mkdir(exist_ok=True) # 创建目录，如果存在则不报错

    img_scores = [] # 用于存储 (图片路径, 清晰度分数) 的列表

    # --- 第一步：计算分数 (Laplacian Variance) ---
    # 遍历每一张图片进行评分
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path)) # 使用 OpenCV 读取图片
        if img is None: continue # 如果读取失败（如文件损坏），跳过
        
        # 将图片转为灰度图，因为清晰度检测主要看梯度，不需要颜色信息
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape # 获取图片的高度和宽度

        # 算法逻辑：九宫格评分法
        # 为了避免背景虚化导致整张图分数低，我们将图片切成 3x3 的格子，取最清晰的那个格子作为整张图的分数。
        grid_h, grid_w = h // 3, w // 3 # 计算每个格子的尺寸
        max_grid_score = 0 # 初始化当前图片的最大分数为 0
        
        # 双重循环遍历 3x3 网格
        for r in range(3):
            for c in range(3):
                # 切片操作，提取当前格子的图像区域 (Region of Interest)
                roi = gray[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                # 核心算法：使用拉普拉斯算子计算图像的二阶导数，并求方差。
                # 方差越大，说明边缘越锐利，图片越清晰。
                score = cv2.Laplacian(roi, cv2.CV_64F).var()
                if score > max_grid_score:
                    max_grid_score = score # 更新最大分数
        
        # 将结果存入列表
        img_scores.append((img_path, max_grid_score))
        # 每处理 20 张打印一次进度
        if i % 20 == 0:
            print(f"  -> 分析中... {img_path.name}: 局部最高分 {max_grid_score:.1f}")

    # --- 第二步：质量清洗 (剔除废片) ---
    # 提取所有的分数组成一个列表
    scores = [s[1] for s in img_scores]
    if not scores: return

    num_total = len(scores)
    # 使用 numpy 计算百分位数阈值。
    # 例如 keep_ratio=0.85，我们要剔除底部 15% 的分数。
    # np.percentile 计算出第 15% 位置的分数值，低于这个分数的都将被剔除。
    quality_threshold = np.percentile(scores, (1 - keep_ratio) * 100)
    
    print(f"\n📊 统计结果:")
    print(f"   - 图片总数: {num_total}")
    print(f"   - 质量阈值 (Bottom {(1-keep_ratio)*100:.0f}%): {quality_threshold:.2f}")

    good_images = [] # 暂存合格的图片 (路径)
    removed_count_quality = 0 # 记录因质量差被移除的数量

    # 遍历所有图片及其分数，进行筛选
    for img_path, score in img_scores:
        if score < quality_threshold:
            # 如果分数低于阈值，移动到垃圾桶目录
            # shutil.move 实现文件移动操作
            shutil.move(str(img_path), str(trash_dir / img_path.name))
            removed_count_quality += 1
        else:
            good_images.append(img_path) # 质量合格，暂时保留

    print(f"   -> 第一轮清洗完成: 剔除 {removed_count_quality} 张废片，剩余 {len(good_images)} 张合格图片。")

    # --- 第三步：数量控制 (均匀采样) ---
    removed_count_quantity = 0 # 记录因超出数量限制被移除的数量
    
    # 如果合格的图片数量依然超过全局最大限制 (MAX_IMAGES)
    if len(good_images) > max_images:
        print(f"   ⚠️ 合格图片 ({len(good_images)}) 仍超过上限 ({max_images})")
        print(f"   -> 执行【均匀采样】以保证视角覆盖...")
        
        # 算法逻辑：均匀采样 (Uniform Sampling)
        # np.linspace 在 [0, len-1] 区间内均匀生成 max_images 个数字。
        # 作用：确保保留的图片在时间轴/空间轴上是分布均匀的，而不是只保留前200张。
        indices_to_keep = set(np.linspace(0, len(good_images) - 1, max_images, dtype=int))
        
        # 遍历当前所有合格图片
        for idx, img_path in enumerate(good_images):
            if idx not in indices_to_keep: # 如果索引不在保留列表中
                # 虽然质量合格，但为了数量限制不得不删
                shutil.move(str(img_path), str(trash_dir / img_path.name))
                removed_count_quantity += 1
    else:
        print(f"   ✅ 合格图片数量 ({len(good_images)}) 未超标，全部保留。")

    # 统计最终结果
    total_removed = removed_count_quality + removed_count_quantity
    final_count = num_total - total_removed
    print(f"✨ 清洗结束: 共移除 {total_removed} 张 (废片 {removed_count_quality} + 采样 {removed_count_quantity})，最终保留 {final_count} 张。")

# 🔥 强制开启球体切割配置
# 如果设为 True，无论场景被判断为什么类型，都会在最后执行点云切割。
FORCE_SPHERICAL_CULLING = True

# 🔥 核心参数：保留百分比 (0.0 ~ 1.0)
# 用于最后的点云裁剪算法。
# 0.9 表示计算所有点到中心的距离，只保留距离最近的 90% 的点，去除最远的 10% (通常是天空或极远处的伪影)。
KEEP_PERCENTILE = 0.9

# 检查依赖：plyfile 库
# plyfile 用于读写 .ply 格式的 3D 模型文件。
try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True # 标记库已安装
except ImportError:
    HAS_PLYFILE = False # 标记库未安装
    print("❌ 严重警告: 未安装 plyfile 库！无法执行切割。请运行: pip install plyfile")

# ================= 核心算法 1: 训练参数计算 =================
def analyze_and_calculate_adaptive_collider(json_path):
    """
    功能：根据相机轨迹 (transforms.json) 分析场景类型（是物体 Object 还是场景 Scene），
    并计算动态的 collider 参数 (near/far planes)，以优化 NeRF/Splat 训练效果。
    
    参数: json_path: transforms.json 文件的路径。
    返回: (参数列表, 场景类型字符串)
    """
    print(f"\n🤖 [AI 分析] 解析相机轨迹...")
    try:
        # 读取 json 文件
        with open(json_path, 'r') as f: data = json.load(f)
        frames = data["frames"] # 获取所有帧的信息
        if not frames: return [], "unknown" # 如果没帧，返回未知

        positions = [] # 存储相机位置 (XYZ)
        forward_vectors = [] # 存储相机朝向向量
        dists_to_origin = [] # 存储相机到世界原点的距离
        
        for frame in frames:
            c2w = np.array(frame["transform_matrix"]) # 读取 4x4 变换矩阵 (Camera-to-World)
            positions.append(c2w[:3, 3]) # 提取平移向量 (相机位置)
            # 计算相机的前方向量 (假设 OpenCV 坐标系：Z轴向内，所以 -Z 是前方)
            # 矩阵乘法：旋转矩阵 @ [0,0,-1]
            forward_vectors.append(c2w[:3, :3] @ np.array([0, 0, -1]))
            # 计算位置到原点 (0,0,0) 的欧几里得距离
            dists_to_origin.append(np.linalg.norm(c2w[:3, 3]))
            
        positions = np.array(positions) # 转为 numpy 数组
        forward_vectors = np.array(forward_vectors)
        
        # 计算所有相机位置的几何中心 (重心)
        center = np.mean(positions, axis=0)
        # 计算从相机位置指向中心的向量
        vec_to_center = center - positions
        # 归一化向量 (除以模长)，防止除以0加了个 1e-6
        vec_to_center /= (np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-6)
        
        # 逻辑判断：计算“相机朝向”与“指向中心向量”的点积。
        # 点积 > 0 表示夹角小于 90度，即相机是看着中心的。
        # 统计看着中心的相机的比例。
        ratio = np.sum(np.sum(forward_vectors * vec_to_center, axis=1) > 0) / len(frames)
        
        print(f"    -> 相机聚合度: {ratio:.2f}")

        # 如果超过 60% 的相机都看着中心，或者强制开启了切割，则认为是“物体模式”(Object Mode)
        is_object_mode = ratio > 0.6 or FORCE_SPHERICAL_CULLING

        if is_object_mode:
            # 物体模式下的逻辑
            avg_dist = np.mean(dists_to_origin) # 平均拍摄距离
            min_dist = np.min(dists_to_origin) # 最近拍摄距离
            scene_radius = 1.0 * SCENE_RADIUS_SCALE # 场景半径
            
            # 动态计算近平面 (near) 和远平面 (far)
            # near: 避免切掉太近的物体
            calc_near = max(0.05, min_dist - scene_radius)
            # far: 涵盖平均距离 + 半径
            calc_far = avg_dist + scene_radius
            
            # 返回 nerfstudio 的训练参数
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", str(round(calc_near, 2)), 
                    "far_plane", str(round(calc_far, 2))], "object"
        else:
            # 场景模式 (如航拍、漫游)，设置很大的远平面
            return ["--pipeline.model.enable-collider", "True", 
                    "--pipeline.model.collider-params", "near_plane", "0.05", "far_plane", "100.0"], "scene"

    except Exception as e:
        # 出错时的默认参数
        return ["--pipeline.model.enable-collider", "True", 
                "--pipeline.model.collider-params", "near_plane", "0.1", "far_plane", "50.0"], "unknown"

# ================= 核心算法 2: 基于分位数的暴力切割 (New!) =================
def perform_percentile_culling(ply_path, json_path, output_path):
    """
    功能：对生成的点云进行后处理切割。
    逻辑：计算所有点云到相机轨迹中心的距离，保留最近的 X% (KEEP_PERCENTILE)，删除远处的背景噪声。
    这是解决 3DGS 生成大量漂浮背景噪点的有效方法。
    """
    if not HAS_PLYFILE: # 检查依赖
        print("❌ 缺少 plyfile 库，跳过切割。")
        return False
        
    print(f"\n✂️ [后处理] 正在执行【分位数暴力切割】...")
    print(f"🔥 目标: 只保留离圆心最近的 {KEEP_PERCENTILE*100:.0f}% 点云")

    try:
        # 1. 计算切割中心
        # 依然使用相机位置的重心作为球心，因为对于绕物拍摄，相机重心通常就是物体中心。
        with open(json_path, 'r') as f: frames = json.load(f)["frames"]
        cam_pos = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
        center = np.mean(cam_pos, axis=0)
        
        print(f"    -> 切割圆心 (相机重心): {center}")

        # 2. 读取原始 PLY 点云文件
        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex'] # 获取顶点数据
        
        # 提取 x, y, z 坐标
        x, y, z = vertex['x'], vertex['y'], vertex['z']
        # 堆叠成 (N, 3) 的矩阵
        points = np.stack([x, y, z], axis=1)
        original_count = len(points)
        
        # 3. 计算所有点到中心的欧几里得距离
        print("    -> 正在计算所有点的距离分布...")
        dists_pts = np.linalg.norm(points - center, axis=1)
        
        # 4. === 核心逻辑：计算分位数阈值 ===
        # np.percentile 找到一个距离值 D，使得只有 KEEP_PERCENTILE (如 90%) 的点距离小于 D。
        threshold_radius = np.percentile(dists_pts, KEEP_PERCENTILE * 100)
        
        print(f"    -> 统计结果: {KEEP_PERCENTILE*100:.0f}% 的点集中在半径 {threshold_radius:.4f} 以内")
        print(f"    -> 执行切割: 所有大于 {threshold_radius:.4f} 的点将被删除")
        
        # 5. 执行切割
        # 获取点的不透明度 (opacity)。在 Gaussian Splatting 中，opacity 通常经过 sigmoid 激活，所以这里用 sigmoid 还原一下（或者直接取值，取决于存储格式）。
        # 这里代码写的是逆 Sigmoid 的逻辑：1 / (1 + exp(-x))，其实应该是 ply 里存的是 logit，这里转成概率。
        opacities = 1 / (1 + np.exp(-vertex['opacity']))
        
        # 组合过滤条件 (Mask)：
        # 条件1: 距离 < 阈值 (保留近处的点)
        # 条件2: 不透明度 > 0.05 (保留比较实、看得见的点，剔除半透明幽灵点)
        mask = (dists_pts < threshold_radius) & (opacities > 0.05)
        
        # 应用 Mask，只保留符合条件的顶点
        filtered_vertex = vertex[mask]
        new_count = len(filtered_vertex)
        
        print(f"    -> 原始点数: {original_count}")
        print(f"    -> 剩余点数: {new_count} (删除了 {original_count - new_count} 个背景点)")
        
        # 6. 保存新的 PLY 文件
        # PlyElement.describe 用于创建新的 PLY 元素结构
        PlyData([PlyElement.describe(filtered_vertex, 'vertex')]).write(str(output_path))
        return True # 返回成功

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
    # 定义核心目录变量
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen" 

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
    
    # 接上文：all_candidates = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")))
    
    # 251 行左右：获取临时目录下所有图片的总数
    total_candidates = len(all_candidates)
    # MAX_IMAGES = 200 # 全局变量，之前定义过
    
    final_images_list = [] # 用于存储最终决定送入 COLMAP 的图片路径列表
    
    # 逻辑：如果清洗后的图片数量依然超过设定的上限 (200张)
    if total_candidates > MAX_IMAGES:
        print(f"    ⚠️ 图片过多 ({total_candidates}), 正在均匀选取 {MAX_IMAGES} 张...")
        # 【核心算法】均匀采样 (Uniform Sampling)
        # np.linspace(start, stop, num): 在 0 到 总数-1 之间生成 num 个均匀分布的数字
        # 例如：从 1000 张里取 200 张，它会算出 [0, 5, 10, 15...] 这样的索引
        indices = np.linspace(0, total_candidates - 1, MAX_IMAGES, dtype=int)
        # set 去重 + sorted 排序：防止极端情况下产生重复索引
        indices = sorted(list(set(indices)))
        
        # 根据计算出的索引，从候选列表中提取图片
        for idx in indices:
            final_images_list.append(all_candidates[idx])
    else:
        # 如果数量未超标，则直接保留所有图片
        print(f"    ✅ 图片数量 ({total_candidates}) 未超标，全部保留。")
        final_images_list = all_candidates

    # 执行文件复制操作：只把最终选中的 "精英图片" 放入 extracted_images_dir (即 raw_images)
    # 这样做是为了隔离脏数据，确保 COLMAP 只处理最好的图片
    for img_path in final_images_list:
        shutil.copy2(str(img_path), str(extracted_images_dir / img_path.name))
        
    print(f"    ✅ 已将 {len(final_images_list)} 张干净图片移入 COLMAP 专用目录。")
    print(f"    🧹 正在清理临时文件...")
    # 删除临时目录 temp_dir，节省磁盘空间并保持工作区整洁
    shutil.rmtree(temp_dir) # 删掉脏区，防止混淆

    # =========================================================
    # 🚀 COLMAP 启动
    # =========================================================
    
    print(f"    ✅ 准备启动 COLMAP (Linux GPU 模式)...")
    
    # 定义 COLMAP 输出的数据库路径 (database.db 是 COLMAP 存储特征点和匹配关系的核心文件)
    colmap_output_dir = data_dir / "colmap"
    colmap_output_dir.mkdir(parents=True, exist_ok=True)
    database_path = colmap_output_dir / "database.db"
    
    # 尝试使用绝对路径调用系统安装的 COLMAP
    system_colmap_exe = "/usr/local/bin/colmap" 
    
    # 双重保险机制：检查该路径下的文件是否存在
    if not os.path.exists(system_colmap_exe):
        # 如果指定路径不存在，使用 shutil.which 在系统 PATH 中自动查找 "colmap" 命令
        found_path = shutil.which("colmap")
        # 排除掉 conda 环境自带的阉割版 colmap (通常 conda 的 colmap 没有 CUDA 支持)
        if found_path and "conda" not in found_path:
            system_colmap_exe = found_path
            print(f"    ⚠️ 警告: /usr/local/bin/colmap 不存在，尝试使用: {system_colmap_exe}")
        else:
            # 如果也没找到，后续执行可能会报错，但这里不做处理，依赖系统抛出异常
            pass

    full_log_content = [] # 用于存储 COLMAP 的所有日志输出，以便后续进行正则分析

    # 定义一个内部函数，用于执行 COLMAP 的各个子命令
    def run_colmap_step(cmd, step_desc):
        """
        参数:
            cmd: 命令行参数列表 (list)
            step_desc: 步骤描述字符串 (用于打印日志)
        """
        print(f"\n⚡ {step_desc}...")
        try:
            # subprocess.Popen: 启动子进程
            # stdout=subprocess.PIPE, stderr=subprocess.STDOUT: 将标准输出和错误输出合并捕获
            # text=True: 以文本形式读取输出
            # bufsize=1: 行缓冲，实时输出
            with subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True, 
                env=env, # 传入环境变量 (包含 QT_QPA_PLATFORM="offscreen" 防止弹窗)
                bufsize=1 
            ) as process:
                # 实时逐行读取日志并打印，让用户看到进度
                for line in process.stdout:
                    print(line, end='') 
                    full_log_content.append(line) # 同时保存到列表中
                
                process.wait() # 等待子进程结束
                if process.returncode != 0: # 如果返回值不为0，说明执行出错
                    raise subprocess.CalledProcessError(process.returncode, cmd)
        except Exception as e:
            print(f"\n❌ {step_desc} 执行异常: {e}")
            raise e # 向上抛出异常，中断流程

    # 3. 手动运行 Feature Extractor (特征提取)
    # COLMAP 第一步：分析每张图片，提取 SIFT 特征点
    # 注意：这里移除了 --SiftExtraction.use_gpu 参数，因为新版 COLMAP 如果检测到 CUDA 会自动开启，
    # 显式指定在某些 CPU 机器上反而会报错。
    run_colmap_step([
        system_colmap_exe, "feature_extractor",
        "--database_path", str(database_path), # 数据库文件
        "--image_path", str(extracted_images_dir), # 图片目录
        "--ImageReader.camera_model", "OPENCV", # 指定相机模型为 OpenCV (常用且兼容性好)
        "--ImageReader.single_camera", "1" # 假设所有图片来自同一个相机 (共用内参)，有助于提高重建稳定性
    ], "[1/4] GPU 特征提取")

    # 4. 手动运行 Sequential Matcher (顺序匹配)
    # COLMAP 第二步：匹配特征点。因为是视频抽帧，图片之间有时间连续性，
    # 所以使用 sequential_matcher 比 exhaustive_matcher (穷举) 快得多且效果更好。
    run_colmap_step([
        system_colmap_exe, "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "25" # 假设相邻的 25 张图片可能有重叠，只在这些范围内进行匹配
    ], "[2/4] GPU 顺序匹配")

    # 4.5 手动运行 Mapper (稀疏重建) 
    # COLMAP 第三步：利用匹配关系计算相机位姿和稀疏点云
    # 我们需要创建 sparse/0 目录，这是 Nerfstudio 默认读取 COLMAP 数据的标准结构
    sparse_output_dir = colmap_output_dir / "sparse" / "0"
    sparse_output_dir.mkdir(parents=True, exist_ok=True)
    
    run_colmap_step([
        system_colmap_exe, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(extracted_images_dir),
        "--output_path", str(sparse_output_dir) # 强制输出到 sparse/0
    ], "[3/4] 稀疏重建 (Mapper)")

    print(f"✅ COLMAP 计算完成！正在检查并修正目录结构...")

    # =========================================================
    # 🔧 [3.5] 目录结构强力修正 (Auto-Fixer)
    # 背景：COLMAP 的 mapper 命令在不同版本行为不一致，有时它会在 output_path 下再建一层 '0'，
    # 有时则直接输出。为了保证后续步骤不出错，这里写了一段逻辑来“纠正”文件位置。
    # =========================================================
    
    colmap_root = colmap_output_dir  # .../data/colmap
    sparse_root = colmap_root / "sparse"
    target_dir_0 = sparse_root / "0"
    target_dir_0.mkdir(parents=True, exist_ok=True) # 确保目标目录存在

    # 定义 COLMAP 模型的两套标准文件名 (二进制 bin 或 文本 txt)
    required_files_bin = ["cameras.bin", "images.bin", "points3D.bin"]
    required_files_txt = ["cameras.txt", "images.txt", "points3D.txt"]
    
    model_found = False # 标记是否找到了完整的模型文件

    # 1. 检查是不是已经在 sparse/0 (完美情况，不需要动)
    if all((target_dir_0 / f).exists() for f in required_files_bin):
        print("    ✅ 模型文件 (BIN) 位置正确。")
        model_found = True
    elif all((target_dir_0 / f).exists() for f in required_files_txt):
        print("    ✅ 模型文件 (TXT) 位置正确。")
        model_found = True
        
    # 2. 检查是不是在 sparse 根目录 (常见错误情况) -> 需要搬运到 sparse/0
    if not model_found:
        if all((sparse_root / f).exists() for f in required_files_bin):
            print("    🔧 检测到 BIN 模型在 sparse 根目录，正在归位...")
            for f in required_files_bin:
                shutil.move(str(sparse_root / f), str(target_dir_0 / f)) # 移动文件
            model_found = True
        elif all((sparse_root / f).exists() for f in required_files_txt):
            print("    🔧 检测到 TXT 模型在 sparse 根目录，正在归位...")
            for f in required_files_txt:
                shutil.move(str(sparse_root / f), str(target_dir_0 / f))
            model_found = True

    # 3. 递归搜索：如果上面都没找到，可能在更深的子目录 (如 sparse/1 或 sparse/0/0)
    if not model_found:
        # os.walk 遍历所有子目录
        for root, dirs, files in os.walk(sparse_root):
            # 检查当前目录是否有 bin 模型
            if all(f in files for f in required_files_bin):
                src_path = Path(root)
                if src_path == target_dir_0: continue # 跳过目标目录自己
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

    # 如果找了一圈还是没找到，说明 COLMAP 彻底失败了
    if not model_found:
        print("❌ [严重错误] 在 sparse 目录下找不到完整的 COLMAP 模型文件！")
        print("    -> 可能原因：Mapper 失败，未能重建出场景。")
        # 抛出文件未找到异常，终止程序
        raise FileNotFoundError("COLMAP Mapper failed to generate valid model files.")

    # [3.6] 提前同步图片 (为了让 ns-process-data 能找到)
    # Nerfstudio 要求图片必须在 data/images 目录下，而我们之前是在 raw_images
    print(f"    -> 正在同步图片: raw_images -> data/images ...")
    dest_images_dir = data_dir / "images"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    
    valid_images = []
    # 再次搜索所有图片文件
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        valid_images.extend(list(extracted_images_dir.glob(ext)))
        
    for img_path in valid_images:
        shutil.copy2(str(img_path), str(dest_images_dir / img_path.name))
    print(f"    ✅ 已同步 {len(valid_images)} 张图片。")

    print(f"✅ 数据准备就绪！正在生成 transforms.json (用于后续切割)...")

    # 5. 运行 ns-process-data (生成 transforms.json)
    # 功能：将 COLMAP 的二进制数据转换为 Nerfstudio 所需的 JSON 格式
    # 关键参数：
    # --skip-colmap: 我们之前手动跑过 COLMAP 了，这里跳过
    # --skip-image-processing: 图片也处理过了，跳过
    # --num-downscales 0: 不生成缩略图，节省时间
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
    # 将之前的日志列表合并成一个长字符串
    full_log = "".join(full_log_content)
    
    # 1. 检测 "No convergence" (不收敛)
    # 这是 COLMAP 常见的失败报错，表示无法从图片中解算出 3D 结构
    if "Termination : No convergence" in full_log:
        print("\n❌ [严重错误] COLMAP 无法收敛 (No convergence)！")
        
        # 尝试提取匹配率，告知用户有多糟糕
        match_pct = re.search(r"COLMAP only found poses for (\d+\.?\d*)% of the images", full_log)
        if match_pct:
            print(f"    -> 成功注册图片比例: {match_pct.group(1)}% (质量过低)")
        else:
            # 备选方案：通过正则查找 "Registered images" 关键词来计算比例
            reg_match = re.findall(r"Registered images.*?(\d+)", full_log)
            if reg_match:
                registered_count = int(reg_match[-1])
                # 注意：这里 num_images 变量可能是引用之前的 total_candidates，代码此处可能有上下文依赖
                ratio = (registered_count / total_candidates) * 100 if total_candidates > 0 else 0
                print(f"    -> 成功注册图片: {registered_count}/{total_candidates} ({ratio:.2f}%)")
            
        print("🛑 任务已终止，因为生成的稀疏点云质量无法满足训练要求。")
        
        # 失败后清理工作目录
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print(f"🧹 清理完成: 已删除工作区 {work_dir}")
        return None

    # 2. 检测匹配率过低
    # 即使收敛了，如果只有 10% 的图片被匹配上，训练出来的结果也会很差
    match = re.search(r"COLMAP only found poses for (\d+\.?\d*)% of the images", full_log)
    if match:
        matched_percentage = float(match.group(1))
        print(f"\n📊 COLMAP 匹配率检测: {matched_percentage:.2f}%")
        
        if matched_percentage < 35.0: # 阈值：35%
            print(f"❌ [质量警告] 匹配率过低 (< 35%)！")
            print("    -> 这意味着大部分图片无法被定位，生成的 3D 场景将严重残缺。")
            print("🛑 任务已终止。建议：增加图片数量、保证图片清晰度或增加重叠率。")
            
            if work_dir.exists():
                shutil.rmtree(work_dir)
                print(f"🧹 清理完成: 已删除工作区 {work_dir}")
            return None

    step1_duration = time.time() - step1_start
    print(f"⏱️ [Step 1 完成] 耗时: {format_duration(step1_duration)}")

    # [Step 2] 训练
    step2_start = time.time()
    # 定义训练输出路径
    search_path = output_dir / project_name / "splatfacto"
    # 检查是否已经训练过 (如果有断点续传的需求)
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []
    
    scene_type_detected = "unknown"

    if run_dirs:
        # 如果找到旧的训练目录，跳过训练 (此处逻辑是为了防重复，或者调试用)
        print(f"\n⏩ [训练跳过] 已完成")
        _, scene_type_detected = analyze_and_calculate_adaptive_collider(transforms_file)
    else:
        # 调用之前定义的 AI 分析函数，计算场景参数 (collider args)
        collider_args, scene_type_detected = analyze_and_calculate_adaptive_collider(transforms_file)
        print(f"\n🧠 [2/3] 开始训练...")
        
        # 启动 ns-train (Nerfstudio 训练主程序)
        subprocess.run([
            "ns-train", "splatfacto", # 指定模型为 splatfacto (Nerfstudio 的 3DGS 实现)
            "--data", str(data_dir), # 数据路径
            "--output-dir", str(output_dir), # 输出路径
            "--experiment-name", project_name, # 实验名称
            "--pipeline.model.random-init", "False", # 关闭随机初始化，利用稀疏点云初始化
            "--pipeline.model.cull-alpha-thresh", "0.005", # 透明度剔除阈值
            *collider_args, # 解包传入动态计算的 collider 参数 (near/far planes)
            "--max-num-iterations", "15000", # 最大迭代次数 15000 步
            "--vis", "viewer+tensorboard", # 开启可视化支持
            "--viewer.quit-on-train-completion", "True", # 训练完自动关闭 Viewer 后台
            
            # 👇 子命令：指定使用 colmap 数据解析器
            "colmap", 
            
            # 👇 参数修正：只需写短名，并且必须放在 "colmap" 后面
            "--downscale-factor", "1" # 不对图片进行缩放，使用原图分辨率训练
        ], check=True, env=env)

    step2_duration = time.time() - step2_start
    print(f"⏱️ [Step 2 完成] 耗时: {format_duration(step2_duration)}")

    # [Step 3] 导出
    step3_start = time.time()
    print(f"\n💾 [3/3] 导出结果")
    # 重新查找训练结果目录 (确保拿到最新一次训练的文件夹)
    if not run_dirs: run_dirs = sorted(list(search_path.glob("*")))
    if not run_dirs: return None # 如果还是空的，说明训练失败
    latest_run = run_dirs[-1] # 取最新的那个
    
    # 运行 ns-export：将训练好的 checkpoint 转换为通用的 .ply 或 .splat 文件
    subprocess.run([
        "ns-export", "gaussian-splat", # 导出模式
        "--load-config", str(latest_run/"config.yml"), # 加载对应的配置文件
        "--output-dir", str(work_dir) # 导出到工作根目录
    ], check=True, env=env)
    time.sleep(5) # 等待文件系统同步

    # [Step 3.5] 分位数暴力切割 (核心后处理)
    # 检查导出的文件可能是 point_cloud.ply 或 splat.ply
    raw_ply = work_dir / "point_cloud.ply"
    if not raw_ply.exists(): raw_ply = work_dir / "splat.ply"

    cleaned_ply = work_dir / "point_cloud_cleaned.ply" # 切割后的文件名
    final_ply_to_use = raw_ply # 默认使用原始文件

    # 判断是否需要切割：如果是物体模式，或者强制开启了切割
    should_clean = (scene_type_detected == "object") or FORCE_SPHERICAL_CULLING
    
    if should_clean:
        if raw_ply.exists():
            # 调用之前的 perform_percentile_culling 函数
            if perform_percentile_culling(raw_ply, transforms_file, cleaned_ply):
                print("✨ 暴力切割成功！")
                final_ply_to_use = cleaned_ply # 指向切割后的文件
        else:
            print(f"❌ 警告：未找到 PLY 文件")
    else:
        print(f"ℹ️ 跳过切割")

    step3_duration = time.time() - step3_start
    print(f"⏱️ [Step 3 完成] 耗时: {format_duration(step3_duration)}")
# ... 接上文 Run Pipeline 函数的末尾部分 ...

    # =========================================================
    # 📦 [Step 4] 结果回传与环境清理
    # 这一步非常关键，它负责将 Linux (WSL/Server) 算好的结果
    # 搬运回 Windows 或结果目录，并清理庞大的临时文件。
    # =========================================================
    
    print(f"\n📦 [IO 同步] 回传至 Windows...")
    
    # 1. 定义结果存放的目标目录
    # Path(__file__).parent 获取当前脚本所在的文件夹
    # / "results" 在脚本同级目录下创建一个 results 文件夹
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True, parents=True) # 确保目录存在，不存在则创建
    
    # 2. 定义源文件路径 (在工作区中) 和 目标文件路径 (在结果目录中)
    transforms_src = data_dir / "transforms.json" # 源：Nerfstudio 生成的相机参数
    
    # 目标：WebGL 前端专用的姿态文件 (简化版)
    final_webgl_poses = target_dir / "webgl_poses.json" 
    # 目标：最终的点云模型文件
    final_ply_dst = target_dir / f"{project_name}.ply"
    # 目标：标准的相机参数文件备份
    final_transforms = target_dir / "transforms.json"
    
    # --- 核心逻辑：生成 WebGL 友好姿态文件 ---
    # Nerfstudio 的 transforms.json 包含很多训练参数，前端 WebGL 展示时不需要那么多
    # 这里我们生成一个轻量级的 json，只包含相机位姿矩阵。
    if transforms_src.exists():
        print("🔄 正在生成 WebGL 友好姿态文件 (webgl_poses.json)...")
        try:
            with open(transforms_src, 'r') as f:
                data = json.load(f) # 读取原始 JSON
            
            webgl_frames = []
            # 遍历每一帧 (每一张照片)
            for frame in data["frames"]:
                # 提取 4x4 变换矩阵 (Camera to World)
                c2w_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
                
                # 将矩阵转为 list 并存入新结构
                webgl_frames.append({
                    "file_path": frame["file_path"], # 图片路径
                    "pose_matrix_c2w": c2w_matrix.tolist() # 矩阵数据
                })
                
            # 构造精简版的数据字典
            webgl_data = {
                "camera_model": data.get("camera_model", "OPENCV"), # 相机模型
                "w": data.get("w", 0), # 宽
                "h": data.get("h", 0), # 高
                "fl_x": data.get("fl_x", 0), # 焦距 X
                "fl_y": data.get("fl_y", 0), # 焦距 Y
                "frames": webgl_frames # 帧数据
            }
            
            # 写入 webgl_poses.json
            with open(final_webgl_poses, 'w') as f:
                json.dump(webgl_data, f, indent=4)
            print(f"✅ WebGL 姿态文件已保存至: {final_webgl_poses.resolve()}")
        except Exception as e:
            print(f"❌ 姿态预处理失败: {e}")

    # --- 文件复制与工作区清理 ---
    # final_ply_to_use 是在上一步 (Step 3) 中确定的最终 PLY 路径 (可能是原版，也可能是切割版)
    if final_ply_to_use and final_ply_to_use.exists():
        try:
            # 1. 复制最终 PLY 模型到结果目录
            # shutil.copy2 会保留文件的元数据 (创建时间等)
            shutil.copy2(str(final_ply_to_use), str(final_ply_dst))
            
            # 2. 额外备份原始模型 (Raw Model)
            # 如果我们进行了切割 (Culling)，为了防止切坏了没法补救，
            # 这里把未切割的原始 point_cloud.ply 也复制一份，命名为 *_raw.ply
            final_raw_ply_dst = target_dir / f"{project_name}_raw.ply"
            if raw_ply.exists():
                shutil.copy2(str(raw_ply), str(final_raw_ply_dst))
                print(f"    -> 原始模型已备份: {final_raw_ply_dst.name}")
            
            # 3. 复制 transforms.json
            if transforms_src.exists():
                shutil.copy2(str(transforms_src), str(final_transforms))
            
            # 4. 🔥【重要】清理 Linux 工作区
            # braindance_workspace 目录通常包含数千张解压的图片和巨大的 checkpoint 文件
            # 任务完成后必须删除，否则硬盘很快会满。
            shutil.rmtree(work_dir)
            print(f"🧹 清理完成: 已删除工作区 {work_dir}")
            
            # --- 最终统计 ---
            total_time = time.time() - global_start_time # 计算总耗时
            print(f"\n✅ =============================================")
            print(f"🎉 任务全部完成！安心睡觉吧。")
            print(f"📂 最终模型: {final_ply_dst}")
            print(f"⏱️ 总共耗时: {format_duration(total_time)}")
            print(f"✅ =============================================")
            
            return str(final_ply_dst) # 返回最终路径，供外部调用者使用
        except Exception as e:
            print(f"❌ 回传失败: {e}")
            return None
    else:
        # 如果找不到 PLY 文件，说明之前的步骤肯定出错了
        print("❌ 导出失败，未找到 PLY 文件 (point_cloud.ply 或 splat.ply)。")
        return None

# =========================================================
# 🎬 程序主入口 (Main Entry)
# =========================================================
if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    script_dir = Path(__file__).resolve().parent
    
    # 默认视频文件路径: 当前目录下的 test.mp4
    video_file = script_dir / "test.mp4" 
    
    # 命令行参数支持
    # 如果用户运行: python script.py my_video.mov
    # sys.argv[1] 就会获取到 "my_video.mov"，覆盖默认值
    if len(sys.argv) > 1: video_file = Path(sys.argv[1])

    # 检查视频是否存在
    if video_file.exists():
        # 启动主流程！
        # 项目名称定为 "scene_auto_sync"，这意味着每次运行都会覆盖这个项目名下的数据
        # (因为前面的代码里有 shutil.rmtree(work_dir) 的重置逻辑)
        run_pipeline(video_file, "scene_auto_sync")
    else:
        print(f"❌ 找不到视频: {video_file}")