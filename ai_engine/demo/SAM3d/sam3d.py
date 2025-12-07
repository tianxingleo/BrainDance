import os
import sys
import shutil
import time
import datetime
from pathlib import Path
import logging
import torch
from types import ModuleType
import numpy as np
from PIL import Image

# ================= 🔥 [RTX 5070 兼容性补丁 V32] 终极 Mock 🔥 =================
def inject_mocks():
    print("⚠️ [系统检测] 正在注入 Kaolin 和 PyTorch3D 的 V32 Mock 模块...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class MockClass:
        def __init__(self, *args, **kwargs): self.device = device
        def __call__(self, *args, **kwargs): return torch.zeros(1, 3, device=device, requires_grad=True)
        def compose(self, *args, **kwargs): return self
        def inverse(self): return self
        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def cuda(self): return self
        def clone(self): return self
        def detach(self): return self
        def get_matrix(self): return torch.eye(4, device=device).unsqueeze(0)
        def transform_points(self, x): return x 
        def transform_normals(self, x): return x
        def __getattr__(self, name):
            def method_mock(*args, **kwargs): return self
            return method_mock
    
    def mock_func(*args, **kwargs): return torch.tensor(0.0, device=device)
    def mock_check_func(*args, **kwargs): return False 

    # --- Kaolin Mock ---
    if "kaolin" not in sys.modules:
        mock_kaolin = ModuleType("kaolin")
        submodules = ["ops", "ops.mesh", "ops.spc", "metrics", "metrics.pointcloud", "render", "render.camera", "render.mesh", "visualize", "io", "io.obj", "io.usd", "utils", "utils.testing"]
        for name in submodules:
            parts = name.split(".")
            parent = mock_kaolin
            for i, part in enumerate(parts):
                if not hasattr(parent, part):
                    new_mod = ModuleType(f"kaolin.{'.'.join(parts[:i+1])}")
                    setattr(parent, part, new_mod)
                    sys.modules[f"kaolin.{'.'.join(parts[:i+1])}"] = new_mod
                parent = getattr(parent, part)
        mock_kaolin.ops.mesh.TriangleHash = MockClass
        mock_kaolin.ops.mesh.check_sign = mock_func
        mock_kaolin.ops.mesh.sample_points = mock_func
        mock_kaolin.metrics.pointcloud.chamfer_distance = mock_func
        mock_kaolin.visualize.IpyTurntableVisualizer = MockClass
        mock_kaolin.render.camera.Camera = MockClass
        mock_kaolin.render.camera.CameraExtrinsics = MockClass
        mock_kaolin.render.camera.PinholeIntrinsics = MockClass
        mock_kaolin.render.mesh.dibr_rasterization = mock_func
        mock_kaolin.io.obj.import_mesh = lambda *args, **kwargs: (None, None)
        mock_kaolin.utils.testing.check_tensor = mock_func
        mock_kaolin.__path__ = []
        sys.modules["kaolin"] = mock_kaolin

    # --- PyTorch3D Mock ---
    if "pytorch3d" not in sys.modules:
        mock_p3d = ModuleType("pytorch3d")
        mock_p3d.__path__ = []
        mock_p3d.transforms = ModuleType("pytorch3d.transforms")
        mock_p3d.structures = ModuleType("pytorch3d.structures")
        mock_p3d.renderer = ModuleType("pytorch3d.renderer")
        mock_p3d.renderer.cameras = ModuleType("pytorch3d.renderer.cameras")
        mock_p3d.renderer.camera_utils = ModuleType("pytorch3d.renderer.camera_utils")
        mock_p3d.renderer.mesh = ModuleType("pytorch3d.renderer.mesh")
        mock_p3d.renderer.mesh.textures = ModuleType("pytorch3d.renderer.mesh.textures")
        mock_p3d.renderer.mesh.rasterizer = ModuleType("pytorch3d.renderer.mesh.rasterizer")
        mock_p3d.renderer.mesh.shader = ModuleType("pytorch3d.renderer.mesh.shader")
        sys.modules["pytorch3d.renderer.cameras"] = mock_p3d.renderer.cameras
        sys.modules["pytorch3d.renderer.camera_utils"] = mock_p3d.renderer.camera_utils
        sys.modules["pytorch3d.renderer.mesh"] = mock_p3d.renderer.mesh
        sys.modules["pytorch3d.renderer.mesh.textures"] = mock_p3d.renderer.mesh.textures
        sys.modules["pytorch3d.renderer.mesh.rasterizer"] = mock_p3d.renderer.mesh.rasterizer
        sys.modules["pytorch3d.renderer.mesh.shader"] = mock_p3d.renderer.mesh.shader
        mock_p3d.vis = ModuleType("pytorch3d.vis")
        mock_plotly_vis = ModuleType("pytorch3d.vis.plotly_vis")
        mock_plotly_vis.AxisArgs = MockClass
        mock_plotly_vis.Lighting = MockClass
        mock_plotly_vis.plot_scene = mock_func
        mock_plotly_vis.get_camera_wireframe = mock_func
        mock_plotly_vis._add_camera_trace = mock_func
        mock_plotly_vis._add_ray_bundle_trace = mock_func
        mock_plotly_vis._add_pointcloud_trace = mock_func
        mock_plotly_vis._add_mesh_trace = mock_func
        mock_plotly_vis._scale_camera_to_bounds = mock_func
        mock_plotly_vis._update_axes_bounds = mock_func
        mock_plotly_vis._is_ray_bundle = mock_check_func
        mock_plotly_vis._is_pointclouds = mock_check_func
        mock_plotly_vis._is_meshes = mock_check_func
        mock_plotly_vis._is_cameras = mock_check_func
        mock_p3d.vis.plotly_vis = mock_plotly_vis
        sys.modules["pytorch3d.vis"] = mock_p3d.vis
        sys.modules["pytorch3d.vis.plotly_vis"] = mock_plotly_vis
        mock_p3d.transforms.Transform3d = MockClass
        mock_p3d.transforms.Rotate = MockClass
        mock_p3d.transforms.Translate = MockClass
        mock_p3d.transforms.Scale = MockClass
        mock_p3d.transforms.quaternion_multiply = lambda q1, q2: q1 
        mock_p3d.transforms.quaternion_invert = lambda q: q
        mock_p3d.transforms.matrix_to_quaternion = lambda m: torch.tensor([1., 0., 0., 0.], device=m.device).repeat(m.shape[0], 1)
        mock_p3d.transforms.quaternion_to_matrix = lambda q: torch.eye(3, device=q.device).unsqueeze(0).repeat(q.shape[0], 1, 1)
        mock_p3d.transforms.axis_angle_to_quaternion = lambda a: torch.tensor([1., 0., 0., 0.], device=a.device).repeat(a.shape[0], 1)
        mock_p3d.transforms.quaternion_to_axis_angle = lambda q: torch.zeros((q.shape[0], 3), device=q.device)
        mock_p3d.transforms.axis_angle_to_matrix = lambda a: torch.eye(3, device=a.device).unsqueeze(0).repeat(a.shape[0], 1, 1)
        mock_p3d.renderer.look_at_view_transform = lambda **kwargs: (torch.eye(3, device=device).unsqueeze(0), torch.zeros(1, 3, device=device))
        mock_p3d.renderer.look_at_rotation = lambda **kwargs: torch.eye(3, device=device).unsqueeze(0)
        mock_p3d.renderer.camera_position_from_spherical_angles = lambda **kwargs: torch.zeros(1, 3, device=device)
        mock_p3d.renderer.ray_bundle_to_ray_points = lambda **kwargs: torch.zeros(1, 3, device=device)
        mock_p3d.renderer.ray_points_to_depth = lambda **kwargs: torch.zeros(1, device=device)
        mock_p3d.renderer.camera_utils.camera_to_eye_at_up = lambda **kwargs: (torch.zeros(1, 3, device=device), torch.zeros(1, 3, device=device), torch.zeros(1, 3, device=device))
        mock_p3d.renderer.camera_utils.join_cameras_as_batch = mock_func
        renderer_classes = ["FoVPerspectiveCameras", "PerspectiveCameras", "CamerasBase", "OrthographicCameras", "PointsRenderer", "PointsRasterizationSettings", "PointsRasterizer", "AlphaCompositor", "RasterizationSettings", "MeshRenderer", "MeshRasterizer", "MeshRendererWithFragments", "SoftPhongShader", "HardPhongShader", "SoftSilhouetteShader", "TexturesVertex", "TexturesAtlas", "TexturesUV", "PointLights", "DirectionalLights", "AmbientLights", "Materials", "BlendParams", "HeterogeneousRayBundle", "RayBundle", "ImplicitRenderer", "NDCGridRaysampler", "MonteCarloRaysampler"]
        for cls_name in renderer_classes:
            mock_cls = MockClass
            setattr(mock_p3d.renderer, cls_name, mock_cls)
            if "Cameras" in cls_name: setattr(mock_p3d.renderer.cameras, cls_name, mock_cls)
            if "Textures" in cls_name: setattr(mock_p3d.renderer.mesh.textures, cls_name, mock_cls)
            if "Shader" in cls_name: setattr(mock_p3d.renderer.mesh.shader, cls_name, mock_cls)
            if "Rasterizer" in cls_name and "Mesh" in cls_name: setattr(mock_p3d.renderer.mesh.rasterizer, cls_name, mock_cls)
        mock_p3d.structures.Meshes = MockClass
        mock_p3d.structures.Pointclouds = MockClass
        mock_p3d.structures.join_meshes_as_scene = mock_func
        mock_p3d.structures.join_meshes_as_batch = mock_func
        mock_p3d.structures.list_to_padded = mock_func
        mock_p3d.structures.padded_to_list = mock_func
        sys.modules["pytorch3d"] = mock_p3d
        sys.modules["pytorch3d.transforms"] = mock_p3d.transforms
        sys.modules["pytorch3d.structures"] = mock_p3d.structures
        sys.modules["pytorch3d.renderer"] = mock_p3d.renderer
    print("✅ [Mock V32] 注入完成")

# 注入 Mocks
inject_mocks()

# ================= 🔧 配置区域 =================
INPUT_IMAGE_NAME = "input.jpg"  # 脚本会自动优先查找 input.png
LINUX_WORK_ROOT = Path.home() / "sam3d_workspace"
SAM3D_REPO_PATH = Path.home() / "workspace/ai/sam-3d-objects"
CONFIG_PATH = SAM3D_REPO_PATH / "checkpoints/hf/pipeline.yaml"

# 🔥🔥 [修复点] CPU 配置保存路径改为与 CONFIG_PATH 同级目录 🔥🔥
CPU_CONFIG_PATH = CONFIG_PATH.parent / "cpu_pipeline.yaml"

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def setup_environment():
    if not SAM3D_REPO_PATH.exists():
        print(f"❌ 错误: 找不到 SAM 3D 仓库路径: {SAM3D_REPO_PATH}")
        sys.exit(1)
    
    sys.path.append(str(SAM3D_REPO_PATH))
    sys.path.append(str(SAM3D_REPO_PATH / "notebook"))
    os.chdir(SAM3D_REPO_PATH)

def prepare_cpu_config():
    """创建一个强制使用 CPU 的临时配置文件"""
    print(f"📝 [Config Hack] 正在创建 CPU 初始化配置: {CPU_CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"❌ 找不到原始配置: {CONFIG_PATH}")
        return False
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 强制替换 device: cuda 为 device: cpu
        new_content = content.replace("device: cuda", "device: cpu")
        new_content = new_content.replace('device: "cuda"', 'device: "cpu"')
        
        # 写入到 checkpoints/hf/cpu_pipeline.yaml
        with open(CPU_CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"    ✅ 已生成 CPU 配置")
        return True
    except Exception as e:
        print(f"❌ 创建 CPU 配置失败: {e}")
        return False

def auto_generate_mask(image_np):
    """自动生成 Mask (简单的去白/黑背景)"""
    # image_np 是 (H, W, 3) 的 uint8
    
    # 策略1: 亮度阈值 (去除接近白色的背景)
    # 计算每个像素的亮度
    intensity = image_np.mean(axis=2)
    
    # 假设背景是白色的 (亮度 > 240)
    is_white_bg = intensity > 240
    
    # 假设背景是黑色的 (亮度 < 15)
    is_black_bg = intensity < 15
    
    # 生成 mask: 背景部分为 0，物体部分为 255
    # 如果大部分是白色背景，就去除白色；如果是黑色，就去除黑色
    white_pixel_count = np.sum(is_white_bg)
    black_pixel_count = np.sum(is_black_bg)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    
    if white_pixel_count > total_pixels * 0.1: # 如果有超过10%的白色，假设是白背景
        print("    🎨 检测到浅色背景，正在自动抠图...")
        mask = np.where(is_white_bg, 0, 255).astype(np.uint8)
    elif black_pixel_count > total_pixels * 0.1: # 否则假设黑背景
        print("    🎨 检测到深色背景，正在自动抠图...")
        mask = np.where(is_black_bg, 0, 255).astype(np.uint8)
    else:
        print("    ⚠️ 背景颜色不明确，使用全图 Mask (可能会生成方块)")
        mask = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.uint8) * 255
        
    return mask

def run_pipeline():
    global_start_time = time.time()
    
    windows_dir = Path(__file__).resolve().parent
    source_img_path = windows_dir / INPUT_IMAGE_NAME
    # 尝试寻找 png 文件 (带透明通道)
    source_png_path = windows_dir / "input.png"
    
    if source_png_path.exists():
        print(f"✨ 发现 input.png，将使用 Alpha 通道作为 Mask (推荐)")
        source_img_path = source_png_path
        INPUT_EXT = ".png"
    else:
        INPUT_EXT = ".jpg"
    
    project_name = source_img_path.stem 
    work_dir = LINUX_WORK_ROOT / project_name
    
    print(f"\n🚀 [RTX 5070 Pipeline V44] 启动任务: {source_img_path.name}")
    
    if not source_img_path.exists():
        print(f"❌ 错误: 找不到图片 {source_img_path}")
        return

    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    target_img_path = work_dir / source_img_path.name
    shutil.copy2(str(source_img_path), str(target_img_path))

    print(f"\n🧠 [2/3] 加载模型推理 (强制 CPU 初始化)...")
    setup_environment()
    
    if not prepare_cpu_config():
        return

    try:
        from inference import Inference, load_image
        import numpy as np
        from PIL import Image
        
        # 🔥🔥🔥 终极拦截器：强制 torch.load 使用 CPU 🔥🔥🔥
        original_torch_load = torch.load
        def cpu_load_hook(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = 'cpu'
            return original_torch_load(*args, **kwargs)
        
        print("    🛡️ 已激活显存拦截器：强制所有权重加载至 RAM...")
        torch.load = cpu_load_hook
        
        try:
            # 初始化 (所有模型进 RAM)
            inference = Inference(str(CPU_CONFIG_PATH))
            pipeline = inference._pipeline
            pipeline.device = torch.device('cuda') # 欺骗 Pipeline
            print("    ✅ 模型初始化完成，显存占用 0GB")
        finally:
            # 恢复 torch.load
            torch.load = original_torch_load
            print("    🛡️ 显存拦截器已解除")

        # 读取并处理图片
        pil_image = Image.open(str(target_img_path)).convert("RGBA") # 读取 RGBA 以防万一
        orig_w, orig_h = pil_image.size
        
        target_size = 512
        if max(orig_w, orig_h) > target_size:
            scale = target_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"    📉 [显存保护] 图片降采样至: {new_w} x {new_h}")
        
        image_rgba = np.array(pil_image)
        image = image_rgba[:, :, :3] # 取 RGB
        h, w = image.shape[:2]
        
        # 🔥🔥🔥 智能 Mask 生成 🔥🔥🔥
        if INPUT_EXT == ".png" and image_rgba.shape[2] == 4:
            # 如果是 PNG 且有 Alpha 通道，直接用 Alpha 作为 Mask
            print("    🎭 使用 PNG Alpha 通道作为 Mask")
            mask = image_rgba[:, :, 3]
        else:
            # 如果是 JPG，尝试自动去除背景
            mask = auto_generate_mask(image)
        
        # =========================================================
        # 🔥 第一阶段：搬运 Stage 1 模型到 GPU
        # =========================================================
        print("\n🚚 [Stage 1] 正在将 Stage 1 模型搬运到 GPU...")
        torch.cuda.empty_cache()
        
        pipeline.models["ss_generator"].to('cuda')
        pipeline.models["ss_decoder"].to('cuda')
        if "ss_encoder" in pipeline.models and pipeline.models["ss_encoder"] is not None:
            pipeline.models["ss_encoder"].to('cuda')
        
        if "ss_condition_embedder" in pipeline.condition_embedders:
            pipeline.condition_embedders["ss_condition_embedder"].to('cuda')
            
        print("🚀 [Step 1/2] 正在运行 Stage 1 (生成结构)...")
        stage1_output = pipeline.run(
            image=image, 
            mask=mask, 
            stage1_only=True, 
            seed=42
        )
        print("    ✅ Stage 1 完成！")

        # =========================================================
        # 🔥 第二阶段：撤回 Stage 1，搬运 Stage 2
        # =========================================================
        print("\n🔄 [显存切换] 卸载 Stage 1，加载 Stage 2...")
        pipeline.models["ss_generator"].cpu()
        pipeline.models["ss_decoder"].cpu()
        if "ss_encoder" in pipeline.models and pipeline.models["ss_encoder"] is not None:
             pipeline.models["ss_encoder"].cpu()
        if "ss_condition_embedder" in pipeline.condition_embedders:
             pipeline.condition_embedders["ss_condition_embedder"].cpu()
        torch.cuda.empty_cache()
        
        pipeline.models["slat_generator"].to('cuda')
        pipeline.models["slat_decoder_gs"].to('cuda')
        if "slat_condition_embedder" in pipeline.condition_embedders:
            pipeline.condition_embedders["slat_condition_embedder"].to('cuda')
        
        print("    ✅ 模型切换完毕！")
        pipeline.decode_formats = ["gaussian"]
        
        print("🚀 [Step 2/2] 正在运行 Stage 2 (生成 Gaussian)...")
        original_sample_ss = pipeline.sample_sparse_structure
        pipeline.sample_sparse_structure = lambda *args, **kwargs: stage1_output
        
        try:
            output = pipeline.run(
                image=image,
                mask=mask,
                stage1_only=False, 
                seed=42,
                with_mesh_postprocess=False, 
                with_texture_baking=False 
            )
        finally:
            pipeline.sample_sparse_structure = original_sample_ss

        # 保存结果
        if "gs" in output:
            gaussian_splats = output["gs"]
            ply_output_path = work_dir / f"{project_name}_3d.ply"
            gaussian_splats.save_ply(str(ply_output_path))
            print(f"    ✅ 生成成功: {ply_output_path.name}")
        else:
            print("    ❌ 错误: 输出中没有 Gaussian Splats 数据")
            return
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return

    final_windows_path = windows_dir / f"{project_name}_3dgs.ply"
    if ply_output_path.exists():
        shutil.copy2(str(ply_output_path), str(final_windows_path))
        print(f"\n🎉 成功！模型已保存: {final_windows_path}")
    else:
        print("    ❌ 失败: 未生成 PLY 文件")

    print(f"\n📊 总耗时: {format_duration(time.time() - global_start_time)}")

if __name__ == "__main__":
    run_pipeline()