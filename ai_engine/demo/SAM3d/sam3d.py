import os
import sys
import shutil
import time
import datetime
from pathlib import Path
import logging
import torch
import torch.nn.functional as F
from types import ModuleType
import numpy as np

# ================= 🔥 [RTX 5070 兼容性补丁 V32] 修复 Mask 溢出 + 终极 Mock 🔥 =================
def inject_mocks():
    print("⚠️ [系统检测] 正在注入 Kaolin 和 PyTorch3D 的 V32 Mock 模块...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. 定义智能 Mock 类 (处理链式调用) ---
    class MockClass:
        def __init__(self, *args, **kwargs): 
            self.device = device
        
        # 场景 A: 直接调用 t(points) -> 返回 Tensor
        def __call__(self, *args, **kwargs): 
            return torch.zeros(1, 3, device=device, requires_grad=True)
        
        # 场景 B: 显式转换方法 -> 返回 self (支持链式调用)
        def compose(self, *args, **kwargs): return self
        def inverse(self): return self
        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def cuda(self): return self
        def clone(self): return self
        def detach(self): return self
        
        # 场景 C: 显式计算方法 -> 返回 Tensor 或原值
        def get_matrix(self): return torch.eye(4, device=device).unsqueeze(0)
        def transform_points(self, x): return x # 直接返回输入，保证数据流不断
        def transform_normals(self, x): return x
        
        # 场景 D: 未知属性/方法
        def __getattr__(self, name):
            def method_mock(*args, **kwargs):
                return self
            return method_mock

    def mock_func(*args, **kwargs): 
        return torch.tensor(0.0, device=device)
    
    def mock_check_func(*args, **kwargs):
        return False 

    # --- 2. 伪造 Kaolin ---
    if "kaolin" not in sys.modules:
        mock_kaolin = ModuleType("kaolin")
        submodules = [
            "ops", "ops.mesh", "ops.spc", 
            "metrics", "metrics.pointcloud", 
            "render", "render.camera", "render.mesh",
            "visualize", "io", "io.obj", "io.usd",
            "utils", "utils.testing"
        ]
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

    # --- 3. 伪造 PyTorch3D ---
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
        
        # 填充 vis 模块
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

        # [A] Transforms
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
        
        # [B] Renderer
        def look_at_view_transform(dist=1.0, elev=0.0, azim=0.0, **kwargs):
            R = torch.eye(3, device=device).unsqueeze(0)
            T = torch.zeros(1, 3, device=device)
            return R, T
        
        mock_p3d.renderer.look_at_view_transform = look_at_view_transform
        mock_p3d.renderer.look_at_rotation = lambda **kwargs: torch.eye(3, device=device).unsqueeze(0)
        mock_p3d.renderer.camera_position_from_spherical_angles = lambda **kwargs: torch.zeros(1, 3, device=device)
        mock_p3d.renderer.ray_bundle_to_ray_points = lambda **kwargs: torch.zeros(1, 3, device=device)
        mock_p3d.renderer.ray_points_to_depth = lambda **kwargs: torch.zeros(1, device=device)
        
        mock_p3d.renderer.camera_utils.camera_to_eye_at_up = lambda **kwargs: (torch.zeros(1, 3, device=device), torch.zeros(1, 3, device=device), torch.zeros(1, 3, device=device))
        mock_p3d.renderer.camera_utils.join_cameras_as_batch = mock_func

        renderer_classes = [
            "FoVPerspectiveCameras", "PerspectiveCameras", "CamerasBase", "OrthographicCameras",
            "PointsRenderer", "PointsRasterizationSettings", "PointsRasterizer", 
            "AlphaCompositor", "RasterizationSettings", 
            "MeshRenderer", "MeshRasterizer", "MeshRendererWithFragments",
            "SoftPhongShader", "HardPhongShader", "SoftSilhouetteShader", 
            "TexturesVertex", "TexturesAtlas", "TexturesUV",
            "PointLights", "DirectionalLights", "AmbientLights", "Materials", "BlendParams",
            "HeterogeneousRayBundle", "RayBundle", "ImplicitRenderer",
            "NDCGridRaysampler", "MonteCarloRaysampler"
        ]
        
        for cls_name in renderer_classes:
            mock_cls = MockClass
            setattr(mock_p3d.renderer, cls_name, mock_cls)
            if "Cameras" in cls_name:
                setattr(mock_p3d.renderer.cameras, cls_name, mock_cls)
            if "Textures" in cls_name:
                setattr(mock_p3d.renderer.mesh.textures, cls_name, mock_cls)
            if "Shader" in cls_name:
                setattr(mock_p3d.renderer.mesh.shader, cls_name, mock_cls)
            if "Rasterizer" in cls_name and "Mesh" in cls_name:
                setattr(mock_p3d.renderer.mesh.rasterizer, cls_name, mock_cls)
        
        # [D] Structures
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
        
    print("✅ [Mock V32] 修复 Mask 溢出 + 终极注入完成")

# 注入 Mocks
inject_mocks()
# =========================================================================

# ================= 🔧 配置区域 =================
INPUT_IMAGE_NAME = "input.jpg" 
LINUX_WORK_ROOT = Path.home() / "sam3d_workspace"
SAM3D_REPO_PATH = Path.home() / "workspace/ai/sam-3d-objects"
CONFIG_PATH = SAM3D_REPO_PATH / "checkpoints/hf/pipeline.yaml"

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

def run_pipeline():
    global_start_time = time.time()
    
    windows_dir = Path(__file__).resolve().parent
    source_img_path = windows_dir / INPUT_IMAGE_NAME
    project_name = source_img_path.stem 
    work_dir = LINUX_WORK_ROOT / project_name
    
    print(f"\n🚀 [RTX 5070 Pipeline] 启动任务: {INPUT_IMAGE_NAME}")
    print(f"📍 Windows: {windows_dir}")
    print(f"📍 WSL:     {work_dir}")
    
    if not source_img_path.exists():
        print(f"❌ 错误: 找不到图片 {source_img_path}")
        return

    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    target_img_path = work_dir / INPUT_IMAGE_NAME
    shutil.copy2(str(source_img_path), str(target_img_path))
    print(f"    ✅ 数据迁移完成")

    print(f"\n🧠 [2/3] 加载模型推理 (Dependencies Bypassed)...")
    setup_environment()
    
    try:
        from inference import Inference, load_image
        from PIL import Image
        
        inference = Inference(str(CONFIG_PATH))
        
        print(f"    -> 读取并处理图片: {target_img_path}")
        
        # [1] 处理图片
        pil_image = Image.open(str(target_img_path)).convert("RGB")
        orig_w, orig_h = pil_image.size

        # [2] 降采样 (低显存模式)
        target_size = 256
        if max(orig_w, orig_h) > target_size:
            scale = target_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"       📉 降采样至: {new_w} x {new_h}")
        
        # [3] 关键修复：保持 uint8 格式！不要除以 255.0！
        # 结果形状: (H, W, 3), 类型: uint8, 范围: [0, 255]
        image = np.array(pil_image)
        
        # [4] 处理 Mask：保持 uint8 格式
        # 结果形状: (H, W), 类型: uint8
        # 值设为 1，因为 inference 内部会执行 mask * 255。
        # 如果传 255，会变成 255*255 (溢出)。如果传 1，会变成 255 (完美)。
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)
        
        print(f"    -> 输入状态确认: Image {image.shape} ({image.dtype}), Mask {mask.shape} ({mask.dtype})")
        
        print("    -> 生成 3D Gaussian Splats...")
        # 此时 Image 是 (H, W, 3) uint8
        # 此时 Mask 是 (H, W) uint8
        # inference 内部会拼接它们 -> RGBA (H, W, 4) uint8
        output = inference(image, mask=mask, seed=42) 
        
        gaussian_splats = output["gs"]
        
        ply_output_path = work_dir / f"{project_name}_3d.ply"
        gaussian_splats.save_ply(str(ply_output_path))
        
        print(f"    ✅ 生成成功: {ply_output_path.name}")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return

    print(f"\n💾 [3/3] 结果回传")
    final_windows_path = windows_dir / f"{project_name}_3dgs.ply"
    
    if ply_output_path.exists():
        shutil.copy2(str(ply_output_path), str(final_windows_path))
        print(f"    🎉 成功！模型已保存: {final_windows_path}")
    else:
        print("    ❌ 失败: 未生成 PLY 文件")

    print(f"\n📊 总耗时: {format_duration(time.time() - global_start_time)}")

if __name__ == "__main__":
    run_pipeline()