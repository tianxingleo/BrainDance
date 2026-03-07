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
import cv2
import dashscope
from http import HTTPStatus
import gc
import json

# ================= 🔧 配置区域 (请修改这里) =================
INPUT_IMAGE_NAME = "input.jpg" 
LINUX_WORK_ROOT = Path.home() / "sam3d_workspace"
SAM3D_REPO_PATH = Path.home() / "workspace/ai/sam-3d-objects"
CONFIG_PATH = SAM3D_REPO_PATH / "checkpoints/hf/pipeline.yaml"
CPU_CONFIG_PATH = CONFIG_PATH.parent / "cpu_pipeline.yaml"

# 🔥 [新增] 阿里云 DashScope API Key
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 🔥 [新增] SAM 2 模型路径
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt" 
SAM2_CONFIG = "sam2_hiera_l.yaml"

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

# ================= 🔥 Mock 系统 (保持不变) =================
def inject_mocks():
    print("⚠️ [System] Injecting Virtual Kaolin & PyTorch3D Modules...")
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
    print("✅ [System] Mock Environment Ready")

inject_mocks()

# ================= 🔥 新增：Qwen + SAM 2 智能分割类 =================

class VLMProcessor:
    """负责调用 Qwen-VL API 获取物体坐标"""
    def __init__(self):
        pass

    def get_main_object_box(self, image_path):
        """调用 Qwen3.5-Plus 识别图片主体"""
        print("🤖 [Qwen] Uploading image to API...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_path},
                    {"text": "Detect the main subject in this image. Return the bounding box in [ymin, xmin, ymax, xmax] format within the range [0, 1000]."}
                ]
            }
        ]

        try:
            response = dashscope.MultiModalConversation.call(
                model='qwen-plus', 
                messages=messages,
            )

            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                print(f"🤖 [Qwen] Thinking: {content}")
                return self._parse_coordinates(content, image_path)
            else:
                print(f"❌ [Qwen] API Error: {response.code} - {response.message}")
                return None
        except Exception as e:
            print(f"❌ [Qwen] Exception: {e}")
            return None

    def _parse_coordinates(self, content, image_path):
        import re
        nums = re.findall(r'\d+', content)
        if len(nums) >= 4:
            y1, x1, y2, x2 = map(int, nums[:4])
            with Image.open(image_path) as img:
                w, h = img.size
            x_min = int(x1 / 1000 * w)
            y_min = int(y1 / 1000 * h)
            x_max = int(x2 / 1000 * w)
            y_max = int(y2 / 1000 * h)
            # Padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            return np.array([x_min, y_min, x_max, y_max])
        return None

class SAM2Processor:
    """负责加载 SAM 2 并执行分割"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = None

    def load_model(self):
        if self.predictor is not None: return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print("🔄 [SAM 2] Loading model to GPU...")
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print("✅ [SAM 2] Model loaded.")
        except Exception as e:
            print(f"❌ [SAM 2] Load failed: {e}")

    def segment(self, image_path, box):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        masks, scores, _ = self.predictor.predict(
            point_coords=None, point_labels=None,
            box=box[None, :], multimask_output=False,
        )
        return masks[0].astype(np.uint8) * 255

    def unload_model(self):
        """显存清理：非常重要！"""
        if self.predictor:
            print("🧹 [SAM 2] Unloading model to free VRAM...")
            del self.predictor
            self.predictor = None
            torch.cuda.empty_cache()
            gc.collect()

# ================= 🔧 辅助函数 =================

def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def setup_environment():
    if not SAM3D_REPO_PATH.exists():
        print(f"❌ Error: Repo not found at {SAM3D_REPO_PATH}")
        sys.exit(1)
    sys.path.append(str(SAM3D_REPO_PATH))
    sys.path.append(str(SAM3D_REPO_PATH / "notebook"))
    os.chdir(SAM3D_REPO_PATH)

def prepare_cpu_config():
    """Config Hack to prevent OOM at init"""
    print(f"📝 [Config Hack] Creating CPU config: {CPU_CONFIG_PATH}")
    if not CONFIG_PATH.exists(): return False
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f: content = f.read()
        new_content = content.replace("device: cuda", "device: cpu").replace('device: "cuda"', 'device: "cpu"')
        with open(CPU_CONFIG_PATH, 'w', encoding='utf-8') as f: f.write(new_content)
        return True
    except: return False

# ================= 🚀 主流水线 (整合版) =================

def run_pipeline():
    global_start_time = time.time()
    
    windows_dir = Path(__file__).resolve().parent
    source_img_path = windows_dir / INPUT_IMAGE_NAME
    project_name = source_img_path.stem 
    work_dir = LINUX_WORK_ROOT / project_name
    
    print(f"\n🚀 [SAM 3D + Qwen + SAM2] Start: {source_img_path.name}")
    
    if not source_img_path.exists():
        print(f"❌ Error: Image not found {source_img_path}")
        return

    if work_dir.exists(): shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    target_img_path = work_dir / source_img_path.name
    shutil.copy2(str(source_img_path), str(target_img_path))

    # ================= 🌟 Phase 1: AI 智能抠图 (Qwen + SAM 2) =================
    print(f"\n🧠 [Phase 1] Auto-Segmentation (VLM + SAM 2)...")
    
    vlm = VLMProcessor()
    sam = SAM2Processor()
    clean_image_path = work_dir / "input_clean.png"
    
    # 1. 识别坐标 (API)
    box = vlm.get_main_object_box(str(target_img_path))
    
    if box is not None:
        print(f"📍 Box found: {box}")
        # 2. 分割 (Local GPU)
        sam.load_model()
        try:
            mask = sam.segment(str(target_img_path), box)
            
            # 3. 应用 Mask 并保存 PNG
            pil_image = Image.open(str(target_img_path)).convert("RGBA")
            image_np = np.array(pil_image)
            
            if mask.shape != image_np.shape[:2]:
                mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            image_np[:, :, 3] = mask
            clean_img = Image.fromarray(image_np)
            
            # 自动裁剪透明边缘
            bbox = clean_img.getbbox()
            if bbox: clean_img = clean_img.crop(bbox)
            
            clean_img.save(clean_image_path)
            print(f"    ✅ Clean image saved: {clean_image_path}")
            
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
            return
        finally:
            # 4. 关键：清理 SAM 2 显存！
            sam.unload_model()
    else:
        print("❌ Qwen failed to detect object. Aborting.")
        return

    # ================= 🌟 Phase 2: SAM 3D 生成 (Process 3DGS) =================
    print(f"\n🧠 [Phase 2] Loading SAM 3D Pipeline...")
    setup_environment()
    if not prepare_cpu_config(): return

    try:
        from inference import Inference
        
        # 显存拦截器
        original_torch_load = torch.load
        def cpu_load_hook(*args, **kwargs):
            if 'map_location' not in kwargs: kwargs['map_location'] = 'cpu'
            return original_torch_load(*args, **kwargs)
        
        torch.load = cpu_load_hook
        print("    🛡️ Memory Interceptor: Active.")
        
        try:
            inference = Inference(str(CPU_CONFIG_PATH))
            pipeline = inference._pipeline
            pipeline.device = torch.device('cuda')
            print("    ✅ SAM 3D Init Done (RAM only).")
        finally:
            torch.load = original_torch_load

        # 读取我们刚刚生成的“干净”图片
        pil_image = Image.open(str(clean_image_path)).convert("RGBA")
        
        # 降采样到 256px 防止 OOM
        target_size = 512
        orig_w, orig_h = pil_image.size
        if max(orig_w, orig_h) > target_size:
            scale = target_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"    📉 Resized input for 3D: {new_w} x {new_h}")
        
        image_rgba = np.array(pil_image)
        image = image_rgba[:, :, :3]
        mask = image_rgba[:, :, 3] # 直接使用刚才扣好的 Alpha 通道

        # --- Stage 1 ---
        print("\n🚚 [Stage 1] Loading Sparse Structure Models...")
        torch.cuda.empty_cache()
        pipeline.models["ss_generator"].to('cuda')
        pipeline.models["ss_decoder"].to('cuda')
        if "ss_encoder" in pipeline.models and pipeline.models["ss_encoder"]: pipeline.models["ss_encoder"].to('cuda')
        if "ss_condition_embedder" in pipeline.condition_embedders: pipeline.condition_embedders["ss_condition_embedder"].to('cuda')
            
        print("🚀 [Stage 1] Inference Start...")
        stage1_output = pipeline.run(image=image, mask=mask, stage1_only=True, seed=42)
        print("    ✅ Stage 1 Complete.")

        # --- Stage 2 ---
        print("\n🔄 [Swap] Offloading Stage 1 -> Loading Stage 2...")
        pipeline.models["ss_generator"].cpu()
        pipeline.models["ss_decoder"].cpu()
        if "ss_encoder" in pipeline.models and pipeline.models["ss_encoder"]: pipeline.models["ss_encoder"].cpu()
        if "ss_condition_embedder" in pipeline.condition_embedders: pipeline.condition_embedders["ss_condition_embedder"].cpu()
        torch.cuda.empty_cache()
        
        pipeline.models["slat_generator"].to('cuda')
        pipeline.models["slat_decoder_gs"].to('cuda')
        if "slat_condition_embedder" in pipeline.condition_embedders: pipeline.condition_embedders["slat_condition_embedder"].to('cuda')
        
        pipeline.decode_formats = ["gaussian"]
        original_sample_ss = pipeline.sample_sparse_structure
        pipeline.sample_sparse_structure = lambda *args, **kwargs: stage1_output
        
        print("🚀 [Stage 2] Inference Start...")
        try:
            output = pipeline.run(image=image, mask=mask, stage1_only=False, seed=42, with_mesh_postprocess=False, with_texture_baking=False)
        finally:
            pipeline.sample_sparse_structure = original_sample_ss

        # --- Save ---
        if "gs" in output:
            gaussian_splats = output["gs"]
            ply_output_path = work_dir / f"{project_name}_3d.ply"
            gaussian_splats.save_ply(str(ply_output_path))
            final_windows_path = windows_dir / f"{project_name}_3dgs.ply"
            shutil.copy2(str(ply_output_path), str(final_windows_path))
            print(f"\n🎉 Success! Saved to: {final_windows_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()

    print(f"\n📊 Total Time: {format_duration(time.time() - global_start_time)}")

if __name__ == "__main__":
    run_pipeline()