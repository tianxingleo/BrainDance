import sys
import torch
from types import ModuleType

def inject_rtx50_mocks():
    """注入 Kaolin 和 PyTorch3D 的兼容性 Mock (针对 RTX 50 系列/WSL2)"""
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

    # --- setuptools / pkg_resources 兼容层 ---
    # 某些精简环境只装了 lightning 但没有携带 pkg_resources，
    # 会在 import lightning.pytorch 时直接失败。
    if "pkg_resources" not in sys.modules:
        mock_pkg_resources = ModuleType("pkg_resources")
        mock_pkg_resources.declare_namespace = lambda *args, **kwargs: None
        sys.modules["pkg_resources"] = mock_pkg_resources

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
    print("✅ [Mock System] RTX 50 系列兼容层已激活")
