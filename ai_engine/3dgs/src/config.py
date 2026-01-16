# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import os

# 确保在导入 config 时就加载环境变量
load_dotenv()

@dataclass
class PipelineConfig:
    # 1. 【必填项】用户初始化时必须给我的
    project_name: str
    video_path: Path
    
    # 2. 【选填项】
    work_root: Path = Path("output")
    
    # 🟢 [修改] 默认值改为从 os.getenv 读取，如果没有则使用备用值
    max_images: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGES", 500)))
    
    # 🟢 [新增] 训练迭代步数
    training_iterations: int = field(default_factory=lambda: int(os.getenv("TRAINING_ITERATIONS", 15000)))

    enable_ai: bool = False
    
    # 🟢 [新增] 场景理解开关与 API Key
    enable_scene_analysis: bool = True 
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    
    # 🟢 [新增] 质检阈值
    min_quality_score: int = field(default_factory=lambda: int(os.getenv("MIN_QUALITY_SCORE", 40)))

    # 🟢 [新增] 接收共享模型路径
    shared_model_dir: Path = field(default_factory=lambda: Path("./models"))

    # 引擎核心参数
    force_spherical_culling: bool = False 
    scene_radius_scale: float = 1.0
    keep_percentile: float = 0.8

    @property
    def project_dir(self) -> Path:
        return self.work_root

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "masks"

    @property
    def transforms_file(self) -> Path:
        return self.data_dir / "transforms.json"

    @property
    def vocab_tree_path(self) -> Path:
        return self.work_root / "vocab_tree_flickr100k_words.bin"

    def __post_init__(self):
        """
        这个函数会在类初始化完成之后，自动执行！
        我们在这里集中处理环境设置。
        """
        # --- B. 环境修正 (对应原代码的 PATH 设置逻辑) ---
        # 把设置环境变量的逻辑搬到这里，保证 config 一加载，环境就是对的
        # sys_path = "/usr/local/bin"
        # current_path = os.environ.get("PATH", "")
        # if sys_path not in current_path.split(os.pathsep)[0]:
        #     print(f"⚡ [Config] 自动优化 PATH 优先级: {sys_path}")
        #     os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"
            
        # 设置 Setuptools 修复 (对应原代码 env["SETUPTOOLS_USE_DISTUTILS"])
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
