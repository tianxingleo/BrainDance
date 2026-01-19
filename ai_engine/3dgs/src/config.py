# src/config.py
# 功能：定义Pipeline配置类，管理项目的所有配置参数
# 实现：使用dataclass定义配置项，从环境变量加载默认值
# 逻辑：1. 定义配置项及其默认值 2. 从环境变量加载配置 3. 设置必要的环境变量
# 包含：PipelineConfig数据类、环境变量加载逻辑、目录路径属性
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os

# 确保在导入 config 时就加载环境变量
load_dotenv()

# 项目根目录 (用于计算相对路径)
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class PipelineConfig:
    # 1. 【基本属性】
    project_name: str = "default_project"
    video_path: Optional[Path] = None
    
    # 2. 【核心配置】
    work_root: Path = Path("./temp_workspace")
    
    # 🟢 [修复] 必须添加这一行，给一个默认训练步数 (兼容旧代码中的 .iterations)
    iterations: int = 30000
    
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
    shared_model_dir: Path = field(default_factory=lambda: BASE_DIR.parent.parent / "models")

    # 引擎核心参数
    force_spherical_culling: bool = False
    scene_radius_scale: float = 1.0
    keep_percentile: float = 0.8

    # [新增] SAM3D 相关配置
    sam3d_repo_path: Path = field(default_factory=lambda: BASE_DIR / "src/libs/sam-3d-objects")
    sam3d_checkpoint_dir: Path = field(default_factory=lambda: BASE_DIR.parent.parent / "models/sam3d/checkpoints")

    # [新增] SHARP 相关配置
    sharp_repo_path: Path = field(default_factory=lambda: BASE_DIR / "src/libs/ml-sharp")

    @property
    def project_dir(self) -> Path:
        return self.work_root

    @project_dir.setter
    def project_dir(self, value: Path):
        self.work_root = value

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
