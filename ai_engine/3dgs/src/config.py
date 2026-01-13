# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class PipelineConfig:
    # 1. 【必填项】用户初始化时必须给我的
    project_name: str
    video_path: Path
    
    # 2. 【选填项】有默认值的配置 (对应你原代码的全局变量)
    work_root: Path = Path.home() / "braindance_workspace"
    max_images: int = 180
    force_spherical_culling: bool = True 
    scene_radius_scale: float = 1.8
    keep_percentile: float = 0.9
    enable_ai: bool = True  # 新增控制开关
    
    # 3. 【自动计算项】用户不用传，我自己算出来的路径
    # field(init=False) 的意思是：这个变量存在，但在初始化(__init__)时不需要作为参数传入
    project_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    images_dir: Path = field(init=False)
    masks_dir: Path = field(init=False)
    transforms_file: Path = field(init=False)
    vocab_tree_path: Path = field(init=False)

    def __post_init__(self):
        """
        这个函数会在类初始化完成之后，自动执行！
        我们在这里集中处理所有的路径拼接和环境设置。
        """
        # --- A. 自动计算路径 (再也不用在主函数里写一遍了) ---
        self.project_dir = self.work_root / self.project_name
        self.data_dir = self.project_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.transforms_file = self.data_dir / "transforms.json"
        
        # 词汇树路径 (对应原代码 VOCAB_TREE_PATH)
        self.vocab_tree_path = self.work_root / "vocab_tree_flickr100k_words.bin"

        # --- B. 环境修正 (对应原代码的 PATH 设置逻辑) ---
        # 把设置环境变量的逻辑搬到这里，保证 config 一加载，环境就是对的
        # sys_path = "/usr/local/bin"
        # current_path = os.environ.get("PATH", "")
        # if sys_path not in current_path.split(os.pathsep)[0]:
        #     print(f"⚡ [Config] 自动优化 PATH 优先级: {sys_path}")
        #     os.environ["PATH"] = f"{sys_path}{os.pathsep}{current_path}"
            
        # 设置 Setuptools 修复 (对应原代码 env["SETUPTOOLS_USE_DISTUTILS"])
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
