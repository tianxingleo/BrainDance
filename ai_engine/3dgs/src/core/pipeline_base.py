import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class BasePipeline(ABC):
    """
    【流水线基类】
    
    这是所有具体业务流水线（如 Video3DGSPipeline, MultiImagePipeline）的父类。
    它定义了所有流水线必须遵守的接口规范。
    
    设计模式：策略模式 (Strategy Pattern) + 模板方法 (Template Method)
    """

    def __init__(self, context: Dict[str, Any]):
        """
        初始化流水线。
        
        :param context: 一个字典，包含当前任务的所有上下文环境信息。
                        由 Worker 在工厂创建流水线时注入。
                        通常包含:
                        - 'task_id': 数据库中的任务 ID
                        - 'scene_id': 场景唯一 ID (用于文件命名)
                        - 'work_root': 本地的工作目录路径 (用于存放临时文件)
                        - 'log_callback': (函数) 用于向数据库发送实时日志的回调函数
                        - 'supabase': (可选) Supabase 客户端实例，如果流水线需要直接查库
        """
        self.context = context
        
        # 提取常用的上下文，方便子类直接 self.xxx 调用
        self.task_id = context.get('task_id')
        self.scene_id = context.get('scene_id')
        self.work_dir = context.get('work_root', './temp')
        
        # 提取日志回调函数，如果没有提供，则默认使用 print 打印到控制台
        self._log_callback = context.get('log_callback', print)

    def log(self, message: str, level: str = "INFO"):
        """
        【通用工具方法】记录日志。
        
        子类在执行过程中，应该调用 self.log("正在训练...")，
        而不是 print()。这样日志才能通过 Worker 实时回传到 Supabase 数据库。
        
        :param message: 日志内容
        :param level: 日志等级 (INFO, WARN, ERROR)
        """
        # 1. 格式化消息，加上流水线名称前缀，方便调试
        formatted_msg = f"[{self.__class__.__name__}] {message}"
        
        # 2. 调用外部传入的回调函数 (Worker 传进来的)
        if self._log_callback:
            # 可以在这里扩展，比如把 level 也传回去
            self._log_callback(formatted_msg)

    @abstractmethod
    def run(self, input_path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        【核心抽象方法】执行流水线逻辑。
        
        ⚠️ 注意：这是一个抽象方法 (@abstractmethod)。
        这意味着：BasePipeline 本身不知道怎么运行，
        所有继承它的子类（如 Video3DGSPipeline）**必须强制重写**这个方法。
        如果不重写，程序启动时会直接报错。
        
        :param input_path: 输入文件的本地路径。
                           - 对于视频任务，这是 .mp4 文件的路径
                           - 对于多图任务，这可能是 .zip 压缩包的路径
                           
        :param params: 任务参数字典 (对应数据库 processing_tasks 表的 task_params 字段)。
                       例如: {"use_mask": True, "iterations": 7000, "quality": "high"}
                       子类需要根据这些参数决定具体的处理逻辑。
                       
        :return: 返回一个元组 (output_ply_path, metadata)
                 - output_ply_path: 生成的最终 .ply 模型文件的本地绝对路径
                 - metadata: 字典，包含需要存入数据库的元数据 (如 quality_score, tags, file_size)
        """
        pass

    def cleanup(self):
        """
        【可选钩子方法】清理资源。
        
        流水线执行结束后，Worker 会调用此方法。
        子类可以重写此方法来删除中间产生的临时文件 (colmap 数据、临时图片等)。
        默认实现为空。
        """
        pass