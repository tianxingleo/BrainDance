import torch
import contextlib

@contextlib.contextmanager
def force_cpu_load():
    """
    上下文管理器：强制所有 torch.load 加载到 CPU RAM，
    防止模型初始化时直接爆显存。
    """
    original_load = torch.load
    
    def cpu_load_hook(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cpu'
        return original_load(*args, **kwargs)
    
    print("    🛡️ 已激活显存拦截器：强制所有权重加载至 RAM...")
    try:
        torch.load = cpu_load_hook
        yield
    finally:
        torch.load = original_load
        print("    🛡️ 显存拦截器已解除")
