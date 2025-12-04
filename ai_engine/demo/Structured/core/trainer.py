from pathlib import Path
from utils.common import run_command
from config.settings import TRAIN_ITERATIONS

def run_training(project_name, data_dir, output_dir, collider_args):
    print(f"\n🧠 [2/3] 开始训练 ({project_name})")
    
    cmd_train = [
        "ns-train", "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(output_dir),
        "--experiment-name", project_name,
        
        # 强制 COLMAP 初始化
        "--pipeline.model.random-init", "False",
        "--pipeline.model.cull-alpha-thresh", "0.005",
        
        # 插入分析得出的裁剪参数
        *collider_args,
        
        "--max-num-iterations", str(TRAIN_ITERATIONS),
        "--vis", "viewer+tensorboard",
        
        # 训练完成后自动退出
        "--viewer.quit-on-train-completion", "True",
        
        "colmap", # Dataparser
    ]
    
    run_command(cmd_train)
