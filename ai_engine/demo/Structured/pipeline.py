import shutil
from pathlib import Path
from config.settings import LINUX_WORK_ROOT
from utils.common import setup_logging
from core import preprocessor, analyzer, trainer, exporter

def run(video_path, project_name):
    # 1. 初始化日志
    setup_logging()
    print(f"\n🚀 [BrainDance Engine] 启动任务: {project_name}")
    
    # 2. 路径准备
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    transforms_file = data_dir / "transforms.json"

    # ================= Step 1: 预处理 (支持断点) =================
    if transforms_file.exists():
        print(f"\n⏩ [断点续传] 检测到已存在的 COLMAP 数据: {transforms_file}")
    else:
        # 清理旧工作区并新建
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        
        preprocessor.process_video(video_src, work_dir, data_dir)

    # ================= Step 2: 训练 =================
    # 检查是否已有训练结果
    is_trained = False
    if (output_dir / project_name / "splatfacto").exists():
        # 简单检查是否有内容
        if list((output_dir / project_name / "splatfacto").glob("*")):
            is_trained = True
            
    if is_trained:
        print(f"\n⏩ [训练跳过] 检测到已完成的训练结果")
    else:
        # 智能分析场景参数
        collider_args, scene_type = analyzer.analyze_scene_type(transforms_file)
        # 开始训练
        trainer.run_training(project_name, data_dir, output_dir, collider_args)

    # ================= Step 3: 导出与同步 =================
    # 结果回传到 pipeline.py 所在的同级目录
    target_root = Path(__file__).parent 
    exporter.export_results(project_name, work_dir, output_dir, data_dir, target_root)
    
    # ================= 清理 =================
    # 训练完成且导出成功后，清理庞大的工作区 (可选)
    print(f"\n🧹 [清理] 正在移除临时工作区: {work_dir}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    print("✨ 全部流程完成！")
