from pathlib import Path
import os

# === 基础路径配置 ===
# Linux 下的高速工作区 (训练时的临时文件放这里，速度快 10 倍)
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"

# === 环境变量设置 ===
ENV = os.environ.copy()
# 防止无头模式下 Qt 报错
ENV["QT_QPA_PLATFORM"] = "offscreen" 

# === 默认参数 ===
FFMPEG_FPS = 4           # 抽帧频率
FFMPEG_WIDTH = 1920      # 抽帧宽度 (1080P)
TRAIN_ITERATIONS = 15000 # 训练迭代次数
