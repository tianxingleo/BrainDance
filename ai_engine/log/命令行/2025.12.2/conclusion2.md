这份文档总结了我们从 Windows 原生环境一路踩坑到 WSL2 完美运行的所有经验。这是一份**“工业级快速部署方案”**，专门适配 **RTX 5070 (Blackwell)** 等最新硬件。

只要在新设备上严格按照此流程操作，可以跳过所有编译报错、环境冲突和网络问题，在 **30 分钟内** 完成环境搭建。

---

# 🚀 BrainDance Engine 极速部署指南 (WSL2 + RTX 5070 版)

**适用场景**：Windows 10/11, NVIDIA 显卡 (推荐 RTX 40/50 系)
**核心策略**：Windows 编写代码 -> WSL2 (Linux) 运行计算

## 第一阶段：系统层准备 (5分钟)

### 1. 开启 WSL2
在 Windows **管理员 PowerShell** 中执行：
```powershell
wsl --install
```
*执行完后**立即重启电脑**。重启后会自动弹出 Ubuntu 窗口，按提示设置用户名 (纯英文) 和密码。*

### 2. 初始化 Linux 编译环境
在 **Ubuntu (WSL) 终端**中执行（补全基础工具链）：
```bash
# 更新源并安装基础库 (含 FFmpeg)
sudo apt-get update && sudo apt-get install -y build-essential git ffmpeg pkg-config
```

### 3. 安装 Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 接受服务条款 (防止报错)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```

---

## 第二阶段：Python 环境构建 (15分钟)

**注意**：这是避坑的关键，请严格遵守版本号和参数。

### 1. 创建“黄金版本”环境
**必须使用 Python 3.10** (避开 3.13 的兼容性地狱)。
```bash
conda create -n gs_linux python=3.10 -y
conda activate gs_linux
```

### 2. 安装 PyTorch Nightly (适配 RTX 5070)
因为显卡太新，需要预览版 PyTorch (cu126)。
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

### 3. 安装 CUDA 编译器 (用于编译 gsplat)
PyTorch 不带编译器，必须手动补装。
```bash
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit -y
```

### 4. 安装核心组件 gsplat (防坑指令)
使用 `--no-build-isolation` 强制使用当前环境编译，防止找不到 PyTorch。
```bash
pip install gsplat==1.5.3 --no-binary=gsplat --no-cache-dir --no-build-isolation
```
*(注意：此步会触发编译，屏幕滚动白字属正常，需等待约 5-10 分钟)*

### 5. 安装 Nerfstudio & Colmap
使用 `no-deps` 防止它降级我们刚装好的包。
```bash
# 1. 安装 Colmap (GPU版)
conda install -c conda-forge colmap -y

# 2. 安装 Nerfstudio (使用镜像加速下载)
GIT_SSL_NO_VERIFY=true pip install git+https://mirror.ghproxy.com/https://github.com/nerfstudio-project/nerfstudio.git --no-deps

# 3. 补齐 Nerfstudio 的其他依赖
pip install tyro wandb tensorboard rich visdom matplotlib mediapy msgpack-numpy
```

---

## 第三阶段：部署运行脚本 (5分钟)

### 1. 建立工作区 (避开 IO 性能陷阱)
**严禁**直接在 `/mnt/c/` 下运行训练！必须在 Linux 主目录下运行。
```bash
mkdir -p ~/braindance_workspace
cd ~/braindance_workspace
```

### 2. 部署终极版脚本
在当前目录下创建 `process_3dgs.py`，粘贴以下**Linux 专用优化版代码**：

```python
import subprocess
import sys
import shutil
import os
from pathlib import Path

# ================= 配置区域 =================
# 工作目录直接设为当前 Linux 目录，保证 IO 速度
WORK_ROOT = Path.home() / "braindance_workspace"

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine] 启动任务: {project_name}")
    
    # 1. 路径处理与搬运
    video_src = Path(video_path).resolve()
    project_dir = WORK_ROOT / project_name
    data_dir = project_dir / "data"
    images_dir = data_dir / "images"
    colmap_dir = data_dir / "colmap"
    output_dir = project_dir / "outputs"

    # 如果已存在，清理旧数据 (根据需求可改为断点续传)
    if project_dir.exists():
        shutil.rmtree(project_dir)
    
    images_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    print(f"📂 数据准备中...")
    
    # ================= [Step 1] FFmpeg 预处理 (抽帧+缩放) =================
    # 将视频统一处理为 1080p，降低帧率以加快 COLMAP 速度
    print(f"\n🎥 [1/3] 正在预处理视频 (FFmpeg)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_src),
        "-vf", "scale=1920:-1,fps=5",  # 1080p, 5fps
        "-q:v", "2",
        str(images_dir / "frame_%05d.jpg")
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # ================= [Step 2] COLMAP 位姿解算 =================
    # 注入 QT 离屏模式环境变量，防止 WSL2 无头模式崩溃
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    print(f"\n📐 [2/3] 正在解算相机位姿 (COLMAP)...")
    cmd_colmap = [
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(data_dir),
        "--verbose"
    ]
    subprocess.run(cmd_colmap, check=True, env=env)

    # ================= [Step 3] 模型训练 =================
    print(f"\n🧠 [3/3] 开始训练 (RTX 5070 加速中)...")
    cmd_train = [
        "ns-train", "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(output_dir),
        "--experiment-name", project_name,
        "--pipeline.model.cull_alpha_thresh", "0.005", # 瘦身模型
        "--max-num-iterations", "7000",                # 快速训练 (可改 15000)
        "--vis", "viewer+tensorboard",
    ]
    subprocess.run(cmd_train, check=True, env=env)

    # ================= [Step 4] 导出结果 =================
    print(f"\n💾 正在导出 PLY 模型...")
    # 自动寻找最新的 config
    search_path = output_dir / project_name / "splatfacto"
    latest_run = sorted(list(search_path.glob("*")))[-1]
    config_path = latest_run / "config.yml"
    
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(project_dir)
    ]
    subprocess.run(cmd_export, check=True, env=env)
    
    final_ply = project_dir / "model.ply"
    print(f"\n✅ 任务完成！模型路径: {final_ply}")
    return str(final_ply)

if __name__ == "__main__":
    # 示例：传入 Windows 路径下的视频
    # 在 WSL 中，Windows C盘是 /mnt/c，D盘是 /mnt/d
    video_input = "/mnt/c/Users/ltx/Documents/test.mp4"  # <-- 修改这里为实际路径
    
    if Path(video_input).exists():
        run_pipeline(video_input, "demo_scene")
    else:
        print(f"❌ 找不到输入视频: {video_input}")
```

---

## 第四阶段：如何运行

1.  **将视频放在 Windows 的任意位置** (例如 C 盘文档)。
2.  **修改脚本**：把 `process_3dgs.py` 底部的 `video_input` 路径改为你的视频路径。
3.  **一键启动**：
    ```bash
    conda activate gs_linux
    python process_3dgs.py
    ```

**享受极速训练吧！** 这一套方案避开了所有 Windows 编译坑、网络代理坑和 IO 性能坑，是目前最稳健的版本。