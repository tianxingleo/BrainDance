# 【首发】针对 Windows 环境下 NVIDIA RTX 50 系列图形处理单元的 Nerfstudio (3DGS) 部署流程

> **适用硬件规格**：NVIDIA GeForce RTX 5070 / 5080 / 5090 **操作系统要求**：Windows 10 / 11 **核心架构方案**：WSL2 (Ubuntu) + PyTorch Nightly + 本地编译加速 (Local Compilation Acceleration)

鉴于搭载最新 Blackwell 架构（即 RTX 50 系列）的图形处理单元在 Windows 原生环境下的配置存在显著的复杂性及较高的编译失败率，本方案确立了利用 **Windows Subsystem for Linux (WSL2)** 的标准化部署流程。此架构旨在实现图形处理单元性能的无损调用，并利用 Linux 编译生态系统的稳定性。本文档详述了构建全自动、生产级 3D Gaussian Splatting (3DGS) 训练管线的规范程序。

## 🛠️ 第一阶段：Windows Subsystem for Linux (WSL2) 子系统的启用与配置

WSL2 作为集成于 Windows 的 Linux 内核环境，提供了直接访问底层图形硬件的能力。

1. **WSL2 子系统的安装**： 须在 Windows **PowerShell (以管理员权限运行)** 中执行以下指令：

   ```
   wsl --install
   ```

   *系统安装程序执行完毕后，**必须重启计算机**以完成配置。重启后，Ubuntu 终端将自动启动，须按照提示完成用户凭证（用户名及密码）的设定。*

2. **数据迁移策略（建议执行）**： 尽管 WSL 具备直接访问宿主机 C 盘 (`/mnt/c`) 的能力，但为优化 I/O 读写速率（预计可提升训练效率一个数量级），建议将**源代码及执行脚本**部署于 Linux 内部文件系统（即 `~` 目录），仅将**最终产物**同步回传至 Windows 文件系统。

## 📦 第二阶段：Linux 基础运行环境的配置

在 **Ubuntu 终端**环境中，须依次执行以下指令以配置基础依赖。

### 1. 系统级依赖库的安装

Nerfstudio 的数据预处理模块依赖于 FFmpeg 及 COLMAP，因此必须部署支持 CUDA 的底层库文件。

```
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    colmap \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx
```

### 2. Miniconda 环境管理系统的部署

此步骤旨在建立 Python 运行环境的管理机制。

```
mkdir -p ~/miniconda3
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

## ⚡ 第三阶段：构建适配 RTX 50 系列架构的 AI 引擎

此阶段为部署流程的核心。鉴于 RTX 50 系列硬件需要 **CUDA 12.6+** 的支持，必须部署 PyTorch 预览版（Nightly Build）并执行渲染器的**源码编译**。

### 1. 隔离环境的创建

```
conda create -n gs_linux python=3.10 -y
conda activate gs_linux
```

### 2. PyTorch  的安装 (适配 Blackwell 架构)

### （这里需要注意！！！12.6版本在2025.12月会存在兼容性问题例如识别不到gpu型号，13.0版本能识别但是后续版本也跑不通，只能使用12.8版本，不要按照下面的命令直接复制黏贴，应该直接在PyTorch官网找到12.8版本Linux x64的安装命令安装）

```
pip install --upgrade pip
# 强制指定 CUDA 12.6 版本以确保兼容性
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu126](https://download.pytorch.org/whl/nightly/cu126)
```

### 3. gsplat 的编译 (3DGS 渲染核心组件)

由于硬件架构的更新，预编译包无法直接运行，必须执行现场编译以生成适配的指令集。

```
# --no-binary 参数强制执行本地编译，以适配当前的显卡架构
pip install gsplat --no-binary=gsplat --no-cache-dir --no-build-isolation
```

### 4. Nerfstudio 框架的安装

```
# 从 GitHub 获取最新开发版本，以确保最佳兼容性
pip install git+[https://github.com/nerfstudio-project/nerfstudio.git](https://github.com/nerfstudio-project/nerfstudio.git)
```

### 5. 💉 安全补丁注入 (修正 PyTorch 2.6+ 兼容性问题)

PyTorch 新版本增强了安全验证机制，可能导致模型导出功能异常。须运行以下指令自动修改源代码，以解除相关限制：

```
# 自动定位并修改 eval_utils.py 文件
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
sed -i 's/map_location="cpu")/map_location="cpu", weights_only=False)/g' $SITE_PACKAGES/nerfstudio/utils/eval_utils.py
echo "✅ 补丁已应用：允许加载自定义模型权重"
```

## 🤖 第四阶段：自动化流水线脚本的部署

为规避手动操作可能引发的参数错误及路径混淆，建议部署 Python 脚本以接管以下全流程：**视频输入 -> 帧提取 -> COLMAP 位姿解算 -> 模型训练 -> 模型导出 -> 结果回传 Windows**。

在 WSL 终端中执行：

```
mkdir -p ~/braindance
cd ~/braindance
code process_3dgs.py  # 若未安装 VS Code，可使用 nano process_3dgs.py
```

须将以下代码写入并保存。该脚本针对 **RTX 5070** 进行了特定优化：

1. **Offscreen 模式**：解决 WSL 无显示设备导致的 COLMAP 进程崩溃问题。
2. **自动数据迁移**：自动将视频源文件从 Windows 存储介质复制至 Linux 高速存储区域进行训练。
3. **参数优化**：采用 `3.0 FPS` 采样率配合 `2K` 分辨率，以在处理速度与模型质量之间取得最佳平衡。

```
import subprocess
import sys
import shutil
import os
import time
from pathlib import Path

# ================= 配置区域 =================
# Linux 下的临时高速工作区路径定义
LINUX_WORK_ROOT = Path.home() / "braindance_workspace"

def run_pipeline(video_path, project_name):
    print(f"\n🚀 [BrainDance Engine] 启动 RTX 5070 加速任务: {project_name}")
    
    # 1. 路径解析与环境准备
    video_src = Path(video_path).resolve()
    work_dir = LINUX_WORK_ROOT / project_name
    data_dir = work_dir / "data"
    output_dir = work_dir / "outputs"
    
    # 环境变量配置：强制 Qt 使用离屏模式，以防止 GUI 初始化失败
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    # ================= 智能断点续传逻辑 =================
    transforms_file = data_dir / "transforms.json"
    if transforms_file.exists():
        print(f"\n⏩ 检测到现有数据，跳过预处理阶段...")
    else:
        print(f"🆕 初始化高速工作区...")
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        
        # 将视频复制到 Linux 本地 (旨在解决 Windows 跨系统 I/O 瓶颈)
        video_dst = work_dir / video_src.name
        shutil.copy(str(video_src), str(video_dst))

        # ================= [Step 1] 数据预处理 (鲁棒模式) =================
        print(f"\n🎥 [1/3] 视频抽帧与位姿解算")
        
        # 1.1 手动执行 FFmpeg: 2K 分辨率 + 3.0 FPS (平衡精度与速度)
        extracted_images_dir = data_dir / "images"
        extracted_images_dir.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(video_dst), 
            "-vf", "scale=2560:-1,fps=3.0", 
            "-q:v", "2", 
            str(extracted_images_dir / "frame_%05d.jpg")
        ]
        subprocess.run(ffmpeg_cmd, check=True) 
        
        # 1.2 调用 Nerfstudio 包装器以执行 COLMAP
        cmd_colmap = [
            "ns-process-data", "images",
            "--data", str(extracted_images_dir),
            "--output-dir", str(data_dir),
            "--verbose",
        ]
        
        # 捕获输出流以进行质量监控
        result = subprocess.run(cmd_colmap, check=True, env=env, capture_output=True, text=True)
        print(result.stdout)
        
        if "COLMAP only found poses" in result.stdout:
            raise RuntimeError("🚨 COLMAP 数据质量过低，建议重新采集视频素材（增加纹理/减少反光）。")

    # ================= [Step 2] 模型训练 =================
    # 检查是否存在已完成的训练结果
    search_path = output_dir / project_name / "splatfacto"
    run_dirs = sorted(list(search_path.glob("*"))) if search_path.exists() else []

    if run_dirs:
        print(f"\n⏩ 检测到已完成的训练，跳过...")
    else:
        print(f"\n🧠 [2/3] 开始训练 (Splatfacto)")
        cmd_train = [
            "ns-train", "splatfacto",
            "--data", str(data_dir),
            "--output-dir", str(output_dir),
            "--experiment-name", project_name,
            "--pipeline.model.random-init", "False",       # 强制使用 COLMAP 点云进行初始化
            "--pipeline.model.cull-alpha-thresh", "0.005", # 优化模型体积参数
            "--max-num-iterations", "15000",               # 15k 迭代次数足以收敛
            "--vis", "viewer+tensorboard", 
            "--viewer.quit-on-train-completion", "True",   # 训练完成后自动退出进程
            "colmap",                                      # 指定数据解析器类型
        ]
        subprocess.run(cmd_train, check=True, env=env)

    # ================= [Step 3] 导出与回传 =================
    print(f"\n💾 [3/3] 导出结果")
    
    # 重新获取最新的配置文件
    run_dirs = sorted(list(search_path.glob("*")))
    latest_run = run_dirs[-1]
    config_path = latest_run / "config.yml"
    
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(work_dir)
    ]
    subprocess.run(cmd_export, check=True, env=env)
    
    # 回传结果至 Windows 目录
    print(f"\n📦 同步结果至 Windows...")
    temp_ply = work_dir / "splat.ply" # 新版本默认文件名
    if not temp_ply.exists(): temp_ply = work_dir / "point_cloud.ply"

    # 自动获取脚本所在的 Windows 目录路径
    target_dir = Path(__file__).parent / "results"
    target_dir.mkdir(exist_ok=True)
    final_ply = target_dir / f"{project_name}.ply"
    final_transforms = target_dir / "transforms.json"
    
    if temp_ply.exists():
        # 使用 cp 命令绕过 Python 权限检查机制
        subprocess.run(f"cp {str(temp_ply)} {str(final_ply)}", check=True, shell=True)
        # 同时回传姿态文件以供 WebGL 使用
        if (data_dir / "transforms.json").exists():
            subprocess.run(f"cp {str(data_dir / 'transforms.json')} {str(final_transforms)}", check=True, shell=True)
            
        print(f"✅ 成功！模型已保存至: {final_ply}")
        # 清理 Linux 缓存，释放存储空间
        shutil.rmtree(work_dir)
    else:
        print("❌ 导出失败，未找到 PLY 文件")

if __name__ == "__main__":
    # 自动定位当前目录下的 test.mp4 文件
    script_dir = Path(__file__).resolve().parent
    video_file = script_dir / "test.mp4" 
    
    if video_file.exists():
        run_pipeline(video_file, "scene_rtx50")
    else:
        print(f"❌ 请将视频 test.mp4 放入此脚本同级目录: {script_dir}")
```

## 🚀 第五阶段：执行程序

操作者仅需在 Windows 资源管理器中，将视频源文件重命名为 `test.mp4`，并置于 `process_3dgs.py` 同级目录下。

随后在 WSL 终端中执行以下指令：

```
cd /mnt/c/Users/您的用户名/Documents/您的项目路径/demo
python process_3dgs.py
```



# 🩸 RTX 5070 + Windows + Nerfstudio：一场从入门到放弃再到重生的踩坑实录

**硬件环境**：Intel i5-14600KF + 64GB RAM + **NVIDIA RTX 5070** **软件目标**：跑通 3D Gaussian Splatting (Splatfacto) 训练流：视频 -> 训练 -> 导出模型。

------

## 第一阶段：Windows 原生环境的“死胡同”

既然是 Windows 电脑，第一反应当然是直接在 PowerShell 里跑。

### 坑位 1：COLMAP 的“找不到指定文件”与 GPU 缺失

起初，我想直接用 `pip install nerfstudio` 然后运行 `ns-process-data`。 **报错：**

Plaintext

```
[WinError 2] 系统找不到指定的文件
```

**原因**：Windows 下的 Python `subprocess` 调用外部命令（如 `colmap`）时，如果没有把路径加到 System PATH，或者没有开启 `shell=True`，它根本找不到命令。 **尝试修复**：

1. 手动下载 COLMAP Windows 二进制包。
2. 配置环境变量。
3. **新报错**：`Failed to parse options - unrecognised option '--SiftExtraction.use_gpu'`。 **原因**：Windows 上很多预编译的 COLMAP 版本（包括 pip 安装的）默认是不带 CUDA 支持的，或者参数版本过旧。 **结论**：在 Windows 上配置一个能被 Python 正确调用的、带 CUDA 的 COLMAP 极其痛苦。

### 坑位 2：RTX 5070 的架构兼容性 (SM_120)

搞定 COLMAP 后，尝试开始训练。 **报错：**

Plaintext

```
UserWarning: NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 ... sm_90.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因**：RTX 5070 太新了！目前主流的 PyTorch Stable (2.4/2.5) 对应的 CUDA 11.8 或 12.1/12.4 的二进制包里，根本没有编译针对 **Blackwell (SM_120)** 架构的内核。这就好比拿着 PS6 的光盘塞进 PS5 里。 **尝试修复**：必须安装 **PyTorch Nightly (预览版)**，配合 CUDA 12.6+。

### 坑位 3：Windows MSVC 与 PyTorch Nightly 的“世纪大战”

升级到 PyTorch Nightly 后，Nerfstudio 依赖的 `gsplat` 库需要重新编译（JIT Compile）。 **报错：**

Plaintext

```
error C2872: “std”: ambiguous symbol
...
C:/.../torch/include/torch/csrc/dynamo/compiled_autograd.h(1134): error C2872: “std”: 不明确的符号
```

**原因**：这是 PyTorch Nightly 在 Windows 上的一个严重 Bug。它的 C++ 头文件代码写得不严谨，与 Windows 的 Visual Studio 编译器 (MSVC) 的标准库发生了命名空间冲突。 **尝试修复**：

1. 试图手动修改 PyTorch 源码，把 `std` 改成 `::std`。改了一个文件，后面还有十个文件报错。
2. 试图升级/降级 Visual Studio 版本，无效。
3. 试图强制升级 `gsplat` 到最新版，依然编译失败。

**阶段一结论**：在 PyTorch 官方修复 Windows 编译问题之前，**RTX 50 系显卡无法在 Windows 原生环境下运行需要编译 CUDA 扩展的库（如 gsplat）**。此路不通。

------

## 第二阶段：转战 WSL2 (Windows Subsystem for Linux)

既然 Windows 编译器不行，那就用 Linux 的 GCC。于是我开启了 WSL2。

### 坑位 4：COLMAP 在无头模式下的“自杀”

在 WSL2 里配置好环境，运行 `ns-process-data`。 **报错：**

Plaintext

```
qt.qpa.xcb: could not connect to display
...
Aborted (core dumped)
ERROR:root:Feature extraction failed with code 34304. Exiting.
```

**原因**：WSL2 默认是**无头模式 (Headless)**，没有连接物理显示器。但 COLMAP（即便是命令行版）在初始化特征提取时，居然试图通过 Qt 库去初始化图形界面或 OpenGL 上下文，找不到显示器就直接崩了。 **尝试修复**：

1. 既然是 Qt 的锅，那就禁用 GUI。
2. **解决方案**：设置环境变量 `QT_QPA_PLATFORM=offscreen`。告诉它“我在后台跑，别找显示器”。

### 坑位 5：Opengl 上下文创建失败

解决了 Qt 报错后，COLMAP 依然报错： **报错：**

Plaintext

```
opengl_utils.cc:56] Check failed: context_.create()
```

**原因**：即使不显示界面，COLMAP 默认使用 SiftGPU 进行特征提取，这需要 OpenGL 上下文。WSL2 虽然支持 CUDA，但对 OpenGL 的 headless 支持有时候很玄学。 **解决方案**： 放弃 GPU 特征提取，强制使用 CPU。修改调用命令，加入 `--SiftExtraction.use_gpu 0` 和 `--SiftMatching.use_gpu 0`。虽然慢点，但稳。

------

## 第三阶段：Nerfstudio 版本的混乱之治

环境终于稳了，开始运行 Nerfstudio 的 Python 脚本。

### 坑位 6：CLI 参数的“捉迷藏”

我想修改图片分辨率防止显存溢出，于是加了 `--downscale-factor`。 **报错：**

Plaintext

```
Unrecognized options: --downscale-factor
```

我又试了 `--camera-res-scale-factor`。 **报错：**

Plaintext

```
Unrecognized options: --camera-res-scale-factor
```

**原因**：Nerfstudio 的开发版（Git Version）更新极快，命令行参数的位置和名字改来改去。有的参数从一级目录移到了 `pipeline.datamanager` 下，有的又移回了 `dataparser` 下。 **最终解决方案**： **不信任 Nerfstudio 的预处理命令**。直接用 Python 调用 `ffmpeg` 手动抽帧并缩放图片（比如缩放到 2K），然后只让 Nerfstudio 做它最擅长的解算，不让它处理图片缩放。

### 坑位 7：模型变成了“黑色海胆”

终于跑通了训练，但结果出来一看：模型是一团乱糟糟的黑色尖刺，只有依稀一点点轮廓。 **原因**：

1. **COLMAP 数据极差**：日志显示 `COLMAP only found poses for 1.09% of the images`。因为耳机是黑色的、反光的，且我为了追求画质用了 4K 分辨率 + 高帧率。特征点太多且全是噪点，导致匹配彻底失败。
2. **Nerfstudio 自动缩放**：Nerfstudio 默认会自动缩放场景并重新定中心，对于稀疏的点云，这直接把模型“拉爆”了。

**修复方案**：

1. **重拍视频**：增加纹理点（贴纸），降低反光。
2. **降低参数**：分辨率降到 2K，帧率降到 3.0 FPS（保证重叠率但减少无效帧）。
3. **禁用自动魔法**：在训练命令中加入 `--auto-scale-poses False` 和 `--center-method none`（虽然这又引发了新一轮的参数位置报错，最终只能通过清理数据强制重跑 Step 1 来解决）。

------

## 第四阶段：导出的“最后一公里”

训练完成了 15000 步，最后一步是导出 `.ply` 模型文件。

### 坑位 8：PyTorch 2.6+ 的安全拦截

导出时直接 Crash。 **报错：**

Plaintext

```
_pickle.UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default.
```

**原因**：为了适配 RTX 5070，我们安装了最新的 PyTorch Nightly (2.6+)。新版 PyTorch 默认开启了 `weights_only=True` 的安全检查，禁止加载包含旧版 NumPy 格式的模型权重文件。Nerfstudio 还没适配这个改动。 **尝试修复**：

1. 设置环境变量 `TORCH_ALLOW_UNSAFE_SERIALIZATION=1`？**无效**，似乎没传进去。
2. **最终解决方案**：**外科手术式修改**。直接找到 Conda 环境里的 `nerfstudio/utils/eval_utils.py` 源码，把 `torch.load(..., map_location="cpu")` 强行改成 `torch.load(..., map_location="cpu", weights_only=False)`。

### 坑位 9：CLI 导出模式不存在

尝试用 `ns-export zip-nerf` 导出。 **报错**：`invalid choice: 'zip-nerf'`。 **原因**：版本差异。 **解决**：回归最原始的 `ns-export gaussian-splat`。

### 坑位 10：文件回传的权限拒绝

最后一步，想把 WSL2 里生成的 `model.ply` 复制回 Windows 的 `D:\results`。 **报错：**

Plaintext

```
PermissionError: [Errno 1] Operation not permitted
```

**原因**：Python 的 `shutil.copy` 试图在复制文件内容的同时复制 Linux 的文件权限（chmod），但 Windows 的 NTFS 文件系统不支持这些操作，导致报错。 **最终解决方案**： 放弃 Python 的 `shutil`，直接调用 Shell 命令 `cp`。Linux 的 `cp` 命令在处理跨文件系统复制时更“懂事”，只拷数据，不拷权限。

------

## 🏁 总结

要在 Windows 上用 RTX 5070 跑通 Nerfstudio，你必须：

1. **放弃 Windows 原生环境**，拥抱 **WSL2**。
2. **放弃 PyTorch 稳定版**，拥抱 **Nightly + CUDA 12.6**。
3. **放弃 `pip install gsplat`**，必须 **从源码编译**。
4. **放弃 Nerfstudio 的自动数据处理**，自己写 Python 脚本调用 FFmpeg。
5. **手动修改 Nerfstudio 源码**，以绕过 PyTorch 2.6 的安全检查。
6. **在无头模式下给 COLMAP 戴上“眼罩” (`offscreen`)**，并放弃 GPU 特征提取。