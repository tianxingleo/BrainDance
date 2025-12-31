# 关于首次在 Google Colab 平台基于 Tesla T4 加速器全流程部署三维高斯泼溅（3DGS）技术的实施协议（2025.12）

**关键词**：三维高斯泼溅 (3DGS)；Google Colab；计算架构部署；图形处理单元；三维重建



## 最快入门（2025.12，请注意时效性）：

访问：https://colab.research.google.com/drive/1iktXf6qhJXwB0io6YR-FtoGcGooXvakR?usp=sharing



本项目基于：https://github.com/yassgan/3DGaussianSplatting-INRIA-Method-Colab修改实现

## 绪论

本文档旨在阐述在 Google Colab 云端计算环境中部署并执行三维高斯泼溅（3D Gaussian Splatting，以下简称 3DGS）技术的标准化操作流程。鉴于部分研究人员缺乏高端 NVIDIA 图形处理单元（GPU）等本地硬件资源，Google Colab 提供的 Tesla T4 GPU 实例被确认为一种可行的替代方案。

然而，在云端环境部署该技术时，常面临环境依赖冲突、显存资源溢出及底层软件兼容性故障等技术障碍。经系统性验证，本协议确立了一套具有高稳定性的实施方案。该方案无需操作人员具备深层代码逻辑知识或故障排查能力，仅需严格遵循本文档之规定步骤执行指令，即可完成高精度三维模型的构建。

本实施方案之核心技术优势如下：

- **系统稳定性保障**：采用经官方验证的特定软件环境组合（Python 3.7 与 CUDA 11.8），以规避新版本环境可能引发的兼容性偏差。
- **资源溢出防御机制**：集成了优化的数据预处理脚本，通过自动执行视频分辨率的压缩及强制调用中央处理器（CPU）进行运算，确保在仅有 12GB 内存限制的免费版 Colab 实例中维持进程的稳定性。

## 前置准备条件

1. **Google 账户凭证**：需持有有效的 Google 账户以获取 Google Colab 及 Google Drive 的访问权限。
2. **源视频素材**：
   - **时长规范**：建议视频时长控制在 30 至 60 秒之间。
   - **采集规范**：需围绕目标物体或场景进行 360 度环绕拍摄，保持运镜平缓稳定，严禁剧烈晃动或产生运动模糊。
   - **光照条件**：必须确保拍摄环境光照充足且分布均匀。
3. **网络连接环境**：需具备访问 Google 服务的网络条件。

## 第一阶段：计算实例的创建与环境配置

鉴于 Colab 默认运行时环境版本过高，不符合 3DGS 直接运行之要求，须构建一个基于旧版本软件的稳定运行环境。

1. 访问 [Google Colab](https://colab.research.google.com/) 平台，并初始化一个新的笔记本实例。
2. 导航至菜单栏，选择 **“修改”** 选项，进而点击 **“笔记本设置”**。
3. 在“硬件加速器”配置项中，务必选定 **“T4 GPU”**，随后确认保存设置。

随后，须在代码执行单元中完整复制并执行下列指令集。该过程将自动执行 Python 版本的降级、CUDA 11.8 的部署及 PyTorch 环境的配置（预计耗时 5 至 8 分钟）。

```
# 1.1 执行 Python 环境降级至 3.7 版本及基础依赖安装
!wget -O mini.sh [https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh)
!chmod +x mini.sh
!bash ./mini.sh -b -f -p /usr/local
!conda install -q -y python=3.7
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

# 1.2 部署 CUDA 11.8 工具包
!wget [https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run)
!chmod +x cuda_11.8.0_520.61.05_linux.run
!./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --no-drm --no-man-page
import os
os.environ['PATH'] += ':/usr/local/cuda-11.8/bin'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.8/lib64:/usr/lib64-nvidia'

# 1.3 安装 PyTorch 框架及 3DGS 核心代码库
!pip uninstall torch torchvision torchaudio -y
!pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)

%cd /content
!rm -rf gaussian-splatting # 清除旧目录以防冲突
!git clone --recursive [https://github.com/camenduru/gaussian-splatting](https://github.com/camenduru/gaussian-splatting)
!pip install -q plyfile

# 编译核心组件
%cd /content/gaussian-splatting
!pip install -q /content/gaussian-splatting/submodules/diff-gaussian-rasterization
!pip install -q /content/gaussian-splatting/submodules/simple-knn

# 安装数据处理必备工具集
!apt-get update && apt-get install -y colmap ffmpeg
```

## 第二阶段：视频数据导入与自动化预处理

此阶段为关键环节。为防止 Colab 实例发生崩溃，严禁直接调用官方脚本，而应执行下列经优化的综合处理脚本。该脚本将自动执行视频抽帧、相机位姿解算（COLMAP）及图像去畸变等必要工序。

### 2.1 视频数据导入

1. 展开 Colab 界面左侧边栏的 **文件夹图标** 📁。
2. 定位至 `/content/gaussian-splatting` 目录。
3. 在此目录下新建名为 `my_data` 的子目录。
4. 将源视频文件重命名为 **`video.mp4`**，并将其上传至 `my_data` 目录中。

### 2.2 执行预处理脚本

新建代码执行单元，复制下列代码并执行。该脚本已被配置为自动降低 4K 视频分辨率，并强制使用 CPU 进行运算，虽处理速率略有降低（约需 5 至 10 分钟），但能确保系统的绝对稳定性。

```
import os

# 路径配置
PROJECT_PATH = "/content/gaussian-splatting/my_data"
VIDEO_PATH = os.path.join(PROJECT_PATH, "video.mp4")

%cd /content/gaussian-splatting

print("🎬 1. 启动视频抽帧程序 (已启用智能降采样)...")
!mkdir -p {PROJECT_PATH}/input
# 使用 ffmpeg 执行每秒 2 帧的抽样，并将宽度缩放至 1600px，以防内存溢出
if os.path.exists(VIDEO_PATH):
    !ffmpeg -i {VIDEO_PATH} -vf "fps=2,scale=1600:-1" -qscale:v 1 -qmin 1 -qmax 1 {PROJECT_PATH}/input/%04d.jpg -hide_banner -loglevel error
    print(f"✅ 抽帧程序执行完毕。共生成 {len(os.listdir(f'{PROJECT_PATH}/input'))} 张图像。")
else:
    raise FileNotFoundError("未检测到视频文件，请核实路径是否为 my_data/video.mp4")

print("⚙️ 2. 启动 COLMAP 运算流程 (CPU 稳健模式)...")
# 强制指定 CPU (--use_gpu 0) 运算，以规避 Colab 无头模式下可能发生的 OpenGL 崩溃
!colmap feature_extractor --database_path {PROJECT_PATH}/database.db --image_path {PROJECT_PATH}/input --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 0
!colmap exhaustive_matcher --database_path {PROJECT_PATH}/database.db --SiftMatching.use_gpu 0

print("Kv 3. 执行稀疏重建...")
!mkdir -p {PROJECT_PATH}/distorted/sparse
!colmap mapper --database_path {PROJECT_PATH}/database.db --image_path {PROJECT_PATH}/input --output_path {PROJECT_PATH}/distorted/sparse

print("📐 4. 执行图像去畸变处理 (训练数据准备)...")
!colmap image_undistorter --image_path {PROJECT_PATH}/input --input_path {PROJECT_PATH}/distorted/sparse/0 --output_path {PROJECT_PATH} --output_type COLMAP --max_image_size 1600

print("\n🎉🎉🎉 数据预处理流程圆满结束，准备启动训练程序。")
```

## 第三阶段：模型训练程序的执行

在完成数据准备工作后，仅需执行单行指令即可启动模型训练程序。

新建代码执行单元并运行下列指令：

```
# 启动训练程序
# -s 指定数据源路径，-m 指定模型输出路径
!python train.py -s my_data -m output/my_model
```

**操作提示**：

- 控制台将显示进度条信息。
- 默认训练迭代次数设定为 30,000 次，预计耗时 30 至 40 分钟。
- 若迭代至 7,000 次左右，模型效果已初步显现。如时间紧迫，可随时终止程序，系统将自动保存当前进度的模型文件。

## 第四阶段：成果数据的导出与验证

训练程序终止后，三维模型文件即已生成。

### 4.1 模型文件下载

1. 在左侧文件栏中，依次展开如下路径： `content/gaussian-splatting/output/my_model/point_cloud/iteration_30000/` *(注：若中途终止，文件可能位于 iteration_7000 目录中)*
2. 定位至目录内的 **`point_cloud.ply`** 文件。
3. 右键单击该文件，选择 **“下载”**。文件体积通常介于 100MB 至 500MB 之间。

### 4.2 在线可视化验证

下载的 `.ply` 文件不应使用常规图像浏览器打开。建议使用专业的在线查看器进行验证：

👉 [**SuperSplat (访问链接)**](https://playcanvas.com/supersplat/editor)

**操作规程**：

1. 访问上述网页地址。
2. 将下载的 `point_cloud.ply` 文件直接拖拽至网页视窗内。
3. 等待数据加载完成后，即可预览生成的三维场景。
4. 可通过鼠标左键旋转视角、右键平移视图、滚轮缩放比例，亦可录制演示视频。

## 操作风险提示与注意事项

- **会话活跃度维护**：Colab 免费版实例若处于非活动状态超过一定时限（约 90 分钟），连接将被强制断开。建议保持网页处于前台运行状态，或进行间歇性交互。
- **内存资源预警**：若源视频时长过长或清晰度过高，可能导致 Colab 内存溢出。如遇程序崩溃，务必在第二阶段代码中将 `scale=1600:-1` 修改为 `scale=1080:-1`，或缩减视频时长。
- **数据及时归档**：Colab 虚拟机环境具有临时性，网页关闭后所有数据将被清除。训练完成后，务必立即下载 `.ply` 模型文件进行本地归档。

愿诸位在三维重建领域取得理想成果。



## Google Colab 环境中 3D Gaussian Splatting 部署的技术性障碍分析

**关键词**：三维高斯泼溅 (3DGS)；Nerfstudio；Google Colab；Inria；COLMAP；CUDA；计算架构兼容性

## 摘要

本文旨在探讨在 Google Colab 云端计算环境中部署三维高斯泼溅（3D Gaussian Splatting，以下简称 3DGS）技术的可行性与实施路径。鉴于当前主流 3DGS 实现对 NVIDIA CUDA 生态系统的深度依赖，持有 AMD 图形处理单元（GPU）或其他非 CUDA 兼容硬件的研究人员面临显著的架构壁垒。本文系统性地分析了在 Colab 免费层级（基于 Tesla T4 加速卡）部署该技术时所遇到的环境配置、存储输入/输出（I/O）、内存资源分配及图形接口依赖等关键技术障碍，并提出了相应的解决方案。经实验验证，通过特定的环境回退策略与预处理流程优化，可在此受限环境中实现模型的有效训练。

## 技术障碍 I：硬件架构的不兼容性

**现象描述**：在本地 AMD 硬件环境中尝试部署时，常出现编译失败或训练吞吐量为零的异常情况。

**成因分析**：3DGS 的核心栅格化库（`diff-gaussian-rasterization`）及 Nerfstudio 的依赖库（`tiny-cudann`）均基于 NVIDIA CUDA C++ 编写，与 CUDA 架构存在底层深度绑定。尽管 ROCm 理论上提供兼容层，但在 Windows 操作系统下的配置极其繁复且性能损耗严重。

**结论**：鉴于硬件架构的根本性差异，采用基于 NVIDIA GPU 的云端计算服务（如 Colab 或 AutoDL）被确认为当前最为可行的替代方案。

## 技术障碍 II：网络挂载存储的数据库锁定限制

**场景描述**：将工作目录挂载至 `/content/drive/MyDrive/` 以实现数据持久化。

**现象描述**：在执行 COLMAP 数据预处理阶段，进程在特征提取环节停滞，或在无错误日志的情况下静默终止。

**成因分析**：COLMAP 在运行过程中需频繁对 SQLite 数据库文件（`database.db`）执行读写操作。Google Drive 作为网络挂载文件系统（Network-Attached Storage），并不完全支持 SQLite 所需的原子性文件锁定（Atomic File Locking）机制，导致数据库死锁及进程挂起。

**实施方案**：

1. **本地化作业**：必须将输入视频数据复制至虚拟机本地临时存储空间（`/content/workspace`）进行处理。
2. **异步持久化**：仅在本地计算任务全部完成后，通过脚本将最终产物复制回云端存储。
3. **I/O 性能考量**：本地临时存储的读写速率显著优于网络挂载存储，可大幅提升处理效率。

## 技术障碍 III：内存资源溢出 (OOM)

**场景描述**：尝试处理高分辨率（4K）视频素材。

**现象描述**：

1. **COLMAP 阶段**：特征提取过程中，系统内存占用率瞬间达到峰值，导致会话崩溃。
2. **训练阶段**：JIT（即时编译）过程中内存耗尽，进程被系统内核终止。

**成因分析**：

- Colab 免费实例通常仅分配 12GB 系统内存。
- 高分辨率图像解压后的内存占用极高，且 Nerfstudio 在首次运行时需编译 `gsplat` 库，此过程对 CPU 及内存资源消耗巨大。

**实施方案**：

1. **降采样处理**：严禁直接使用原始 4K 图像。必须通过 FFmpeg 将输入素材降采样至 1080p 或 1600px 宽度。

   ```
   ffmpeg -i input.mp4 -vf "fps=2,scale=1600:-1" ...
   ```

2. **数量控制**：将抽帧数量严格限制在 100 至 200 帧之间（`--num-frames-target 150`）。

3. **虚拟内存扩容**：在物理内存不足的情况下，通过 `dd` 命令创建交换文件（Swap File）以利用磁盘空间扩展虚拟内存。

   ```
   # 强制创建 8GB 交换文件
   !dd if=/dev/zero of=/swapfile bs=1G count=8 status=progress
   !chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
   ```

## 技术障碍 IV：无头服务器环境下的图形界面依赖

**现象描述**：数据处理过程中，程序因 `qt.qpa.xcb: could not connect to display` 或 `Aborted (core dumped)` 错误而终止。

**成因分析**：COLMAP 包含图形用户界面（GUI）组件。即便在命令行模式下运行，某些模块仍尝试初始化 Qt 窗口系统或检测显示设备。鉴于 Colab 为无头（Headless）Linux 服务器环境，缺乏显示服务器支持，从而导致初始化失败。

**实施方案**： 须在执行相关命令前注入环境变量，强制指定 Qt 平台插件为离屏模式：

```
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# 或在命令行中显式声明
!QT_QPA_PLATFORM=offscreen ns-process-data ...
```

## 技术障碍 V：OpenGL 上下文缺失导致的 GPU 加速失效

**场景描述**：在解决 GUI 初始化问题后，COLMAP 仍报错 `Check failed: context_.create()`。

**成因分析**：COLMAP 默认利用 GPU（SiftGPU）加速特征提取，此操作依赖于 OpenGL 上下文。Colab 的 Tesla T4 GPU 仅配置用于通用计算（Compute），并未配置图形渲染管线（X Server），因此无法创建有效的 OpenGL 上下文。

**实施方案**： **强制 CPU 运算**是当前环境下唯一可行的解决方案。必须手动构建 COLMAP 命令管线，并显式禁用 GPU 加速选项：

```
--SiftExtraction.use_gpu 0
--SiftMatching.use_gpu 0
```

*注：Nerfstudio 的自动化脚本较难注入此参数，建议采用手动编写的 COLMAP 处理脚本。*

## 技术障碍 VI：交互式进程阻塞

**场景描述**：

1. `ns-process-data` 在无错误提示的情况下中途停止。此系 Python 封装层掩盖了底层 C++ 进程的崩溃信息所致。
2. 训练进程长时间停滞于 `Using --data alias...` 阶段，GPU 负载为零。此系启动命令中包含 `--vis wandb` 参数，导致 Weights & Biases 工具在后台等待用户输入 API 密钥，而非交互式 Shell 无法显示输入提示，进而引发进程死锁。

**实施方案**：

- 调试阶段应直接调用 COLMAP 二进制可执行文件以获取完整错误日志。
- 训练命令中应移除 `wandb` 参数，并启用 `--viewer.quit-on-train-completion True` 选项。

## 技术障碍 VII：Inria 官方实现的目录结构规范性

**场景描述**：在使用 Inria 官方代码库时，训练脚本报错 `Could not recognize scene type!`。

**成因分析**：Inria 的 `train.py` 脚本对输入数据的目录结构具有严格的规范要求。其必须识别到以下两种特定结构之一方可运行：

1. **COLMAP 格式**：必须存在 `sparse/0/cameras.bin` 等文件路径。若文件直接位于 `sparse/` 目录下而缺失子目录 `0/`，则无法识别。
2. **Blender 格式**：必须存在 `transforms.json` 文件。

**实施方案**： 须在预处理后手动校验目录结构，确保相关二进制文件位于 `output/sparse/0/` 路径下。若自动化脚本将输出置于 `distorted/sparse`，则需手动执行文件迁移操作。

## 结论：优化部署协议综述

综上所述，为在 Google Colab 免费层级环境中成功部署 3DGS 模型训练，建议严格遵循以下标准化操作协议：

1. **环境配置**：采用 Inria 官方复刻版 Notebook 环境（降级至 Python 3.7 及 CUDA 11.8），或使用 Nerfstudio 框架配合手动数据处理流程。
2. **数据预处理**：
   - **弃用**官方一键式脚本（如 `ns-process-data` 或 `convert.py`），因其在无头模式下的稳定性不足。
   - **编写**自定义 Python 脚本，依次调用 `ffmpeg` 进行降采样，随后调用 `colmap`（配置 `--use_gpu 0` 及 `offscreen` 参数），最后手动整理文件目录结构。
3. **模型训练**：
   - 确保数据路径的绝对准确性。
   - 若使用 Nerfstudio 框架，应禁用 Viewer 及 WandB 功能，并启用低内存占用模式。
4. **结果导出**：
   - 下载生成的 `.ply` 文件，并利用本地工具（如 SuperSplat）进行可视化验证。

本报告详尽阐述了云端部署过程中可能遭遇的技术瓶颈及相应的解决策略，旨在为相关研究人员提供一套标准化的操作范式，以降低实验过程中的试错成本。