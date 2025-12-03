
这是一个非常痛并快乐着的配置过程。回顾整个历程，你实际上完成了一次**从“Windows 舒适区”到“Linux 工业级开发”的硬核蜕变**，同时还在**首发硬件（RTX 5070）**上踩平了软件生态滞后的坑。

以下是你在这个项目中踩过的所有坑的详细复盘，这份清单对于任何使用新硬件做 AI 开发的人来说都是宝贵的经验。

------



### 第一阶段：Windows 本地开发（处处碰壁）

试图在 Windows 上直接运行 Nerfstudio，这是最直观但坑最多的路径。

1. **文件路径与执行位置不一致**
   - **坑**：运行 python process_3dgs.py 提示找不到 test.mp4。
   - **原因**：在 VS Code 或终端根目录运行脚本时，cwd（当前工作目录）不是脚本所在目录。
   - **解法**：在代码中使用 Path(__file__).parent 锁定绝对路径，或者手动 cd 到脚本目录。
2. **Subprocess 调用命令失败 ([WinError 2])**
   - **坑**：Python 报错“系统找不到指定的文件”，但文件明明在。
   - **原因**：Windows 下 subprocess.run 默认不走系统 Shell，无法自动解析 PATH 中的命令（如 ns-process-data）。
   - **解法**：加上 shell=True 参数。
3. **Conda 环境“幽灵”**
   - **坑**：明明安装了 nerfstudio，子目录终端却无法识别命令。
   - **原因**：文件在硬盘里，但终端没激活 Conda 环境。
   - **解法**：必须显式执行 conda activate <env_name>，VS Code 可以配置自动激活。
4. **COLMAP 的 GPU 加速之谜（最深的一个坑）**
   - **坑 1**：默认 pip/conda 安装的 COLMAP 是 CPU 版，巨慢。
   - **坑 2**：手动下载了 GPU 版并配了 PATH，Python 依然调用 CPU 版。因为 colmap.bat 脚本抢占了 .exe 的优先级。
   - **坑 3**：删了 .bat，Python 依然调用 CPU 版。因为 Conda 环境内部（Library/bin）藏了一个 CPU 版 COLMAP，且优先级高于系统 PATH。
   - **解法**：物理覆盖，把 GPU 版 exe 强行复制到 Conda 的 Scripts 目录下，或者在 Python 代码里强行注入环境变量。
5. **MSVC 编译器与 PyTorch Nightly 的冲突（劝退点）**
   - **坑**：编译 gsplat 时报错 error C2872: “std”: ambiguous symbol。
   - **原因**：RTX 5070 需要 PyTorch Nightly，但 Nightly 版的 C++ 头文件在 Windows 上写得不规范，与微软 MSVC 编译器标准库冲突。这是 PyTorch 官方的 Bug，短期无解。
   - **结果**：被迫放弃 Windows 原生环境，转向 WSL2。

------



### 第二阶段：迁移至 WSL2（环境搭建）

进入 Linux 子系统，开始标准化的 AI 开发流程。

1. **Windows 与 Linux 傻傻分不清**
   - **坑**：在 PowerShell (PS C:\...) 里输入 Linux 命令（ls, wget）报错。
   - **原因**：没进入 WSL 容器。
   - **解法**：输入 wsl 进入 Linux 终端（提示符变为 user@host:~$）。
2. **WSL 空壳问题**
   - **坑**：输入 wsl 提示没有安装分发版。
   - **解法**：执行 wsl --install 安装 Ubuntu 并重启。
3. **Linux 基础编译库缺失**
   - **坑**：安装 nerfstudio 报错 pkg-config is required。
   - **原因**：纯净版 Ubuntu 缺胳膊少腿。
   - **解法**：sudo apt install pkg-config libavformat-dev ... 补全编译工具链。
4. **Python 版本陷阱**
   - **坑**：安装 Open3D 报错 No matching distribution found。
   - **原因**：Conda base 环境默认为 Python 3.13，太新了，AI 社区还没适配。
   - **解法**：创建 Python 3.10 的环境 (conda create -n gs_linux python=3.10)。
5. **Anaconda 服务条款拦截**
   - **坑**：CondaToSNonInteractiveError，无法创建环境。
   - **原因**：Anaconda 更新了协议。
   - **解法**：运行 conda tos accept ... 命令。

------



### 第三阶段：依赖安装与网络问题（国情特色）

1. **GitHub 连接超时**
   - **坑**：pip install git+... 报 Connection reset by peer 或 SSL 错误。
   - **原因**：国内网络访问 GitHub 不稳定。
   - **解法**：
     - **必杀技**：在 Windows 挂梯子下载好代码，直接 cp 到 WSL 里。
     - **备选**：使用 mirror.ghproxy.com 代理，或设置 GIT_SSL_NO_VERIFY=true。
2. **PyTorch 自动升级背刺**
   - **坑**：明明指定安装 cu126，结果 pip 自动装了 cu130，导致报错 cudaErrorNotSupported（因为显卡驱动还没跟上 CUDA 13）。
   - **解法**：卸载后强制指定 index-url 安装，并检查 torch.version.cuda。
3. **gsplat 编译时的“构建隔离”**
   - **坑**：安装 gsplat 报错 ModuleNotFoundError: No module named 'torch'。
   - **原因**：pip 为了干净，在编译时创建了一个不带 PyTorch 的临时虚拟环境。
   - **解法**：加上 --no-build-isolation 参数，强迫它使用当前环境的 PyTorch。
4. **WSL 中缺少 CUDA 编译器**
   - **坑**：gsplat 编译失败，提示 No CUDA toolkit found。
   - **原因**：PyTorch 自带 CUDA 运行库，但不带编译器 (nvcc)。
   - **解法**：conda install cuda-toolkit。

------



### 第四阶段：运行时与性能优化（最后的磨合）

1. **跨文件系统 I/O 性能瓶颈**
   - **坑**：在 WSL 里直接读取 /mnt/c/... 下的数据，速度极慢。
   - **原因**：WSL 访问 Windows NTFS 文件系统有巨大开销。
   - **解法**：把代码和数据搬到 Linux 原生目录 (~/...) 下运行，或者在脚本里写自动搬运逻辑。
2. **Headless 模式下的 Qt 崩溃**
   - **坑**：运行 COLMAP 时报错 Aborted (core dumped)，涉及 Qt/OpenGL。
   - **原因**：WSL 是无头模式（没有显示器），但 COLMAP 试图启动 GUI 相关的渲染。
   - **解法**：设置环境变量 QT_QPA_PLATFORM="offscreen"。
3. **Nerfstudio 参数版本不兼容**
   - **坑**：ns-process-data 报错 Unrecognized options: --downscale-factor 或 --camera-res-scale-factor。
   - **原因**：Nerfstudio 更新频繁，参数名改来改去，且不同子命令参数不通用。
   - **解法**：放弃 Nerfstudio 的内置缩放，改用 FFmpeg 手动预处理视频，然后仅让 Nerfstudio 做位姿解算。

------



### 🏆 最终成就

经过这一系列“九九八十一难”，你现在的环境是：

- **OS**: WSL2 (Ubuntu)
- **Hardware**: RTX 5070 (Blackwell)
- **Drivers**: CUDA 12.6 + PyTorch Nightly
- **Workflow**: Windows 写代码 -> 自动同步/搬运 -> Linux 高速训练 -> 结果回传。

这是目前针对最新硬件最先进、最科学的开发架构。恭喜通关！