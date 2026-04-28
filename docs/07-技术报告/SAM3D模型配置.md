# SAM3D 模型下载与设置教程

> **项目**: BrainDance - 面向空间计算时代的三维语义记忆引擎  
> **创建时间**: 2026-01-20  
> **最后更新**: 2026-01-20

---

## 目录

1. [概述](#1-概述)
2. [系统要求](#2-系统要求)
3. [模型申请流程](#3-模型申请流程)
4. [模型下载步骤](#4-模型下载步骤)
5. [配置 BrainDance 项目](#5-配置-braindance-项目)
6. [验证配置](#6-验证配置)
7. [故障排除](#7-故障排除)
8. [常见问题](#8-常见问题)
9. [相关链接](#9-相关链接)

---

## 1. 概述

### 1.1 什么是 SAM3D？

**SAM 3D Objects** 是 Meta AI 开发的基础模型，能够从单张图片重建完整的 3D 形状几何、纹理和布局。它在真实世界场景（包含遮挡和杂乱背景）中表现出色，通过渐进式训练和带人类反馈的数据引擎实现。

**核心能力**：
- ✅ 单张图片生成 3D 模型
- ✅ 支持几何、纹理、布局重建
- ✅ 适用于复杂真实场景
- ✅ 优于之前的 3D 生成模型

### 1.2 为什么需要单独配置？

SAM3D 模型（约 12.5GB）由于体积较大，不会随代码仓库一起分发。需要：

1. **单独申请访问权限**（Meta 审核）
2. **单独下载模型文件**
3. **配置环境变量**指向模型位置

---

## 2. 系统要求

### 2.1 硬件要求

| 资源 | 最低要求 | 推荐配置 |
|------|---------|----------|
| **显存** | 12GB | 16GB+ |
| **系统内存** | 48GB | 64GB+ |
| **磁盘空间** | 30GB | 50GB+ |
| **GPU** | RTX 3060 | RTX 4070+ |

**推荐 GPU**：
- NVIDIA RTX 3060 (12GB) - 最低可用
- NVIDIA RTX 4070/5070 (16GB) - 推荐
- NVIDIA RTX 4090 (24GB) - 最佳性能

### 2.2 软件要求

| 组件 | 要求 | 说明 |
|------|------|------|
| **操作系统** | Linux / WSL2 | Windows 原生可能有问题 |
| **Python** | 3.9 - 3.11 | 推荐 3.10 |
| **PyTorch** | 2.2+ | 需匹配 CUDA 版本 |
| **CUDA** | 11.8 / 12.1 | 推荐 12.1 |
| **Git LFS** | 最新版本 | 用于大文件管理 |

### 2.3 存储空间计算

```
模型文件 (12.5GB):
├── ss_generator.ckpt          6.9GB  # 主结构生成器
├── slat_generator.ckpt        4.9GB  # SLAT 结构生成器
├── slat_decoder_mesh.ckpt     364MB  # Mesh 解码器
├── slat_decoder_gs.ckpt       171MB  # Gaussian Splatting 解码器
├── ss_decoder.ckpt            148MB  # SS 解码器
└── 配置文件                    <10MB

推荐预留空间: 50GB+
（包含模型、运行时缓存、中间结果）
```

---

## 3. 模型申请流程

> ⚠️ **重要**: SAM3D 是 gated model，需要向 Meta 申请访问权限。

### 3.1 创建 HuggingFace 账号

1. 访问 https://huggingface.co
2. 点击 **Sign Up** 注册账号
3. 验证邮箱（会发送验证邮件）
4. 完成账号设置

### 3.2 申请模型访问权限

1. 访问模型页面: https://huggingface.co/facebook/sam-3d-objects
2. 点击 **"Agree and access repository"** 或 **"Apply for access"**
3. 填写申请表单（英文填写）：

```
Required Information:
├── First Name:          [你的名]
├── Last Name:           [你的姓]
├── Date of birth:       [出生日期]
├── Country:             [国家]
├── Affiliation:         [组织名称 - 完整名称，避免缩写]
│   └── 例如: "Beijing University" 而不是 "PKU"
│   └── 例如: "Meta Platforms, Inc." 而不是 "Meta"
└── Job title:           [职位]
    └── 例如: "Research Scientist", "Student", "AI Developer"
```

**⚠️ 填写注意事项**：
- 必须使用真实姓名（法定姓名）
- 组织名称必须完整，包含所有公司标识符
- 避免使用缩写和特殊字符
- 提交后**无法修改**，请仔细检查

### 3.3 等待审批

- **审批时间**: 通常 1-3 个工作日
- **通知方式**: HuggingFace 邮箱通知
- **查询状态**: 访问 https://huggingface.co/facebook/sam-3d-objects 查看状态

### 3.4 接受许可证协议

审批通过后：
1. 重新访问 https://huggingface.co/facebook/sam-3d-objects
2. 点击 **"Agree and access repository"**
3. 阅读并接受 SAM License
4. 即可下载模型文件

---

## 4. 模型下载步骤

### 4.1 安装必要工具

```bash
# 1. 安装 Git LFS（用于大文件）
git lfs install

# 2. 安装 HuggingFace Hub CLI
pip install "huggingface_hub[cli]"

# 3. 验证安装
huggingface-cli --version
```

### 4.2 登录 HuggingFace

**方法 1: CLI 登录（推荐）**
```bash
huggingface-cli login
# 会提示输入 Access Token
```

**方法 2: 获取 Access Token**
1. 访问 https://huggingface.co/settings/tokens
2. 点击 **"New token"**
3. 填写名称（建议: "SAM3D-Download"）
4. 选择 **"Read"** 权限
5. 点击 **"Generate"**
6. 复制 token 并保存

**方法 3: 设置环境变量**
```bash
export HF_TOKEN="your_access_token_here"
```

### 4.3 下载模型文件

**方法 1: 使用 HuggingFace CLI（推荐）**

```bash
# 创建工作目录
mkdir -p ~/workspace/ai
cd ~/workspace/ai

# 下载模型（约 12.5GB，需要几分钟）
TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects

# 整理文件结构
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download

# 验证下载
ls -lh checkpoints/${TAG}/checkpoints/
```

**方法 2: 使用 Python 代码**

```python
from huggingface_hub import hf_hub_download
import os

# 创建目录
os.makedirs("checkpoints/hf/checkpoints", exist_ok=True)

# 下载各个文件
files = [
    "checkpoints/hf/checkpoints/ss_generator.ckpt",
    "checkpoints/hf/checkpoints/slat_generator.ckpt",
    "checkpoints/hf/pipeline.yaml",
]

for filename in files:
    print(f"下载中: {filename}")
    hf_hub_download(
        repo_id="facebook/sam-3d-objects",
        filename=filename,
        local_dir=".",
        local_dir_use_symlinks=False
    )
```

### 4.4 下载额外配置文件

```bash
# 下载 ss_generator.yaml
mkdir -p checkpoints/hf
wget -O checkpoints/hf/ss_generator.yaml \
  https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/configs/ss_generator.yaml

# 下载其他必要配置
wget -O checkpoints/hf/slat_generator.yaml \
  https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/configs/slat_generator.yaml
```

### 4.5 验证下载完整性

```bash
# 检查文件大小
ls -lh checkpoints/hf/checkpoints/

# 预期输出：
# total 12G
# -rw-r--r-- 1 user user 6.9G ss_generator.ckpt
# -rw-r--r-- 1 user user 4.9G slat_generator.ckpt
# -rw-r--r-- 1 user user 364M slat_decoder_mesh.ckpt
# -rw-r--r-- 1 user user 171M slat_decoder_gs.ckpt
# -rw-r--r-- 1 user user 148M ss_decoder.ckpt

# 检查配置文件
ls -lh checkpoints/hf/
```

---

## 5. 配置 BrainDance 项目

### 5.1 设置环境变量

**方法 1: 临时设置（当前终端生效）**

```bash
# 替换为你的实际路径
export SAM3D_REPO_PATH=/home/yourname/workspace/ai/sam-3d-objects
export SAM3D_CHECKPOINT_DIR=/home/yourname/workspace/ai/sam-3d-objects/checkpoints/hf
```

**方法 2: 永久设置（推荐）**

编辑 `~/.bashrc` 或 `~/.zshrc`：

```bash
echo 'export SAM3D_REPO_PATH="/home/yourname/workspace/ai/sam-3d-objects"' >> ~/.bashrc
echo 'export SAM3D_CHECKPOINT_DIR="/home/yourname/workspace/ai/sam-3d-objects/checkpoints/hf"' >> ~/.bashrc

# 生效
source ~/.bashrc
```

**方法 3: 使用 .env 文件**

在项目根目录或 `ai_engine/3dgs/` 目录创建 `.env` 文件：

```bash
# ai_engine/3dgs/.env

SAM3D_REPO_PATH=/home/yourname/workspace/ai/sam-3d-objects
SAM3D_CHECKPOINT_DIR=/home/yourname/workspace/ai/sam-3d-objects/checkpoints/hf
```

### 5.2 项目配置示例

**完整配置示例**：

```bash
# ~/.bashrc 或项目 .env 文件

# SAM3D 模型配置
export SAM3D_REPO_PATH="/home/ltx/workspace/ai/sam-3d-objects"
export SAM3D_CHECKPOINT_DIR="/home/ltx/workspace/ai/sam-3d-objects/checkpoints/hf"

# 其他可能需要的配置
export PYTHONPATH="${PYTHONPATH}:/home/ltx/projects/BrainDance/ai_engine/3dgs/src"
```

---

## 6. 验证配置

### 6.1 检查环境变量

```bash
# 检查是否设置成功
echo "SAM3D_REPO_PATH: $SAM3D_REPO_PATH"
echo "SAM3D_CHECKPOINT_DIR: $SAM3D_CHECKPOINT_DIR"

# 检查目录是否存在
ls -ld $SAM3D_REPO_PATH
ls -ld $SAM3D_CHECKPOINT_DIR
```

### 6.2 检查模型文件

```bash
# 检查 checkpoint 文件
ls -lh $SAM3D_CHECKPOINT_DIR/checkpoints/

# 预期输出：
# total 12G
# -rw-r--r-- 1 user user 6.9G ss_generator.ckpt
# -rw-r--r-- 1 user user 4.9G slat_generator.ckpt
# ...
```

### 6.3 快速测试

```bash
# 进入项目目录
cd /home/ltx/projects/BrainDance/ai_engine/3dgs

# 测试配置加载
python -c "
import os
from src.config import PipelineConfig

config = PipelineConfig()
print(f'SAM3D_REPO_PATH: {config.sam3d_repo_path}')
print(f'SAM3D_CHECKPOINT_DIR: {config.sam3d_checkpoint_dir}')
print(f'✓ 配置加载成功！')
"
```

**预期输出**：
```
SAM3D_REPO_PATH: /home/ltx/workspace/ai/sam-3d-objects
SAM3D_CHECKPOINT_DIR: /home/ltx/workspace/ai/sam-3d-objects/checkpoints/hf
✓ 配置加载成功！
```

### 6.4 完整功能测试

```bash
# 测试 SAM3D Engine 初始化
python -c "
from pathlib import Path
import os

# 检查环境变量
repo_path = os.getenv('SAM3D_REPO_PATH')
checkpoint_dir = os.getenv('SAM3D_CHECKPOINT_DIR')

print(f'仓库路径: {repo_path}')
print(f'检查点路径: {checkpoint_dir}')

# 验证路径存在
assert Path(repo_path).exists(), f'仓库路径不存在: {repo_path}'
assert Path(checkpoint_dir).exists(), f'检查点路径不存在: {checkpoint_dir}'

# 验证必要文件
checkpoint_files = [
    'ss_generator.ckpt',
    'slat_generator.ckpt',
    'pipeline.yaml'
]

for file in checkpoint_files:
    file_path = Path(checkpoint_dir) / 'checkpoints' / file
    if file == 'pipeline.yaml':
        file_path = Path(checkpoint_dir) / file
    assert file_path.exists(), f'必要文件不存在: {file_path}'
    print(f'✓ {file}')

print('\\n🎉 所有检查通过！SAM3D 配置成功！')
"
```

---

## 7. 故障排除

### 7.1 CUDA 内存不足

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate 30.00 GiB
```

**解决方案**：
1. 确保系统内存 ≥ 48GB
2. 关闭其他占用 GPU 显存的程序
3. 增加 swap 分区：

```bash
# 检查当前 swap
free -h

# 创建 swap 文件（如果需要）
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 7.2 模型下载失败

**错误信息**：
```
RepositoryNotFoundError: Could not find model facebook/sam-3d-objects
```

**解决方案**：
1. 检查是否已获得访问权限（重新访问模型页面）
2. 确认 HuggingFace 已登录：
   ```bash
   huggingface-cli whoami
   ```
3. 检查网络连接
4. 尝试重新下载

### 7.3 路径配置错误

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/hf/pipeline.yaml'
```

**解决方案**：
1. 检查环境变量是否正确设置：
   ```bash
   echo $SAM3D_REPO_PATH
   echo $SAM3D_CHECKPOINT_DIR
   ```
2. 验证路径存在：
   ```bash
   ls -ld $SAM3D_REPO_PATH
   ls -ld $SAM3D_CHECKPOINT_DIR
   ```
3. 确保模型文件完整下载

### 7.4 CUDA 版本不匹配

**错误信息**：
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**解决方案**：
1. 检查 PyTorch 和 CUDA 版本：
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
   ```
2. 安装匹配的 PyTorch 版本：
   ```bash
   # CUDA 11.8
   pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu118

   # 或 CUDA 12.1
   pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### 7.5 权限错误

**错误信息**：
```
PermissionError: [Errno 13] Permission denied
```

**解决方案**：
1. 检查文件权限：
   ```bash
   ls -l checkpoints/hf/checkpoints/
   ```
2. 修复权限：
   ```bash
   chmod -R u+rwx checkpoints/
   ```

---

## 8. 常见问题

### Q1: 可以不使用环境变量吗？

**答**: 可以。如果不设置环境变量，系统会使用默认路径：
- `sam3d_repo_path`: `ai_engine/3dgs/src/libs/sam-3d-objects`
- `sam3d_checkpoint_dir`: `ai_engine/models/sam3d/checkpoints`

你可以将模型下载到这些默认位置，或创建符号链接。

### Q2: 模型文件可以放在其他位置吗？

**答**: 可以。只需设置 `SAM3D_REPO_PATH` 环境变量指向你的模型位置即可。

### Q3: 下载速度很慢怎么办？

**答**: 
1. 检查网络连接
2. 使用代理（如果需要）：
   ```bash
   export HTTP_PROXY="http://proxy:port"
   export HTTPS_PROXY="http://proxy:port"
   ```
3. 使用国内镜像源（如果有）

### Q4: 需要下载所有文件吗？

**答**: 是的，所有文件都是运行所必需的。缺失任何关键文件都会导致运行失败。

### Q5: 可以使用部分模型吗？

**答**: 不建议。SAM3D 是一个完整的系统，各组件之间有依赖关系。必须下载所有文件。

### Q6: 模型占用空间太大，可以放在外置硬盘吗？

**答**: 可以，只要：
1. 路径固定（不经常更换硬盘）
2. 硬盘读写速度足够快（推荐 SSD）
3. 保持硬盘连接

### Q7: 如何更新模型？

**答**: 
1. 备份当前模型
2. 重新下载最新版本
3. 更新 `SAM3D_REPO_PATH` 指向新版本

### Q8: 在 Windows 上能用吗？

**答**: 推荐使用 WSL2（Windows Subsystem for Linux 2）。Windows 原生环境可能存在兼容性问题。

---

## 9. 相关链接

### 官方资源

| 资源 | 链接 |
|------|------|
| **HuggingFace 模型** | https://huggingface.co/facebook/sam-3d-objects |
| **GitHub 仓库** | https://github.com/facebookresearch/sam-3d-objects |
| **官方网站** | https://ai.meta.com/sam3d/ |
| **在线演示** | https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d |
| **论文** | arXiv:2511.16624 |

### 项目文档

| 文档 | 链接 |
|------|------|
| **开发环境配置** | [docs/06-部署运维/环境配置（详细版）.md](../06-部署运维/环境配置（详细版）.md) |
| **3DGS 引擎文档** | [ai_engine/3dgs/README.md](../ai_engine/3dgs/README.md) |
| **项目架构** | [docs/02-架构设计/项目架构.md](../02-架构设计/项目架构.md) |

### 外部教程

| 资源 | 链接 |
|------|------|
| **Skywork 安装指南** | https://skywork.ai/blog/ai-image/install-sam-3d/ |
| **Roboflow 教程** | https://blog.roboflow.com/sam-3d/ |
| **HuggingFace CLI 文档** | https://huggingface.co/docs/huggingface_hub/en/guides/cli |

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-01-20 | v1.0 | 初始版本，添加完整配置流程 |

---

<div align="center">

**BrainDance 项目**

*物理世界注定走向无序，而我们在比特世界重建永恒。*

</div>
