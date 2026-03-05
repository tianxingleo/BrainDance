# DA3 × DN-Splatter Pipeline 完整使用指南

> **一句话总结**: 本项目提供了一套端到端的自动化 Pipeline，将 **Depth Anything V3 (DA3)** 的深度估计输出直接转化为 **DN-Splatter** 训练所需的数据格式，并完成训练和 3DGS PLY 导出。

---

## 📋 目录

1. [背景与动机](#1-背景与动机)
2. [两条 Pipeline 对比](#2-两条-pipeline-对比)
3. [Pipeline A: DA3 直出流 (本项目的核心)](#3-pipeline-a-da3-直出流)
4. [Pipeline B: COLMAP 标准流 (dn-splatter 原生)](#4-pipeline-b-colmap-标准流)
5. [Pipeline 内部实现细节](#5-pipeline-内部实现细节)
6. [常见问题 FAQ](#6-常见问题-faq)
7. [兼容性修复备忘](#7-兼容性修复备忘)

---

## 1. 背景与动机

### 什么是 DN-Splatter?

[DN-Splatter](https://github.com/maturk/dn-splatter) 是基于 Nerfstudio 框架的 3D Gaussian Splatting 变体，它在训练过程中额外引入了**深度约束**和**法线约束**，从而：

- ✅ 消除白墙/纯色区域的漂浮物 (floaters)
- ✅ 让表面更加平整（适合导出 Mesh）
- ✅ 从内向外看时保持几何一致性

### 为什么需要这个 Pipeline?

DN-Splatter 原生设计是基于 **COLMAP 重建流程** 来获取相机位姿和稀疏点云的。但如果你已经通过 **Depth Anything V3** 的 Streaming 模式拿到了：

- 📷 每帧图片
- 📐 每帧内参 (focal length, principal point)
- 🗺️ 每帧位姿 (camera-to-world 4×4 矩阵)
- 🏔️ 每帧深度图 (metric depth, 米为单位)

那就**完全不需要跑 COLMAP**了。本 Pipeline 直接把 DA3 的输出"翻译"成 DN-Splatter 能接受的 Nerfstudio JSON 格式。

---

## 2. 两条 Pipeline 对比

| 维度 | Pipeline A: DA3 直出流 ⚡ | Pipeline B: COLMAP 标准流 🏗️ |
|------|--------------------------|------------------------------|
| **位姿来源** | DA3 Streaming 直出 | COLMAP SfM 重建 |
| **深度来源** | DA3 Metric Depth (绝对尺度) | ZoeDepth 或 DA3 + 手动对齐 |
| **法线来源** | 从 DA3 深度图直接推导 | Omnidata / DSINE 预训练模型 |
| **是否需要 COLMAP** | ❌ 不需要 | ✅ 必须 |
| **深度对齐** | ❌ 不需要 (DA3 和位姿同源) | ⚠️ 关键步骤 (必须对齐) |
| **数据解析器** | `normal-nerfstudio` | `coolermap` |
| **适用场景** | 已有 DA3 输出 | 只有图片,需要 SfM |
| **自动化程度** | 🟢 一键完成 | 🟡 需要手动多步 |
| **脚本** | `run_da3_to_dn_splatter_pipeline.py` | 手动命令行 |

### 选择建议

```
你有 DA3 的输出吗？(extracted/ + results_output/ + poses)
  ├── ✅ 是 → 使用 Pipeline A (本文档重点)
  └── ❌ 否，只有原始图片/视频
        ├── 有 COLMAP 重建 → 使用 Pipeline B
        └── 没有 → 先跑 COLMAP，再走 Pipeline B
```

---

## 3. Pipeline A: DA3 直出流

### 3.0 前置条件

**环境要求:**
- Conda 环境 `gs_linux_backup` (含 nerfstudio + gsplat + dn-splatter)
- Python 3.10+, numpy, Pillow, tqdm

**DA3 输出目录结构:**

```
output/sugar_streaming/          # DA3 Streaming 输出根目录
├── extracted/                   # 视频抽帧的 RGB 图片
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
├── results_output/              # DA3 深度估计结果
│   ├── frame_0.npz             # 每个 NPZ 包含 "depth" 键 (float32, 米)
│   ├── frame_1.npz
│   └── ...
├── intrinsic.txt                # 每帧内参: fx fy cx cy
└── camera_poses.txt             # 每帧位姿: 16 个数字 = 4x4 矩阵 (row-major)
```

### 3.1 一键运行 (推荐)

```bash
cd /home/ltx/projects/Depth-Anything-3

# 🚀 完整 Pipeline: 数据转换 → 训练(30000步) → 导出 PLY
python run_da3_to_dn_splatter_pipeline.py
```

就这么简单。脚本会自动完成以下三个步骤。

### 3.2 Step 1: 数据格式转换

**做了什么:**

Pipeline 将 DA3 的原始输出转换为 DN-Splatter 需要的 Nerfstudio 数据格式：

```
da3_dn_splatter_dataset/         # 自动生成的数据集目录
├── transforms.json              # Nerfstudio 格式的相机参数 (内参 + 位姿)
├── images/                      # RGB 图片 (从 DA3 output 复制)
│   ├── frame_00000.png
│   └── ...
├── depths/                      # 16-bit PNG 深度图 (毫米, uint16)
│   ├── frame_00000.png
│   └── ...
└── normals_from_pretrain/       # 法线贴图 (从深度图推导, uint8 RGB)
    ├── frame_00000.png
    └── ...
```

**关键处理:**

| 处理项 | 具体操作 |
|--------|----------|
| **分辨率对齐** | DA3 深度图分辨率 (如 280×504) 可能小于原图 (如 720×1280)。Pipeline 自动检测差异，将深度图 resize 到原图分辨率，并按比例缩放内参 |
| **坐标系转换** | DA3 位姿是 OpenCV 坐标系 (Y↓ Z→前)，DN-Splatter 需要 OpenGL 坐标系 (Y↑ Z→后)。Pipeline 自动应用翻转矩阵 |
| **深度格式** | DA3 输出 float32 NPZ (米) → 转为 uint16 PNG (毫米)。`depth_unit_scale_factor=0.001` |
| **法线生成** | 利用深度图梯度 + 内参反投影，直接计算每像素法线方向，映射到 [0, 255] RGB |

**单独运行:**

```bash
# 只做数据转换，不训练
python run_da3_to_dn_splatter_pipeline.py --convert-only

# 清除旧数据后重新转换
python run_da3_to_dn_splatter_pipeline.py --convert-only --clean
```

### 3.3 Step 2: DN-Splatter 训练

**训练命令 (Pipeline 内部自动执行):**

```bash
ns-train dn-splatter \
    --output-dir da3_dn_splatter_output \
    --experiment-name da3_dn_splatter \
    --max-num-iterations 30000 \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss True \
    --pipeline.model.normal-lambda 0.05 \
    --pipeline.model.predict-normals True \
    --pipeline.model.use-normal-tv-loss True \
    --pipeline.model.two-d-gaussians True \
    --pipeline.model.densify-grad-thresh 0.0004 \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.stop-split-at 12000 \
    --pipeline.model.max-gs-num 2000000 \
    --viewer.websocket-port 7007 \
    --vis viewer+tensorboard \
    normal-nerfstudio \
    --data da3_dn_splatter_dataset \
    --load-3D-points False \
    --load-pcd-normals False
```

**关键参数解释:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `dn-splatter` | - | 模型名称。也可用 `dn-splatter-big`(更多高斯球，细节更好) |
| `use-depth-loss` | True | **开启深度监督** — 利用 DA3 深度图约束几何 |
| `depth-lambda` | 0.2 | 深度损失权重 (越大几何越准，越小颜色越好) |
| `use-normal-loss` | True | **开启法线监督** — 约束表面平整 |
| `normal-lambda` | 0.05 | 法线损失权重 |
| `predict-normals` | True | 从高斯球朝向提取预测法线 |
| `use-normal-tv-loss` | True | 法线 TV 正则 — 进一步平滑法线 |
| `two-d-gaussians` | True | 鼓励 2D 薄片高斯 — 更好的表面 |
| `densify-grad-thresh` | 0.0004 | **分裂阈值** — 越高越不容易分裂，控制增长速度 |
| `cull-alpha-thresh` | 0.005 | **剪枝阈值** — 低于此透明度的高斯球被清理 |
| `stop-split-at` | 12000 | **停止分裂步数** — 到此步后不再分裂/复制 |
| `max-gs-num` | 2000000 | **⚡ 高斯球数量上限** — 超过此数时裁剪最低透明度的 |
| `normal-nerfstudio` | - | **数据解析器**: 读取 transforms.json + depths + normals |
| `load-3D-points` | False | DA3 流无 SfM 点云,使用随机初始化 |

**单独运行训练:**

```bash
# 假设数据集已经转换好
python run_da3_to_dn_splatter_pipeline.py --train-only

# 自定义迭代数
python run_da3_to_dn_splatter_pipeline.py --train-only --max-iterations 10000
```

**实时查看效果:**

训练启动后会在 `http://localhost:7007` 打开 Viser 查看器，可以实时看到 3DGS 的重建效果。

### 3.4 Step 3: 导出 PLY

训练完成后,Pipeline 自动查找 `config.yml` 并导出标准 3DGS PLY 文件:

```bash
ns-export gaussian-splat \
    --load-config da3_dn_splatter_output/da3_dn_splatter/dn-splatter/YYYY-MM-DD_HHMMSS/config.yml \
    --output-dir da3_dn_splatter_output/export/
```

**导出结果:**

```
da3_dn_splatter_output/export/
└── splat.ply          # 标准 3DGS PLY 文件
```

这个 `splat.ply` 可以用以下工具查看:
- [SuperSplat](https://playcanvas.com/supersplat/editor) (在线)
- [Polycam](https://poly.cam/) (移动端)
- Unity / Unreal 的 3DGS 插件
- [3DGS Viewer](https://github.com/antimatter15/splat) (网页)

**跳过导出:**

```bash
python run_da3_to_dn_splatter_pipeline.py --skip-export
```

### 3.5 完整 CLI 参数速查

```bash
python run_da3_to_dn_splatter_pipeline.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source-dir` | `output/sugar_streaming` | DA3 输出目录 |
| `--output-name` | `da3_dn_splatter` | 实验名称 |
| `--max-iterations` | 30000 | 最大训练迭代数 |
| `--convert-only` | - | 只做数据转换 |
| `--train-only` | - | 只做训练 (数据已准备好) |
| `--skip-export` | - | 训练完不导出 PLY |
| `--clean` | - | 清除旧数据后重新转换 |

### 3.6 生成目录总览

运行完整 Pipeline 后，项目目录结构:

```
Depth-Anything-3/
├── output/sugar_streaming/              # [输入] DA3 原始输出
│   ├── extracted/
│   ├── results_output/
│   ├── intrinsic.txt
│   └── camera_poses.txt
│
├── da3_dn_splatter_dataset/             # [中间] 转换后的数据集
│   ├── transforms.json
│   ├── images/
│   ├── depths/
│   └── normals_from_pretrain/
│
├── da3_dn_splatter_output/              # [输出] 训练结果
│   ├── da3_dn_splatter/
│   │   └── dn-splatter/
│   │       └── YYYY-MM-DD_HHMMSS/
│   │           ├── config.yml
│   │           └── nerfstudio_models/   # checkpoints
│   └── export/
│       └── splat.ply                    # 最终 3DGS 文件
│
├── run_da3_to_dn_splatter_pipeline.py   # 🚀 主脚本
└── run_direct_dn_splatter.py            # 简化训练入口
```

---

## 4. Pipeline B: COLMAP 标准流

> 如果你**没有** DA3 输出，只有原始图片或视频，需要走这条传统路线。

### 4.1 整理数据目录 (Standardize)

创建以下目录结构:

```
my_room_dataset/
├── images/                 # 原始 RGB 图片 (如 frame_00001.png)
├── colmap/
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
└── mono_depth/             # (可选) 放入你的 DA3 深度图
```

> 💡 **提示**: 如果你还没有 COLMAP 重建，可以使用 `ns-process-data` 一键处理:
> ```bash
> ns-process-data images --data ./my_images/ --output-dir ./my_room_dataset/
> ```

### 4.2 深度对齐 (Align Depth) — ⚠️ 关键步骤

**为什么需要对齐?**

COLMAP 的世界坐标系有**任意尺度** (arbitrary scale)，而 DA3 输出的是**绝对尺度** (metric depth)。不对齐的话训练直接崩溃。

dn-splatter 自带了对齐脚本:

```bash
python dn_splatter/scripts/align_depth.py --data path/to/my_room_dataset
```

这个脚本会:
1. 从 `colmap/sparse/0/points3D.bin` 提取 SfM 稀疏深度 → `sfm_depths/`
2. 用 ZoeDepth 生成单目深度 → `mono_depth/`
3. 将两者对齐 → `mono_depth/*_aligned.npy`

**用 DA3 深度替代默认深度:**

如果你认为 DA3 的深度比 ZoeDepth 更好,可以:
1. 先让脚本跑完,确保 `sfm_depths/` 和对齐参数正确
2. 把 `mono_depth/` 里的文件替换成你的 DA3 深度图
3. 文件名必须保持一致 (`.npy` 格式)

> ⚠️ **风险提示**: 如果跳过对齐直接用 DA3 深度,因为 COLMAP 的世界坐标尺度与 DA3 不一致,训练大概率会发散。除非你确信两者都是 Metric Scale。

### 4.3 生成法线图 (可选但强烈推荐)

法线约束能显著消除白墙上的伪影:

```bash
python dn_splatter/scripts/normals_from_pretrain.py \
    --data-dir path/to/my_room_dataset \
    --resolution low
```

支持的法线模型:
- `omnidata` (默认) — 需要下载模型权重
- `dsine` — 更新的法线估计模型

这会生成 `normals_from_pretrain/` 目录。

### 4.4 开始训练

```bash
# 标准版
ns-train dn-splatter \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss True \
    --pipeline.model.use-normal-tv-loss True \
    --pipeline.model.normal-supervision mono \
    coolermap --data path/to/my_room_dataset

# 加大版 (更多高斯球,更多细节)
ns-train dn-splatter-big \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss True \
    --pipeline.model.use-normal-tv-loss True \
    --pipeline.model.normal-supervision mono \
    coolermap --data path/to/my_room_dataset
```

**数据解析器差异:**

| 解析器 | 适用场景 | 输入格式 |
|--------|----------|----------|
| `normal-nerfstudio` | Pipeline A (DA3 直出流) | `transforms.json` + images + depths + normals |
| `coolermap` | Pipeline B (COLMAP 标准流) | `colmap/sparse/0/` + images + mono_depth + normals |

> `coolermap` 会自动读取 COLMAP 重建，加载对齐后的深度图和法线图。如果 `mono_depth/` 或 `normals_from_pretrain/` 不存在,它会**自动触发生成**。

### 4.5 导出 PLY

```bash
ns-export gaussian-splat \
    --load-config outputs/my_experiment/dn-splatter/YYYY-MM-DD_HHMMSS/config.yml \
    --output-dir exports/
```

导出的 `splat.ply` 与 Pipeline A 产出的格式完全相同。

---

## 5. Pipeline 内部实现细节

### 5.1 数据转换流程图

```
DA3 Streaming Output
        │
        ▼
┌─────────────────────┐
│  intrinsic.txt      │──→ 读取 fx, fy, cx, cy
│  camera_poses.txt   │──→ 读取 4x4 c2w 矩阵
│  extracted/*.png    │──→ 检测实际图片分辨率
│  results_output/*.npz│──→ 检测深度图分辨率
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 分辨率对齐检查                               │
│   if depth_size ≠ image_size:               │
│     scale_x = img_w / depth_w               │
│     scale_y = img_h / depth_h               │
│     fx' = fx * scale_x                      │
│     fy' = fy * scale_y                      │
│     cx' = cx * scale_x                      │
│     cy' = cy * scale_y                      │
│     depth → resize to (img_w, img_h)        │
└─────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Per-frame processing    │  × N 帧
│  ├── copy image          │
│  ├── depth (m→mm, u16)   │
│  ├── normal = f(depth,K) │
│  └── pose × flip_matrix  │  OpenCV→OpenGL
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  transforms.json         │  Nerfstudio 标准格式
│  ├── fl_x, fl_y, cx, cy  │  (缩放后内参)
│  ├── w, h                │  (图片实际分辨率)
│  └── frames[].matrix     │  (OpenGL 坐标系)
└──────────────────────────┘
```

### 5.2 法线计算方法

我们从深度图直接推导法线 (不借助预训练模型):

```python
# Central difference 梯度
zy, zx = np.gradient(depth)

# 反投影到 3D 空间的梯度
nx = -zx * fx / depth    # X 方向法线分量
ny = -zy * fy / depth    # Y 方向法线分量
nz = 1.0                 # Z 方向法线分量 (相机方向)

# 归一化
normal = normalize([nx, ny, nz])

# 映射到 [0, 255] 颜色空间
normal_img = (normal + 1) / 2 * 255
```

**与 Omnidata 的对比:**

| 方法 | 优点 | 缺点 |
|------|------|------|
| 深度推导 (Pipeline A) | 快速、无需额外模型、与深度图完全一致 | 在深度不连续处有伪影 |
| Omnidata (Pipeline B) | 在复杂场景表面更平滑 | 需要下载模型、推理较慢 |

### 5.3 坐标系转换

```
DA3 坐标系 (OpenCV):        DN-Splatter 坐标系 (OpenGL):
    Z → 前方                     Z → 后方
    Y ↓ 下方                     Y ↑ 上方
    X → 右方                     X → 右方

翻转矩阵:
    ┌ 1   0   0  0 ┐
    │ 0  -1   0  0 │     c2w_opengl = c2w_opencv × flip_mat
    │ 0   0  -1  0 │
    └ 0   0   0  1 ┘
```

---

## 6. 常见问题 FAQ

### Q1: Pipeline A 与 B 可以混用吗?

> 不建议。两条路线的坐标系来源和深度尺度标定方式不同。如果你有 DA3 输出，直接用 Pipeline A 即可。

### Q2: 训练速度大约是多少?

> 在有 GPU 的机器上，约 **89ms/iter**, **~10M rays/sec**。30000 步约需 45 分钟。
> 
> ⚠️ 但如果看到速度逐渐变慢（如从 89ms 涨到 1-2s），说明高斯球数量在密度化阶段爆炸增长。Pipeline 已内置了以下保护措施：
> - `densify-grad-thresh=0.0004` — 提高分裂阈值，减少分裂频率
> - `cull-alpha-thresh=0.005` — 更激进地清理无用高斯球
> - `stop-split-at=12000` — 12000 步后完全停止分裂
> - `max-gs-num=2000000` — 硬上限 200 万个高斯球，超过时自动裁剪最低透明度的

### Q2.5: 训练越来越慢怎么办?

> 这通常是高斯球数量爆炸导致的。日志中出现 `XXX GSs duplicated, XXX GSs split` 时如果数字很大（>50k），说明密度化太激进。解决方案：
>
> 1. **调高分裂阈值**: `--pipeline.model.densify-grad-thresh 0.001` (更保守)
> 2. **降低数量上限**: `--pipeline.model.max-gs-num 1000000` (硬限制 100 万)
> 3. **更早停止分裂**: `--pipeline.model.stop-split-at 8000`
> 4. **更激进剪枝**: `--pipeline.model.cull-alpha-thresh 0.01`
>
> 💡 **注意**: `DefaultStrategy`（gsplat 原版）本身没有数量上限机制，我们在 DN-Splatter 中额外实现了 `max_gs_num` 硬上限: 超过时自动移除透明度最低的高斯球。

### Q3: 出现 `image size does not match camera parameters` 怎么办?

> 这是分辨率不匹配问题。Pipeline 已经自动处理了这个问题（自动检测并缩放内参 + resize 深度图）。但如果你手动准备数据，确保 `transforms.json` 中的 `w/h` 与你的图片实际像素尺寸一致。

### Q4: 出现 `AttributeError: 'DNSplatterModel' object has no attribute 'k_nearest_sklearn'`?

> 这是 DN-Splatter 与新版 nerfstudio/gsplat 的兼容性问题，已经在本项目中修复。见[兼容性修复备忘](#7-兼容性修复备忘)。

### Q5: 可以用 `dn-splatter-big` 获得更好的效果吗?

> 可以！只需修改训练命令中的模型名: 在 `run_da3_to_dn_splatter_pipeline.py` 中将 `"dn-splatter"` 改为 `"dn-splatter-big"`。`dn-splatter-big` 使用更宽松的剪枝阈值 (`cull_alpha_thresh=0.005`),保留更多高斯球,细节更好。

### Q6: 深度对齐 (align_depth.py) 在 Pipeline A 中需要吗?

> **不需要!** 这是 Pipeline A 相比 Pipeline B 的一大优势。因为 DA3 的深度和位姿是**同一个模型同时输出的**,它们天然就是对齐的。而 COLMAP 的位姿是 SfM 独立估计的,与深度模型的输出尺度不同,所以需要对齐。

### Q7: 如果我想用 COLMAP 的位姿 + DA3 的深度怎么办?

> 走 Pipeline B。把 DA3 深度图放入 `mono_depth/`，然后运行 `align_depth.py` 来对齐尺度。这样你就能利用 COLMAP 的精确位姿 + DA3 的高质量深度。

---

## 7. 兼容性修复备忘

本项目对 `dn_splatter/dn_model.py` 进行了以下修复，以兼容新版 nerfstudio (≥0.3.4) 和 gsplat (≥1.0.0):

| # | 原始问题 | 修复方式 | 影响的方法 |
|---|----------|---------|------------|
| 1 | `self.k_nearest_sklearn()` 不存在 | 改为独立函数 `from nerfstudio.utils.math import k_nearest_sklearn` | `populate_modules()` |
| 2 | `self.after_train` 回调不存在 | 移除，使用继承的 `step_post_backward` | `get_training_callbacks()` |
| 3 | `refinement_after()` 使用已移除的 gsplat API | 删除，改为继承 `SplatfactoModel.step_post_backward`，底层使用 `gsplat.strategy.DefaultStrategy` | `refinement_after()` (已删除) |
| 4 | 旧版 `rasterize_gaussians` 无法渲染法线 | 使用 `rasterization()` 替代 | `get_outputs()` |
| 5 | 缺少 strategy 初始化 | 添加 `DefaultStrategy` + `strategy_state` 初始化 | `populate_modules()` |

**修改的文件:**
- `/home/ltx/my_envs/gs_linux_backup/lib/python3.10/site-packages/dn_splatter/dn_model.py`
- `/home/ltx/my_envs/gs_linux_backup/lib/python3.10/site-packages/gsplat/__init__.py` (添加 `rasterize_gaussians` 兼容 wrapper)

---

## 附录: 快速上手命令汇总

```bash
# ==================== Pipeline A: DA3 直出流 ====================

# 一键全流程 (转换 + 训练 + 导出)
python run_da3_to_dn_splatter_pipeline.py

# 只转换数据
python run_da3_to_dn_splatter_pipeline.py --convert-only --clean

# 只训练 (数据已转换)
python run_da3_to_dn_splatter_pipeline.py --train-only --max-iterations 30000

# 只训练不导出
python run_da3_to_dn_splatter_pipeline.py --train-only --skip-export

# 指定自定义路径
python run_da3_to_dn_splatter_pipeline.py \
    --source-dir /path/to/da3/output \
    --output-name my_scene


# ==================== Pipeline B: COLMAP 标准流 ====================

# Step 1: 对齐深度
python dn_splatter/scripts/align_depth.py --data path/to/dataset

# Step 2: 生成法线
python dn_splatter/scripts/normals_from_pretrain.py \
    --data-dir path/to/dataset --resolution low

# Step 3: 训练
ns-train dn-splatter-big \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss True \
    --pipeline.model.normal-supervision mono \
    coolermap --data path/to/dataset

# Step 4: 导出
ns-export gaussian-splat \
    --load-config outputs/.../config.yml \
    --output-dir exports/
```

---

*最后更新: 2026-02-17 | 作者: DA3 × DN-Splatter Pipeline*
