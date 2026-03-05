# Depth Anything 3 -> SuGaR Pipeline - 快速开始

## 一键运行

将Depth Anything 3的输出转换为SuGaR重建：

```bash
cd /home/ltx/projects/Depth-Anything-3

# 快速预览模式（约30分钟）
./da3_to_sugar_pipeline.sh output/sugar_streaming my_scene dn_consistency short false true

# 标准质量（约1小时）
./da3_to_sugar_pipeline.sh output/sugar_streaming my_scene dn_consistency short true false

# 高质量（约2小时）
./da3_to_sugar_pipeline.sh output/sugar_streaming my_scene dn_consistency long true false
```

## Pipeline流程

```
Depth Anything 3 输出
    ├── camera_poses.txt (相机位姿)
    ├── intrinsic.txt (相机内参)
    ├── pcd/combined_pcd.ply (点云)
    └── extracted/ (图像)
           ↓
   [步骤1] 转换为COLMAP文本格式
           ↓
   sparse/0/
   ├── cameras.txt
   ├── images.txt
   └── points3D.txt
           ↓
   [步骤2] 转换为COLMAP二进制格式
           ↓
   sparse/0/
   ├── cameras.bin  ← SuGaR使用这个
   ├── images.bin   ← SuGaR使用这个
   └── points3D.bin ← SuGaR使用这个
           ↓
   [步骤3] 整理到SuGaR数据目录
           ↓
   SuGaR/data/my_scene/
   ├── images/
   └── sparse/0/*.bin
           ↓
   [步骤4] SuGaR训练
           ↓
   SuGaR/output/my_scene/
   ├── refined_ply/*.ply (3DGS模型)
   └── refined_mesh/*.obj (网格模型)
```

## 单独使用组件

### 1. 转换为COLMAP文本格式

```bash
python3 convert_da3_to_colmap.py \
  --base_dir /path/to/da3/output \
  --output_dir /path/to/colmap/output
```

### 2. 转换为COLMAP二进制格式

```bash
python3 colmap_text_to_binary.py \
  /path/to/colmap/output/sparse/0
```

### 3. 测试COLMAP二进制文件

```bash
python3 test_colmap_binary.py \
  /path/to/colmap/output/sparse/0
```

## 输入要求

**Depth Anything 3输出目录必须包含：**

```
your_da3_output/
├── camera_poses.txt      # 相机位姿 (N x 4 x 4, c2w格式)
├── intrinsic.txt         # 相机内参 (N x 4, fx fy cx cy)
├── pcd/
│   └── combined_pcd.ply  # 点云文件 (二进制格式)
└── extracted/           # 图像目录
    ├── frame_000001.png
    ├── frame_000002.png
    └── ...
```

## 输出结果

**SuGaR输出目录：**

```
SuGaR/output/my_scene/
├── coarse_sugar/              # 粗糙模型
├── refined_ply/              # 精炼3DGS文件
│   └── my_scene_*.ply        # 可在3DGS viewer中查看
└── refined_mesh/             # 精炼网格文件
    └── my_scene_*.obj        # 可在Blender中编辑
```

## 查看结果

### 使用SuGaR查看器

```bash
cd /home/ltx/projects/SuGaR
python run_viewer.py -p output/refined_ply/my_scene/*.ply
```

### 在线查看

- SuperSplat: https://playcanvas.com/supersplat/editor

### 在Blender中编辑

1. 安装Blender add-on: https://github.com/Anttwo/sugar_frosting_blender_addon
2. 导入OBJ文件

## 参数说明

| 参数 | 选项 | 说明 |
|------|------|------|
| 正则化方法 | dn_consistency | 推荐，最佳网格质量 |
|  | density | 密度正则化 |
|  | sdf | SDF正则化（理论最优，慢） |
| 精炼时间 | short | 2k迭代，快速 |
|  | medium | 7k迭代 |
|  | long | 15k迭代，高质量 |
| 高精度 | true | 1M顶点 |
|  | false | 200k顶点 |
| 快速模式 | true | 跳过mesh，节省50%时间 |
|  | false | 完整流程 |

### SDF正则化

**SDF（有符号距离场）正则化**提供理论最优的表面表示，但计算开销较大。

**使用方法**:
```bash
./da3_to_sugar_pipeline.sh <DA3输出> <场景名> sdf <其他参数>
```

**特点**:
- ✅ 理论最优的表面表示
- ✅ 适合复杂场景
- ⚠️ 计算开销大（慢2-3倍）
- ⚠️ 训练时间长

**使用场景**:
- ✅ 需要最高质量
- ✅ 复杂背景场景
- ✅ 有充足时间和计算资源
- ⚠️ 不适合快速迭代

**详细说明请参考**: [DA3_TO_SUGAR_PIPELINE.md](DA3_TO_SUGAR_PIPELINE.md#sdf约束详细使用指南)

## 故障排除

### 错误：找不到输入文件

```bash
# 检查DA3输出目录结构
ls -la /path/to/da3/output/
```

### 错误：图像数量较少

```bash
# 使用更长的输入视频
# 或降低DA3抽帧率
```

### 错误：SuGaR训练失败

```bash
# 检查CUDA
nvidia-smi

# 检查Conda环境
conda activate gs_linux_backup

# 检查磁盘空间
df -h
```

## 性能参考

RTX 5070上的时间估计：

| 模式 | 图像数 | 时间 |
|------|--------|------|
| 快速模式 | 50帧 | 15-30分钟 |
| 标准模式 | 50帧 | 30-60分钟 |
| 高质量 | 100帧 | 60-120分钟 |

## 详细文档

完整文档请参考：[DA3_TO_SUGAR_PIPELINE.md](DA3_TO_SUGAR_PIPELINE.md)

## 联系方式

如有问题，请查看：
- [SuGaR GitHub](https://github.com/Anttwo/SuGaR)
- [Depth Anything 3 GitHub](https://github.com/DepthAnything/Depth-Anything-V3)
- [COLMAP文档](https://colmap.github.io/)
