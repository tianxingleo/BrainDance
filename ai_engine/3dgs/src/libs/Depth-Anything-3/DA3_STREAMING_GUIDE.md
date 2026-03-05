# DA3-Streaming 完整说明

## 核心特性

### 1. 处理方式对比

| 方面 | 单帧处理 | DA3-Streaming |
|------|---------|--------------|
| 前后帧关系 | ❌ 独立处理 | ✅ **Sliding Window + Overlap** |
| 状态管理 | ❌ 无状态 | ✅ **跨 Chunk 管理状态** |
| 连续性 | ❌ 逐帧独立 | ✅ **闭环检测，防止漂移** |
| 显存需求 | 低（单帧） | 中等（120 帧 + 60 重叠） |

### 2. 单视角 vs 多视角

**DA3-Streaming 是单视角多帧处理**

- ❌ **不是**多相机多视角
- ✅ **是**单相机连续帧（时间序列）
- ✅ 类似 SLAM 系统（视觉里程计）
- ✅ 估计相机运动轨迹（位姿估计）

### 3. 3D Gaussian Splatting 支持

**点云输出 → 3DGS 输入**

```bash
# 输出点云
pcd/combined_pcd.ply  # 所有帧合并的点云

# 转换为 3DGS
# 方法 1: 直接在点云上训练 3DGS
# 方法 2: 使用 Gaussian Splatting 工具
```

## 处理流程

```
视频 → 抽帧 → DA3-Streaming → 点云/深度图 → 3DGS
          ↓
       分块处理
          ↓
      [120帧 + 60重叠]
          ↓
       闭环检测
          ↓
      位姿优化
```

## 关键参数解释

### base_config.yaml

```yaml
Model:
  chunk_size: 120      # 每块 120 帧
  overlap: 60          # 块间重叠 60 帧（50%）
  loop_chunk_size: 20   # 闭环检测块大小
  loop_enable: True     # 启用闭环检测

  save_depth_conf_result: True  # 保存每帧深度

Pointcloud_Save:
  sample_ratio: 0.015  # 点云采样率（1.5%）
  conf_threshold_coef: 0.75  # 置信度阈值系数
```

### Chunking 机制

```
Frame 0-179:   Chunk 1
                  ↓
Frame 120-239:  Chunk 2（重叠 60 帧）
                  ↓
Frame 240-359:  Chunk 3（重叠 60 帧）
                  ↓
         ...
```

**优势**：
- 显存峰值可控
- 长视频可处理
- 重叠保证连续性

### Loop Closure

```
检测回环 → 优化位姿 → 消除漂移
   ↓
   识别相似场景（SALAD）
   ↓
   优化变换矩阵（SIM3）
   ↓
   更新全局轨迹
```

**相似度阈值**：0.85（可调）

## 性能数据

### KITTI 评估（A100 GPU）

| Chunk Size | Peak VRAM | ATE RMSE [m] |
|-----------|------------|---------------|
| 120 | 15.9 GB | 10.42 |
| 90 | 14.3 GB | 9.38 |
| 60 | 12.7 GB | 10.64 |
| 30 | 11.5 GB | 19.39 |

**RTX 5070 预估**：
- Peak VRAM: ~12-13 GB（你 11.94GB 足够）
- Processing Speed: 6-8 FPS（720p）
- 总处理时间：~视频时长 / (6-8)

## 与其他方法对比

### KITTI Odometry (ATE RMSE [m])

| Method | ATE RMSE |
|--------|-----------|
| VGGT-Long | 25.60 |
| Pi-Long | 21.17 |
| **DA3-Streaming** | **18.63** ✅ |

### TUM RGB-D (ATE RMSE [m])

| Method | 360 | desk2 | floor | plant | room |
|--------|-----|-------|-------|-------|------|
| Droid-SLAM | 0.163 | 0.032 | 0.091 | 0.064 | 0.045 |
| Mast3r-SLAM | 0.060 | 0.035 | 0.055 | 0.056 | 0.035 |
| VGGT-Long | 0.110 | 0.058 | 0.111 | 0.118 | 0.071 |
| Pi-long | 0.094 | 0.115 | 0.052 | 0.160 | 0.085 |
| **DA3-Streaming** | **0.087** ✅ | 0.059 | **0.034** ✅ | **0.042** ✅ | **0.060** ✅ |

## 输出文件详解

### 必须输出

```
output_dir/
├── camera_poses.txt      # 相机外参 [N, 3, 4] (w2c)
├── intrinsic.txt         # 相机内参 [N, 4] (fx, fy, cx, cy)
└── pcd/
    └── combined_pcd.ply  # 合并点云（所有帧）
```

**格式说明**：
- `camera_poses.txt`: 每行一个矩阵
- `intrinsic.txt`: 每行 fx, fy, cx, cy
- `combined_pcd.ply`: PLY 格式点云

### 可选输出（save_depth_conf_result=True）

```
output_dir/results_output/
├── rgb/          # RGB 图像
├── depth/        # 深度图
└── conf/         # 置信度图
```

## 运行脚本

### 完整自动化

```bash
cd ~/projects/Depth-Anything-3
bash run_sugar_streaming.sh
```

### 手动步骤

```bash
# 1. 抽帧
ffmpeg -i ~/projects/SuGaR/video.mp4 \
  -vf "fps=1,scale=720:-1" \
  ~/projects/Depth-Anything-3/output/sugar_streaming/extracted/frame_%06d.png

# 2. 运行 DA3-Streaming
cd ~/projects/Depth-Anything-3/da3_streaming
HF_ENDPOINT=https://hf-mirror.com python da3_streaming.py \
  --image_dir ~/projects/Depth-Anything-3/output/sugar_streaming/extracted \
  --config configs/base_config.yaml \
  --output_dir ~/projects/Depth-Anything-3/output/sugar_streaming
```

## 3DGS 使用建议

### 点云转 3DGS

```python
import torch
from plyfile import PlyData

# 读取点云
plydata = PlyData.read('pcd/combined_pcd.ply')
points = plydata['vertex']

# 使用 GSplat 训练
# 参考: https://github.com/nerfstudio-project/gsplat
```

### 直接集成

DA3-Streaming 输出的位姿可直接用于 3DGS：

```python
# 位姿矩阵可以直接用于 3DGS 渲染
poses = np.loadtxt('camera_poses.txt').reshape(-1, 3, 4)
intrinsics = np.loadtxt('intrinsic.txt').reshape(-1, 4)
```

## 常见问题

### Q: 为什么需要分块处理？

A: 长视频需要大显存，分块后：
- 峰值显存可控（~12-15GB）
- 超长视频可处理（理论上无限）
- 重叠保证连续性

### Q: Loop Closure 有什么用？

A: 闭环检测：
- 识别回环（回到之前位置）
- 优化位姿消除漂移
- 类似 SLAM 系统的关键特性

### Q: 能实时处理吗？

A: 可以！
- A100: 8.5 FPS
- RTX 5070: 预估 6-8 FPS（720p）
- 虽然不是 30 FPS，但接近实时

### Q: 与传统 SLAM 相比如何？

A: 优势：
- 不需要标定
- 精度接近标定 SLAM
- 内存效率高

劣势：
- 不构建全局地图
- 仅局部优化

## 下一步

1. **运行处理**：`bash run_sugar_streaming.sh`
2. **查看结果**：`MeshLab pcd/combined_pcd.ply`
3. **转换 3DGS**：使用 gsplat 训练 3D Gaussian
4. **可视化**：生成 Novel Views

## 参考文献

- [DA3-Streaming README](../da3_streaming/README.md)
- [Depth Anything 3 Paper](https://arxiv.org/abs/2511.10647)
- [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)
