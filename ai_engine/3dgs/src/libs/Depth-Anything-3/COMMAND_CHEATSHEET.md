# DA3 → SuGaR Pipeline - 命令速查表

## 正则化方法快速选择

### 1分钟决策树

```
开始
  ↓
场景复杂吗？
  ├─ 是 → 需要最高质量吗？
  │        ├─ 是 → 使用 sdf ⚠️（慢）
  │        └─ 否 → 使用 dn_consistency ✅（推荐）
  └─ 否 → 需要快速输出吗？
           ├─ 是 → 使用 density ⚡（快）
           └─ 否 → 使用 dn_consistency ✅（推荐）
```

## SDF约束命令

### 快速预览（无mesh）

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  sdf \
  short \
  false \
  true
```

**时间**: 20-40分钟
**输出**: PLY文件（快速查看）

### 标准质量（推荐）

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  sdf \
  short \
  true \
  false
```

**时间**: 40-80分钟
**输出**: PLY + OBJ文件

### 高质量（最优）

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  sdf \
  long \
  true \
  false
```

**时间**: 80-160分钟
**输出**: 最高质量PLY + OBJ

## dn_consistency命令（推荐）

### 快速预览（无mesh）

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  dn_consistency \
  short \
  false \
  true
```

**时间**: 15-30分钟
**输出**: PLY文件

### 标准质量

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  dn_consistency \
  short \
  true \
  false
```

**时间**: 30-60分钟
**输出**: PLY + OBJ文件

### 高质量

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  dn_consistency \
  long \
  true \
  false
```

**时间**: 60-120分钟
**输出**: PLY + OBJ文件

## density命令（快速）

### 快速预览（无mesh）

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  density \
  short \
  false \
  true
```

**时间**: 15-30分钟
**输出**: PLY文件

### 标准质量

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  density \
  short \
  true \
  false
```

**时间**: 30-60分钟
**输出**: PLY + OBJ文件

### 高质量

```bash
cd /home/ltx/projects/Depth-Anything-3

./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  density \
  long \
  true \
  false
```

**时间**: 60-120分钟
**输出**: PLY + OBJ文件

## 参数说明

| 参数 | 选项 | 说明 |
|------|------|------|
| **1. DA3输出** | `output/sugar_streaming` | DA3输出目录路径 |
| **2. 场景名** | `my_scene` | 输出场景名称 |
| **3. 正则化** | `sdf` / `dn_consistency` / `density` | **SDF / dn_consistency / density** |
| **4. 精炼时间** | `short` / `medium` / `long` | Refinement时间 |
| **5. 高精度** | `true` / `false` | 1M顶点 / 200k顶点 |
| **6. 快速模式** | `true` / `false` | 跳过mesh / 完整流程 |

## 正则化方法对比

| 方法 | 质量 | 速度 | VRAM | 推荐度 |
|------|-----|------|------|--------|
| **sdf** | ⭐⭐⭐⭐⭐ | 慢 | 高 | ⭐⭐⭐ |
| **dn_consistency** | ⭐⭐⭐⭐⭐ | 中等 | 中等 | ⭐⭐⭐⭐⭐ |
| **density** | ⭐⭐⭐⭐ | 快 | 低 | ⭐⭐⭐⭐⭐ |

## 时间对比（RTX 5070）

| 模式 | sdf | dn_consistency | density |
|------|-----|---------------|---------|
| **快速预览** | 20-40分钟 | 15-30分钟 | 15-30分钟 |
| **标准质量** | 40-80分钟 | 30-60分钟 | 30-60分钟 |
| **高质量** | 80-160分钟 | 60-120分钟 | 60-120分钟 |

## 场景建议

| 场景类型 | 推荐 | 备选 |
|---------|------|------|
| **简单场景** | dn_consistency | density |
| **复杂背景** | sdf | dn_consistency |
| **快速测试** | density | dn_consistency |
| **追求最优** | sdf | dn_consistency |
| **平衡质量速度** | dn_consistency | density |

## SDF特别说明

### 何时使用SDF

✅ **适合**:
- 复杂背景场景
- 多个遮挡物体
- 薄结构（纸张、布料）
- 追求理论最优
- 有充足时间和计算资源

⚠️ **不适合**:
- 快速测试和迭代
- 计算资源有限（<16GB VRAM）
- 简单场景
- 需要快速输出

### SDF注意事项

⚠️ **重要**: SDF约束计算开销大，训练时间比其他方法长2-3倍。

**建议**:
1. 先使用`dn_consistency`测试pipeline
2. 确认场景复杂性后考虑使用SDF
3. 确保有充足的时间和计算资源
4. 监控VRAM使用，必要时调整参数

### SDF详细文档

完整SDF使用指南请参考: [SDF_REGULARIZATION_GUIDE.md](SDF_REGULARIZATION_GUIDE.md)

## 常见命令模板

### 模板1: 最常用（推荐）

```bash
./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  dn_consistency \
  short \
  true \
  false
```

### 模板2: 使用SDF

```bash
./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  sdf \
  long \
  true \
  false
```

### 模板3: 快速预览

```bash
./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  density \
  short \
  false \
  true
```

### 模板4: 高质量快速输出

```bash
./da3_to_sugar_pipeline.sh \
  output/sugar_streaming \
  scene_name \
  dn_consistency \
  long \
  true \
  false
```

## 故障排除

### SDF训练太慢？

1. 使用`dn_consistency`代替
2. 减少SDF采样点数
3. 降低图像分辨率
4. 使用快速模式

### VRAM不足？

1. 使用`density`（最低VRAM占用）
2. 减少图像数量
3. 降低mesh顶点数（使用`false`）
4. 减少SDF采样点数

### 质量不满意？

1. 延长refinement时间（`long`）
2. 使用SDF（复杂场景）
3. 增加mesh顶点数（`true`）
4. 尝试不同的正则化方法

## 相关文档

- [SDF Regularization Guide](SDF_REGULARIZATION_GUIDE.md) - SDF详细使用指南
- [DA3 → SuGaR Pipeline Guide](DA3_TO_SUGAR_PIPELINE.md) - 完整pipeline文档
- [DA3 → SuGaR Quick Start](DA3_TO_SUGAR_QUICKSTART.md) - 快速开始指南
- [SuGaR Modes Technical Details](SUGAR_MODES_TECHNICAL_DETAILS.md) - 模式技术对比

## 总结

**选择正则化方法**:
- **默认推荐**: `dn_consistency`（最佳平衡）
- **追求最优**: `sdf`（复杂场景，慢）
- **快速输出**: `density`（简单场景，快）

**选择质量模式**:
- **快速预览**: `short` + `false`（15-30分钟）
- **标准质量**: `short` + `true`（30-60分钟）
- **高质量**: `long` + `true`（60-120分钟）
