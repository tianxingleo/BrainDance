# 高斯椭球数量限制配置指南

本文档总结了项目中各个3DGS训练脚本的高斯椭球数量限制配置。

## 📊 配置对比

### 1. Nerfstudio Splatfacto (推荐)

**脚本**: [train_3dgs_nerfstudio.sh](train_3dgs_nerfstudio.sh)

**高斯椭球限制参数**:
```bash
--pipeline.model.densify-grad-thresh 0.0004     # 分裂阈值（默认0.0002）
--pipeline.model.cull-alpha-thresh 0.005        # 清理低透明度高斯球（默认0.005）
--pipeline.model.stop-split-at $((ITER-3000))   # 停止分裂迭代（默认15000）
--pipeline.model.max-gs-num 2000000             # 高斯球硬上限（默认1000000）
```

**效果**:
- 分裂阈值提高 → 减少新高斯球生成
- 设置硬上限 → 最多200万高斯球
- 提前停止分裂 → 避免后期过度增长

**适用场景**: 需要Web查看器、实时监控、显存有限（12GB）

---

### 2. 原始3DGS (Inria版)

**脚本**: [train_3dgs_from_colmap.sh](train_3dgs_from_colmap.sh)

**高斯椭球限制参数**:
```bash
--densify_until_iter $((ITERATIONS - 3000))     # 停止分裂迭代（默认15000）
--densify_grad_threshold 0.0004                 # 分裂梯度阈值（默认0.0002）
```

**效果**:
- 提高分裂阈值 → 只在梯度大的地方分裂
- 提前停止分裂 → 控制最终数量

**适用场景**: 追求速度、不需要Web查看器、显存非常有限

**注意**: 原始3DGS没有硬上限机制，依赖上述软限制

---

### 3. DN-Splatter Pipeline

**脚本**: [run_da3_to_dn_splatter_pipeline.py](run_da3_to_dn_splatter_pipeline.py)

**高斯椭球限制参数**:
```python
--pipeline.model.densify-grad-thresh 0.0004
--pipeline.model.cull-alpha-thresh 0.005
--pipeline.model.stop-split-at 12000
--pipeline.model.max-gs-num 2000000
```

**特点**:
- 使用MCMC策略，有更强的高斯球管理
- 支持法向量、2D高斯等高级特性
- 显存优化目标: ~10GB (RTX 5070 12GB)

---

### 4. 前馈3DGS (Feed-Forward)

**脚本**: [feed_forward_3dgs_fixed.py](feed_forward_3dgs_fixed.py)

**高斯椭球限制参数**:
```bash
--conf-threshold 0.85      # 置信度阈值
--sample-ratio 1.0         # 采样比例（1.0=全部，0.5=50%）
--frame-interval 10        # 抽帧间隔
```

**特点**:
- 不进行训练，直接生成高斯球
- 通过降采样和置信度过滤控制数量
- 生成后数量固定，不会增长

---

## 🔧 参数调优建议

### 显存 < 8GB
```bash
--pipeline.model.max-gs-num 1000000              # 降低上限到100万
--pipeline.model.densify-grad-thresh 0.001       # 提高分裂阈值
--pipeline.model.stop-split-at $((ITER-5000))    # 更早停止分裂
```

### 显存 8-12GB (默认)
```bash
--pipeline.model.max-gs-num 2000000              # 200万上限
--pipeline.model.densify-grad-thresh 0.0004      # 适中分裂阈值
--pipeline.model.stop-split-at $((ITER-3000))    # 前3000步停止分裂
```

### 显存 > 12GB (追求质量)
```bash
--pipeline.model.max-gs-num 5000000              # 500万上限
--pipeline.model.densify-grad-thresh 0.0002      # 默认分裂阈值
--pipeline.model.stop-split-at $ITERATIONS       # 训练全程分裂
```

---

## 📈 数量控制策略对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **提高分裂阈值** | 简单有效 | 可能影响细节 | 显存紧张 |
| **提前停止分裂** | 控制精确 | 后期无法优化细节 | 固定场景 |
| **硬上限** | 绝对安全 | 可能强制删除重要高斯 | 所有场景 |
| **降采样** | 前期控制 | 损失信息 | 前馈方法 |

---

## 🎯 推荐配置

### 快速测试 (显存 6GB)
```bash
./train_3dgs_nerfstudio.sh output/sugar_streaming1_colmap test 7000
# 脚本会自动配置: max-gs-num=200万, stop-split-at=4000
```

### 标准训练 (显存 12GB)
```bash
./train_3dgs_nerfstudio.sh output/sugar_streaming1_colmap scene 15000
# 脚本会自动配置: max-gs-num=200万, stop-split-at=12000
```

### 高质量训练 (显存 24GB)
```bash
# 手动修改脚本中的 max-gs-num 为 5000000
./train_3dgs_nerfstudio.sh output/sugar_streaming1_colmap scene 30000
```

---

## 🔍 监控训练中的高斯球数量

### Nerfstudio
```bash
# TensorBoard 会显示实时高斯球数量
tensorboard --logdir output/nerfstudio_3dgs/<scene>/outputs
```

### 原始3DGS
```bash
# 查看训练日志
tail -f /path/to/output/point_cloud/iteration_*/test_output.log
```

---

## 📝 更新日志

- **2025-02-21**: 为 `train_3dgs_nerfstudio.sh` 和 `train_3dgs_from_colmap.sh` 添加高斯球限制
- **参考**: [run_da3_to_dn_splatter_pipeline.py:294-297](run_da3_to_dn_splatter_pipeline.py#L294-L297) 的成功配置
