# SuGaR 论文和仓库默认迭代次数

## 官方默认参数（来自代码和论文）

### 完整Pipeline（论文默认）

```
1. Vanilla 3DGS 优化
   ↓ 7_000 次迭代

2. SuGaR Coarse 训练
   ↓ 7_000 次迭代

3. Mesh 提取
   ↓ 从density场提取isosurface（Marching Cubes）

4. SuGaR Refinement
   ↓ 15_000 次迭代

5. 导出 PLY + OBJ
```

**总迭代次数**: 7_000 (Vanilla) + 7_000 (Coarse) + 15_000 (Refinement) = **29_000**

### 各阶段默认参数

| 阶段 | 默认迭代次数 | 说明 |
|--------|------------|------|
| **Vanilla 3DGS** | 7,000 | 初始优化，让高斯定位到场景 |
| **SuGaR Coarse** | 7,000 | 添加表面对齐正则化 |
| **SuGaR Refinement** | 15,000 | 与mesh联合优化 |

### 代码中的默认值

#### train.py（完整pipeline）
```python
# 默认值
iteration_to_load = 7000  # Vanilla 3DGS加载的迭代次数
refinement_iterations = 15000  # Refinement迭代次数
```

#### train_full_pipeline.py
```python
# Vanilla 3DGS训练
--iterations 7_000

# SuGaR Coarse训练
-i 7_000

# Refinement（默认15k，可通过--refinement_time调整）
-f 15_000  # 默认
-f 2_000   # short模式
-f 7_000   # medium模式
-f 15_000  # long模式
```

#### coarse_density.py / coarse_sdf.py
```python
# Coarse训练默认迭代次数（可通过args.num_iterations覆盖）
num_iterations = 15_000  # 代码中的硬编码默认值
```

### 论文中提到的迭代次数

根据README和代码注释：

> "As we explain in the paper, SuGaR optimization starts with first optimizing a 3D Gaussian Splatting model for **7k iterations** with no additional regularization term."

论文中提到的关键点：
- **7k iterations**: Vanilla 3DGS初始优化
- 没有提到coarse训练的具体迭代次数（但代码中默认是7k）
- Refinement: 代码中默认15k

---

## Fork中添加的模式

### 快速预览模式（Fast Mode）

```
1. Vanilla 3DGS: 7_000 次迭代
   ↓
2. SuGaR Coarse: 7_000 次迭代（--fast_mode时为7k）
   ↓
3. 完成（无mesh，无refinement）
```

**总迭代次数**: 7_000 + 7_000 = **14,000**

**代码实现**:
```python
# train_fast.py
if args.fast_mode:
    args.iterations = 7000  # Vanilla和Coarse都使用7k
    args.eval = False  # 禁用评估以加速
```

---

## 原始Gaussian Splatting对比

### 原始3DGS
- **默认迭代次数**: 30_000
- **代码位置**: `gaussian_splatting/arguments/__init__.py`

```python
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000  # 原始默认值
```

### SuGaR
- **Vanilla 3DGS**: 7_000（原始的1/4）
- **总迭代次数**: 29_000（完整pipeline）

**为什么SuGaR使用更少的迭代？**
1. SuGaR使用7k作为初始优化就足够了
2. Coarse和Refinement阶段进一步优化
3. Mesh约束加速收敛
4. 总体质量仍然优秀

---

## 三种Refinement模式对比

| 模式 | Refinement迭代 | 总迭代次数 | 时间（RTX 5070） |
|--------|--------------|------------|-----------------|
| **Paper默认** | 15,000 | 29,000 | 60-120分钟 |
| **Short** | 2,000 | 16,000 | 30-60分钟 |
| **Medium** | 7,000 | 21,000 | 45-90分钟 |
| **Long** | 15,000 | 29,000 | 60-120分钟 |

**注意**: Paper默认 = Long模式

---

## 代码中的关键参数

### train.py（主训练脚本）
```python
parser.add_argument('-i', '--iteration_to_load',
    type=int, default=7000,  # ← Vanilla 3DGS加载的迭代次数
    help='iteration to load.')

parser.add_argument('-f', '--refinement_iterations',
    type=int, default=15_000,  # ← Refinement默认值
    help='Number of refinement iterations.')
```

### train_full_pipeline.py（完整pipeline）
```bash
# Vanilla 3DGS训练
python ./gaussian_splatting/train.py \
    -s {args.scene_path} \
    -m {gs_checkpoint_dir} \
    --iterations 7_000  # ← 硬编码7k

# SuGaR完整训练
python train.py \
    -s {args.scene_path} \
    -c {gs_checkpoint_dir} \
    -i 7_000  # ← Coarse迭代次数
    -r {args.regularization_type} \
    -f {args.refinement_iterations}  # ← Refinement迭代次数（默认15k）
```

### train_fast.py（快速模式）
```python
# 快速模式调整
if args.fast_mode:
    args.iterations = 7000  # ← Vanilla和Coarse都使用7k
    args.eval = False
```

---

## 总结

### SuGaR论文/仓库官方默认

| 阶段 | 迭代次数 |
|--------|----------|
| Vanilla 3DGS | 7,000 |
| SuGaR Coarse | 7,000 |
| SuGaR Refinement | 15,000 |
| **总计** | **29,000** |

### 原始Gaussian Splatting

| 阶段 | 迭代次数 |
|--------|----------|
| Vanilla 3DGS | 30,000 |
| **总计** | **30,000** |

### 对比

- SuGaR使用更少的Vanilla迭代（7k vs 30k）
- 但增加了Coarse和Refinement阶段
- 总迭代次数相近（29k vs 30k）
- 质量更好，因为有Mesh约束

---

## 推荐配置

### 完整质量（论文默认）
```bash
python train_full_pipeline.py \
    -s <scene_path> \
    -r dn_consistency \
    --high_poly
```
**迭代**: 7k (Vanilla) + 7k (Coarse) + 15k (Refinement) = 29k

### 标准质量（推荐）
```bash
python train_full_pipeline.py \
    -s <scene_path> \
    -r dn_consistency \
    --high_poly \
    --refinement_time short
```
**迭代**: 7k (Vanilla) + 7k (Coarse) + 2k (Refinement) = 16k

### 快速预览
```bash
python train_fast.py \
    -s <scene_path> \
    -r dn_consistency \
    --fast_mode
```
**迭代**: 7k (Vanilla) + 7k (Coarse) = 14k
