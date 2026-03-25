# BrainDance 三维重建测试文档

本文档基于 `ai_engine/3dgs/PAPER_QUANT_EVIDENCE.md` 中整理的论文证据，以及当前 `ai_engine/3dgs/src/core/factory.py` 的流水线映射，给出 BrainDance 现阶段三维重建测试方案。

适用范围：
- 视频三维重建
- 多图 / 少图三维重建
- 单图 3D 生成
- 网格提取与精修
- 深度 / 位姿 / 视图空间恢复

## 1. 测试目标

测试目标分为四层：
- 功能正确：流水线能跑通，能输出可交付文件。
- 工程稳定：任务状态、日志、上传、回写链路稳定。
- 重建质量：渲染质量、几何质量、位姿质量达到预期。
- 路线选型：验证不同 pipeline 是否符合对应论文宣称的优势场景。

## 2. Pipeline 与论文映射

| task_type | 代码文件 | 对应论文 | 主要能力 | 重点验证项 |
| --- | --- | --- | --- | --- |
| `video_3dgs` | `ai_engine/3dgs/src/pipelines/video_3dgs.py` | Nerfstudio | 默认视频 3DGS 重建 | 跑通率、训练稳定性、基础渲染质量 |
| `video_3dgs` + `mapper_type=da3` | `ai_engine/3dgs/src/pipelines/video_3dgs.py` | Depth Anything 3 | DA3 作为位姿/深度前端 | 位姿质量、长视频稳定性 |
| `da3_feed_forward_3dgs` | `ai_engine/3dgs/src/pipelines/da3_feed_forward_pipeline.py` | Depth Anything 3 | 无需 Nerfstudio 训练的直接 3DGS 输出 | 极速出结果、空间一致性 |
| `da3_sugar` | `ai_engine/3dgs/src/pipelines/da3_sugar_pipeline.py` | Depth Anything 3 + SuGaR | 高质量 mesh / refinement | 网格质量、导出质量、耗时 |
| `da3_2dgs` | `ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py` | Depth Anything 3 + 2DGS | 少图 / 中短序列高几何质量重建 | 几何一致性、渲染质量 |
| `sparse2dgs` | `ai_engine/3dgs/src/pipelines/sparse2dgs.py` | Sparse2DGS | 极稀疏视角重建 | 3-view / 5-view 几何质量 |
| `single_image_sharp` | `ai_engine/3dgs/src/pipelines/single_image_sharp.py` | SHARP | 单图快速生成 3D Gaussian 结果 | 速度、邻近视角观感 |
| `single_image_sam3d` | `ai_engine/3dgs/src/pipelines/single_image_sam3d.py` | SAM 3D Objects | 单图对象级 3D 生成 | 单对象质量、布局合理性 |

补充：
- `Segment Anything in 3D Scenes` 当前没有独立 `task_type`，仅作为 3D 分割方向参考，不纳入主回归流水线。

## 3. 测试输入分层

### A. 视频重建集
- `V1 室内小场景`：桌面、椅子、柜子，遮挡较少，30-60 秒。
- `V2 室内复杂场景`：房间、舞蹈教室、多人/动态干扰，30-90 秒。
- `V3 户外场景`：建筑、雕塑、树木，光照变化明显，30-60 秒。
- `V4 长视频压力集`：2-5 分钟，验证 DA3 长序列可扩展性。

### B. 多图重建集
- `M3`：3 张图片，环绕角度尽量分散。
- `M5`：5 张图片，轻度稀疏。
- `M10`：10 张图片，中等覆盖。
- `M30`：30 张图片，近似完整覆盖。

### C. 单图生成集
- `S1 单物体白底或简单背景`
- `S2 单物体复杂背景`
- `S3 多物体场景`
- `S4 遮挡明显场景`

## 4. 测试输出物与检查项

每次测试至少检查以下产物：
- 主输出文件是否存在：`point_cloud.ply` / mesh / preview 图。
- 过程日志是否完整：下载、解算、训练、导出、上传。
- `processing_tasks.status` 是否正确流转：`pending -> processing -> completed/failed`。
- 预览图、`transforms.json`、`webgl_poses.json` 是否存在且可解析。
- `model_assets` 是否成功回写必要字段。

## 5. 统一测试指标

### 功能与工程指标
- 成功率：成功任务数 / 总任务数。
- 平均耗时：从任务启动到上传完成。
- 峰值显存：用于识别 pipeline 是否超出设备预算。
- 输出完整率：PLY、预览图、位姿文件、日志文件的存在率。

### 渲染质量指标
- `PSNR`
- `SSIM`
- `LPIPS`
- 单图快速视图生成场景增加 `DISTS`

### 几何质量指标
- `Chamfer Distance`
- `F1`
- `Accuracy / Completion / Avg`
- 若输出 mesh，再补充 watertight、洞数、异常面占比

### 位姿 / 空间恢复指标
- `Auc3 / Auc30` 或等价位姿误差统计
- 注册成功帧比例
- 轨迹连续性
- 点云 / 视角对齐可视检查

## 6. 各 Pipeline 专项测试要求

### 6.1 `video_3dgs`

目标：验证默认视频重建主线稳定可用。

必测项：
- `V1`、`V2`、`V3` 各跑 1 次。
- 分别测试 `fast_mode=true` 与默认参数。
- 检查是否稳定导出 `point_cloud.ply`。

关注指标：
- 成功率
- 总耗时
- 预览图质量
- NVS 指标：`PSNR / SSIM / LPIPS`

验收建议：
- 作为基线流水线，必须优先保证成功率和稳定性。
- 若与其他论文型 pipeline 对比，允许质量不是最优，但不能成为明显短板。

### 6.2 `video_3dgs` + `mapper_type=da3`

目标：验证 DA3 替代传统 SfM/GLOMAP 作为前端时，对视频空间恢复是否更稳。

必测项：
- `V2` 与 `V4`。
- 与默认 `mapper_type=glomap` 做 A/B 对比。

关注指标：
- 注册成功帧比例
- 位姿连续性
- 重建中断率
- 长视频耗时与显存

验收建议：
- 在复杂视频和长视频上，DA3 版本不应比 GLOMAP 版本更差。
- 若 GLOMAP 丢帧明显而 DA3 保持稳定，则判定 DA3 前端有效。

### 6.3 `da3_feed_forward_3dgs`

目标：验证“快速直接出 3DGS”路线。

必测项：
- `V1`、`V2` 各 1 次。
- 与 `video_3dgs` 比较首个可用结果时间。

关注指标：
- 首个可用结果时间
- 总耗时
- 空间一致性
- 预览观感

验收建议：
- 应明显快于标准训练型流水线。
- 可接受最终质量不如 `da3_sugar` / `da3_2dgs`，但必须足够适合快速预览。

### 6.4 `da3_sugar`

目标：验证高质量 mesh 提取与精修能力。

必测项：
- `V1`、`V2` 各 1 次。
- `regularization=dn_consistency`、`sdf` 各测 1 次。
- `fast_mode=true/false` 各测 1 次。

关注指标：
- mesh 是否成功导出
- 网格洞/碎片数量
- mesh 渲染质量：`PSNR / SSIM / LPIPS`
- 几何质量：`Chamfer / F1`
- 总耗时

验收建议：
- 若目标是后续编辑/资产化导出，`da3_sugar` 应作为首选高质量链路。
- 允许耗时较高，但 mesh 质量必须显著优于普通点云导出路线。

### 6.5 `da3_2dgs`

目标：验证 2DGS 路线在少图/短序列场景下的几何优势。

必测项：
- `M5`、`M10`、`M30`。
- `iterations=7000` 与更高训练步数各测 1 次。

关注指标：
- `PSNR / SSIM / LPIPS`
- `Chamfer / Accuracy / Completion`
- 训练耗时
- 输出体积

验收建议：
- 几何质量应优于 `video_3dgs` 基线。
- 在中小规模输入上，应表现出接近论文中“几何更强、画质不显著下降”的趋势。

### 6.6 `sparse2dgs`

目标：验证极稀疏视角场景下的专用路线。

必测项：
- `M3`、`M5`、`M10`。
- `M3` 是主评测项。

关注指标：
- `Chamfer Distance`
- `Accuracy / Completion / Avg`
- `PSNR / SSIM / LPIPS`
- COLMAP 成功率
- 总耗时

验收建议：
- 在 `M3` / `M5` 条件下，应优于 `da3_2dgs` 或默认 3DGS 基线。
- 如果 `M3` 下几何仍无法闭合或出现严重漂浮点，判定该任务失败。

### 6.7 `single_image_sharp`

目标：验证单图快速 3D / 多视图预览能力。

必测项：
- `S1`、`S2`、`S3` 各 1 次。
- 记录模型核心推理时间与整体任务时间。

关注指标：
- `DISTS / LPIPS`
- 近邻视角视觉连续性
- 推理时间
- 伪影数量

验收建议：
- 重点是“快”和“预览好看”。
- 不要求其几何质量达到多视图重建 pipeline 水平。

### 6.8 `single_image_sam3d`

目标：验证单图对象级 3D 生成与简单场景布局恢复能力。

必测项：
- `S1`、`S2`、`S4`。
- 若有 mask 输入，再补 1 组 mask / no-mask 对照。

关注指标：
- 单物体几何可用性
- 多视角观感
- 布局合理性
- 纹理连续性

验收建议：
- 目标不是高度精确测绘，而是快速生成可用 3D 对象资产。
- 对复杂背景和遮挡，应优先看对象主体是否完整、轮廓是否稳定。

## 7. 回归测试矩阵

建议最小回归集如下：

| 编号 | pipeline | 输入集 | 目标 |
| --- | --- | --- | --- |
| R1 | `video_3dgs` | `V1` | 默认视频重建可用 |
| R2 | `video_3dgs` | `V2` | 复杂场景稳定性 |
| R3 | `video_3dgs` + `mapper_type=da3` | `V4` | DA3 长视频稳定性 |
| R4 | `da3_feed_forward_3dgs` | `V1` | 快速预览 |
| R5 | `da3_sugar` | `V1` | mesh 导出质量 |
| R6 | `da3_2dgs` | `M10` | 少图高质量重建 |
| R7 | `sparse2dgs` | `M3` | 极稀疏视角重建 |
| R8 | `single_image_sharp` | `S2` | 单图快速预览 |
| R9 | `single_image_sam3d` | `S1` | 单物体 3D 生成 |

## 8. 测试记录模板

每个任务建议记录：

```text
任务编号:
测试日期:
task_type:
输入数据:
关键参数:
GPU / CUDA:
总耗时:
是否成功:
输出文件:
PSNR/SSIM/LPIPS:
Chamfer/F1/Accuracy/Completion:
位姿质量:
主观结论:
失败日志摘要:
```

## 9. 最终判定原则

判定优先级如下：
- 第一优先级：能否稳定跑通并交付结果。
- 第二优先级：该 pipeline 是否在自己的目标场景内优于基线。
- 第三优先级：是否符合论文宣称的优势方向。

按这个原则，当前推荐的判断口径是：
- `video_3dgs`：看稳定性与通用性。
- `da3_feed_forward_3dgs`：看速度与可预览性。
- `da3_sugar`：看 mesh 质量与资产化价值。
- `da3_2dgs`：看少图情况下的几何质量。
- `sparse2dgs`：看极稀疏视角几何质量。
- `single_image_sharp`：看单图快速视图合成质量。
- `single_image_sam3d`：看单图对象 3D 资产生成能力。
