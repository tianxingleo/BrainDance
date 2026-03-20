# BrainDance 3DGS 相关论文定量证据整理

本文档从 `ai_engine/3dgs/THIRD_PARTY_ATTRIBUTIONS.md` 中提到的论文出发，提取各论文正文或补充材料中的定量实验数据，用于支撑 BrainDance 当前 3DGS 相关技术路线的选择。

说明：
- 只整理论文原文中直接给出的实验设置、指标和数值。
- 重点保留最能体现“为什么采用该方案”的数据。
- 若论文本身不是以 SOTA 定量对比为主，或没有给出充分定量结果，会在对应小节明确说明。

## 1. Nerfstudio / Nerfacto

论文：Nerfstudio: A Modular Framework for Neural Radiance Field Development  
链接：<https://arxiv.org/abs/2302.04264>

### 当前 ai_engine 对应 pipeline
- `task_type=video_3dgs` -> `ai_engine/3dgs/src/pipelines/video_3dgs.py`
- 模块角色：默认视频三维重建主流水线，底层训练引擎为 Nerfstudio `splatfacto`。

### 适合支撑 BrainDance 的结论
- Nerfstudio 的核心优势不是“单一新算法绝对 SOTA”，而是模块化框架 + 默认 `Nerfacto` 管线提供了很强的速度/质量折中。
- 对 BrainDance 而言，它最重要的价值是：能以较低训练成本快速得到可用结果，并作为后续 3DGS、2DGS、SuGaR 等路线的工程底座。

### 关键定量结果
- Mip-NeRF 360 平均结果（7 个场景）：
  - `Nerfacto 70K iter (~30 min)`：PSNR `27.98`，SSIM `0.800`，LPIPS `0.291`
  - `Nerfacto 5K iter (~2 min)`：PSNR `25.38`，SSIM `0.688`，LPIPS `0.390`
  - `MipNeRF-360`：PSNR `29.23`，SSIM `0.844`，LPIPS `0.207`
- 这组结果说明：虽然 Nerfacto 不是该数据集上的最佳精度方法，但它能在 `2 分钟` 产出可用质量，在 `30 分钟` 达到较强基线，而 MipNeRF-360 训练通常需要数小时。

### 模块化带来的量化收益
- Nerfstudio 自建真实场景数据集 10 场景平均消融：
  - `Nerfacto(default)`：PSNR `20.99`，SSIM `0.663`，LPIPS `0.389`
  - `no contraction`：PSNR `18.59`，SSIM `0.534`，LPIPS `0.506`
- 这说明框架中的关键组件并非装饰项，像 scene contraction 这类设计对真实场景质量影响明显。

## 2. 2D Gaussian Splatting

论文：2D Gaussian Splatting for Geometrically Accurate Radiance Fields  
链接：<https://arxiv.org/abs/2403.17888>

### 当前 ai_engine 对应 pipeline
- `task_type=da3_2dgs` / `task_type=da3+2dgs` -> `ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py`
- 模块角色：先用 DA3 做位姿/深度解算，再训练 2DGS，作为 Nerfstudio 3DGS 的少图替代路线。

### 适合支撑 BrainDance 的结论
- 2DGS 最强的证据是：相比传统 3DGS，它显著提升几何精度，同时保持分钟级训练和接近的渲染质量。
- 对 BrainDance 而言，这正对应“既要可渲染，又要几何更可靠”的核心诉求。

### 几何精度：DTU
- DTU Mean Accuracy（越低越好）：
  - `3DGS`：`1.96`
  - `SuGaR`：`1.33`
  - `2DGS-15k`：`0.83`
  - `2DGS-30k`：`0.80`
- 训练时间：
  - `3DGS`：`11.2 min`
  - `SuGaR`：`1 h`
  - `2DGS-15k`：`5.5 min`
  - `2DGS-30k`：`18.8 min`
- 结论：2DGS 在 DTU 上把 3DGS 的几何误差从 `1.96` 降到 `0.80`，而训练仍保持在分钟级。

### 几何 + 存储效率：DTU
- DTU Table 3：
  - `3DGS`：CD `1.96`，PSNR `35.76`，Time `11.2 min`，Storage `113 MB`
  - `SuGaR`：CD `1.33`，PSNR `34.57`，Time `1 h`，Storage `1247 MB`
  - `2DGS-30k`：CD `0.80`，PSNR `34.52`，Time `18.8 min`，Storage `52 MB`
- 结论：2DGS 相比 SuGaR 不仅更准，而且更轻；相比 3DGS，几何显著更强，PSNR 仍处于可接受范围。

### 真实复杂场景几何：Tanks and Temples
- Mean F1（越高越好）：
  - `3DGS`：`0.09`
  - `SuGaR`：`0.19`
  - `2DGS`：`0.30`
  - `Neuralangelo`：`0.50`
- 训练时间：
  - `3DGS`：`14.3 min`
  - `SuGaR`：`1 h`
  - `2DGS`：`34.2 min`
  - 隐式/SDF 类方法通常约 `24 h`
- 结论：2DGS 在真实场景几何上显著优于 3DGS/SuGaR，同时远快于需要 24 小时级训练的隐式方法。

### 新视角合成质量：Mip-NeRF360
- Outdoor：
  - `3DGS`：PSNR `24.24`，SSIM `0.705`，LPIPS `0.283`
  - `2DGS`：PSNR `24.33`，SSIM `0.709`，LPIPS `0.284`
- Indoor：
  - `3DGS`：PSNR `30.99`，SSIM `0.926`，LPIPS `0.199`
  - `2DGS`：PSNR `30.39`，SSIM `0.924`，LPIPS `0.182`
- 结论：2DGS 的渲染质量与 3DGS 基本同档，Indoor 上 LPIPS 甚至更优，说明“几何变强”没有明显牺牲画质。

## 3. Sparse2DGS

论文：Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views  
链接：<https://arxiv.org/abs/2504.20378>

### 当前 ai_engine 对应 pipeline
- `task_type=sparse2dgs` -> `ai_engine/3dgs/src/pipelines/sparse2dgs.py`
- 模块角色：面向少图/稀疏视角输入的专用重建流水线，包含 COLMAP 稀疏重建与 Sparse2DGS 训练。

### 适合支撑 BrainDance 的结论
- Sparse2DGS 的价值非常明确：在 `3-view` 极稀疏视角下，相比 2DGS、PGSR、SparseNeuS 等方法，几何重建更准，同时训练更快。
- 对 BrainDance 而言，这直接支撑“少图/稀疏视角重建”路线。

### DTU 3-view 重建：核心结果
- 设置：DTU 15 个场景，固定 `3` 张视图 `23/24/33` 训练，分辨率 `576x768`。
- Mean Chamfer Distance：
  - `2DGS`：`2.81`
  - `GOF`：`2.82`
  - `PGSR`：`2.08`
  - `CLMVSNet`：`1.26`
  - `SparseNeuS`：`1.27`
  - `ReTR`：`1.17`
  - `Sparse2DGS`：`1.13`
- 相比 `2DGS`：`2.81 -> 1.13`，约下降 `59.8%`
- 相比 `PGSR`：`2.08 -> 1.13`，约下降 `45.7%`
- 相比 `SparseNeuS`：`1.27 -> 1.13`，约下降 `11.0%`

### 同时兼顾渲染与几何
- DTU 3-view 补充实验：
  - `2DGS`：PSNR `16.55`，SSIM `0.601`，LPIPS `0.385`，Avg `2.81`
  - `PGSR`：PSNR `15.30`，SSIM `0.536`，LPIPS `0.409`，Avg `2.08`
  - `DNGaussian`：PSNR `19.09`，SSIM `0.664`，LPIPS `0.390`，Avg `5.68`
  - `Sparse2DGS`：PSNR `17.49`，SSIM `0.726`，LPIPS `0.275`，Avg `1.13`
- 结论：Sparse2DGS 不是单纯追求 PSNR，而是在稀疏视角下同时拿到最好的几何指标与更好的感知质量。

### 对 MVS 初始化可继续增益
- `CLMVSNet`：Avg `1.26`
- `CLMVSNet + Sparse2DGS`：Avg `1.13`
- `TransMVSNet`：Avg `1.11`
- `TransMVSNet + Sparse2DGS`：Avg `1.03`
- 结论：Sparse2DGS 不是简单替代 MVS，而是可以叠加在现有 MVS 初始化上继续提升。

### 训练效率
- `Sparse2DGS`：CD `1.13`，训练 `10 min`
- `SparseNeuS`：CD `1.27`，训练 `25 min`
- `NeuSurf`：CD `0.99`，训练 `10 h`
- 结论：Sparse2DGS 相比 SparseNeuS 约快 `2.5x`，相比 NeuSurf 约快 `60x`，但仍保持很强几何质量。

## 4. SuGaR

论文：SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering  
链接：<https://arxiv.org/abs/2311.12775>

### 当前 ai_engine 对应 pipeline
- `task_type=da3_sugar` / `task_type=da3+sugar` -> `ai_engine/3dgs/src/pipelines/da3_sugar_pipeline.py`
- 模块角色：先用 DA3 做位姿/深度，再由 SuGaR 执行 mesh/SDF 约束训练与网格导出。

### 适合支撑 BrainDance 的结论
- SuGaR 的主要价值不是替代所有 Gaussian 方法，而是提供“高质量可编辑网格提取 + 贴附式高质量渲染”。
- 对 BrainDance 的网格提取、网格精修、面向编辑/导出的链路最有价值。

### 网格提取与精修时间
- 第一阶段优化：`15,000 iterations`，约 `15-45 min/scene`
- 网格提取：通常 `5-10 min/scene`
- 联合精修：`2,000 / 7,000 / 15,000 iterations`，耗时从几分钟到约 `1 h`
- 结论：SuGaR 把“从高斯到可编辑网格”的生产时间压到单场景小时内。

### 带网格渲染质量：Mip-NeRF360
- `R-SuGaR-15K`：平均 PSNR `27.27`，SSIM `0.820`，LPIPS `0.253`
- 对比：
  - `NeRFMeshing`：平均 PSNR `23.15`
  - `Mobile-NeRF`（户外）：PSNR `21.95`，SSIM `0.470`，LPIPS `0.470`
- 结论：在“要求最终输出 mesh”的路线里，SuGaR 的画质明显优于传统网格方案。

### 表面对齐高斯优于传统 UV 贴图
- `1M vertices`：
  - `Surface Gaussians`：PSNR `24.51`，SSIM `0.768`，LPIPS `0.295`
  - `UV`：PSNR `21.24`，SSIM `0.609`，LPIPS `0.478`
- `200K vertices`：
  - `Surface Gaussians`：PSNR `24.24`，SSIM `0.757`，LPIPS `0.300`
  - `UV`：PSNR `21.44`，SSIM `0.656`，LPIPS `0.419`
- 结论：如果 BrainDance 在网格阶段仍希望保留较高可视质量，SuGaR 的“surface Gaussians”路线显著优于直接烘焙 UV。

### 真实场景网格几何：Tanks and Temples
- Chamfer Distance：
  - Barn：`0.2279` vs Instant-NeuS `0.8894`
  - Caterpillar：`0.1611` vs `0.2034`
  - Ignatius：`0.0380` vs `0.0930`
  - Meetingroom：`0.2394` vs `2.7102`
  - Truck：`0.0888` vs `0.2119`
- 结论：在复杂真实场景中，SuGaR 的网格几何稳定性明显更强，尤其适合无 mask、背景复杂的采集数据。

## 5. Depth Anything 3

论文：Depth Anything 3: Recovering the Visual Space from Any Views  
链接：<https://arxiv.org/abs/2511.10647>

### 当前 ai_engine 对应 pipeline
- `task_type=da3_feed_forward_3dgs` -> `ai_engine/3dgs/src/pipelines/da3_feed_forward_pipeline.py`
- `task_type=da3_sugar` -> `ai_engine/3dgs/src/pipelines/da3_sugar_pipeline.py`
- `task_type=da3_2dgs` -> `ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py`
- `task_type=video_3dgs` + `mapper_type=da3` -> `ai_engine/3dgs/src/pipelines/video_3dgs.py`
- 模块角色：作为位姿、深度、统一空间恢复的前置模块，服务多条三维重建流水线。

### 适合支撑 BrainDance 的结论
- DA3 不只是单目深度模型，而是“视图空间恢复”模型：同时解决位姿、深度、点云/空间一致性、新视角合成等问题。
- 对 BrainDance 而言，它最适合作为多视图/视频输入时的深度与空间恢复模块。

### 位姿估计
- Auc3 / Auc30：
  - HiRoom：`DA3-Giant 80.3 / 95.9`，`VGGT 49.1 / 88.0`，`Pi3 67.0 / 94.8`
  - ETH3D：`DA3-Giant 48.4 / 91.2`，`VGGT 26.3 / 80.8`，`Pi3 35.2 / 87.3`
  - ScanNet++：`DA3-Giant 85.0 / 98.1`，`VGGT 62.6 / 95.1`，`Pi3 50.7 / 92.1`
- 结论：DA3 在多数据集上都能更稳定地恢复相机位姿。

### 3D 几何恢复
- ScanNet++ F1：`DA3-Giant 77.0 (w/o p.) / 79.3 (w/ p.)`，`VGGT 66.4 / 70.7`
- HiRoom F1：`85.1 / 95.6`，`VGGT 56.7 / 70.2`
- ETH3D F1：`79.0 / 87.1`，`VGGT 57.2 / 66.7`
- DTU CD(mm)：`DA3-Giant 1.85 / 1.85`，`VGGT 2.05 / 1.44`，`Pi3 3.28 / 1.72`
- 结论：尤其在 `w/o pose` 的设定下，DA3 仍能恢复高质量几何，这很贴近真实视频输入条件。

### 单目深度
- DA3 综合成绩：KITTI `95.3`，NYU `97.4`，SINTEL `75.5`，ETH3D `98.6`，DIODE `95.4`，Rank `2.20`
- 对比：`DA2 Rank 2.60`，`VGGT Rank 3.75`
- 结论：即使退化为单目输入，DA3 也仍是强基线。

### 新视角合成
- DL3DV：`DA3 21.33 / 0.711 / 0.241`，`VGGT 20.96 / 0.697 / 0.253`，`DepthSplat 19.24 / 0.620 / 0.322`
- Tanks and Temples：`DA3 18.10 / 0.578 / 0.311`，`VGGT 17.18 / 0.550 / 0.347`
- MegaDepth：`DA3 17.89 / 0.561 / 0.351`，`VGGT 16.45 / 0.500 / 0.417`
- 指标顺序均为 `PSNR / SSIM / LPIPS`
- 结论：DA3 作为几何 backbone，不仅恢复空间更准，还能带来更好的重渲染结果。

### 工程可用性
- 最大可处理图像数 / 速度：
  - `DA3-Base`：`2100-2200` 张，`126.5 FPS`
  - `DA3-Large`：`1500-1600` 张，`78.37 FPS`
  - `DA3-Giant`：`900-1000` 张，`37.6 FPS`
  - `VGGT`：`400-500` 张，`34.1 FPS`
- 结论：DA3 在长视频和多视图输入下更具工程可扩展性。

## 6. SHARP / ml-sharp

论文：Sharp Monocular View Synthesis in Less Than a Second  
链接：<https://arxiv.org/abs/2512.10685>

### 当前 ai_engine 对应 pipeline
- `task_type=single_image_sharp` -> `ai_engine/3dgs/src/pipelines/single_image_sharp.py`
- 模块角色：单图快速生成 3D Gaussian 结果。

### 适合支撑 BrainDance 的结论
- SHARP 的核心优势是：单图输入下，跨数据集新视角合成质量显著领先，同时满足亚秒级推理需求。
- 这直接对应 BrainDance 的“单图快速生成 3D Gaussian 结果”能力。

### 跨数据集零样本新视角合成
- 论文在 6 个测试集上报告 DISTS / LPIPS，均为越低越好。
- `SHARP`：
  - Middlebury：`0.097 / 0.358`
  - Booster：`0.119 / 0.270`
  - ScanNet++：`0.071 / 0.154`
  - WildRGBD：`0.069 / 0.190`
  - Tanks and Temples：`0.122 / 0.421`
  - ETH3D：`0.258 / 0.554`
- 对比 `Gen3C`：
  - Middlebury：`0.164 / 0.545`
  - Booster：`0.207 / 0.384`
  - ScanNet++：`0.090 / 0.227`
  - WildRGBD：`0.106 / 0.285`
  - Tanks and Temples：`0.177 / 0.566`
  - ETH3D：`0.408 / 0.734`
- 结论：SHARP 在各测试集上都拿到最优或显著更优的感知质量，说明它很适合作为单图快速 3D/多视角生成前端。

### 论文对速度的直接定位
- 论文标题即强调 `Less Than a Second`。
- 正文还指出其感知损失设计在提升 DISTS 的同时，可进一步降低渲染延迟；附录中给出该类设计带来 `49%` 和 `36%` 的 latency reduction。
- 结论：SHARP 的定位不是高耗时优化式重建，而是近实时单图视图生成。

## 7. SAM 3D Objects

论文：SAM 3D: 3Dfy Anything in Images  
链接：<https://arxiv.org/abs/2511.16624>

### 当前 ai_engine 对应 pipeline
- `task_type=single_image_sam3d` -> `ai_engine/3dgs/src/pipelines/single_image_sam3d.py`
- 模块角色：单图 / 单对象导向的 3D 生成流水线。

### 适合支撑 BrainDance 的结论
- SAM 3D 的证据分为两部分：单图 3D 物体生成质量，以及单图场景中物体布局恢复质量。
- 它直接支撑 BrainDance 的“单图/少图 3D 物体生成”和“对象级 3D 场景恢复”能力。

### 单物体 3D 形状质量
- SA-3DAO / ISO3D 指标：F1@0.01、vIoU、Chamfer、EMD、ULIP、Uni3D
- `SAM 3D`：
  - F1@0.01 `0.2344`
  - vIoU `0.2311`
  - Chamfer `0.0400`
  - EMD `0.1211`
  - ULIP `0.1488`
  - Uni3D `0.3707`
- 对比方法：
  - `Hi3DGen`：F1 `0.1629`，vIoU `0.1531`，Chamfer `0.0937`
  - `TripoSG`：F1 `0.1533`，vIoU `0.1445`，Chamfer `0.0844`
  - `HY3D-2.0`：F1 `0.1574`，vIoU `0.1504`，Chamfer `0.0866`
- 结论：SAM 3D 在物体几何精度上明显强于同期 image-to-3D 方法。

### 单图场景布局恢复
- SA-3DAO：
  - `Joint SAM 3D`：3D IoU `0.4254`，ICP-Rot `20.7667`，ADD-S `0.2661`，ADD-S@0.1 `0.7232`
  - `Pipeline HY3D-2.0 + FoundationPose`：`0.2937 / 32.9444 / 0.3705 / 0.5396`
  - `Pipeline Trellis + Megapose`：`0.2449 / 39.3866 / 0.5391 / 0.2831`
- ADT：
  - `Joint SAM 3D`：3D IoU `0.4970`，ICP-Rot `15.2515`，ADD-S `0.0765`，ADD-S@0.1 `0.7673`
  - `Pipeline SAM 3D + FoundationPose`：`0.3661 / 18.9102 / 0.0930 / 0.6495`
  - `Joint MIDI`：`0.0336 / 44.2353 / 2.5278 / 0.0175`
- 结论：SAM 3D 不只是把物体单独重建得更好，还能在单图场景里恢复更准确的对象布局。

### 人类偏好结果
- 单物体与场景重建上，论文报告相对既有 SOTA 至少达到 `5:1` 和 `6:1` 的人类偏好胜率。
- 这类结果虽然不是传统几何指标，但对产品观感非常有参考意义。

### 多阶段训练的量化收益
- 形状质量随训练阶段逐步提升：
  - Pre-training：F1 `0.1349`，vIoU `0.1202`，Chamfer `0.1036`，EMD `0.2396`
  - + Mid-training：F1 `0.1705`，vIoU `0.1683`，Chamfer `0.0760`，EMD `0.1821`
  - + SFT(MITL-3DO)：F1 `0.2027`，vIoU `0.2025`，Chamfer `0.0578`，EMD `0.1510`
  - + DPO(MITL-3DO)：F1 `0.2156`，vIoU `0.2156`，Chamfer `0.0498`，EMD `0.1367`
  - + SFT(Art-3DO)：F1 `0.2331`，vIoU `0.2337`，Chamfer `0.0445`，EMD `0.1257`
  - 最终模型：F1 `0.2344`，vIoU `0.2311`，Chamfer `0.0400`，EMD `0.1211`
- 结论：SAM 3D 的性能提升不是偶然，而是由完整训练链条逐步堆出来的。

## 8. Segment Anything in 3D Scenes

论文：SAM3D: Segment Anything in 3D Scenes  
链接：<https://arxiv.org/abs/2306.03908>

### 当前 ai_engine 对应 pipeline
- 当前 `ai_engine/3dgs/src/core/factory.py` 中没有独立 `task_type` 直接映射到这篇论文的实现。
- 在当前仓库中，它更适合作为 `SAM 3D Objects` 与 3D 分割方向的技术脉络引用，而不是一条独立可调度生产流水线。

### 适合支撑 BrainDance 的结论
- 这篇论文更适合作为“3D 分割方向脉络引用”，不适合作为 BrainDance 量化优越性的主要证据来源。
- 论文重点在于提出一个无需训练/微调 SAM 的 3D mask 投影与融合框架。

### 定量证据情况
- 论文未提供足够系统的标准 benchmark 数值表，不能像 2DGS、Sparse2DGS、SuGaR 那样支撑强定量比较。
- 文中更适合引用其工程路径：`2D SAM mask -> 多视角投影 -> 3D mask 合并 -> 可选几何 over-segmentation 融合`。

## 9. 对 BrainDance 技术路线的综合结论

从论文定量证据看，BrainDance 当前方案的优势可以概括为：

- `Nerfstudio` 提供高效、稳定、模块化的 3DGS 训练底座。
- `2DGS` 提供显著更强的几何一致性，适合需要可靠表面的主干重建。
- `Sparse2DGS` 在少图/稀疏视角条件下显著优于原始 2DGS 和多种 sparse-view 方法。
- `SuGaR` 负责把高斯表示稳定转成高质量、可编辑网格，并保持较好的渲染质量。
- `Depth Anything 3` 提供多视图/视频场景下的深度、位姿与统一空间恢复能力。
- `SHARP` 提供单图输入到新视角结果的快速生成能力，适合低门槛和高响应场景。
- `SAM 3D` 强化了单图对象级 3D 生成和对象布局恢复能力。

整体上，这条路线不是押注单一论文，而是把“快速建模、几何质量、稀疏视角、网格导出、单图生成、多视图空间恢复”分别交给最适合的模块，因此在能力覆盖面和工程可落地性上更强。
