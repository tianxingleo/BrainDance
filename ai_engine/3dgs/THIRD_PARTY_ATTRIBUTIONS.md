# BrainDance 3DGS 第三方依赖与论文引用

本文档整理 `ai_engine/3dgs/src/libs/` 中核心第三方项目的用途、官方仓库、论文、引用与许可证信息，便于：

- 补充 README 致谢与参考文献
- 核对上游项目许可证
- 准备比赛材料、答辩材料和论文引用清单

## 使用说明

- 本文档优先以当前仓库中实际集成和调用的项目为准。
- 若本地子模块工作树不完整，则补充参考其官方仓库公开信息。
- 具体使用时，请同时遵守各项目代码许可证、模型许可证和非商业限制。

## 核心依赖清单

### Nerfstudio

**Paper Title**  
[2302.04264] Nerfstudio: A Modular Framework for Neural Radiance Field Development

**Paper**  
arXiv:2302.04264  
Submitted: 2023-02-08  
Last revised: 2023-10-17  
Link: <https://arxiv.org/abs/2302.04264>

**Project**  
Role in BrainDance: 默认视频 3DGS 训练与 `splatfacto` 管线基础框架  
Local path: `ai_engine/3dgs/src/libs/nerfstudio`  
Official repo: <https://github.com/nerfstudio-project/nerfstudio>

**Local docs**  
- `src/libs/nerfstudio/README.md`
- `src/libs/nerfstudio/LICENSE`

**Citation / License**  
- README 中包含论文链接和 BibTeX  
- License: Apache 2.0

### 2D Gaussian Splatting

**Paper Title**  
[2403.17888] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

**Paper**  
arXiv:2403.17888  
Submitted: 2024-03-26  
Last revised: 2025-02-22  
Link: <https://arxiv.org/abs/2403.17888>

**Project**  
Role in BrainDance: 2DGS 表达、几何约束与相关渲染/训练能力  
Local path: `ai_engine/3dgs/src/libs/2d-gaussian-splatting`  
Official repo: <https://github.com/hbb1/2d-gaussian-splatting>

**Local docs**  
- `src/libs/2d-gaussian-splatting/README.md`
- `src/libs/2d-gaussian-splatting/LICENSE.md`

**Citation / License**  
- README 顶部提供 Paper 链接  
- README 包含 `Citation` / BibTeX  
- README 包含 `Acknowledgements`  
- License: Gaussian-Splatting License，偏非商业研究用途

### Sparse2DGS

**Paper Title**  
[2504.20378] Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views

**Paper**  
arXiv:2504.20378  
Submitted: 2025-04-29  
Last revised: 2025-04-29  
Link: <https://arxiv.org/abs/2504.20378>

**Project**  
Role in BrainDance: 少图 / 稀疏视角下的 2DGS 重建路线  
Local path: `ai_engine/3dgs/src/libs/Sparse2DGS`  
Official repo: <https://github.com/Wuuu3511/Sparse2DGS>  
Integrated fork: <https://github.com/tianxingleo/Sparse2DGS>

**Local docs**  
- `src/libs/Sparse2DGS/README.md`
- `src/libs/Sparse2DGS/MVS/CLMVSNet/README.md`
- `src/libs/Sparse2DGS/submodules/diff-surfel-rasterization/README.md`

**Citation / License**  
- README 包含 arXiv 链接  
- README 包含 `Citation` / BibTeX  
- README 包含 `Acknowledgments`  
- README 包含 `License` 说明  
- 主项目 README 声明沿用原始 Sparse2DGS 仓库许可  
- 内含 `diff-surfel-rasterization` 等子依赖，需分别遵守其许可证

### SuGaR

**Paper Title**  
[2311.12775] SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

**Paper**  
arXiv:2311.12775  
Submitted: 2023-11-21  
Last revised: 2023-12-02  
Link: <https://arxiv.org/abs/2311.12775>

**Project**  
Role in BrainDance: 网格提取、基于表面对齐的高质量 Gaussian 优化与精修  
Local path: `ai_engine/3dgs/src/libs/SuGaR`  
Official repo: <https://github.com/Anttwo/SuGaR>  
Integrated fork: <https://github.com/tianxingleo/SuGaR>

**Local docs**  
- `src/libs/SuGaR/README.md`
- `src/libs/SuGaR/LICENSE.md`

**Citation / License**  
- README 顶部提供 arXiv  
- README 包含 BibTeX  
- README 包含 `Acknowledgments`  
- README 包含 `License`  
- License: Gaussian-Splatting License，偏非商业研究用途

**Notes**  
- 内嵌 `gaussian_splatting`、`SIBR_viewers`、`diff-gaussian-rasterization` 等上游代码  
- 使用时需一并遵守这些子目录里的许可证

### Depth Anything 3

**Paper Title**  
[2511.10647] Depth Anything 3: Recovering the Visual Space from Any Views

**Paper**  
arXiv:2511.10647  
Submitted: 2025-11-13  
Last revised: 2025-11-13  
Link: <https://arxiv.org/abs/2511.10647>

**Project**  
Role in BrainDance: 深度估计 / 视图空间恢复，服务于部分增强型三维流水线  
Local path: `ai_engine/3dgs/src/libs/Depth-Anything-3`  
Official repo: <https://github.com/ByteDance-Seed/Depth-Anything-3>

**Local docs**  
- 当前工作树未看到顶层 README / LICENSE  
- 子目录 `da3_streaming/loop_utils/salad` 有独立 README / LICENSE

**Citation / License**  
- 官方 README 包含论文链接和 Citation  
- 官方仓库声明代码 Apache 2.0  
- 官方模型卡区分 Apache 2.0 与 CC BY-NC 4.0 等约束

**Notes**  
- 对外文档建议同时明确“代码许可”和“模型许可”可能不同

### SHARP / ml-sharp

**Paper Title**  
[2512.10685] Sharp Monocular View Synthesis in Less Than a Second

**Paper**  
arXiv:2512.10685  
Submitted: 2025-12-11  
Last revised: 2026-02-27  
Link: <https://arxiv.org/abs/2512.10685>

**Project**  
Role in BrainDance: 单图快速生成 3D Gaussian 结果  
Local path: `ai_engine/3dgs/src/libs/ml-sharp`  
Official repo: <https://github.com/apple/ml-sharp>

**Local docs**  
- 当前工作树未完整展开顶层 README / LICENSE  
- `src/sharp.egg-info/PKG-INFO` 中已包含 README 主要内容

**Citation / License**  
- 本地元数据中可确认论文、arXiv、`Citation` / BibTeX、`Acknowledgements`  
- 明确提到 `LICENSE` 与 `LICENSE_MODEL`

**Notes**  
- 对外文档使用官方仓库和论文链接  
- 代码许可证与模型许可证需分别查看

### SAM 3D Objects

**Paper Title**  
[2511.16624] SAM 3D: 3Dfy Anything in Images

**Paper**  
arXiv:2511.16624  
Submitted: 2025-11-20  
Last revised: 2025-11-20  
Link: <https://arxiv.org/abs/2511.16624>

**Project**  
Role in BrainDance: 单图 / 少图 3D 物体生成相关实验与推理能力  
Local path: `ai_engine/3dgs/src/libs/sam-3d-objects`  
Official repo: <https://github.com/facebookresearch/sam-3d-objects>

**Local docs**  
- 当前工作树未看到 README / LICENSE / CITATION

**Citation / License**  
- 官方 README 包含 Citation  
- 官方 README 包含 Contributors / 致谢  
- 官方 README 声明 SAM License

**Notes**  
- 项目文档引用官方仓库与论文即可  
- 若后续恢复完整子模块，建议补齐本地文档文件

## 补充引用

### Segment Anything in 3D Scenes

**Paper Title**  
[2306.03908] SAM3D: Segment Anything in 3D Scenes

**Paper**  
arXiv:2306.03908  
Submitted: 2023-06-06  
Last revised: 2023-06-06  
Link: <https://arxiv.org/abs/2306.03908>

**Project**  
Role in BrainDance: 如果项目材料中提到 “SAM3D” 的早期三维分割脉络，可同时引用此工作  
Official repo: <https://github.com/Pointcept/SegmentAnything3D>

**Citation / License**  
- README 包含 Citation  
- 仓库使用 MIT License

## 在 BrainDance 中的推荐写法

### 根 README

适合保留简版致谢，只列核心上游项目：

- Nerfstudio
- 2D Gaussian Splatting
- Sparse2DGS
- SuGaR
- Depth Anything 3
- SHARP
- SAM 3D Objects
- Supabase

### ai_engine/3dgs/README.md

适合保留“模块级说明”，强调：

- 本目录依赖多个第三方研究项目与子模块
- 详细论文引用、许可证与致谢见本文件

### 许可证风险提示

以下项目应特别注意非商业或额外模型许可限制：

- 2D Gaussian Splatting
- SuGaR
- SHARP（代码许可与模型许可分离）
- Depth Anything 3（代码许可与模型许可可能不同）
- SAM 3D Objects（需看官方 SAM License）

## 维护建议

- 每次更新 `src/libs/` 子模块后，同步检查：
  - 是否新增或变更 `LICENSE`
  - 是否新增 `CITATION.cff`
  - 是否更新论文链接
  - 是否新增模型许可证
- 如果某子模块本地工作树不完整，优先以官方仓库 README 为准补文档
