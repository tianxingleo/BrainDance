# 🧠 BrainDance AI Engine

> **面向空间计算时代的智能3D重建与语义搜索引擎**

本引擎是BrainDance项目的核心计算后端，基于3D高斯泼溅（3D Gaussian Splatting）技术，结合多模态AI和向量数据库，实现了从视频输入到语义搜索的完整AI流水线。

## 🚀 项目概述

BrainDance AI Engine是一个时空记忆引擎，致力于通过AI技术将物理世界的光场信息转化为可永久保存的数字3D资产。我们使用最先进的3DGS技术，结合Qwen-VL等多模态AI模型，实现视频到3D模型的高质量转换，并通过语义向量检索实现自然语言查询。

### 核心价值
- **个人记忆保存**：将珍贵的空间记忆数字化，对抗时间熵增
- **智能检索能力**：通过自然语言搜索3D场景中的特定物体
- **云端分布式处理**：支持大规模3D重建任务的自动化处理
- **医疗辅助应用**：为阿尔茨海默症患者提供怀旧疗法支持

## 🌟 核心特性

### AI驱动的3D重建
- 视频输入自动转换为高质量3D高斯泼溅模型
- 集成COLMAP/GLOMAP进行精确位姿估计
- 支持多种场景类型的自适应优化

### 单图3DGS生成
- 基于SAM3D的单张照片3D重建，无需视频
- 支持自定义Mask输入实现精确控制
- 内嵌推理库，零外部依赖
- 智能抠图：YOLO World / SAM 2.1 / Simple 降级

### 语义理解与搜索
- 集成Qwen-VL多模态AI进行场景理解
- 自动生成场景描述、物体清单和标签
- 基于pgvector的语义向量检索系统

### 云端任务调度
- 基于Supabase的消息队列系统
- 分布式任务处理与负载均衡
- 实时任务进度监控与日志同步

### 智能质检系统
- AI前置质量评估与自动过滤
- 场景完整性检测与重建成功率预测
- 自动参数优化与错误恢复

## 🏗️ 系统架构

```
[输入] → 根据 task_type 分支
   │
   ├── video_3dgs ─→ 图像预处理 → AI场景分析 → 位姿解算 → 3DGS训练 → 模型输出
   │      ↓                           ↓              ↓              ↓
   │      ↓                      [质量评估]    [智能清洗]    [参数优化]
   │
   └── single_image_sam3d ─→ 智能抠图 → SAM3D推理 → 模型输出
           ↓                       ↓            ↓
           ↓                 [YOLO/SAM]   [Stage1+Stage2]
```

### 架构组件
- **任务调度层**：Supabase数据库与实时消息队列，支持多任务类型
- **AI处理层**：多模态AI模型、3DGS训练引擎、SAM3D单图引擎
- **存储层**：向量数据库与模型资产管理系统
- **接口层**：REST API与WebSocket实时通信

## 🛠️ 技术栈

### AI与3D框架
- **3D重建**：Nerfstudio, Gaussian Splatting, Splatfacto
- **位姿估计**：COLMAP, GLOMAP
- **AI模型**：Qwen-VL, SAM 2.1, YOLO World, text-embedding-v2
- **单图重建**：SAM3D, SHARP, DINOv2

### 数据库与存储
- **向量数据库**：Supabase + pgvector
- **文件存储**：Supabase Storage
- **消息队列**：Supabase Realtime

### 开发环境
- **编程语言**：Python 3.10+
- **CUDA支持**：CUDA 11.8+ / 12.x
- **依赖管理**：Conda, pip

## 📋 安装与部署

### 环境要求
- **GPU**：NVIDIA RTX 30/40/50系列 (推荐12GB+显存，测试显卡: RTX5070)
- **操作系统**：Linux (Ubuntu 22.04) / Windows WSL2
- **CUDA版本**：11.8 或 12.x

### 快速开始

1. **克隆项目**
```bash
git clone https://github.com/tianxingleo/BrainDance.git
cd BrainDance/ai_engine/3dgs
```

<<<<<<< HEAD
=======
> ⚠️ **重要**：所有后续命令都在 `ai_engine/3dgs` 目录下执行

>>>>>>> origin/tianxingleo-da3
2. **创建conda环境**
```bash
conda create -n braindance python=3.10
conda activate braindance
```

3. **安装PyTorch (根据CUDA版本)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **安装项目依赖**
```bash
pip install -r requirements.txt
```

或者逐个安装：
```bash
pip install nerfstudio supabase python-dotenv openai ultralytics plyfile opencv-python dashscope
```

5. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入Supabase和AI服务的API密钥
```

必要配置项：
- `SUPABASE_URL`: Supabase项目URL
- `SUPABASE_KEY`: Supabase服务角色密钥
- `DASHSCOPE_API_KEY`: 阿里云DashScope API密钥（用于Qwen-VL和embedding）
- `MAX_IMAGES`: 单次处理最大图片数量（默认500）
- `TRAINING_ITERATIONS`: 训练迭代次数（默认15000）
- `MIN_QUALITY_SCORE`: AI质检最低分（默认40）

### Supabase配置
1. 启动本地Supabase环境
2. 配置`processing_tasks`表用于任务队列
3. 配置`model_assets`表用于向量存储
4. 创建`braindance-assets`存储桶

### AI模型下载
首次运行程序时，所需AI模型会自动从HuggingFace下载到 `~/braindance_workspace/models/` 目录，包括：
- YOLO World 模型
- SAM 2.1 模型

## 🎮 运行模式

### 本地视频处理模式
```bash
<<<<<<< HEAD
# 直接处理本地视频文件
=======
# 在 ai_engine/3dgs 目录下执行
>>>>>>> origin/tianxingleo-da3
python main.py /path/to/your/video.mp4
```

### 单图本地测试模式
```bash
<<<<<<< HEAD
# 使用 SAM3D 处理单张图片
=======
# 在 ai_engine/3dgs 目录下执行
>>>>>>> origin/tianxingleo-da3
python tests/test_local_single_image.py

# 或指定图片路径
python tests/test_local_single_image.py --file /path/to/image.png
```

### 云端监听模式
```bash
<<<<<<< HEAD
# 监听Supabase任务队列
=======
# 在 ai_engine/3dgs 目录下执行
>>>>>>> origin/tianxingleo-da3
python main.py
```

### 批处理模式
```bash
<<<<<<< HEAD
# 处理多个视频文件
=======
# 在 ai_engine/3dgs 目录下执行
>>>>>>> origin/tianxingleo-da3
python main.py --batch-mode
```

## 📖 使用指南

### 基础用法
```python
from pathlib import Path
from src.config import PipelineConfig
from src.core.pipeline import run_pipeline

# 配置处理参数
cfg = PipelineConfig(
    project_name="my_scene",
    video_path=Path("input.mp4"),
    max_images=500,
    training_iterations=15000,
    enable_ai=True,
    min_quality_score=40
)

# 执行3D重建
result, metadata = run_pipeline(cfg)

# 输出结果
print(f"3D模型路径: {result}")
print(f"AI分析结果: {metadata}")
```

### 单图 SAM3D 用法
```python
from src.core.factory import PipelineFactory

# 创建上下文
context = {
    "task_id": "scene_001",
    "scene_id": "scene_001",
    "work_root": "./output",
    "log_callback": lambda msg: print(f"[LOG] {msg}"),
}

# 获取 SAM3D Pipeline
pipeline = PipelineFactory.get_pipeline("single_image_sam3d", context)

# 执行单图生成
ply_path, metadata = pipeline.run("image.png", {
    "mask_path": "mask.png"  # 可选：指定Mask
})

print(f"3D模型: {ply_path}")
print(f"元数据: {metadata}")
```

### 云端任务处理
```python
from src.core.worker import CloudWorker

# 启动云端监听器
worker = CloudWorker()
worker.start()  # 持续监听Supabase任务队列
```

### 高级配置
- `max_images`: 控制输入图片数量，影响重建精度
- `training_iterations`: 训练迭代次数，平衡质量和时间
- `enable_ai`: 是否启用AI增强功能
- `min_quality_score`: AI质检最低分数阈值

### 云端任务管理
1. 通过前端App或API向`processing_tasks`表插入任务
2. Worker自动监听并处理`status=pending`的任务
3. 实时更新任务状态和日志
4. 处理完成后上传PLY模型到Storage

## 🔌 API接口

### 任务创建
```http
POST /api/tasks
Content-Type: application/json

{
  "scene_id": "unique_scene_identifier",
  "user_id": "user_identifier",
  "status": "pending"
}
```

### 任务状态查询
```http
GET /api/tasks/{task_id}
Response: {status, logs, progress, result_url}
```

### 语义搜索
```http
GET /api/search?q=red cup&start=2025-01-01&end=2025-12-31
Response: {scene_id, description, similarity_score, model_url}
```

## 📁 项目结构

```
ai_engine/
├── 3dgs/                   # 3DGS核心引擎
│   ├── main.py            # 程序入口
│   ├── search_demo.py     # 语义搜索演示
│   ├── requirements.txt   # 依赖配置
│   ├── .env.example       # 环境变量模板
│   ├── src/               # 源代码
│   │   ├── config.py      # 配置管理
│   │   ├── core/          # 核心业务逻辑
│   │   │   ├── pipeline.py  # 3DGS处理流水线
│   │   │   ├── pipeline_base.py  # Pipeline基类
│   │   │   ├── worker.py    # 云端任务处理器
│   │   │   └── factory.py   # Pipeline工厂
│   │   ├── pipelines/      # Pipeline实现
│   │   │   ├── video_3dgs.py      # 视频3DGS Pipeline
│   │   │   ├── single_image_sam3d.py  # 单图SAM3D Pipeline
│   │   │   └── image_to_3d.py    # 多图Pipeline
│   │   ├── modules/       # 功能模块
│   │   │   ├── image_proc.py      # 图像预处理
│   │   │   ├── glomap_runner.py   # 位姿解算
│   │   │   ├── scene_analyzer.py  # 场景分析
│   │   │   ├── ai_segmentor.py    # AI分割
│   │   │   ├── nerf_engine.py     # 3D训练引擎
│   │   │   ├── knowledge_base.py  # 知识库
│   │   │   └── sam3d_engine/      # SAM3D引擎模块
│   │   │       ├── core.py        # SAM3DEngine主引擎
│   │   │       ├── masking.py     # 智能抠图
│   │   │       ├── mocks.py       # RTX50兼容层
│   │   │       └── memory.py      # CPU内存加载
│   │   ├── libs/          # 内嵌依赖库
│   │   │   └── sam-3d-objects/    # SAM3D推理库
│   │   └── utils/         # 工具函数
│   │       ├── common.py      # 通用工具
│   │       ├── cv_algorithms.py  # CV算法
│   │       ├── geometry.py      # 几何计算
│   │       └── ply_utils.py     # PLY处理
│   └── tests/             # 测试脚本
│       ├── test_local_single_image.py  # 单图测试
│       └── test_pipeline_cases.py      # 单元测试
└── demo/                  # 演示脚本
    ├── process_3dgs.py    # 3D处理脚本
    └── results/           # 输出结果
```

## 🧩 模块详解

### ImageProcessor
- **功能**：图像预处理与质量清洗
- **算法**：混合策略模糊检测、拉普拉斯算子分析
- **优化**：均匀采样降维、智能去噪

### GlomapRunner  
- **功能**：相机位姿解算与轨迹优化
- **流程**：特征提取→匹配→全局重建→结构修正
- **特点**：支持COLMAP和GLOMAP双引擎

### SceneAnalyzer
- **功能**：AI场景质量评估与元数据生成
- **模型**：Qwen-VL-Max多模态大模型
- **输出**：质量分数、场景描述、物体清单

### AISegmentor
- **功能**：语义分割与主要物体提取
- **技术栈**：YOLO World + SAM 2.1
- **优化**：中心物体优先、Mask清洗验证

### NerfEngine
- **功能**：3DGS模型训练与导出
- **算法**：Splatfacto训练、自适应Collider计算
- **后处理**：点云切割、质量优化

### SAM3DEngine
- **功能**：基于SAM3D的单图3DGS生成
- **流程**：智能抠图 → Stage1结构生成 → Stage2高斯生成
- **优化**：
  - 显存保护：强制权重加载至CPU RAM
  - 图片自动降采样（最大400px）
  - 分阶段GPU切换（Stage1/Stage2）
- **支持**：自定义Mask输入

### SharpEngine
- **功能**：基于SHARP的单图3DGS生成
- **流程**：封装 sharp predict 命令行工具
- **特性**：
  - 子进程调用管理
  - 自动GPU设备分配 (CUDA_VISIBLE_DEVICES=0)
  - 自动查找生成的 .ply 文件
- **输入**：单张图片 (image.png)
- **输出**：3DGS 模型 (model.ply)

### MaskGenerator
- **功能**：智能抠图生成
- **支持**：YOLO World / SAM 2.1 / Simple 降级
- **特性**：自动检测模型可用性
- **降级策略**：优先使用YOLO World，失败则SAM 2.1，最后Simple

### KnowledgeBase
- **功能**：RAG向量存储与语义检索
- **技术**：text-embedding-v2、pgvector索引
- **策略**：加权文本、混合搜索

### CloudWorker
- **功能**：云端任务队列监听与处理
- **架构**：生产者-消费者模式
- **特性**：实时日志同步、错误恢复、任务类型路由

## ⚡ 性能优化

### 硬件加速
- **GPU优化**：CUDA内存管理和显存分配
- **并行计算**：多任务并发执行
- **缓存策略**：模型权重共享与预加载

### 网络优化
- **传输效率**：视频压缩与分块上传
- **CDN集成**：模型文件加速分发
- **断点续传**：大文件传输可靠性保障

### 内存管理
- **垃圾回收**：临时文件自动清理
- **资源监控**：实时资源使用追踪
- **批处理优化**：内存复用与预分配

## 🐛 故障排除

### 常见问题
- **GPU兼容性**：CUDA版本不匹配或驱动问题
- **环境配置**：依赖库版本冲突
- **权限问题**：API密钥未正确配置

### 调试技巧
- **日志分析**：查看Supabase任务日志
- **进度监控**：实时任务状态跟踪
- **资源监控**：GPU显存与CPU使用情况

### 性能瓶颈
- **显存不足**：降低输入图片分辨率或减少max_images
- **处理缓慢**：检查网络连接与API限流
- **重建失败**：验证视频质量与运动轨迹

## 🤝 开发贡献

### 代码规范
- **模块化设计**：功能分离与单一职责原则
- **类型注解**：完整的类型提示支持
- **文档字符串**：详细的函数与类说明

### 测试方案
- **单元测试**：各模块功能验证
- **集成测试**：端到端流水线测试
- **性能测试**：处理速度与质量基准

### 贡献流程
1. Fork项目并创建功能分支
2. 实现功能并添加相应测试
3. 提交PR并等待代码审查
4. 修复问题并合并代码

## 🌍 应用场景

### 个人记忆保存
- 宿舍/租房搬迁前的空间数字化
- 家庭聚会与重要时刻的3D记录
- 旅行景点的沉浸式回忆保存

### 医疗辅助
- 阿尔茨海默症患者的怀旧疗法
- 熟悉环境的VR重现与情感支持
- 认知康复训练的3D场景

### 文化遗产保护
- 即将拆迁的街区与建筑数字化
- 历史场所的虚拟重建与展示
- 城市变迁的3D时间轴记录

### 教育科研
- 3D教学场景的快速创建
- 科研数据的可视化展示
- 虚拟实验室的构建与共享

## 🚀 未来发展

### 功能路线图
- **AI能力增强**：更精确的场景理解和物体识别
- **XR设备集成**：原生支持Vision Pro、Quest等头显
- **社交功能**：多人协作的空间编辑与分享

### 性能优化
- **处理速度**：分钟级3D重建能力
- **重建质量**：更精细的纹理与几何表现
- **资源消耗**：更低的硬件要求与更快的处理

### 生态扩展
- **插件系统**：第三方功能模块支持
- **API开放**：开发者友好的接口服务
- **云服务**：商业化SaaS平台建设

## 📞 社区与支持

### 文档资源
<<<<<<< HEAD
- [技术文档](docs/)
- [API参考](docs/API_DOC.md)  
- [部署指南](docs/deployment_guide.md)
=======
- [快速开始指南](docs/快速开始指南.md) - 30 分钟上手教程
- [API 参考文档](docs/API_DOC.md) - 完整接口说明
- [本地部署指南](docs/LOCAL_DEPLOYMENT.md) - 开发环境配置
- [SAM3D 模型设置](docs/SAM3D_MODEL_SETUP.md) - 模型下载配置
- [开发环境配置](docs/开发环境配置.md) - 详细环境搭建
>>>>>>> origin/tianxingleo-da3

### 交流渠道
- GitHub Issues: bug报告与功能建议
- Discord: 开发者社区交流
- 邮箱: tianxingleo@gmail.com

### 版本更新
- **发布周期**：每月定期更新
- **变更日志**：详细的版本说明
- **兼容性**：向后兼容保证

---

<div align="center">

**Made with ❤️ by the BrainDance Team**  
*"物理世界注定走向无序，而我们在比特世界重建永恒。"*

</div>