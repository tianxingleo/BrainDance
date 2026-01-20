# 项目：BrainDance | 流光·记

## 技术栈

### AI 引擎 (Python Worker)
- **核心框架**：Python 3.10+ / CUDA 11.8+ / 12.x
- **3DGS 框架**：Nerfstudio / gsplat / gaussian-splatting
- **单图重建**：SAM 3D / SHARP
- **多模态 AI**：Qwen-VL (场景理解与自动打标)
- **位姿估计**：GLOMAP / COLMAP
- **向量检索**：pgvector (PostgreSQL 扩展)

### 前端
- **移动端**：Flutter 3.10+ (正在开发中)
- **Web 3D 查看器**：Vue 3 + TypeScript + Vite + Three.js
- **3DGS 渲染库**：@mkkellogg/gaussian-splats-3d

### 后端 (BaaS)
- **数据库**：Supabase (PostgreSQL + pgvector)
- **存储**：Supabase Storage (Raw 视频 / 输出模型)
- **认证与安全**：Supabase Auth / RLS (行级安全策略)
- **实时通信**：Supabase Realtime (WebSocket)
- **边缘函数**：Deno Deploy (语义搜索接口)

## 代码规范

### Python 代码风格
- **文件头注释**：使用中文描述功能、实现逻辑和流程
  ```python
  # src/core/pipeline.py
  # 功能：实现3DGS生成主流水线
  # 实现：按顺序调用各个功能模块
  # 逻辑：1. 视频抽帧与预处理 2. AI质检...
  ```
- **导入顺序**：1) 配置模块 2) 功能模块 3) 工具模块
- **日志规范**：使用带时间戳的回调函数输出进度
- **配置管理**：使用 `dataclass` + `field(default_factory=lambda: os.getenv(...))` 模式
- **架构模式**：工厂模式、模块化流水线设计

### 数据库规范
- **任务队列**：`processing_tasks` 表，状态流转为 pending → processing → completed
- **资产存储**：`model_assets` 表，使用 pgvector 存储语义向量
- **存储路径**：`{user_id}/{scene_id}/{raw,output}/...` 格式
- **安全策略**：所有表启用 RLS，确保用户数据隔离

### Git 规范
- **分支策略**：`main` (保护分支) → `dev` (集成分支) → `feat/<名称>/<功能>`
- **提交信息**：中文描述，格式为 `类型(范围): 主题`
- **提交类型**：feat (新功能)、fix (修复)、docs (文档)、refactor (重构)、chore (维护)
- **禁止操作**：严禁执行 `git push` (已在 `.opencode.json` 中禁用)

### 安全规范
- **环境变量**：所有密钥通过 `.env` 文件管理 (已加入 `.gitignore`)
- **硬编码禁止**：严禁在代码中硬编码 API Key，使用 `os.getenv()` 获取

## 目录结构

```
BrainDance/
├── ai_engine/            # [Python] 核心算法引擎 (Worker)
│   ├── 3dgs/             #   - 3DGS 核心引擎
│   │   ├── src/          #   - 源代码
│   │   │   ├── core/         #       - Pipeline 基类、工厂、Worker
│   │   │   ├── pipelines/    #       - Pipeline 实现
│   │   │   ├── modules/      #       - 功能模块 (SAM3D/NeRF/GLOMAP/知识库)
│   │   │   ├── libs/         #       - 内嵌依赖库
│   │   │   └── utils/        #       - 工具函数
│   │   ├── requirements.txt  #   - Python 依赖
│   │   └── main.py           #   - 程序入口
│   ├── demo/              #   - 演示脚本与测试数据
│   ├── models/            #   - AI 模型缓存目录
│   └── log/               #   - 日志文件
│
├── supabase/              # [BaaS] 云端基础设施
│   ├── migrations/        #   - SQL 数据库结构变更历史
│   ├── seed.sql           #   - 初始化测试数据
│   └── config.toml        #   - Supabase 本地开发配置
│
├── docs/                  # [Doc] 项目文档
│   ├── API_DOC.md         #   - API 接口文档
│   ├── BrainDance 项目协作规范与开发协议 (v1.0).md  #   - 开发规范
│   └── 技术报告/           #   - 技术报告
│
└── README.md              #   - 项目主文档
```

## 常用命令

### Supabase 本地开发
```bash
# 启动本地 Supabase 服务
supabase start

# 停止本地服务
supabase stop

# 查看本地服务状态
supabase status
```

### AI 引擎 (3DGS Worker)
```bash
cd ai_engine/3dgs

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 Supabase URL 和 Keys

# 启动 Worker
python src/worker.py
# 输出: 🚀 [Worker] Connected to Supabase Local. Listening for tasks...
```

### Web 3D 查看器
```bash
cd ai_engine/demo/webgl/my-3dgs-viewer

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

### Flutter 移动端
```bash
cd app

# 配置连接信息 (编辑 lib/config.dart 或 .env)

# 运行 App
flutter run
```

## 重要说明

- **项目文档**：`docs/BrainDance 项目协作规范与开发协议 (v1.0).md` 包含完整的开发协作规范
- **API 文档**：`docs/API_DOC.md` 包含边缘函数接口规范
- **环境变量**：`ai_engine/3dgs/.env.example` 提供了环境变量配置模板
- **后端部署**：`supabase/README.md` 包含云端部署指南
- **项目架构**：参考 `README.md` 中的时序图了解完整的数据流转
- **IDE 配置**：`.opencode.json` 中已禁用 `git push` 操作
