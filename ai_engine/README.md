# BrainDance AI Engine

`ai_engine/` 是 BrainDance 的计算侧目录，包含当前接入主链路的 3DGS Worker，以及实验脚本、模型资源和运行日志。

## 当前定位

如果把整个项目拆开看，`ai_engine/` 负责的是“下载素材、执行重建、回写结果”这一段计算链路。它和 `app/`、`supabase/`、`dashboard/` 的关系大致如下：

- `app/` 创建任务、上传素材
- `supabase/` 保存任务和文件
- `ai_engine/3dgs/` 监听任务并执行处理，同时向 `worker_nodes` 汇报心跳和控制状态
- `dashboard/` 观察处理状态、资产结果和 Worker 集群状态

## 当前主用目录

- `3dgs/`：当前主用的 AI Worker 与 3D/2D 重建流水线
- `demo/`：不同方向的实验代码和外部项目试验目录
- `models/`：本地模型资源
- `log/`：运行日志与历史记录

其中，真正接入主链路的是 `3dgs/`。

## `3dgs/` 的角色

从当前代码和文档看，`ai_engine/3dgs/` 主要负责：

- 监听 `processing_tasks`
- 根据 `task_type` 选择对应流水线
- 下载视频、图片或压缩包素材
- 执行 3DGS / 2DGS / 单图相关处理
- 上传模型与相关输出
- 回写日志、质量分、标签和资产信息

## 使用入口

如果你要看实际部署、环境要求、任务类型和运行方式，请直接看：

- [ai_engine/3dgs/README.md](/home/ltx/projects/BrainDance/ai_engine/3dgs/README.md)

这个子目录 README 里保留了当前主文档应该承载的内容，包括：

- 安装步骤
- `.env` 配置
- 环境要求
- 任务类型
- 本地模式 / 云端监听模式
- 论文型 pipeline 的参数说明
- 第三方依赖与引用

## 说明

- `demo/` 下很多内容是实验材料，不等同于当前稳定链路
- `ai_engine/README.md` 只负责说明这个目录在整个项目中的位置
- `ai_engine/3dgs/README.md` 才是当前 AI Engine 的主说明文档
