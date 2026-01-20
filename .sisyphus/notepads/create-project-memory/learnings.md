# 项目记忆创建 - 学习记录

## 任务完成总结

**完成时间**: 2026-01-20
**任务**: 创建 BrainDance 项目的 CLAUDE.md 项目记忆文件

## 发现的模式与约定

### 1. Python 代码风格
- **文件头注释**: 必须使用中文描述功能、实现逻辑和调用流程
- **配置模式**: 使用 `dataclass` + `field(default_factory=lambda: os.getenv(...))` 模式
- **导入顺序**: 1) 配置模块 2) 功能模块 3) 工具模块

### 2. Git 规范
- **提交语言**: 必须使用中文
- **分支策略**: `feat/<名称>/<功能>` 格式
- **提交类型**: feat, fix, docs, refactor, chore
- **关键约束**: 禁止执行 `git push` (已在 `.opencode.json` 中禁用)

### 3. 项目架构
- **Monorepo 结构**: `ai_engine/` (Python), `supabase/` (BaaS), `docs/` (文档)
- **技术栈**: 3DGS + RAG + 多模态 AI 的空间记忆引擎
- **后端模式**: Supabase BaaS (PostgreSQL + pgvector + Storage + Realtime)

### 4. 数据库设计
- **任务队列**: `processing_tasks` 表实现异步任务处理
- **安全策略**: 所有表启用 RLS (行级安全策略)
- **存储路径**: `{user_id}/{scene_id}/{raw,output}/...` 格式

## 成功实践
1. 使用中文描述技术文档，符合项目现有惯例
2. 完整覆盖技术栈、代码规范、目录结构和常用命令
3. 明确标注禁止 `git push` 的安全约束

## 参考资料
- `docs/BrainDance 项目协作规范与开发协议 (v1.0).md`
- `ai_engine/3dgs/src/core/pipeline.py` (Python 风格示例)
- `README.md` (项目概述与架构)
