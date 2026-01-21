PR 标题： feat(3dgs): 统一 BasePipeline，集中 RAG 分析与上传逻辑并修复 RAG 写入目标

PR 正文：

概要
--
本 PR 将 3DGS 流水线的 RAG（检索增强生成）与 PLY 上传/DB 记录逻辑集中到一个 BasePipeline（pipeline_base），并把单图流水线迁移为复用该基类的 helper，以消除代码重复、提升可靠性与可测试性。同时修复了此前向不存在表写入导致的 Supabase 错误（PGRST205），将写入目标统一为现有的 `model_assets` 表；并做了一些配置与防御性导入改进以提升在本地开发环境的稳健性。

背景 / 问题
--
- 以前代码在多个 pipeline 中重复实现 RAG 分析 + 上传逻辑，导致行为分散、错误处理不一致和维护成本高。
- 部分代码曾尝试将 RAG 数据 upsert 到一个不存在的表（`model_knowledge_base`），在运行时触发 Supabase PGRST205 错误。
- 环境依赖（openai/DashScope/supabase client）在某些开发环境未安装时，会导致模块导入时即失败，影响本地开发与测试。

本次变更（高层）
--
- 新增：BasePipeline（ai_engine/3dgs/src/core/pipeline_base.py）
  - 提供 run_rag_analysis(input_path)：统一做 SceneAnalyzer 调用并返回标准化的 ai_* 元数据。
  - 提供 upload_and_record(ply_path, metadata, params)：统一上传 PLY 到 Supabase Storage、在 model_assets 写入记录，包含幂等与错误处理。
- 调整：single-image pipelines（single_image_sam3d.py、single_image_sharp.py）改为调用 BasePipeline helpers，移除重复实现。
- 部分迁移：video_3dgs.py 在导出阶段已改为调用 upload_and_record（属于分阶段完成的迁移）。
- 修复：将 RAG 写入目标改为 `model_assets`（而非缺失的 `model_knowledge_base`）。
- 配置：归一化 SUPABASE_URL（确保仅保留单个尾部斜线以避免 REST URL 双斜线问题）。
- 防御性改进：对 openai/DashScope 与 supabase 客户端做防御性导入与降级处理（当缺少依赖时不会在 import 阶段直接致命）。
- 文档/补丁：新增 patches/、.sisyphus/ 计划与代办文档，便于后续迁移与审查。

主要修改文件（要点）
--
- ai_engine/3dgs/src/core/pipeline_base.py (新增)
- ai_engine/3dgs/src/pipelines/single_image_sam3d.py (迁移使用 BasePipeline helper)
- ai_engine/3dgs/src/pipelines/single_image_sharp.py (迁移使用 BasePipeline helper)
- ai_engine/3dgs/src/pipelines/video_3dgs.py (导出阶段调用 upload_and_record；部分迁移)
- ai_engine/3dgs/src/modules/rag_memory.py (将 upsert 目标改为 model_assets；防御性导入)
- ai_engine/3dgs/src/config.py (SUPABASE_URL 规范化)
- patches/0001_pipeline_base.patch (补丁文件)
- .sisyphus/plans/* 与 docs/代办/*（文档、计划、任务记录）

测试与验证
--
- 本地运行 tests/test_local_single_image.py：通过（生成 PLY 并产出 RAG 元数据）。
- 手工验证：若重复上传同一路径，Supabase Storage 返回 409（对象已存在），说明 Storage 与 DB 可达。

回滚计划
--
- 回滚单个提交： git revert <commit-hash>
- 恢复到远端分支： git reset --hard origin/<branch> （慎用）

审阅要点
--
- BasePipeline 上传/写表逻辑的幂等性与错误处理。
- run_rag_analysis 在 embedding 不可用时的降级策略。
- single_image 流水线迁移后 meta_info 结构兼容性。

---

使用说明
--
文件已写为 PR_BODY.md。此文件仅写入工作区（未自动提交或推送）。

要创建 PR（手动步骤，示例）：
1) 确认当前分支为你要做 PR 的分支：
   git branch --show-current
2) 将分支推送到远端（示例，不在本脚本中自动执行）：
   git push -u origin $(git branch --show-current)
3) 使用 gh 创建 PR（示例）：
   gh pr create --title "feat(3dgs): 统一 BasePipeline，集中 RAG 分析与上传逻辑并修复 RAG 写入目标" --body "$(sed -n '1,400p' PR_BODY.md)" --base main --head $(git branch --show-current)

如果你希望我代为执行 push 与创建 PR，请明确授权（注意：我不会执行 push 到远端，除非你明确允许）。
