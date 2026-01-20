# Plan: Pipeline Unification (高优先级 1.1)

目标：把所有单图/视频 3DGS 重建流水线统一到 ai_engine/3dgs/src/pipelines 下，定义统一 Pipeline 接口，降低重复代码并便于测试与扩展。

范围：只做计划和任务拆解，**不修改源代码**。生成可执行的 TODO 清单与补丁草案（*.patch）供后续实现。

时间估计：总体 3-5 天（根据硬件可用性），分为 5 个子任务。

任务列表：

1. 设计阶段（0.5d）
   - 定义 Pipeline 接口：
     - 方法：run(input_path: str, params: dict) -> Tuple[str, dict]
     - 约定：所有 pipeline 返回 final_ply_path 与 metadata
   - 确定公共上下文字段（task_id, scene_id, work_root, log_callback, shared_model_dir）

2. 发现与提取（0.5d）
   - 在代码库中找到所有现有 Pipeline 实现（single_image_sam3d, single_image_sharp, video_3dgs 等）。
   - 抽取公共逻辑（下载/上传/日志回调/ply 查找）并列出差异点。

3. 逐步重构（1.5-2d）
   - 新增 PipelineBase（ai_engine/3dgs/src/core/pipeline_base.py），实现抽象基类与默认实现。
   - 把每个具体 pipeline 迁移为继承 PipelineBase 的子类，重用公共逻辑。
   - 保持接口兼容，做到最小修改，逐个迁移并运行测试。

4. 测试与回归（0.5-1d）
   - 运行本地单图测试与 worker 模式，确保行为一致。
   - 添加单元测试覆盖新基类的关键分支。

5. 文档与 PR（0.5d）
   - 更新 docs，写明 Pipeline 接口与示例。
   - 生成变更补丁并创建本地分支/PR 草案（不 push）。

验收标准：
- 所有 pipeline 通过 `python ai_engine/3dgs/tests/test_local_single_image.py` 和 `python ai_engine/3dgs/tests/test_pipeline_cases.py`（若存在）
- CloudWorker 在处理同一任务时行为无差异（上传文件，写入 model_assets，更新 processing_tasks）。

风险与缓解：
- 风险：重构过程中引入行为差异。缓解：每次迁移只改一个 pipeline，保持回滚点。

下一步（我将自动执行）:
1. 生成 docs/代办/pipeline_unify_todo.md（包含分解任务 + patch 草案模板）。
2. 在工作区创建补丁草案文件（不应用到代码）供你 review。
