# Pipeline 统一化 TODO（草案）

目标：把所有 Pipeline 实现迁移到 `ai_engine/3dgs/src/pipelines/`，并实现共享基类 `PipelineBase`。

任务拆解：

1) 新建基类文件：
- 路径： `ai_engine/3dgs/src/core/pipeline_base.py`
- 内容： 定义 `class PipelineBase:` 与抽象方法 `run(self, input_path: str, params: dict) -> (str, dict)`，实现 download/upload/logging hooks。

2) 迁移 single_image_sam3d：
- 把现有的 `ai_engine/3dgs/src/pipelines/single_image_sam3d.py` 修改为继承 `PipelineBase` 并提取公共逻辑。

3) 迁移 single_image_sharp：
- 同上。

4) 迁移 video_3dgs：
- 同上，注意 video pipeline 的 input 是视频文件，run 返回 PLY 路径。

5) 增加单元测试：
- 为 PipelineBase 提供 mock supabase/文件系统的单元测试，验证日志回调、结果上报行为。

Patch 草案占位文件：`patches/0001_pipeline_base.patch`（我将生成模板供你 review）

验收条件：见 .sisyphus/plans/pipeline-unify.md
