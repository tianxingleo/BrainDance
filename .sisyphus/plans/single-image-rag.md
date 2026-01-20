# 单图 3DGS 流水线 RAG 功能集成计划

## 1. 上下文 (Context)

### 原始需求
目前的 `SingleImageSAM3DPipeline` (SAM3D) 和 `SingleImageSharpPipeline` (SHARP) 仅负责生成 `.ply` 模型，没有调用多模态大模型进行语义分析，导致生成的单图 3D 资产无法在系统中通过自然语言检索。

### 调研摘要
- **SceneAnalyzer**: 目前仅支持对目录下的多张图片进行随机采样分析，不支持单图输入。
- **RAG 系统**: 拥有 `RagMemory` (基础向量存取) 和 `KnowledgeBase` (富文本加权存储) 两个模块。`worker.py` 负责在任务完成后调用这些模块进行入库。
- **流水线接口**: `BasePipeline` 定义了 `run` 方法返回 `(ply_path, metadata)`。目前单图流水线的 `metadata` 过于简单，不包含 AI 语义信息。

### Metis 评审建议
- **健壮性**: RAG 分析应包装在 `try-except` 中，确保即使 VLM 接口失败，3D 模型生成任务也能正常完成。
- **一致性**: 确保 `SceneAnalyzer` 的单图分析结果字段 (`description`, `tags`, `objects`) 与 `KnowledgeBase` 所需的字段名对齐。
- **去重逻辑**: 使用 `upsert` 操作（基于 `scene_id`）防止同一场景重复入库。

---

## 2. 工作目标 (Work Objectives)

### 核心目标
升级单图流水线，使其具备自动打标、生成语义描述并存入向量数据库的能力。

### 交付物
- `src/modules/scene_analyzer.py`: 新增 `analyze_single_image` 方法。
- `src/pipelines/single_image_sam3d.py`: 集成 RAG 分析逻辑。
- `src/pipelines/single_image_sharp.py`: 集成 RAG 分析逻辑。

### 验收标准
- [ ] 调用 `analyze_single_image` 能返回包含 `tags`, `description`, `objects` 的 JSON。
- [ ] 运行单图流水线后，Supabase 的 `model_assets` 表中出现对应的语义记录。
- [ ] 任务日志中清晰显示 “[RAG] 正在进行单图语义分析...” 等过程信息。
- [ ] 如果 API Key 缺失或网络故障，流水线不应崩溃，仅跳过 RAG 步骤。

---

## 3. 验证策略 (Verification Strategy)

### 测试方案
本项目后端使用 Python 编写，将在 `ai_engine/3dgs/tests` 下进行验证。

### 手动验证流程 (Manual QA)
1. **模块测试**: 运行脚本验证 `SceneAnalyzer.analyze_single_image` 对本地单张图片的识别效果。
2. **集成测试**: 模拟 Worker 调用 `SingleImageSAM3DPipeline`，检查返回的 `metadata` 是否包含 `ai_description`, `ai_tags`, `ai_objects`。
3. **数据库验证**: 使用 Supabase Studio 检查 `model_assets` 表中是否生成了带有 `embedding` 向量的记录。
4. **检索验证**: 使用现有的检索接口，输入图片中出现的物体关键词，观察是否能匹配到该 `scene_id`。

---

## 4. 任务流程 (Task Flow)

### 任务 1: 升级 SceneAnalyzer 模块
**目标**: 在 `src/modules/scene_analyzer.py` 中添加单图分析接口。

- **实施步骤**:
    - 实现 `analyze_single_image(image_path)`。
    - 针对单图优化 Prompt，要求模型返回 JSON 格式的描述、标签和物体列表。
    - 添加 Base64 编码逻辑。
- **参考代码**: `src/modules/scene_analyzer.py` 现有的 `run` 方法（多图逻辑）。
- **验收点**: `python -c "from src.modules.scene_analyzer import SceneAnalyzer; ..."` 调用无误。

### 任务 2: SAM3D 流水线集成
**目标**: 在 `SingleImageSAM3DPipeline` 中注入分析逻辑。

- **实施步骤**:
    - 在 `run` 方法末尾引入 `SceneAnalyzer`。
    - 调用分析接口，获取结果。
    - 将结果映射为 `ai_score`, `ai_description`, `ai_tags`, `ai_objects`, `ai_reason` 并更新到 `metadata`。
- **注入点**: `src/pipelines/single_image_sam3d.py` 第 32 行 `return` 之前。

### 任务 3: SHARP 流水线集成
**目标**: 对 SHARP 流水线进行同样的升级。

- **实施步骤**: 同任务 2。
- **注入点**: `src/pipelines/single_image_sharp.py` 第 26 行。

### 任务 4: Worker 适配与 RAG 验证
**目标**: 确保 Worker 能正确持久化流水线返回的元数据。

- **逻辑检查**: `src/core/worker.py` (226-301行) 已经包含处理 `metadata` 并调用 `KnowledgeBase.add_asset` 的逻辑。
- **验证**: 确保 `metadata` 中的键名与 `worker.py` 预期的 `ai_description` 等一致。

---

## 5. 待办事项 (TODOs)

- [ ] 1. 修改 `src/modules/scene_analyzer.py`
  - [ ] 实现 `analyze_single_image(self, image_path: str) -> dict`。
  - [ ] 编写针对单图的专用 Prompt（包含 description, tags, objects）。
  - [ ] 处理 JSON 解析异常和清洗逻辑。
  - **参考**: `src/modules/scene_analyzer.py:24` (原 run 方法)。

- [ ] 2. 修改 `src/pipelines/single_image_sam3d.py`
  - [ ] 引入 `SceneAnalyzer`。
  - [ ] 在 `run` 返回前添加 RAG 分析块。
  - [ ] 映射字段：`ai_description`, `ai_tags`, `ai_objects`, `ai_score`, `ai_reason`。
  - **平行性**: 与任务 3 独立。

- [ ] 3. 修改 `src/pipelines/single_image_sharp.py`
  - [ ] 引入 `SceneAnalyzer`。
  - [ ] 添加 RAG 分析逻辑。
  - [ ] 映射字段。
  - **平行性**: 与任务 2 独立。

- [ ] 4. 联调验证
  - [ ] 使用本地测试脚本模拟任务。
  - [ ] 验证日志回传。
  - [ ] 验证 Supabase 数据库入库。
