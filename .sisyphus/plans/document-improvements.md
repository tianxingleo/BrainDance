# Plan: Document Project Improvements

## Context

### Original Request
请看看这个项目有没有什么改进空间

### Interview Summary
**Key Discussions**:
- User asked for improvement suggestions.
- User decided to record these suggestions in `docs/代办/` instead of implementing them now.

**Research Findings**:
- The project is a 3D memory engine using 3DGS, Supabase, and AI.
- Code redundancies exist in `ai_engine/demo/` and `src/pipelines/`.
- Potential concurrency issues in task processing (lack of `SKIP LOCKED`).
- RAG system can be enhanced with spatial awareness.
- Some potential unbound variable bugs in `worker.py`.

### Metis Review
- (Consultation failed due to system error, proceeding with self-review)

---

## Work Objectives

### Core Objective
Document all identified project improvement suggestions into `docs/代办/项目改进建议_20260120.md`.

### Concrete Deliverables
- `docs/代办/项目改进建议_20260120.md`

### Definition of Done
- [ ] File `docs/代办/项目改进建议_20260120.md` exists and contains categorized suggestions.

### Must Have
- Categories: Engineering & Refactoring, Architecture & Scaling, AI & Spatial RAG, Functional Loop.
- Specific examples and rationale for each suggestion.

### Must NOT Have (Guardrails)
- Do NOT modify any existing source code.
- Do NOT implement any of the suggested changes.

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (documentation task)
- **User wants tests**: NO
- **QA approach**: Manual verification

### Manual QA

| Type | Verification Tool | Procedure |
|------|------------------|-----------|
| **Documentation** | Bash | `ls docs/代办/项目改进建议_20260120.md` and `cat` to verify content |

---

## Task Flow

```
Task 1 (Write Doc)
```

---

## TODOs

- [ ] 1. Create `docs/代办/项目改进建议_20260120.md`

  **What to do**:
  - Write the following content to the file:
    ```markdown
    # BrainDance 项目改进建议 (2026-01-20)

    ## 1. 代码重构与工程化 (Engineering & Refactoring)
    ### 1.1 核心 Pipeline 统一化
    - 统一入口：将所有重建逻辑收敛到 `ai_engine/3dgs/src/pipelines/` 下。
    - 接口标准化：确保所有 Pipeline 遵循相同的协议。
    ### 1.2 增强异常处理与重试机制
    - 引入可重试异常分类，增加 `retrying` 状态和 `retry_count`。
    ### 1.3 自动化测试
    - 增加单元测试和 E2E 测试。

    ## 2. 后端架构优化 (Infrastructure & Scaling)
    ### 2.1 数据库性能优化
    - 为 `processing_tasks` 添加联合索引 `(status, created_at)`。
    - 使用 `FOR UPDATE SKIP LOCKED` 实现并发抢占。
    - 启用 `pg_cron` 自动清理历史任务。
    ### 2.2 存储管理优化
    - 实施存储生命周期策略，压缩 PLY 模型。

    ## 3. AI 与 RAG 深度增强 (AI & Spatial RAG)
    ### 3.1 增强空间语义理解
    - 引入方位元数据（如“X在Y左边”），支持多模态融合搜索。
    ### 3.2 AI 质检系统升级
    - 建立反馈闭环，分析失败原因并提供拍摄指导。

    ## 4. 端到端功能闭环 (End-to-End Loop)
    ### 4.1 Edge Functions 落地
    - 完成 `supabase/functions/search/` 开发。
    ### 4.2 前端渲染优化
    - 探索移动端高效高斯泼溅渲染方案。
    ```

  **Parallelizable**: NO

  **References**:
  - `ai_engine/3dgs/src/core/worker.py` - Current worker logic.
  - `docs/代办/supabase消息队列优化_整理版.md` - Existing optimization ideas.

  **Acceptance Criteria**:
  - [ ] File exists at `docs/代办/项目改进建议_20260120.md`.
  - [ ] Content matches the drafted suggestions.

  **Commit**: YES
  - Message: `docs: add project improvement suggestions for 2026-01-20`
  - Files: `docs/代办/项目改进建议_20260120.md`

---

## Success Criteria

### Verification Commands
```bash
ls docs/代办/项目改进建议_20260120.md  # Expected: file exists
```
