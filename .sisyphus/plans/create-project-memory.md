# Plan: Create BrainDance Project Memory (CLAUDE.md)

## Context

### Original Request
The user wants to create a `CLAUDE.md` file in the project root following a specific template to provide project memory for AI assistants.

### Interview Summary
**Key Discussions**:
- Follow the provided template (Tech Stack, Code Style, Structure, Important Info).
- Language: Mixed (Technical terms in English, descriptions and rules in Chinese as per project convention).

**Research Findings**:
- **Project**: BrainDance (流光·记) - 3D Semantic Memory Engine.
- **Tech Stack**:
    - Frontend: Flutter (Mobile), Vue 3 + TypeScript (Web Viewer).
    - Backend: Supabase (BaaS).
    - AI Engine: Python 3.10+, CUDA, Nerfstudio (3DGS).
- **Conventions**: 
    - Python: File-header Chinese docstrings, dataclass config.
    - Git: Commit messages in Chinese, `git push` forbidden.
    - Architecture: Monorepo with `ai_engine`, `supabase`, `docs`.

---

## Work Objectives

### Core Objective
Create a comprehensive `CLAUDE.md` in the project root to serve as the primary context for AI development sessions.

### Concrete Deliverables
- `/home/ltx/projects/BrainDance/CLAUDE.md`

### Definition of Done
- [x] `CLAUDE.md` exists in the root directory.
- [x] Content covers Tech Stack, Code Style, Directory Structure, and Operational Commands.
- [x] Reflects the "No git push" rule and Chinese commit message convention.

### Must Have
- Explicit mention of forbidden `git push`.
- Tech stack breakdown for AI, Frontend, and Backend.
- Coding conventions for Python and general collaboration.

### Must NOT Have (Guardrails)
- Do not include sensitive API keys (use placeholders/env var references).
- Do not deviate from the existing `docs/BrainDance 项目协作规范与开发协议 (v1.0).md`.

---

## Verification Strategy

### Manual QA Only
Since this is a markdown file creation, verification will be manual inspection of the generated content.

**Procedure**:
- [x] Read `CLAUDE.md` content.
- [x] Verify it matches the project structure found in `explore` results.
- [x] Check for accuracy of tech stack and conventions.

---

## Task Flow

```
Task 1 (Generate CLAUDE.md)
```

---

## TODOs

- [x] 1. Generate `CLAUDE.md`

  **What to do**:
  - Create the file in the project root.
  - Fill with content based on the template and research findings:
    - **Project Name**: BrainDance (流光·记).
    - **Tech Stack**: AI (Python/Nerfstudio), Web (Vue3/TS), Mobile (Flutter), Backend (Supabase).
    - **Commands**: Build, Test, and Lint commands for each sub-module.
    - **Style**: Python Chinese docstrings, Chinese commit messages.
    - **Structure**: Map `ai_engine`, `supabase`, `docs`, `app`.
    - **Warnings**: Forbidden `git push`.

  **Parallelizable**: NO

  **References**:
  - `README.md`: Project overview.
  - `docs/BrainDance 项目协作规范与开发协议 (v1.0).md`: Collaboration rules.
  - `ai_engine/3dgs/src/core/pipeline.py`: Python style example.
  - `AGENTS.md`: Existing assistant rules.

  **Acceptance Criteria**:
  - [x] File exists at `/home/ltx/projects/BrainDance/CLAUDE.md`.
  - [x] Content accurately reflects the "BrainDance" project context.
  - [x] Includes commands for `nerfstudio` and `supabase` if applicable.

  **Commit**: YES
  - Message: `docs: 创建项目记忆文件 CLAUDE.md`
  - Files: `CLAUDE.md`
