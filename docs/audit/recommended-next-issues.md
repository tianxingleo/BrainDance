# BrainDance 审计汇总：建议下一步行动计划

## 审计总览

| 审计报告 | 问题总数 | P0 | P1 | P2 | P3 |
|---------|:-------:|:--:|:--:|:--:|:--:|
| [Flutter 深度审计](flutter-audit.md) | 24 | 5 | 8 | 7 | 4 |
| [AI Worker 审计](ai-worker-audit.md) | 22 | 4 | 5 | 7 | 6 |
| [Dashboard 审计](dashboard-audit.md) | 23 | 2 | 5 | 9 | 7 |
| [Supabase 安全审计](supabase-security-audit.md) | 21 | 4 | 7 | 6 | 4 |
| [跨模块契约审查](contract-audit.md) | 14 | 1 | 2 | 8 | 3 |
| **合计** | **104** | **16** | **27** | **37** | **24** |

---

## 全部 P0 问题清单（必须优先修复）

| # | 来源 | 问题 | 涉及模块 |
|---|------|------|---------|
| 1 | Flutter | MyApp.build() 重复注册 WidgetsBindingObserver | app/lib/main.dart |
| 2 | Flutter | downloadEventBus StreamController 从未关闭 | app/lib/services/download_event_bus.dart |
| 3 | Flutter | MaterialApp.builder child! 强制解包 | app/lib/main.dart |
| 4 | Flutter | transformMatrix 强制类型转换不安全 | app/lib/pages/recall/recall_model_actions.dart |
| 5 | Flutter | Agent 流式 onDone 回调缺少 null 保护 | app/lib/pages/recall/recall_search.dart |
| 6 | Worker | 任务抢单竞态条件 | ai_engine/3dgs/src/core/worker.py |
| 7 | Worker | mask_guided.py 空壳调用必崩 | ai_engine/3dgs/src/pipelines/mask_guided.py |
| 8 | Worker | da3_2dgs_pipeline 使用 Unix-only pty 模块 | ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py |
| 9 | Worker | knowledge_base.py 顶层硬性 import openai | ai_engine/3dgs/src/pipelines/knowledge_base.py |
| 10 | Dashboard | 无认证 + worker_nodes 全开 RLS | dashboard/src/App.vue, supabase/migrations/ |
| 11 | Dashboard | 全表扫描性能退化 | dashboard/src/App.vue |
| 12 | Supabase | model_assets 未启用 RLS | supabase/migrations/20260118144558_init_schema.sql |
| 13 | Supabase | rag_docs 未启用 RLS | supabase/migrations/20260118144558_init_schema.sql |
| 14 | Supabase | Storage "Enable all" 全开策略 | supabase/migrations/20260118144558_init_schema.sql |
| 15 | Supabase | worker_nodes "Allow all for dev" 策略 | supabase/migrations/20260320143000_create_worker_nodes.sql |
| 16 | 契约 | Storage 全开策略覆盖 user folder 策略 | supabase/migrations/ |

---

## 建议修复顺序

### 第一优先级：安全 + 会炸的（Sprint 1）

1. **Supabase Storage 全开策略删除** — 一条 SQL，立即消除最大安全漏洞
2. **model_assets / rag_docs 启用 RLS** — 保护核心数据
3. **Flutter 5 个 P0 崩溃修复** — 防御性修改，低风险
4. **Worker 任务抢单原子化** — 多 Worker 必改
5. **mask_guided.py 修复或从 Factory 移除** — 消除必崩点
6. **worker_nodes RLS 收紧** — 防止远程操控

### 第二优先级：演示体验（Sprint 2）

7. **Dashboard 认证** — 添加登录页
8. **任务失败原因展示** — Flutter + Dashboard
9. **Flutter 静默 catch 补全** — 10+ 处错误处理
10. **task_type 枚举统一** — 消除跨模块不一致
11. **Dashboard App.vue 拆分** — 2015 行 → 可维护
12. **processing_tasks updated_at 触发器** — Dashboard 排序准确

### 第三优先级：亮点功能（Sprint 3）

13. **time-compare-agent 前端对接** — 展示核心卖点
14. **Dashboard 美化** — 专业感
15. **Viewer 稳定性** — 加载进度、错误恢复
16. **端侧 AI 断网演示** — 差异化亮点
17. **集成测试框架** — 改动有保障

---

## 适合 Claude Code 自动修复的 Issue

以下 Issue 结构明确、改动范围可控，适合直接交给 Claude Code 实现：

| Issue | 改动范围 | 自动化难度 |
|-------|---------|:---------:|
| Flutter P0 崩溃修复（5 处） | app/lib 5 个文件 | ⭐ 低 |
| processing_tasks updated_at 触发器 | supabase/migrations 1 个文件 | ⭐ 低 |
| task_type CHECK 约束启用 | supabase/migrations 1 个文件 | ⭐ 低 |
| Storage 全开策略删除 | supabase/migrations 1 个文件 | ⭐ 低 |
| 静默 catch 补全 | app/lib 多处 | ⭐⭐ 中 |
| Dashboard App.vue 拆分 | dashboard/src 多个文件 | ⭐⭐⭐ 高 |
| Worker 任务抢单原子化 | supabase + worker | ⭐⭐⭐ 高 |

---

## 审计报告文件清单

```
docs/audit/
├── flutter-audit.md          # Flutter 深度代码审计（24 个问题）
├── ai-worker-audit.md        # AI Worker / 3DGS pipeline 审计（22 个问题）
├── dashboard-audit.md        # Dashboard 审计（23 个问题）
├── supabase-security-audit.md # Supabase 安全与 schema 审计（21 个问题）
├── contract-audit.md         # 跨模块契约审查（14 个不一致）
└── recommended-next-issues.md # 本文件：汇总与行动计划

docs/product-roadmap.md       # 产品扩展路线图
```

---

## 总结

本次审计共发现 **104 个问题**，其中 **16 个 P0 级**需立即修复。最紧迫的是：
1. **安全**：Storage/RLS 全开策略 + Dashboard 无认证
2. **稳定性**：Flutter 5 个崩溃点 + Worker 竞态 + 空壳 Pipeline
3. **一致性**：task_type 枚举散落各处，命名风格不统一

建议按 Sprint 1-3 的节奏逐步修复，优先消除安全和崩溃风险，再补全功能体验，最后做亮点功能展示。
