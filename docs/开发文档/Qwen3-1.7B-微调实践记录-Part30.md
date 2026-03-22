# Qwen3-1.7B 微调实践记录 Part 30

## Part 30：仓库文档补齐与当前部署口径统一

### 本 part 目标

前面的 Part 27-29 已经把量化路线、strict 集回退定位、部署候选选择基本做完，但仓库根文档和模块文档仍停留在较早阶段，存在明显口径滞后：

- 根 README 还写成 `Part16-19`
- `docs/README.md` 仍停留在 `2026-03-13`
- `app/README.md` 还没有写 Recall 本地 AI 入口已经恢复
- 多处链接仍指向旧路径 `/home/ltx/projects/BrainDance/...`

这一 part 的目标不是新增模型能力，而是把“代码已经完成的事实”同步进仓库文档，避免后续继续沿旧口径协作。

---

## 一、本次补齐的文档范围

本轮只修改 3 类高频入口文档：

1. `README.md`
2. `docs/README.md`
3. `app/README.md`

这 3 份文档分别承担：

- 仓库级概览
- 文档索引入口
- Flutter 客户端运行与结构说明

---

## 二、统一后的核心口径

### 1. Qwen3 目录不再只是早期实验脚本

`ai_engine/finetune_qwen3` 目前已经覆盖：

- `Qwen3-1.7B + LoRA`
- `Qwen3-0.6B + LoRA`
- merged HF 模型导出
- GGUF 转换
- `Q4_K_M` / `Q5_K_M` 量化
- `imatrix` 量化复测
- strict 无泄漏 benchmark
- Part 29 小样本部署候选验证

所以仓库文档不能再继续只写 `Part16-19`。

### 2. 当前部署主线已经确定

截至 `2026-03-22`，部署候选结论已经明确：

- 当前部署主线：`1.7B Q5_K_M + imatrix GGUF`
- 备用方案：`0.6B LoRA`
- 质量基线：`1.7B merged`

这不是预估，而是结合以下结果得出的工程结论：

- strict 集退化定位
- `Q5` 与 `Q5 + imatrix` 复测
- 真实调用链小样本验证
- Recall Flutter 本地 AI 入口的现有接入方式

### 3. Flutter 端当前接法是 `GGUF + llamadart + local RAG`

合并 `tianxingleo` 分支后，Recall 本地 AI 入口已经恢复，当前链路可以明确描述为：

- 页面入口：`app/lib/pages/recall.dart`
- 本地 AI 面板：`app/lib/pages/recall/local_ai_panel.dart`
- 本地索引：`app/lib/services/local_rag_index.dart`
- 关键依赖：`llamadart`、`dio`、`sqflite`

因此从工程接入角度看，当前最顺的模型产物就是 GGUF，而不是 HF adapter。

---

## 三、这次文档补齐具体改了什么

### 1. 根 README

补齐内容：

- 把 Qwen3 描述从 `Part16-19` 更新到 `Part16-29`
- 写明 Flutter Recall 本地 AI 入口已恢复
- 写明当前部署主线、备用方案、质量基线
- 增加 Part 27/28/29 和两份总评测报告入口
- 修复项目结构中 `app/` 重复列出的问题

### 2. `docs/README.md`

补齐内容：

- 增加“Qwen3 微调与部署”专项入口
- 把文档更新时间同步到 `2026-03-22`
- 增加 Part 27/28/29 和两份总评测报告入口说明
- 修复旧绝对路径，统一改到当前仓库路径

### 3. `app/README.md`

补齐内容：

- 增加 Recall 本地 AI 问答现状说明
- 明确当前路线是 `GGUF + llamadart + local RAG`
- 补充关键依赖 `dio`、`sqflite`、`llamadart`
- 列出对应代码入口
- 标注本轮未执行 `flutter analyze` / 打包验证的客观原因

---

## 四、这次没有做的事情

为了控制文档补齐范围，本轮没有继续做这些扩张动作：

- 没有重写 `ai_engine/finetune_qwen3/README.md`
- 没有新增新的 benchmark
- 没有重新训练模型
- 没有做 Flutter 真机构建验证

原因很直接：

1. 这轮任务是“把最近已经落地的修改补进文档”
2. 当前机器缺少 `flutter` / `dart` CLI，无法在这一轮补齐 analyze/build 验证
3. benchmark 与部署结论在 Part 27-29 已经有明确记录，不需要重复开新实验

---

## 五、本 part 结论

这次工作解决的不是模型能力问题，而是**文档事实与代码现实脱节**的问题。

完成后，仓库顶层、文档索引和 App 模块文档已经统一到同一口径：

- Qwen3 本地问答已经推进到部署候选选择阶段
- 当前 Recall 本地 AI 链路已经恢复
- 当前最顺的部署主线是 `1.7B Q5_K_M + imatrix GGUF`
- `0.6B LoRA` 和 `1.7B merged` 仍分别保留为备用方案与质量基线

这让后续继续做 Recall 本地 AI 接入、GGUF 分发和端侧验证时，不需要再靠历史聊天记录恢复上下文。
