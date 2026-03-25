# BrainDance Agent 额外规范

本文件补充仓库根目录下的 Agent 工作规则，重点约束 LangChain / Agent 编排相关工作的文档维护方式。

## 1. LangChain 相关工作必须同步写文档

凡是涉及以下内容之一，都视为 LangChain 相关工作：

- `langchain`、`LangChain`
- `bindTools`
- `DynamicStructuredTool`
- `ChatOpenAI`
- `agent-recall`
- `spatial-search-agent`
- `time-compare-agent`
- 其他基于大模型工具调用、Agent 路由、结构化动作输出的 Supabase Edge Function

只要修改了这类代码，就不能只改代码不改文档。

## 2. 文档统一收口到 `docs/09-LangChain专题/`

LangChain 相关的实现现状、阶段总结、联调记录、未完成事项说明，统一写入：

- [docs/09-LangChain专题/README.md](/home/ltx/projects/BrainDance/docs/09-LangChain专题/README.md)

新增或更新文档时，优先放在这个目录，不要继续把实现状态零散写进多个无关目录。

## 3. 没做完也必须先总结

如果一次 LangChain 相关工作还没做完，但已经出现以下任一情况：

- 需要暂停
- 需要切换任务
- 当前上下文已经复杂
- 已经完成一部分实现，但还有剩余 work items

则必须先写阶段总结文档，再结束当前轮工作。

阶段总结至少要写清楚：

- 已完成什么
- 还没完成什么
- 当前阻塞点是什么
- 下一步建议先做什么
- 哪些文件已经改过

## 4. 文档过长时必须开新文档

如果同一篇 LangChain 文档已经过长，或者开始同时混入多个主题，例如：

- 既有规划，又有现状，又有联调细节
- 既有空间检索，又有资产操作，又有时间对比
- 既有代码说明，又有回归记录，又有问题排查

则不要继续无上限追加，而是新开文档承接。

建议命名方式：

- `LangChain实现现状-YYYY-MM-DD.md`
- `LangChain阶段总结-YYYY-MM-DD-01.md`
- `LangChain联调记录-YYYY-MM-DD-01.md`

## 5. 文档内容必须以代码现状为准

LangChain 相关文档不能只写规划口径，必须明确区分：

- 已实现
- 部分实现
- 实验中
- 未实现

如果代码和旧文档不一致，应先修正文档口径，避免继续漂移。
