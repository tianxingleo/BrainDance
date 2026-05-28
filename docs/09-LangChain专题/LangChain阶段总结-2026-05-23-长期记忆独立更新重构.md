# LangChain 阶段总结 - 2026-05-23 - 长期记忆独立更新重构

## 背景

现有 `user_long_term_memory` 只在前几次调用里更新，后续常常不再变化。根因是旧逻辑把长期记忆是否写库，绑定到了短期记忆的 `turnCount` 和短期偏好变化；而实际请求链路里，短期记忆上下文并不稳定，导致长期记忆几乎停摆。

## 本次调整

- 长期记忆更新改为独立于短期记忆，只看当前 turn 的 query、候选对象、标签和响应模式。
- `preferred_regions / preferred_objects / preferred_time_ranges / preferred_asset_types` 现在由当前 turn 的独立信号直接抽取并频次合并。
- `recent_searches` 继续按 turn 追加，保留最近 10 条搜索型响应，作为可回溯的长期轨迹。
- `spatial_search` 响应补充了 `objects` 字段，便于长期记忆从候选结果中稳定抽取对象信号。

## 影响

- 长期记忆不再依赖短期记忆状态，更新链路更稳定。
- 记忆偏好会随着真实检索内容持续演进，而不是只在早期几轮生效。
- 目前仍保留频次合并和长度上限，避免长期记忆被单轮噪声冲垮。

## 验证

- 已补充长期记忆纯函数测试，覆盖 turn 级信号抽取与持续写入策略。
- 后续需要再跑一次 `deno test` 和 `deno check`，确认 Edge Function 侧类型与语法都能通过。
