# Flutter 前端优化计划

## 问题 1：底栏组件大小溢出 + 布局调整

**当前状态：** `floating_nav_bar.dart` 中 `_NavBarItem` 在选中态使用水平 `Row`（图标 + 文字横排），未选中态仅显示图标。用户希望改为垂直布局（文字在图标下方），增大图标，并允许图标超出底栏上边界。

**修改方案：**
- 文件：`app/lib/floating_nav_bar.dart`
- 将 `_NavBarItem` 的选中态布局从水平 `Row` 改为垂直 `Column`（icon 在上、label 在下）
- 增大图标尺寸（例如从 21-22 改为 26-28）
- 未选中态同样改为垂直布局（图标 + 文字），选中/未选中通过颜色和字重区分
- 允许图标超出底栏上边界：在 nav bar 容器的 Stack 上设置 `clipBehavior: Clip.none`，并适当调整图标的垂直位置
- 调整 `_kNavBarInnerHeight` 或内部 padding 以适应新的垂直布局
- 移除或重构选中态的 pill 动画（因为垂直布局下 pill 可能不再合适，或改为底部指示器）

---

## 问题 2：AgentChatPage 开场白缺失

**当前状态：** `agent_chat.dart` 中新建对话时仅显示空状态 (empty state with suggestion chips)，没有主动发送开场白消息。而旧的 recall 页面的 agent mode 有 `_fetchAgentGreeting()` 会主动调用 API 获取开场白。

**修改方案：**
- 文件：`app/lib/pages/agent_chat.dart` 和 `app/lib/pages/agent_chat/chat_runtime.dart`
- 在新建对话 `_createNewConversation` 或初次加载空历史后，调用类似 `_fetchAgentGreeting()` 的逻辑
- 具体实现：
  1. 在创建新对话后，创建一个 agent ChatMessage（`isUser: false, liveStatus: '正在加载...'`）
  2. 设置 `_activeChatMessage` 为该消息
  3. 调用 `AgentRecallService().query('你好', sessionId: _agentSessionId)` 获取开场白
  4. 收到结果后设置 `_activeChatMessage.finalAnswer = result.answer`
  5. 调用 `_persistAgentResponse` 持久化开场白（需要先修复问题3，避免重复）

---

## 问题 3：Agent 回答重复显示

**当前状态：** `chat_runtime.dart` 的 `_persistAgentResponse` 方法在将消息添加到 `_messages` 列表后，没有清除 `_activeChatMessage` 和 `_activeResult`。而 `chat_view.dart` 的 ListView `itemCount` 为 `_messages.length + (_activeChatMessage != null ? 1 : 0)`，导致同一条回复同时出现在 `_messages` 列表和 active bubble 中。

**修改方案：**
- 文件：`app/lib/pages/agent_chat/chat_runtime.dart`
- 在 `_persistAgentResponse` 的 `setState` 中，添加 `_activeChatMessage = null;` 和 `_activeResult = null;`

```dart
setState(() {
  _messages.add(AgentMessageRecord(...));
  _activeChatMessage = null;
  _activeResult = null;
});
```

---

## 执行顺序

1. 先修复问题 3（最简单，一行代码）
2. 再修复问题 2（依赖问题 3 修复后不重复）
3. 最后处理问题 1（UI 重构，改动最大）
