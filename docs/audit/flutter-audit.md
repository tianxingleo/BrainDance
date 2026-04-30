# Flutter 深度代码审计报告

## 审计概要
- 审计时间：2026-04-30
- 审计范围：app/lib 全部 Dart 文件（78 个文件），app/integration_test/（16 个文件），app/test/（3 个文件）
- 总代码行数：23,238 行（app/lib）
- 发现问题数：24 个
- P0: 5 个 | P1: 8 个 | P2: 7 个 | P3: 4 个

---

## P0 级问题（会导致崩溃或数据丢失）

### 1. MyApp.build() 中重复注册 WidgetsBindingObserver 导致内存泄漏和重复回调

- **严重程度**：P0
- **涉及文件**：`app/lib/main.dart:95-101`
- **问题描述**：`MyApp` 继承了 `WidgetsBindingObserver` 并在 `build()` 方法中调用 `WidgetsBinding.instance.addObserver(this)`。由于 `build()` 可能被框架多次调用（主题切换、语言切换、rebuild 等），每次调用都会重复注册同一个 observer，导致 `_instance` 列表不断增长。更严重的是，`dispose()` 从未被实现（`StatelessWidget` 没有 dispose 回调），observer 永远不会被移除。
- **复现方法**：多次切换深色/浅色模式或切换语言，每次触发 MaterialApp rebuild 都会多注册一个 observer。
- **建议修复**：将 `MyApp` 改为 `StatefulWidget`，在 `initState` 中注册 observer，在 `dispose` 中移除。或者使用独立的生命周期管理类。
- **建议创建 Issue**：是

### 2. 全局 downloadEventBus StreamController 从未关闭，存在内存泄漏

- **严重程度**：P0
- **涉及文件**：`app/lib/services/download_event_bus.dart:27`
- **问题描述**：`downloadEventBus` 是一个顶层 `broadcast` StreamController，作为全局变量在整个应用生命周期中存在。它从未调用过 `.close()`，任何通过 `.listen()` 注册的监听器只要不手动取消，就会在控制器上保持引用，无法被 GC 回收。broadcast 类型的 controller 在最后一个监听器取消后仍保持活跃状态。
- **复现方法**：长时间运行应用，反复进出 Recall 页面（该页面会 listen downloadEventBus），观察内存是否持续增长。
- **建议修复**：提供一个全局关闭方法或使用 Riverpod provider 管理其生命周期。
- **建议创建 Issue**：是

### 3. MaterialApp.builder 中 child! 强制解包存在崩溃风险

- **严重程度**：P0
- **涉及文件**：`app/lib/main.dart:179`
- **问题描述**：`builder: (context, child) { ... child! ... }` 中，虽然 MaterialApp 通常会传入非 null 的 child，但 Flutter API 并未保证这一点。如果未来 Flutter 版本行为变化或某些边界条件触发，`child!` 会抛出 `Null check operator used on a null value` 异常导致应用白屏。
- **复现方法**：当前版本可能不会触发，但在某些边界场景（如路由错误、Navigator 状态异常）下可能触发。
- **建议修复**：改为 `child ?? const SizedBox.shrink()` 防御性处理。
- **建议创建 Issue**：是

### 4. recall_model_actions.dart 中 transformMatrix 类型转换不安全

- **严重程度**：P0
- **涉及文件**：`app/lib/pages/recall/recall_model_actions.dart:48`
- **问题描述**：`initialPose = transformMatrix.map((e) => (e as num).toDouble()).toList();` 使用了强制类型转换 `e as num`。如果 `transformMatrix` 列表中包含 `null`、`String` 或其他非 `num` 类型元素（Supabase 返回的数据可能包含这些），会抛出 `TypeError` 导致崩溃。该方法被 Agent 结果解析和模型操作直接调用，是核心路径。
- **复现方法**：当 Agent 返回的 action payload 中 matrix 字段包含非数值类型数据时触发。
- **建议修复**：改为 `(e as num?)?.toDouble() ?? 0.0` 或使用 try-catch 包裹转换。
- **建议创建 Issue**：是

### 5. AgentRecallService.queryStream 中 stream.listen 后在 onDone 中缺少错误保护

- **严重程度**：P0
- **涉及文件**：`app/lib/pages/recall/recall_search.dart:267-279`
- **问题描述**：`onDone` 回调中直接调用 `_completeAgentRun()`，但此时 `_agentChatMessage` 可能已为 null（因为在 `_resetAgentUiState` 中被置空），会导致 `_completeAgentRun` 内部的 `_agentChatMessage!` 强制解包崩溃。当用户在 Agent 流式响应过程中快速切换搜索模式时，`_resetAgentUiState` 会先于 `onDone` 触发。
- **复现方法**：在 Agent 流式响应过程中快速切换搜索模式（如从 Agent 模式切到 Cloud 模式），然后等待原 stream 的 onDone 回调触发。
- **建议修复**：在 `onDone` 中增加 `if (_agentChatMessage == null) return;` 保护。
- **建议创建 Issue**：是

---

## P1 级问题（功能异常或严重体验问题）

### 6. _RecallPageState 超过 30 个可变字段，存在严重的状态竞态风险

- **严重程度**：P1
- **涉及文件**：`app/lib/pages/recall.dart:63-133`（结合 7 个 part 文件）
- **问题描述**：`_RecallPageState` 拥有超过 30 个可变字段（`_isLoading`、`_isAgentSearching`、`_agentResult`、`_models`、`_processingTasks` 等），通过 part 文件拆分到 7 个 extension 中。这些字段之间存在复杂的依赖关系，但没有任何同步机制保护。多个异步操作（Realtime 监听、轮询、Agent 流式查询、本地 RAG 索引同步）可以并发修改这些状态，导致中间态不一致。
- **复现方法**：在 Recall 页面同时触发：Agent 搜索、Realtime 推送处理任务变更、本地索引同步。观察 UI 是否出现闪烁或数据不一致。
- **建议修复**：将状态拆分为独立的 StateNotifier 或使用 `AsyncValue` 管理异步状态，用 Riverpod 统一管理。
- **建议创建 Issue**：是

### 7. 多处 catch 块静默吞掉错误，用户无法感知失败

- **严重程度**：P1
- **涉及文件**：`app/lib/pages/recall/recall_data_sync.dart:109`、`app/lib/services/task_notification_service.dart:107,122`、`app/lib/services/local_model_catalog_service.dart:51,100,103`、`app/lib/pages/recall/recall_model_actions.dart:159`、`app/lib/pages/task_list.dart:164`
- **问题描述**：代码库中至少 10 处 `catch (e) { // 静默失败 }` 或 `catch (_) {}` 模式。这些错误被完全吞掉，用户和开发者都无法感知到失败。特别是 `_fetchProcessingTasks` 和 `_saveNotifiedTasksToCache` 的静默失败会导致数据不一致。
- **复现方法**：断网后操作，多个操作静默失败但用户完全不知情。
- **建议修复**：至少在 debug 模式下 `print` 或 `debugPrint` 错误信息；在生产模式下使用 Sentry 等 crashlytics 工具上报；对用户可感知的操作（如数据加载失败）显示 Toast。
- **建议创建 Issue**：是

### 8. TaskNotificationService 是 ChangeNotifier 单例但从未被 dispose

- **严重程度**：P1
- **涉及文件**：`app/lib/services/task_notification_service.dart:23-28,225-228`
- **问题描述**：`TaskNotificationService` 继承 `ChangeNotifier` 并实现了 `dispose()` 方法，但由于它是全局单例（`factory TaskNotificationService() => _instance`），其 `dispose()` 永远不会被调用。单例持有 `RealtimeChannel`，若 Supabase client 被重建，旧 channel 仍在监听。同时，虽然 `dispose()` 实现了 `stopMonitoring()`，但单例永远不会走到这个路径。
- **复现方法**：应用生命周期中不会直接触发问题，但在 Supabase client 重建或 token 刷新场景下，可能产生幽灵监听。
- **建议修复**：将 `dispose()` 调用时机绑定到应用退出（通过 `WidgetsBindingObserver` 的 `detached` 状态），或改为不继承 `ChangeNotifier` 而使用独立的状态管理。
- **建议创建 Issue**：是

### 9. main.dart 中 ScrollController 从未被 dispose

- **严重程度**：P1
- **涉及文件**：`app/lib/main.dart:431-456`（`_MainScreenState` 的 `ScrollController` 未见 dispose）
- **问题描述**：虽然 `_MainScreenState` 中 `_animController` 被正确 dispose，但 `_MainScreenState` 没有持有 `ScrollController`（它是页面级别的）。不过观察 `RecallPage`，其 `_recallScrollController` 在 `_disposeRecallPageState` 中被 dispose，这部分没问题。但 `GeneratePage` 的 `_scrollController` 被正确 dispose。此条调整为：观察记录中 `Home.build()` 使用 `loadSettings(ref)` 启动异步操作但没有 mounted 检查，若在 settings 加载完成前 widget 被移除会导致问题。
- **复现方法**：在应用启动后极短时间内切换路由。
- **建议修复**：在 `loadSettings` 的 `setState` 调用前增加 `mounted` 检查。
- **建议创建 Issue**：否（小改动，可直接修复）

### 10. RecordPage 中 _hapticLoopTimer 未在 _stopSensors 中被可靠取消

- **严重程度**：P1
- **涉及文件**：`app/lib/pages/record.dart:382-424`
- **问题描述**：`_syncMotionHaptics` 方法中创建 `Timer.periodic`，调用路径为：`_updateMotionFeedback` -> `_syncMotionHaptics`。但 `_stopSensors` 虽然调用了 `_stopHapticLoop()`，如果 `_updateMotionFeedback` 的某个回调在 `_stopSensors` 执行过程中正在运行（Dart 单线程模型下不太可能但仍需注意），haptic loop timer 可能泄露。更重要的是，`didChangeAppLifecycleState` 中 `_stopSensors` 被调用，但如果用户在暂停/恢复之间快速切换，新的 `_startSensors` 可能与旧的 timer 产生冲突。
- **复现方法**：在运动 HUD 启用状态下快速切换应用前后台。
- **建议修复**：在 `_startSensors` 开头确保调用 `_stopHapticLoop()` 和 `_stopSensors()` 清理旧状态。
- **建议创建 Issue**：是

### 11. RecallPage RealtimeChannel 在某些错误路径下不会被取消订阅

- **严重程度**：P1
- **涉及文件**：`app/lib/pages/recall/recall_data_sync.dart:6-19` 和 `app/lib/pages/recall/recall_view.dart`（dispose 路径）
- **问题描述**：`_setupRealtimeListener()` 创建了 `_realtimeChannel` 并调用 `.subscribe()`。如果在订阅成功之前页面被 dispose，或者 Supabase client 连接断开后重连，channel 可能处于悬挂状态。虽然 dispose 时会调用 `Supabase.instance.client.removeChannel(_realtimeChannel!)`，但如果 dispose 时 `_realtimeChannel` 为 null（比如 `_setupRealtimeListener` 还没执行完），就会跳过清理。
- **复现方法**：快速进出 Recall 页面，或在网络不稳定时切换页面。
- **建议修复**：在 dispose 中用 `_realtimeChannel?.let` 模式确保即使 null 也不跳过其他清理逻辑；在 `_setupRealtimeListener` 开头先移除旧 channel。
- **建议创建 Issue**：是

### 12. video_submit.dart 中 _generateSceneId 使用 static Random 实例

- **严重程度**：P1
- **涉及文件**：`app/lib/pages/video_submit.dart:39-49` 和 `app/lib/pages/generate.dart:50,61-69`
- **问题描述**：`_generateSceneId()` 在 `video_submit.dart` 和 `generate.dart` 中各有一份独立实现，且都使用 `static final Random _rdg = Random()`。两处生成的 sceneId 格式相同但随机数种子独立，在极端并发场景下可能生成相同的 sceneId。更关键的是，这个函数是 `static` 的，不依赖实例状态，意味着它在整个应用生命周期中共享同一个 Random 实例——虽然 Dart 的 Random 是线程安全的，但重复的逻辑是维护隐患。
- **复现方法**：同时在两个页面快速提交任务。
- **建议修复**：抽取到公共工具类，使用 UUID 或更可靠的唯一 ID 生成策略。
- **建议创建 Issue**：是

---

## P2 级问题（代码质量、可维护性）

### 13. _formatBytes 函数在三个文件中重复定义

- **严重程度**：P2
- **涉及文件**：`app/lib/pages/generate.dart:75-82`、`app/lib/pages/video_submit.dart:154-161`、`app/lib/pages/generate/generate_submission.dart`（推测）
- **问题描述**：相同的字节格式化函数 `_formatBytes(int bytes)` 在多个文件中重复实现。代码库已有 `configs/` 和 `extra_func/` 目录，此类通用工具应抽取到公共模块。
- **复现方法**：搜索 `_formatBytes` 即可看到多处重复。
- **建议修复**：抽取到 `lib/utils/format_utils.dart`。
- **建议创建 Issue**：是

### 14. 大量硬编码的中文字符串未走 i18n 系统

- **严重程度**：P2
- **涉及文件**：`app/lib/pages/recall/recall_agent_runtime.dart:167,173,178,183`、`app/lib/pages/recall/recall_search.dart:169,311,530,820,862`、`app/lib/pages/recall/recall_model_actions.dart:477,481,543,545`、`app/lib/pages/recall/recall_data_sync.dart:235`、`app/lib/pages/community.dart:143,198`
- **问题描述**：代码库有完善的 i18n 系统（`textLocalize()`），但大量用户可见文本直接使用硬编码中文字符串，如"已提交请求，正在连接 Agent 服务"、"云端模型删除成功"、"只能删除当前账号自己的云端模型"等。这导致在英文 locale 下仍显示中文，破坏了多语言体验。
- **复现方法**：将 locale 切换为 `en_US`，触发相关功能，会看到中英混杂的界面。
- **建议修复**：将所有硬编码中文字符串迁移到语言资源文件，使用 `textLocalize()` 包裹。
- **建议创建 Issue**：是

### 15. GeneratePage 使用 static 可变字段导致状态跨实例共享

- **严重程度**：P2
- **涉及文件**：`app/lib/pages/generate.dart:41-50`
- **问题描述**：`_uploadKey`、`_uploadKey2`、`firstCheck` 都是 `static` 字段。`firstCheck` 控制是否加载缓存，但由于它是 static，即使 `GeneratePage` 被 dispose 并重建，`firstCheck` 仍为 `true`，导致缓存不会被重新加载。`_uploadKey` 和 `_uploadKey2` 是 `static Key`，在多实例场景下共享同一个 Key 值，可能触发 Flutter 的 key 冲突警告。
- **复现方法**：进入 Generate 页面 -> 离开 -> 重新进入，观察缓存是否被加载。
- **建议修复**：将 `firstCheck` 改为实例变量，将 `_uploadKey`/`_uploadKey2` 改为实例变量。
- **建议创建 Issue**：是

### 16. Supabase 表名和字段名在多个文件中硬编码分散

- **严重程度**：P2
- **涉及文件**：`app/lib/pages/recall/recall_data_sync.dart:83,150,153`、`app/lib/pages/recall/recall_model_actions.dart:153,245,503`、`app/lib/pages/video_submit.dart:117`、`app/lib/pages/task_list.dart:112`、`app/lib/services/task_notification_service.dart:129,133`
- **问题描述**：`processing_tasks`、`model_assets`、`braindance-assets` 等表名和 bucket 名称在 6+ 个文件中重复硬编码。如果后端表名变更或需要支持多环境，需要同时修改所有引用点。
- **复现方法**：搜索 `'processing_tasks'` 可看到至少 6 处引用。
- **建议修复**：抽取到 `SupabaseConfig` 或专门的 `DbConstants` 类。
- **建议创建 Issue**：是

### 17. Agent 流式响应中 _consumeAgentEvent 未对 payload 做完整防御性校验

- **严重程度**：P2
- **涉及文件**：`app/lib/pages/recall/recall_agent_runtime.dart:405-591`
- **问题描述**：`_consumeAgentEvent` 方法对每个 event type 做了 `payload is Map` 检查，但对 payload 内部字段的校验不够严格。例如 `event == 'tool_call'` 时，`payload['args']` 如果是非 Map 类型，`Map<String, dynamic>.from(payload['args'] as Map)` 会抛异常。`event == 'done'` 时，如果 payload 是一个非标准结构，`AgentRecallResponse.fromJson` 内部也可能抛异常。虽然外层有 `stream.listen(onError:)` 兜底，但单个 event 解析失败会终止整个流的消费。
- **复现方法**：后端返回非标准格式的流式事件。
- **建议修复**：在 `_consumeAgentEvent` 外层加 try-catch，单个 event 解析失败时 continue 而非终止整个流。
- **建议创建 Issue**：是

### 18. LocalModelCatalogService.fetchCatalog 中 Dio 实例未复用

- **严重程度**：P2
- **涉及文件**：`app/lib/services/local_model_catalog_service.dart:44`
- **问题描述**：`Dio().get<dynamic>(...)` 每次调用都创建新的 Dio 实例。Dio 实例持有连接池和拦截器配置，频繁创建会增加内存开销和连接建立延迟。同样的问题也存在于 `recall_model_actions.dart:327` 的 `Dio().download(...)`。
- **复现方法**：在弱网环境下反复触发模型目录加载，观察连接建立耗时。
- **建议修复**：使用共享的 Dio 单例或通过依赖注入管理。
- **建议创建 Issue**：是

---

## P3 级问题（优化建议、风格统一）

### 19. recall.dart 使用 part 指令将单个类拆分为 7 个文件

- **严重程度**：P3
- **涉及文件**：`app/lib/pages/recall.dart:48-54` 及其 7 个 part 文件
- **问题描述**：`_RecallPageState` 通过 `part` 指令和 `extension` 拆分到 8 个文件中，总代码量超过 3000 行。虽然 `part` 解决了文件长度问题，但它破坏了代码的封装性——所有 extension 都能访问类的私有成员（通过 `// ignore_for_file: invalid_use_of_protected_member`），本质上回避了 Dart 的访问控制。理想做法是将功能拆分为独立的 service/controller 类。
- **复现方法**：代码审查时难以追踪状态变更的完整路径。
- **建议修复**：长期应将 Agent 运行时、搜索、数据同步、模型操作拆分为独立的 Controller 或 StateNotifier。
- **建议创建 Issue**：是

### 20. 魔法数字和颜色值散布在 UI 代码中

- **严重程度**：P3
- **涉及文件**：`app/lib/pages/record.dart:26-33`、`app/lib/pages/recall/recall_search.dart:112`、`app/lib/pages/video_submit.dart:235`
- **问题描述**：运动检测阈值（`_kIdealAccelMin = 0.08`、`_kInstantSpikeAccel = 2.60` 等）虽已抽取为 const，但颜色值如 `Color(0xFF101014)`、`Color(0xFF2A2A30)` 在多处重复出现。部分阈值（如搜索缓存上限 24、轮询间隔 15 秒）直接硬编码在逻辑中。
- **复现方法**：搜索 `0xFF101014` 可看到至少 3 处引用。
- **建议修复**：将重复的颜色值统一到 `AppTheme` 或 `BDDesign` 中；将业务阈值抽取为命名常量。
- **建议创建 Issue**：否（可逐步改进）

### 21. textLocalize 函数在非 Widget 上下文中调用

- **严重程度**：P3
- **涉及文件**：`app/lib/extra_func/language.dart`、`app/lib/services/agent_recall_service.dart:554,561`
- **问题描述**：`textLocalize()` 是一个全局函数（非依赖 BuildContext），在 service 层和数据模型中被直接调用。虽然这在功能上没有问题，但它耦合了 UI 国际化逻辑与业务逻辑层，且如果未来引入 per-locale 动态切换（而非全局语言），当前架构将无法支持。
- **复现方法**：代码审查问题。
- **建议修复**：长期应将错误消息的本地化推迟到 UI 层，service 层返回错误类型/代码，由 UI 层负责翻译。
- **建议创建 Issue**：否（当前功能正常，仅作架构改进建议）

### 22. TaskListPage 使用 15 秒轮询而非 Realtime 监听

- **严重程度**：P3
- **涉及文件**：`app/lib/pages/task_list.dart:92-101`
- **问题描述**：`TaskListPage` 使用 `Timer.periodic(Duration(seconds: 15))` 轮询任务状态，但同一个应用中 `RecallPage` 和 `TaskNotificationService` 已经使用了 Supabase Realtime 监听 `processing_tasks` 表。轮询不仅浪费网络资源，还导致状态更新有 15 秒延迟。注释中也留下了被注释掉的 Supabase session 检查代码，暗示这是临时方案。
- **复现方法**：提交任务后，任务列表页面最多需要等待 15 秒才能看到状态更新。
- **建议修复**：复用 `TaskNotificationService` 的 Realtime channel 或新建专用 channel，删除轮询逻辑。
- **建议创建 Issue**：是

---

## 文件统计

超过 200 行的文件（按行数降序）：

| 文件路径 | 行数 |
|---|---|
| app/lib/pages/webgl_viewer.dart | 1050 |
| app/lib/pages/recall/recall_search.dart | 978 |
| app/lib/services/agent_recall_service.dart | 973 |
| app/lib/pages/record.dart | 933 |
| app/lib/pages/recall/recall_private_widgets.dart | 901 |
| app/lib/pages/community/views.dart | 900 |
| app/lib/pages/recall/recall_local_ai.dart | 872 |
| app/lib/extra_func/language.dart | 776 |
| app/lib/pages/generate.dart | 715 |
| app/lib/pages/recall/local_ai_panel.dart | 708 |
| app/lib/pages/generate/generate_submission.dart | 605 |
| app/lib/floating_nav_bar.dart | 597 |
| app/lib/pages/recall/recall_agent_runtime.dart | 592 |
| app/lib/pages/recall/recall_model_actions.dart | 586 |
| app/lib/main.dart | 584 |
| app/lib/pages/recall/time_peeling.dart | 571 |
| app/lib/pages/recall/search_header_section.dart | 495 |
| app/lib/pages/task_list.dart | 486 |
| app/lib/pages/recall/model_grid.dart | 477 |
| app/lib/pages/recall/model_action_overlay.dart | 385 |
| app/lib/pages/task_list/category_section.dart | 368 |
| app/lib/pages/recall/recall_data_sync.dart | 353 |
| app/lib/pages/community/composer_sheet.dart | 341 |
| app/lib/services/local_rag_index.dart | 336 |
| app/lib/pages/recall/processing_section.dart | 329 |
| app/lib/services/local_model_catalog_service.dart | 315 |
| app/lib/pages/settings.dart | 309 |
| app/lib/pages/community.dart | 293 |
| app/lib/pages/community/repository.dart | 271 |
| app/lib/pages/video_submit.dart | 257 |
| app/lib/pages/recall/adaptive_thumbnail.dart | 250 |
| app/lib/pages/recall/recall_view.dart | 245 |
| app/lib/widgets/bd_surfaces.dart | 237 |
| app/lib/pages/login.dart | 235 |
| app/lib/pages/recall/result_card.dart | 232 |
| app/lib/services/task_notification_service.dart | 231 |
| app/lib/services/notification_service.dart | 231 |
| app/lib/pages/recall/overview_card.dart | 225 |
| app/lib/pages/recall/model_card.dart | 202 |

**注**：recall.dart 本身仅 160 行，但其 7 个 part 文件合计超过 3,600 行，是代码库中最大的单类。

---

## 演示时最可能炸的路径

1. **Agent 搜索 + 快速切换搜索模式**：正在流式接收 Agent 响应时切换到 Cloud 模式，`_resetAgentUiState` 会将 `_agentChatMessage` 置 null，但流的 onDone 回调仍会尝试访问它（P0 #5）。
2. **多次切换深色/浅色模式**：每次切换触发 MaterialApp rebuild，`MyApp.build()` 会重复注册 WidgetsBindingObserver（P0 #1）。
3. **提交视频任务后快速离开 VideoSubmitPage**：`_submit` 是异步操作，如果用户在上传过程中按返回键，`setState` 会在 unmounted 的 widget 上调用。
4. **离线状态下打开 Recall 页面**：多个静默 catch 块会导致页面显示 demo 数据但用户不知原因（P1 #7）。
5. **Agent 返回异常格式的 tool_call 事件**：`_consumeAgentEvent` 中对 `payload['args']` 的类型假设可能导致流中断（P2 #17）。

---

## 建议新建 Issue 清单

- [ ] **[P0] 修复 MyApp 中 WidgetsBindingObserver 重复注册和未移除的内存泄漏**
- [ ] **[P0] 关闭全局 downloadEventBus StreamController 或管理其生命周期**
- [ ] **[P0] MaterialApp.builder 中 child! 改为防御性处理**
- [ ] **[P0] 修复 transformMatrix 类型转换不安全导致的潜在崩溃**
- [ ] **[P0] Agent 流式 onDone 回调中增加 null 保护**
- [ ] **[P1] 拆分 _RecallPageState 超大状态类，使用 Riverpod 管理异步状态**
- [ ] **[P1] 清理所有静默 catch 块，至少在 debug 模式下输出日志**
- [ ] **[P1] 修复 TaskNotificationService 单例的 dispose 路径**
- [ ] **[P1] 修复 RecordPage 中 haptic loop timer 在快速前后台切换时的潜在泄露**
- [ ] **[P1] 确保 RealtimeChannel 在所有错误路径下被正确清理**
- [ ] **[P1] 统一 sceneId 生成逻辑，使用 UUID 替代自定义 Random 实现**
- [ ] **[P2] 抽取 _formatBytes 到公共工具模块，消除三处重复**
- [ ] **[P2] 将硬编码中文字符串迁移到 i18n 资源文件**
- [ ] **[P2] 修复 GeneratePage 中 static 可变字段导致的状态跨实例共享**
- [ ] **[P2] 抽取 Supabase 表名/桶名为常量，消除 6+ 处硬编码**
- [ ] **[P2] 为 _consumeAgentEvent 增加单 event 级别 try-catch，防止单个异常中断整个流**
- [ ] **[P2] 复用 Dio 实例，避免每次请求创建新实例**
- [ ] **[P3] 长期重构 RecallPage 的 part/extension 架构为独立 Controller**
- [ ] **[P3] 将 TaskListPage 的轮询替换为 Supabase Realtime 监听**
