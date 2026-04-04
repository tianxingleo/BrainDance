# Flutter-Supabase 详细集成测试用例表

更新时间：2026-03-27

## 文档说明

本文在 [Flutter-Supabase交互梳理与集成测试方案-2026-03-27.md](/home/ltx/projects/BrainDance/docs/4.测试（待审阅）/Flutter-Supabase交互梳理与集成测试方案-2026-03-27.md) 的基础上，进一步细化为可执行的集成测试用例。

为满足“表格各条目纵向排列”的要求，本文按“每个用例一张纵向表”的形式组织。每张表固定包含：

- 用例编号
- 测试单元描述
- 用例目的
- 前提条件
- 特殊的规程说明
- 用例间的依赖关系

---

## BD-IT-AUTH-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-AUTH-001` |
| 测试单元描述 | 普通用户使用合法邮箱和密码登录 Flutter 应用，验证 Supabase Auth 登录链路、页面跳转和会话建立是否正常。 |
| 用例目的 | 确认 `auth.signInWithPassword`、首屏路由判断、会话驻留和首页进入逻辑在 `RLS 用户模式` 下可正常工作。 |
| 前提条件 | 1. 本地 Supabase 已启动。 2. 已存在测试账号 `user_a@test.local`。 3. Flutter 测试环境使用 `anon key`。 |
| 特殊的规程说明 | 1. 测试前需清理本地缓存和历史会话。 2. 不允许复用前序测试遗留登录态。 3. 成功判定应同时包含 UI 跳转与 Supabase Session 非空。 |
| 用例间的依赖关系 | 无。该用例是 `RLS 用户模式` 基础入口，用于为 `BD-IT-TASK-*`、`BD-IT-RECALL-*`、`BD-IT-COMM-*` 提供登录前置样例。 |

## BD-IT-AUTH-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-AUTH-002` |
| 测试单元描述 | 普通用户使用错误密码登录 Flutter 应用，验证错误提示、页面停留和会话不创建。 |
| 用例目的 | 确认 Supabase Auth 失败路径被正确处理，避免前端误判为登录成功。 |
| 前提条件 | 1. 本地 Supabase 已启动。 2. 测试账号 `user_a@test.local` 已存在。 3. 当前设备无有效登录态。 |
| 特殊的规程说明 | 1. 必须断言 `auth.currentSession == null`。 2. 只允许出现登录错误提示，不允许跳转到首页。 3. 不应污染后续测试环境。 |
| 用例间的依赖关系 | 无。可独立执行，建议在 `BD-IT-AUTH-001` 之前运行以验证失败路径。 |

## BD-IT-AUTH-003

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-AUTH-003` |
| 测试单元描述 | 在 `Admin Mode` 下启动 Flutter 应用，验证是否绕过登录页直接进入主应用。 |
| 用例目的 | 确认 `SupabaseConfig.isAdminMode` 对首屏路由和登录页逻辑的影响符合当前实现。 |
| 前提条件 | 1. Flutter 测试环境使用 `SUPABASE_SECRET_KEY` 或 `SUPABASE_SERVICE_ROLE_KEY`。 2. 本地 Supabase 已启动。 |
| 特殊的规程说明 | 1. 该用例必须在独立环境变量配置下运行，不能与 `anon key` 场景混跑。 2. 判定标准是首屏直接进入 `/`，而不是通过模拟用户点击跳过。 |
| 用例间的依赖关系 | 无。与 `BD-IT-AUTH-001`、`BD-IT-AUTH-002` 平行，属于另一条运行口径。 |

## BD-IT-AUTH-004

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-AUTH-004` |
| 测试单元描述 | 用户在任务页已登录状态下触发登出，验证任务页是否清空并回到未登录状态。 |
| 用例目的 | 覆盖 `task_list.dart` 中 `auth.onAuthStateChange` 的监听逻辑，确认登出后任务页不会继续显示旧数据。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001` 成功登录。 2. 数据库中至少有 1 条属于该用户的 `processing_tasks`。 |
| 特殊的规程说明 | 1. 需要从测试脚本主动执行 `supabase.auth.signOut()` 或触发登出入口。 2. 要同时校验 `_tasksByStatus` 清空和错误提示文案出现。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。该用例完成后需要重新登录，供后续用户态用例继续执行。 |

---

## BD-IT-TASK-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-TASK-001` |
| 测试单元描述 | 已登录用户提交图片转 3D 任务，验证素材上传到 `braindance-assets` 且 `processing_tasks` 成功写入。 |
| 用例目的 | 覆盖 `generate_submission.dart` 中“Storage 上传 + processing_tasks 插入”的主提交流程。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 准备好可访问的测试图片文件。 3. `braindance-assets` bucket 已存在。 |
| 特殊的规程说明 | 1. 需从数据库断言新增任务的 `task_type` 正确。 2. 需从 Storage 断言 `raw/image.png` 已存在。 3. 应记录新生成的 `scene_id`，供后续 `BD-IT-RECALL-*` 用例复用。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。成功后可作为 `BD-IT-RECALL-001`、`BD-IT-TASK-004` 的数据来源。 |

## BD-IT-TASK-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-TASK-002` |
| 测试单元描述 | 已登录用户提交视频转 3D 任务，验证视频上传、任务类型、任务参数和页面跳转。 |
| 用例目的 | 覆盖 `generate_submission.dart` 中视频提交流程和 `task_params` 写入逻辑。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 准备好可访问的测试视频文件。 3. `braindance-assets` bucket 已存在。 |
| 特殊的规程说明 | 1. 需要校验 `raw/video.mp4` 上传成功。 2. 需要断言 `processing_tasks.task_type` 与当前所选视频管线一致。 3. 若使用自定义视频任务类型，需同时校验 `task_params` JSON。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。可为 `BD-IT-RECALL-002` 与 `BD-IT-REALTIME-001` 提供后续任务数据。 |

## BD-IT-TASK-003

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-TASK-003` |
| 测试单元描述 | 从专用视频提交页提交 `video_dual_chain` 任务，验证 `display_name`、`task_type` 和双链路参数是否写库。 |
| 用例目的 | 覆盖 `video_submit.dart` 的旧提交入口，确保其未被主提交流程演进破坏。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 准备测试视频与缩略图。 |
| 特殊的规程说明 | 1. 需要在断言中检查 `display_name`。 2. 需要校验 `task_type = video_dual_chain`。 3. 需要校验 `task_params.slow_pipeline = video_3dgs` 等关键字段存在。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。执行结果可为 `BD-IT-RECALL-005` 提供同名版本样本。 |

## BD-IT-TASK-004

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-TASK-004` |
| 测试单元描述 | 未登录用户尝试提交任务，验证系统是否先引导登录，再在登录完成后继续提交。 |
| 用例目的 | 覆盖 `_requireAuthenticatedUser()` 的导航和回流逻辑，避免无登录态下直接调用 Storage/DB。 |
| 前提条件 | 1. 当前应用无登录态。 2. 已准备测试图片或视频文件。 3. 测试账号有效。 |
| 特殊的规程说明 | 1. 此用例需拆成两个断言阶段：先跳登录页，再验证登录后提交流程完成。 2. 不允许通过预先注入 session 绕过真实导航。 |
| 用例间的依赖关系 | 与 `BD-IT-AUTH-001` 有逻辑关联，但不要求事先登录。建议在重新清空登录态后单独执行。 |

## BD-IT-TASK-005

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-TASK-005` |
| 测试单元描述 | 用户在任务列表页读取自己的 `processing_tasks`，验证按状态分组、日志解析和排序是否正确。 |
| 用例目的 | 覆盖 `task_list.dart` 对 `processing_tasks` 的查询、分组和日志提取逻辑。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 数据库中存在该用户多种状态的任务样本。 |
| 特殊的规程说明 | 1. 应至少准备 `pending`、`processing`、`completed` 三类任务。 2. 若有 `logs` 字段，需验证日志文本被正确提取。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。可复用 `BD-IT-TASK-001`、`BD-IT-TASK-002`、`BD-IT-TASK-003` 产生的数据。 |

---

## BD-IT-RECALL-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-001` |
| 测试单元描述 | Recall 页面首次加载 `model_assets` 列表，并用 `processing_tasks.display_name` 回填展示名。 |
| 用例目的 | 覆盖 Recall 首页最核心的“模型表读取 + 任务表补显示名”逻辑。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 数据库中存在至少 1 条 `model_assets` 与对应 `processing_tasks`。 |
| 特殊的规程说明 | 1. 必须断言模型卡片展示名优先使用 `display_name` 而不是 `scene_id`。 2. 测试数据应包含一个没有 `display_name` 的对照样本。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`，可复用 `BD-IT-TASK-001` 或预置种子数据。 |

## BD-IT-RECALL-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-002` |
| 测试单元描述 | Recall 页面加载 `processing` 状态任务列表，验证处理中任务区和日志显示。 |
| 用例目的 | 覆盖 `_fetchProcessingTasks()` 对 `processing_tasks(status='processing')` 的读取行为。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 数据库中存在至少 1 条 `status = processing` 的任务。 |
| 特殊的规程说明 | 1. 需准备包含 `logs` 数组的任务。 2. 断言不仅看任务存在，还要验证日志文案被解析展示。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。建议在 `BD-IT-REALTIME-001` 之前执行，作为静态基线。 |

## BD-IT-RECALL-003

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-003` |
| 测试单元描述 | Recall 页面搜索走 `search-models` Edge Function，验证云端检索结果可被页面消费。 |
| 用例目的 | 覆盖 Recall 搜索的“前端提交 + Edge Function + 返回结果渲染”链路。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. `search-models` 函数可用。 3. 数据库中存在可命中的 `memory_poses` / `model_assets` 样本。 |
| 特殊的规程说明 | 1. 需验证函数返回结构中 `success` 与 `results`。 2. 若使用真实 embedding，测试环境需提供模型网关密钥；若用 stub，则要显式标记为桩环境。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。与 `BD-IT-EFUNC-001` 使用同一条服务端能力，可共享种子数据。 |

## BD-IT-RECALL-004

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-004` |
| 测试单元描述 | 用户在 Recall 中修改模型显示名，验证 `processing_tasks.display_name` 更新和前端局部刷新。 |
| 用例目的 | 覆盖 Recall 重命名逻辑，确认当前显示名来源链路仍然可写可读。 |
| 前提条件 | 1. 已完成 `BD-IT-RECALL-001`。 2. 目标模型对应 `scene_id` 在 `processing_tasks` 中存在。 |
| 特殊的规程说明 | 1. 需在数据库层断言更新成功。 2. 需在 UI 层断言当前列表与动作弹层同步更新。 3. 测试结束后应回滚原名称，避免影响其他依赖同名的用例。 |
| 用例间的依赖关系 | 依赖 `BD-IT-RECALL-001`。可能影响 `BD-IT-RECALL-005` 的同名版本测试，因此应在后者之前恢复数据。 |

## BD-IT-RECALL-005

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-005` |
| 测试单元描述 | 已完成任务点击进入 Viewer 时，系统查询同名场景兄弟模型，验证 `Time Peeling` 数据集合构建正确。 |
| 用例目的 | 覆盖 `viewer_navigation.dart` 中“任务表查 display_name，再反查同名 scene_id，再查 model_assets”的复合查询。 |
| 前提条件 | 1. 已完成 `BD-IT-TASK-003` 或已有两条同 `display_name` 的历史数据。 2. 至少一条对应 `model_assets.ply_path` 可用。 |
| 特殊的规程说明 | 1. 需准备同名不同时间版本样本。 2. 除了 Viewer 打开，还要断言 `timePeelingModels` 数据长度符合预期。 |
| 用例间的依赖关系 | 依赖 `BD-IT-TASK-003` 或独立种子数据；与 `BD-IT-RECALL-004` 存在数据名称依赖，建议在重命名恢复后执行。 |

## BD-IT-RECALL-006

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-006` |
| 测试单元描述 | Recall 删除当前用户自己的云端模型，验证 Storage 目录和 `model_assets` 记录都被删除。 |
| 用例目的 | 覆盖 Recall 删除链路中的目录递归枚举、Storage 删除、数据库删除和本地状态收敛。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 目标模型属于当前用户。 3. 模型对应 Storage 路径存在。 |
| 特殊的规程说明 | 1. 该用例是破坏性操作，应使用专门的临时模型样本。 2. 需同时断言 Storage 和数据库两侧结果。 3. 删除后要验证列表和本地索引状态更新。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。不应使用其他测试共享样本，避免破坏后续用例。 |

## BD-IT-RECALL-007

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-RECALL-007` |
| 测试单元描述 | 非当前用户尝试删除他人云端模型，验证删除被前端权限校验拦截。 |
| 用例目的 | 确认 Recall 删除入口不会对非本人模型发起危险写操作。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 数据库中存在 `user_b` 的模型样本。 |
| 特殊的规程说明 | 1. 需要伪造或切换到可见但不属于当前用户的模型数据。 2. 应断言没有调用删除副作用，Storage 和数据库均不变化。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`，建议在 `BD-IT-RECALL-006` 之前执行。 |

---

## BD-IT-REALTIME-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-REALTIME-001` |
| 测试单元描述 | Recall 页面在任务状态被更新为 `processing` 时，通过 Realtime 自动显示处理中任务。 |
| 用例目的 | 覆盖 `channel('public:processing_tasks:recall')` 和 `onPostgresChanges(event: all)` 的增量更新逻辑。 |
| 前提条件 | 1. 已完成 `BD-IT-RECALL-002` 基线加载。 2. 测试脚本具备后台更新数据库任务状态的能力。 |
| 特殊的规程说明 | 1. 需要从脚本侧异步把目标任务从 `pending` 改为 `processing`。 2. 要给 UI 留出合理订阅和事件传播时间。 |
| 用例间的依赖关系 | 依赖 `BD-IT-RECALL-002`。完成后可继续执行 `BD-IT-REALTIME-002`。 |

## BD-IT-REALTIME-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-REALTIME-002` |
| 测试单元描述 | Recall 页面在任务从 `processing` 更新为 `completed` 或 `failed` 时，自动将其从处理中列表移除。 |
| 用例目的 | 覆盖 `_handleRealtimeChange()` 中“从 processing 移出”的分支。 |
| 前提条件 | 1. 已完成 `BD-IT-REALTIME-001`。 2. 目标任务当前已在 Recall 的处理中列表中。 |
| 特殊的规程说明 | 1. 需同时验证 `_taskAllLogs` 与 `_expandedTaskLogs` 对应状态被清除。 2. 完成状态与失败状态建议各跑一次。 |
| 用例间的依赖关系 | 强依赖 `BD-IT-REALTIME-001`。 |

## BD-IT-REALTIME-003

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-REALTIME-003` |
| 测试单元描述 | 全局任务通知在任务状态从非终态切换为 `completed` 或 `failed` 时弹出一次通知，进入任务页后清零。 |
| 用例目的 | 覆盖 `task_notification_service.dart` 的待通知计数、去重缓存和路由规避逻辑。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 当前不在 `/tasks` 页面。 3. 有可控的任务状态变更样本。 |
| 特殊的规程说明 | 1. 需要验证同一任务不会重复通知。 2. 需要验证进入 `/tasks` 后计数清零并写入本地缓存。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。可复用 `BD-IT-REALTIME-001` 的状态更新脚本，但不要求 Recall 页面处于打开状态。 |

---

## BD-IT-COMM-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-COMM-001` |
| 测试单元描述 | 社区页面读取帖子流并联表 `model_assets`，验证帖子内容、模型信息和封面地址映射。 |
| 用例目的 | 覆盖 `community_posts` 读取和 `model_assets` 联表查询逻辑。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001` 或处于允许匿名读取的环境。 2. 数据库中存在社区帖子样本。 |
| 特殊的规程说明 | 1. 需准备一条 `cover_image_url` 为空、依赖模型预览图回填的样本。 2. 要断言模型 URL 与 poses URL 推导逻辑正确。 |
| 用例间的依赖关系 | 无强依赖。可独立使用种子数据执行。 |

## BD-IT-COMM-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-COMM-002` |
| 测试单元描述 | 用户从可分享模型列表选择模型并发布社区帖子，验证 `community_posts` 写入成功。 |
| 用例目的 | 覆盖 `CommunityRepository.createPost()` 的服务端成功路径。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 当前用户至少有 1 条可分享的 `model_assets`。 |
| 特殊的规程说明 | 1. 需要断言数据库新增记录字段完整。 2. 需要校验社区列表刷新后能看到新帖子。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001` 与至少一条当前用户模型数据，可复用 `BD-IT-TASK-*` 后由种子脚本补出的 `model_assets`。 |

## BD-IT-COMM-003

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-COMM-003` |
| 测试单元描述 | 社区发帖时服务端插入失败，验证本地 `_localDrafts` 乐观回退仍能显示帖子。 |
| 用例目的 | 覆盖 `createPost()` 的失败降级路径，避免网络或权限异常时页面直接空白。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 已准备一个可分享模型。 3. 测试环境可人为让 `community_posts` 插入失败。 |
| 特殊的规程说明 | 1. 失败应由受控手段制造，例如临时修改测试环境策略或拦截接口返回。 2. 需要同时断言 UI 有帖子、数据库无记录。 |
| 用例间的依赖关系 | 依赖 `BD-IT-COMM-002` 的交互路径认知，但可独立执行。 |

---

## BD-IT-EFUNC-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-EFUNC-001` |
| 测试单元描述 | 直接验证 `search-models` Edge Function 返回结构和 Recall 页面消费结果的一致性。 |
| 用例目的 | 将函数侧协议验证和 Flutter 页面消费验证串联，确保接口字段变化不会悄悄破坏 UI。 |
| 前提条件 | 1. `search-models` 函数可运行。 2. 测试数据中存在可命中的向量样本。 |
| 特殊的规程说明 | 1. 建议同时记录函数原始响应。 2. 若使用模拟 embedding，应在报告中标注非真实模型环境。 |
| 用例间的依赖关系 | 与 `BD-IT-RECALL-003` 相互支撑，建议先跑本用例确认函数稳定，再跑页面用例。 |

## BD-IT-EFUNC-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-EFUNC-002` |
| 测试单元描述 | Flutter 通过流式请求调用 `agent-recall`，验证事件序列、最终回答和页面实时更新。 |
| 用例目的 | 覆盖 `AgentRecallService.queryStream()`、Flutter 事件消费和 `agent-recall` 流式输出协议。 |
| 前提条件 | 1. `agent-recall` 函数可运行。 2. 测试数据中存在可检索场景。 3. 如使用真实上游模型，需配置相关密钥。 |
| 特殊的规程说明 | 1. 必须记录收到的事件顺序。 2. 至少断言 `ping`、`status`、`done`。 3. 如存在 `top_candidates`、`tool_trace`、`follow_up`，需分别校验结构。 |
| 用例间的依赖关系 | 无强依赖，但建议在 `BD-IT-EFUNC-001` 之后执行。 |

## BD-IT-EFUNC-003

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-EFUNC-003` |
| 测试单元描述 | 当 `agent-recall` 流式链路失败时，Flutter 自动回退到非流式 `functions.invoke`，验证仍能拿到结果或明确错误。 |
| 用例目的 | 覆盖 `AgentRecallService.queryStream()` 中的 fallback 分支，防止流式异常导致整页不可用。 |
| 前提条件 | 1. 已具备 `BD-IT-EFUNC-002` 的运行条件。 2. 测试环境可稳定制造流式失败。 |
| 特殊的规程说明 | 1. 必须用可控手段制造“流式失败但函数本体仍可调用”的场景。 2. 需要断言 UI 中出现 fallback 路径标记或最终回答。 |
| 用例间的依赖关系 | 建议依赖 `BD-IT-EFUNC-002`，先确认流式正常，再验证回退异常路径。 |

## BD-IT-EFUNC-004

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-EFUNC-004` |
| 测试单元描述 | 文生图确认链路调用 `confirm-text-image`，验证函数侧完成图片下载、Storage 上传和 `processing_tasks` 写库。 |
| 用例目的 | 覆盖 `confirm-text-image` 的复合事务链路，确认 Flutter 只调函数也能形成完整任务。 |
| 前提条件 | 1. 已完成 `BD-IT-AUTH-001`。 2. 有可用的测试图片 URL。 3. `confirm-text-image` 函数可运行。 |
| 特殊的规程说明 | 1. 建议使用可控静态图片 URL，而非依赖 `text-to-image` 实时生成结果。 2. 必须同时检查函数返回、Storage 对象存在、数据库记录存在。 |
| 用例间的依赖关系 | 依赖 `BD-IT-AUTH-001`。与 `BD-IT-EFUNC-005` 可串联成完整文生图流程。 |

## BD-IT-EFUNC-005

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-EFUNC-005` |
| 测试单元描述 | Flutter 发起 `text-to-image`，验证生图结果可用于后续确认流程，但默认不要求真实联网大模型稳定返回。 |
| 用例目的 | 为完整文生图链路提供前半段验证，同时把外部依赖不稳定性与 Supabase 集成点拆开。 |
| 前提条件 | 1. `text-to-image` 函数可运行。 2. 若使用真实 DashScope，已配置密钥；若使用桩环境，已准备固定返回。 |
| 特殊的规程说明 | 1. 默认推荐 stub 或录制回放。 2. 若执行真实联网测试，应标记为“扩展回归”而非每次 CI 必跑。 |
| 用例间的依赖关系 | 可独立执行；若要形成完整业务链路，则作为 `BD-IT-EFUNC-004` 的上游前置。 |

## BD-IT-EFUNC-006

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-EFUNC-006` |
| 测试单元描述 | Agent 写工具在 `preview` 与 `execute` 模式下分别执行，验证数据库副作用仅在执行模式下产生。 |
| 用例目的 | 覆盖 Agent 写能力的安全边界，避免预览模式误写数据库。 |
| 前提条件 | 1. `agent-recall` 或相关共享 Agent 工具链可运行。 2. 已准备会触发写工具的测试问句和目标模型。 |
| 特殊的规程说明 | 1. 必须先跑 `preview` 再跑 `execute`。 2. 需要在数据库层分别截取前后快照。 3. 若涉及 `model_assets` 重命名或专题创建，测试后要清理痕迹。 |
| 用例间的依赖关系 | 建议依赖 `BD-IT-EFUNC-002` 的基础 Agent 可用性验证。 |

---

## BD-IT-LOCALAI-001

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-LOCALAI-001` |
| 测试单元描述 | 本地 AI 模型目录读取 catalog 成功，验证 `braindance-models/catalog/model_catalog.json` 被正确解析。 |
| 用例目的 | 覆盖本地模型分发的首选读取路径，确保端侧模型列表可稳定展示。 |
| 前提条件 | 1. `braindance-models` bucket 已存在。 2. `catalog/model_catalog.json` 已上传。 |
| 特殊的规程说明 | 1. 建议准备至少 2 个模型条目，其中 1 个为推荐模型。 2. 需校验排序逻辑是否把推荐模型放在前面。 |
| 用例间的依赖关系 | 无。可独立执行。 |

## BD-IT-LOCALAI-002

| 字段 | 内容 |
| --- | --- |
| 用例编号 | `BD-IT-LOCALAI-002` |
| 测试单元描述 | catalog 缺失时，本地 AI 模型目录退回到 bucket 扫描 `.gguf` 文件。 |
| 用例目的 | 覆盖 `LocalModelCatalogService.fetchCatalog()` 的回退路径，确保 catalog 丢失不会导致端侧模型功能完全失效。 |
| 前提条件 | 1. `braindance-models` 或 `braindance-assets` 中存在 `.gguf` 文件。 2. `catalog/model_catalog.json` 不存在或不可读。 |
| 特殊的规程说明 | 1. 要显式确保测试环境没有可读 catalog。 2. 需断言扫描结果被去重且可生成下载 URL。 |
| 用例间的依赖关系 | 建议在 `BD-IT-LOCALAI-001` 之后执行，以先确认正常路径，再验证回退路径。 |

---

## 推荐执行顺序

建议默认执行顺序如下：

1. `BD-IT-AUTH-002`
2. `BD-IT-AUTH-001`
3. `BD-IT-TASK-001`
4. `BD-IT-TASK-002`
5. `BD-IT-TASK-003`
6. `BD-IT-TASK-005`
7. `BD-IT-RECALL-001`
8. `BD-IT-RECALL-002`
9. `BD-IT-REALTIME-001`
10. `BD-IT-REALTIME-002`
11. `BD-IT-REALTIME-003`
12. `BD-IT-RECALL-003`
13. `BD-IT-EFUNC-001`
14. `BD-IT-EFUNC-002`
15. `BD-IT-EFUNC-003`
16. `BD-IT-COMM-001`
17. `BD-IT-COMM-002`
18. `BD-IT-COMM-003`
19. `BD-IT-LOCALAI-001`
20. `BD-IT-LOCALAI-002`
21. `BD-IT-EFUNC-004`
22. `BD-IT-EFUNC-005`
23. `BD-IT-EFUNC-006`
24. `BD-IT-RECALL-007`
25. `BD-IT-RECALL-006`
26. `BD-IT-AUTH-004`

其中：

- `BD-IT-RECALL-006` 是破坏性删除，放在较后位置
- `BD-IT-AUTH-004` 会清空登录态，放在收尾
- `BD-IT-EFUNC-005` 若使用真实联网模型，可从默认回归集中拆出
