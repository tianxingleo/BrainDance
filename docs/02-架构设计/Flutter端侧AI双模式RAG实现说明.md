# Flutter 端侧 AI 双模式 RAG 实现说明

## 1. 背景

BrainDance 现有的回忆检索页面位于 `app/lib/pages/recall.dart`。在原始实现里，用户输入搜索词后，客户端会调用 Supabase Edge Function `search-models`，由云端完成 Embedding 生成和向量检索，再把结果返回给 Flutter 页面。

这个链路能工作，但有两个明显问题：

1. 用户搜索词属于高隐私数据，每次搜索都上传到云端不合适。
2. 搜索强依赖网络，离线或弱网场景体验差。

因此，这次实现的目标不是“替换掉云端搜索”，而是给 `Recall` 页增加两个可切换的检索模式：

1. `云端语义检索`
2. `端侧隐私检索`

这样用户可以按场景选择：

1. 如果希望获得现有云端语义检索能力，继续走 Edge Function。
2. 如果更重视隐私和离线可用性，走端侧本地向量检索。

## 2. 最终效果

现在 `Recall` 页的搜索区域新增了模式切换：

1. `端侧隐私检索`
2. `云端语义检索`

页面行为如下：

1. 页面加载时，仍然从 `model_assets` 拉取模型基础信息。
2. 同时客户端会在后台把模型描述、标签、对象列表和 `meta_info` 中可用文本整理成检索语料。
3. 客户端将这些文本生成本地向量，并存入 SQLite。
4. 当用户选择“端侧隐私检索”时，搜索时只在本地数据库里计算相似度，不调用云端接口。
5. 当用户选择“云端语义检索”时，仍然调用现有 `search-models` Edge Function。

这意味着项目当前已经具备一套双轨检索架构，而不是只有单一路径。

## 3. 我做了什么

这次主要改了 5 个地方。

### 3.1 Recall 页面增加双模式搜索入口

文件：

- `app/lib/pages/recall.dart`

做的事情：

1. 增加 `_RecallSearchMode` 枚举，定义 `local` 和 `cloud` 两种模式。
2. 在搜索框下方增加模式切换 UI。
3. 搜索逻辑 `_searchModels()` 根据当前模式分流：
   - `local` -> 走本地索引服务
   - `cloud` -> 走原来的 Edge Function
4. 空结果提示也按模式分别展示。

### 3.2 新增端侧文本向量器抽象

文件：

- `app/lib/services/local_text_embedder.dart`

做的事情：

1. 定义 `LocalTextEmbedder` 抽象接口。
2. 实现 `HashingTextEmbedder` 作为当前默认 embedder。

这个实现不是 MiniLM 本体，而是一个纯 Dart、本地可运行、零网络依赖的轻量语义向量器。它做了这些事：

1. 对输入文本做归一化。
2. 提取中英文 token。
3. 生成字符级 2-gram / 3-gram 特征。
4. 通过 hashing trick 映射到固定维度向量。
5. 做 L2 归一化，方便余弦相似度检索。

这么做的原因很务实：

1. 先把端侧检索链路搭通。
2. 避免引入过重的本地模型运行依赖，导致这次改动无法落地。
3. 为后续替换为 INT8 MiniLM / ONNX Runtime 预留统一接口。

也就是说，这次实现完成的是“端侧 RAG 框架与数据流”，而 `HashingTextEmbedder` 是当前可运行的默认实现。

### 3.3 新增本地向量索引服务

文件：

- `app/lib/services/local_rag_index.dart`

做的事情：

1. 使用 `sqflite` 创建本地数据库 `braindance_memory_rag.db`。
2. 建表 `memory_scene_vectors`，存储：
   - `model_id`
   - `scene_id`
   - `user_id`
   - `searchable_text`
   - `vector_json`
   - `payload_json`
   - `fingerprint`
   - `updated_at`
3. 提供 `syncModels()`：
   - 把从云端拉到的模型数据同步到本地索引
   - 通过 `fingerprint` 判断是否需要重建向量
   - 清理本地多余旧记录
4. 提供 `search()`：
   - 对 query 做本地 embedding
   - 遍历本地向量
   - 计算余弦相似度
   - 增加少量 lexical boost，提升明确关键词命中
   - 返回排序后的模型列表

这里采用 SQLite 保存向量，而不是 ObjectBox Vector Search，原因是：

1. 当前项目 Flutter 端还没有接入 ObjectBox。
2. SQLite + `sqflite` 接入成本最低，改动最小。
3. 当前数据量不大时，完全能支撑端上扫描和相似度计算。

这版更准确地说，是“SQLite 持久化 + Flutter 端计算余弦相似度”的端侧向量检索方案。

## 4. 数据流是怎么走的

### 4.1 页面初始化

`RecallPage.initState()` 会做三件事：

1. `_fetchModels()`
2. `_fetchProcessingTasks()`
3. `_setupRealtimeListener()`

其中 `_fetchModels()` 现在不只拉基础字段，还额外拉了：

1. `user_id`
2. `objects`
3. `tags`
4. `meta_info`

原因是端侧检索需要足够多的文本线索。

### 4.2 本地索引构建

`_fetchModels()` 成功后，会调用 `_syncLocalIndex(models)`。

`_syncLocalIndex()` 内部会：

1. 把每条模型记录组装成 `searchable_text`
2. 文本来源包括：
   - `scene_id`
   - `description`
   - `tags`
   - `objects`
   - `meta_info` 中可抽取的字符串内容
3. 调用 `LocalTextEmbedder.embed()`
4. 将向量和原始模型 payload 一起写入本地 SQLite

这样做的好处是：

1. 搜索时不需要再去云端取文本。
2. 页面重启后索引仍然存在。
3. 模型信息更新后只需要增量重建。

### 4.3 用户搜索

用户输入搜索词时，会经过一个 180ms debounce，避免每个字符都触发重算。

然后根据模式分成两条路径。

#### 路径 A：端侧隐私检索

1. 调用 `LocalRagIndexService.search(query)`
2. 生成 query 向量
3. 读取本地索引表中的所有向量
4. 计算相似度
5. 返回排序后的结果

这个过程中不会调用 Supabase Function，也不会上传 query。

#### 路径 B：云端语义检索

1. 调用 `Supabase.instance.client.functions.invoke('search-models', body: {'query': query})`
2. 使用原有云端语义搜索逻辑
3. 返回结果到 Flutter 页面展示

所以这次改动保留了历史能力，没有破坏现有云端搜索。

## 5. 我为什么这样做

这次实现遵循的是“先打通架构，再替换模型”的策略。

如果一开始就强行在 Flutter 里直接接 ONNX MiniLM，会遇到这些问题：

1. 需要引入额外运行时依赖。
2. Android / iOS 打包与模型分发会变复杂。
3. 还要处理 tokenizer、量化模型、推理线程和性能问题。
4. 一次改动跨度太大，不利于稳定落地。

因此我拆成两层：

1. 第一层：先实现双模式检索架构
2. 第二层：把本地 embedding 能力抽象成接口，方便未来替换

这样现在已经得到一个可运行版本，并且后面升级成本很低。

## 6. 现在的端侧 Embedding 方案具体是什么

当前默认实现是 `HashingTextEmbedder`，不是深度模型 Embedding。

它的本质是一个轻量文本向量器，优点是：

1. 完全本地
2. 无需网络
3. 无需原生推理框架
4. 接入非常稳定
5. 对中英文关键词检索和短语召回已经有基本效果

但它也有边界：

1. 语义泛化能力不如 MiniLM 这类真正的 embedding 模型
2. 对复杂自然语言理解能力有限
3. 更适合当前阶段的隐私检索兜底和本地离线搜索

因此，它是一个工程上可上线的第一版，而不是终局方案。

## 7. 如果后续接 INT8 MiniLM，应该怎么替换

这次实现已经把替换点留好了，主要只需要动 `app/lib/services/local_text_embedder.dart`。

推荐替换步骤：

1. 新增 `MiniLmOnnxEmbedder implements LocalTextEmbedder`
2. 在该类里接 tokenizer + ONNX Runtime / TFLite 推理
3. 输出固定维度 embedding
4. 保持 `embed(String text)` 接口不变
5. 在 `LocalRagIndexService` 初始化时，将默认 embedder 从 `HashingTextEmbedder()` 换成 `MiniLmOnnxEmbedder()`

这样：

1. `RecallPage` 不用改
2. 本地 SQLite 索引逻辑不用改
3. 双模式切换 UI 不用改
4. 搜索逻辑不用改

这也是我这次把 embedding 单独抽象成服务层接口的原因。

## 8. 这次改动涉及的文件

### 新增文件

1. `app/lib/services/local_text_embedder.dart`
2. `app/lib/services/local_rag_index.dart`

### 修改文件

1. `app/lib/pages/recall.dart`
2. `app/lib/extra_func/language.dart`
3. `app/pubspec.yaml`
4. `app/pubspec.lock`

其中：

1. `pubspec.yaml` 新增了 `sqflite`
2. `language.dart` 增加了双模式搜索文案
3. `recall.dart` 增加了模式切换、本地索引同步和双通道搜索逻辑

## 9. 这次我做过的验证

我实际执行过以下验证：

1. `flutter pub get`
2. `dart format`
3. `flutter analyze lib/pages/recall.dart lib/extra_func/language.dart lib/services/local_rag_index.dart lib/services/local_text_embedder.dart`

结果是：

1. 依赖解析成功
2. 格式化成功
3. 静态分析无报错

## 10. 当前方案的限制

当前版本虽然已经实现了端侧双模式 RAG，但仍有几个限制：

1. 本地向量检索目前是 SQLite 持久化 + Dart 层扫描，不是 ANN 索引。
2. 当前默认 embedder 是轻量哈希向量器，不是 MiniLM。
3. 搜索结果里的 `matched_frames` 仍然主要取决于云端搜索结果结构，本地模式目前以场景级召回为主。
4. 当端侧数据量非常大时，后续可能需要升级为：
   - ObjectBox Vector Search
   - SQLite-VSS
   - 原生 ANN 库

## 11. 总结

这次改动完成的核心不是“简单把搜索从云端搬到本地”，而是把 `Recall` 页面升级成了双模式检索架构：

1. 保留原有云端语义检索能力
2. 新增端侧隐私检索能力
3. 把本地 embedding 和本地索引抽象成独立服务层
4. 为后续接入真正的 INT8 MiniLM / ONNX Runtime 预留稳定替换位

从工程上看，这样的实现有三个价值：

1. 立即可用，能落地
2. 不破坏原有云端能力
3. 后续可以平滑升级到更强的端侧模型

如果后面要继续推进，我建议下一步优先做两件事：

1. 把 `HashingTextEmbedder` 替换为真正的端侧 MiniLM Embedding
2. 把本地召回从全表扫描升级成 ANN 向量索引
