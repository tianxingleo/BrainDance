# Flutter 端侧模型加载与推理流程

## 1. 概述

BrainDance 的 Recall 页面支持在端侧运行微调后的 Qwen3-1.7B 模型，实现本地记忆问答。整个过程不依赖云端 LLM 推理服务，模型加载、RAG 检索、生成回答全部在设备本地完成。

本文档描述端侧模型从发现、下载、加载到推理的完整流程，以及本地 RAG 检索中 embedding 的实现方式。

## 2. 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 推理引擎 | LlamaDart (`llamadart: ^0.6.6`) | llama.cpp 的 Dart FFI 封装，无需编写原生 Kotlin/Swift 代码 |
| 模型格式 | GGUF (Q5_K_M + imatrix 量化) | 将 Qwen3-1.7B 压缩至约 1.2 GB |
| 模型分发 | Supabase Storage | 模型文件存储在 `braindance-models` / `braindance-assets` bucket |
| 本地存储 | SQLite (`sqflite`) | 模型路径、RAG 向量索引 |
| 本地 Embedding | HashingTextEmbedder | 纯 Dart 哈希向量器，192 维，零模型依赖 |

## 3. 模型生命周期

### 3.1 总体流程

```
App 启动
  ├─ _restoreLocalModelPath()     恢复上次模型路径
  ├─ _loadLocalModelCatalog()     从 Supabase 发现可用模型
  │
  ├─ 用户选择模型（或使用默认）
  │     ↓
  ├─ _downloadModelToPrivateDir()  下载到应用私有目录（如未下载）
  │     ↓
  ├─ _loadLocalQnaModel()          加载模型到 LlamaEngine
  │     ↓
  └─ _askLocalQuestion()           用户提问 → RAG 检索 → 推理生成
```

### 3.2 关键文件

| 文件 | 职责 |
|------|------|
| `app/lib/pages/recall/recall_local_ai.dart` | 模型加载、推理、问答 UI 状态管理 |
| `app/lib/services/local_model_catalog_service.dart` | 从 Supabase 发现可用模型 |
| `app/lib/services/local_rag_index.dart` | 本地向量索引与检索 |
| `app/lib/services/local_text_embedder.dart` | 端侧文本向量器 |
| `app/lib/configs/supabase_config.dart` | 模型 URL、bucket 等配置 |

## 4. 模型发现与下载

### 4.1 配置来源

`app/lib/configs/supabase_config.dart`

默认模型配置：

```dart
static const String _defaultLocalModelBucket = 'braindance-models';
static const String _defaultLocalModelObjectPath =
    'releases/qwen3-1.7b-braindance-q5-k-m-imatrix.gguf';
```

可通过 `.env` 文件覆盖：

- `LOCAL_LLM_MODEL_URL` — 直接指定完整下载链接
- `LOCAL_LLM_MODEL_BUCKET` — 指定 Storage bucket
- `LOCAL_LLM_MODEL_OBJECT_PATH` — 指定 bucket 内的对象路径

### 4.2 模型目录服务

`LocalModelCatalogService.fetchCatalog()` 从两个来源发现可用模型：

1. **Catalog JSON**：读取 `catalog/model_catalog.json`，解析其中的模型列表
2. **Bucket 扫描**：遍历 Supabase Storage 的 `braindance-assets` 和 `braindance-models` bucket，扫描所有 `.gguf` 文件

两个来源的结果合并去重后返回，默认模型标记为 `isRecommended`。

### 4.3 模型下载

`_downloadModelToPrivateDir()` 使用 Dio 下载模型：

- 目标路径：`getApplicationDocumentsDirectory()` + URL 中的文件名
- 支持进度回调、30 分钟接收超时
- 下载完成后记录到 `_downloadedLocalModelPathsByUrl` 映射表
- 模型路径通过 `SharedPreferences` 持久化

### 4.4 路径恢复

`_restoreLocalModelPath()` 在页面启动时：

1. 从 `SharedPreferences` 读取上次保存的路径和 URL
2. 检查本地文件是否存在
3. 不存在时使用默认 URL 计算预期路径

## 5. 模型加载

### 5.1 加载流程

`_loadLocalQnaModel()` 是模型加载的核心方法，位于 `recall_local_ai.dart:405`。

```
1. 校验文件存在且 > 100 MB（防止不完整 GGUF）
       ↓
2. 创建 LlamaEngine(LlamaBackend())
       ↓
3. 尝试 GPU 加载（Vulkan, 24 GPU layers）
   ├── 成功 → 使用 GPU 实例
   └── 失败 → 释放 GPU 实例，创建新的 CPU 回退实例
       ↓
4. 设置日志级别，更新 UI 状态为"已加载"
```

### 5.2 加载参数

| 参数 | GPU 模式 | CPU 回退模式 |
|------|---------|-------------|
| `contextSize` | 2048 | 2048 |
| `gpuLayers` | 24 | 0 |
| `preferredBackend` | `GpuBackend.vulkan` | `GpuBackend.cpu` |
| `numberOfThreads` | 4 | 4 |
| `numberOfThreadsBatch` | 4 | 4 |
| `batchSize` | 256 | 256 |
| `microBatchSize` | 256 | 256 |

### 5.3 GPU/CPU 自动回退

代码采用"先尝试 GPU，失败则回退 CPU"的策略：

```dart
final llama = LlamaEngine(LlamaBackend());
try {
  await llama.loadModel(modelPath, modelParams: gpuParams);
} catch (_) {
  await llama.dispose();
  final fallbackLlama = LlamaEngine(LlamaBackend());
  await fallbackLlama.loadModel(modelPath, modelParams: cpuParams);
  // 使用 fallbackLlama
}
```

这样在不支持 Vulkan 的设备上也能正常运行。

### 5.4 模型释放

`_disposeLocalQnaModel()` 负责：

1. 取消正在进行的流式生成订阅
2. 调用 `model.dispose()` 释放 llama.cpp 资源
3. 置空 `_localQnaModel` 引用

在页面退出、切换模型时调用。

## 6. 推理流程

### 6.1 问答入口

`_askLocalQuestion()` 位于 `recall_local_ai.dart:531`，完整流程：

```
用户输入问题
     ↓
_buildRetrievalPayload(question)     构建 RAG 检索结果
     ↓
构建 ChatML 格式 Prompt
     ↓
LlamaEngine.generate() 流式生成
     ↓
_parseLocalModelOutput() 分离思考链和正式回答
     ↓
_shouldLockAnswer() 检测到句末标点时停止生成
     ↓
展示最终回答
```

### 6.2 Prompt 格式

使用 Qwen3 的 ChatML 格式：

```
<|im_start|>system
你是 BrainDance 的本地记忆问答助手...
<|im_end|>
<|im_start|>user
{"question": "...", "retrieval": {"evidence": [...], "hit_count": 3, "intent": "..."}}
<|im_end|>
<|im_start|>assistant
请直接给出最终回答；如果你仍然生成 <think/> 思考链...
```

System Prompt 的核心规则：

1. `hit_count > 0` 时必须回答具体内容
2. `hit_count == 0` 时只能回答"暂无相关记录"
3. 部分命中时只回答证据覆盖的部分
4. 输出必须是自然语言短句，最多两句
5. 不复述问题、不解释规则

### 6.3 生成参数

```dart
GenerationParams(
  maxTokens: 384,
  temp: 0.1,          // 接近贪心，减少幻觉
  topK: 20,
  topP: 0.1,          // 进一步限制采样范围
  penalty: 1.05,      // 对齐 Python 脚本的 repetition_penalty
  stopSequences: ['<|im_end|>', ''],  // Qwen3 停止符
)
```

### 6.4 输出解析

模型输出可能包含 `<think/>` 思考链，`_parseLocalModelOutput()` 负责分离：

1. 检测 `...` 标签
2. 思考链内容赋给 `_localReasoning`（UI 可选展示）
3. `...` 之后的内容作为正式回答

### 6.5 答案锁定

`_shouldLockAnswer()` 检测到以下条件时提前终止生成：

- 回答长度 >= 8 个字符
- 以句末标点结尾（`。！？.!?`）

这避免了模型在完整回答后继续生成冗余内容。

### 6.6 后处理

`_sanitizeLocalAnswer()` 清理原始输出：

1. 截断 `【说明】`、`\n问题：` 等异常标记之后的内容
2. 提取最后一个 `答案：` 之后的部分
3. 检测并去除重复拼接（前半段 == 后半段时只保留一份）

## 7. 本地 RAG 检索

本节描述推理过程中 embedding 的实现和检索细节。关于 RAG 架构的完整说明另见 [Flutter端侧AI双模式RAG实现说明](./Flutter端侧AI双模式RAG实现说明.md)。

### 7.1 Embedding 实现

`app/lib/services/local_text_embedder.dart` — `HashingTextEmbedder`

当前方案是**纯哈希向量器**，不需要任何 ML 模型：

```
输入文本
  ↓ normalize（小写、去标点、保留中英文）
  ↓
  ├─ 分词: [a-z0-9]+ | [\u4e00-\u9fff]  → weight 1.0
  ├─ 字符 2-gram                          → weight 0.45
  └─ 字符 3-gram                          → weight 0.25
  ↓
  每个 token/ngram:
    hash = FNV-1a32(token)
    index = hash % 192
    sign = (hash >> 1) & 1 == 0 ? +1.0 : -1.0
    vector[index] += sign * weight
  ↓
  L2 归一化
  ↓
输出: 192 维 float 向量
```

**优点**：零延迟、无模型依赖、纯 Dart 实现。

**局限**：语义泛化能力有限，更依赖关键词匹配，因此需要查询扩展来补偿。

### 7.2 查询扩展

`_expandQuery()` 使用硬编码的词汇映射表弥补 HashingEmbedder 的语义不足：

```dart
const semanticExpansions = {
  "理工":   ["算法", "数学", "教材", "电脑", ...],
  "计算机":  ["电脑", "显示器", "机械键盘", ...],
  "学习":   ["教材", "地球仪", "白板", ...],
  "书房":   ["办公桌", "椅子", "书架", "电脑", "书"],
};

const objectExpansions = {
  "洛天依":  ["洛天依", "手办", "展台"],
  "桌面设备": ["显示器", "笔记本电脑", "键盘", "办公桌"],
};
```

扩展逻辑：用户原始 query 中包含映射表的 key 时，将对应的 value 列表追加到 query 末尾。

### 7.3 向量索引构建

`app/lib/services/local_rag_index.dart` — `LocalRagIndexService`

每条记忆记录入库时，从以下字段构建可搜索文本：

```
searchable_text = [
  display_name,
  scene_id,
  description,
  ...tags,
  ...objects,
  ...meta_info 中所有字符串值,
].join(' | ')
```

然后调用 `embedder.embed(searchable_text)` 得到 192 维向量，连同原始 payload 存入 SQLite：

| 字段 | 内容 |
|------|------|
| `model_id` | 记录唯一标识 |
| `scene_id` | 场景 ID |
| `user_id` | 用户 ID |
| `searchable_text` | 用于词汇匹配的原始文本 |
| `vector_json` | JSON 序列化的 192 维向量 |
| `payload_json` | 完整模型数据 JSON |
| `fingerprint` | 用于增量同步的内容指纹 |
| `updated_at` | 更新时间 |

### 7.4 检索过程

`_buildRetrievalPayload(question)` 的检索流程：

```
用户原始问题
     ↓
_expandQuery(question)                    查询扩展
     ↓
expandedQuery
     ↓
_localRagIndex.search(expandedQuery, limit: 3, minScore: 0.08)
     │
     ├─ embedder.embed(expandedQuery)     查询向量化
     │
     ├─ 遍历 SQLite 全表:
     │    ① cosine = dot(queryVec, rowVec) 余弦相似度
     │    ② lexical = 词汇命中率
     │    ③ score = cosine × 0.82 + lexical × 0.18
     │    ④ 过滤: score < 0.08 丢弃
     │
     └─ 按 score 降序，取 top 3
     ↓
构建 evidence 列表:
[
  {id, created_at, description, tags, objects, summary, scene_id},
  ...
]
     ↓
返回 retrieval payload:
{
  "evidence": [...],
  "hit_count": 3,
  "intent": "object_lookup"
}
```

### 7.5 检索结果送入 LLM

检索得到的 `retrieval` payload 与用户原始 `question` 组合成 JSON，作为 ChatML Prompt 中的 `user` 消息送入 LlamaEngine。

如果检索无结果（`matches.isEmpty`），则取当前已加载模型列表的前 3 条作为 fallback evidence，避免 LLM 收到空 context。

## 8. 完整数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                        Flutter UI                            │
│                 recall_local_ai.dart                         │
├───────────┬──────────────┬──────────────┬───────────────────┤
│  模型目录  │   模型下载    │   模型加载    │    推理生成        │
│  Service  │    (Dio)     │ LlamaEngine  │  LlamaEngine      │
│           │              │              │  .generate()      │
├───────────┴──────────────┼──────────────┴───────────────────┤
│    Supabase Storage      │       llama.cpp (FFI)            │
│    (.gguf files)         │    GPU (Vulkan) / CPU 自动回退    │
└──────────────────────────┴──────────────────────────────────┘
          ↕ (RAG 检索)
┌─────────────────────────────────────────────────────────────┐
│  SQLite (braindance_memory_rag.db)                          │
│  表: memory_scene_vectors                                    │
│  + HashingTextEmbedder (192 维, FNV-1a 哈希)                │
│  + 查询扩展 (语义扩展 + 对象查找扩展)                          │
│  混合打分: cosine × 0.82 + lexical × 0.18                   │
└─────────────────────────────────────────────────────────────┘
```

## 9. 当前方案的定位

端侧模型加载与推理系统目前是一个**可运行的完整实现**，但不是终局方案：

| 方面 | 当前状态 | 后续优化方向 |
|------|---------|-------------|
| 推理引擎 | LlamaDart (llama.cpp FFI) | 保持，已稳定 |
| 模型 | Qwen3-1.7B Q5_K_M (~1.2 GB) | 可选更小模型 + LoRA |
| Embedding | HashingTextEmbedder (哈希) | INT8 MiniLM / ONNX Runtime |
| 向量存储 | SQLite 全表扫描 | ANN 索引 (ObjectBox Vector Search) |
| 查询扩展 | 硬编码映射表 | 端侧意图识别模型 |
| 语义理解 | 关键词 + 哈希相似度 | 真正的语义 embedding |

关于 Embedding 的替换方案，`LocalTextEmbedder` 已抽象为接口，只需实现 `MiniLmOnnxEmbedder implements LocalTextEmbedder` 并替换初始化处的默认实例即可，无需修改上层代码。
