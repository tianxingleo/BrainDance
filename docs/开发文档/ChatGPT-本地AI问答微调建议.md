# 本地AI问答微调建议

**User:** Anonymous (tianxingleo@proton.me)  
**Created:** 3/21/2026 10:05:29  
**Updated:** 3/21/2026 10:20:51  
**Exported:** 3/21/2026 10:20:58  
**Link:** [https://chatgpt.com/c/69bdfcb1-6a64-8323-86a9-f63f5a3cf0dc](https://chatgpt.com/c/69bdfcb1-6a64-8323-86a9-f63f5a3cf0dc)  

## Prompt:
我的项目BrainDance中我希望实现本地ai问答，下面的本地ai小模型微调建议是否可取：结合你们项目现状，结论很明确：这个问题不能只靠“把模型微调一下”解决，必须把“检索约束 + 微调目标 + 智能体编排”一起改。对
  BrainDance 这类“用户个人时空记忆”场景，微调的对象应该是“如何使用检索结果回答”，而不是把“用户最近拍了什么”这种事实记进
  模型参数里。这一点和 TUNE_DOC.md 里总结的 P-RAG 思路是一致的。

  先说我看到的几个当前根因。

  1. 你们现在的回答链路里，检索和回答没有被强绑定。
     /C:/Projects/VibeBrainDance/BrainDance/app/lib/pages/recall.dart:464 这里 _buildMemoryContext() 检索不到时，会直接
     fallback 到“最新 3 个模型”。这会把“无命中”和“拿最近内容凑上下文”混在一起，训练和推理都会变脏。模型最后既可能乱答，
     也可能明明有资料却判断成“没有明确答案”。
  2. 当前本地 RAG 基本没有时间语义。
     /C:/Projects/VibeBrainDance/BrainDance/app/lib/services/local_rag_index.dart:228 这里构建向量只用了 display_name/
     description/tags/objects/meta_info，created_at 只参与 fingerprint，不参与检索排序和时间理解。所以用户问“最近拍了什
     么”，召回阶段本身就不稳，后面的模型再聪明也救不回来。
  3. 服务端其实已经有“时间解析”，但没有打通到回答智能体。
     /C:/Projects/VibeBrainDance/BrainDance/supabase/functions/search-models/index.ts:337 已经在做“上周/之前”等时间解
     析；
     /C:/Projects/VibeBrainDance/BrainDance/supabase/migrations/20260306123456_match_memory_poses.sql:1 也支持按时间过
     滤。
     但这个能力目前更像“搜索 API 能用”，不是“回答 Agent 必须依赖的证据层”。所以会出现“RAG 查到了，智能体却说没有”。
  4. 循环输出现在只是被前端做了事后修补，不是从生成机制上解决。
     /C:/Projects/VibeBrainDance/BrainDance/app/lib/pages/recall.dart:367 到 /C:/Projects/VibeBrainDance/BrainDance/app/
     lib/pages/recall.dart:421 的 prompt、stopSequences、_sanitizeLocalAnswer()、_shouldLockAnswer() 只能缓解重复，不能
     保证模型稳定停住。它说明当前模型输出格式控制本身不够强。

  基于这些问题，你们引入智能体后，正确的方案不是“让 agent 更自由”，而是“把 agent 做窄”。对于“我最近拍了什么”这类问答，建
  议用一个很薄的 Retrieval-first Answer Agent：

  1. 意图解析
     把问题解析成结构化槽位：question_type=recent_capture、time_range、target_object、need_count/list/detail。
     这一步可以继续复用 /C:/Projects/VibeBrainDance/BrainDance/supabase/functions/search-models/index.ts 现有时间解析思
     路。
  2. 检索执行
     不要只走向量召回，要加时间硬过滤和时间排序。
     “最近”类问题优先按 created_at desc 取 top-k，向量只做辅助手段，不要反过来。
  3. 证据整理
     把检索结果整理成固定格式，例如：
     [{created_at, scene_id, display_name, description, objects, tags}]
     不要给模型原始杂乱文本。
  4. 回答生成
     让模型只做“证据转自然语言”，而不是让它自己决定要不要相信 RAG。
     规则必须是：

  - hit_count > 0 时，禁止回答“没有记录/不知道”
  - hit_count = 0 时，才允许回答“暂无记录”
  - 对“最近拍了什么”统一按时间倒序列出 1-3 条

  微调上，建议这样做。

  1. 只微调“能力”，不微调“事实”
     也就是训练模型学会：

  - 看懂结构化检索结果
  - 按时间排序输出
  - 有证据就答，没证据才拒答
  - 输出短答案，不解释规则，不循环

  2. 数据集不要做成普通对话，要做成“问题 + 检索结果 + 标准回答”
     最重要的训练样本类型有四类：

  - 命中 1 条：我昨天拍了什么
  - 命中多条：我最近拍了什么
  - 有检索但和问题部分错位：要求模型只答证据覆盖的部分
  - 无检索命中：明确回答“暂无记录”

  3. 必须加入“反例监督”
     你提到的两个坏现象都要专门做负样本：

  - RAG 有资料，但模型说没有
  - 模型重复一句话/循环输出
    也就是同一个输入，准备“错误回答”和“正确回答”偏好对，SFT 后再加一轮 DPO/ORPO 会很有价值。

  4. 推荐训练目标
     第一阶段：LoRA SFT

  - 基座：Qwen3-1.7B 或 3B，端侧优先 1.7B
  - rank 8
  - lr 2e-5
  - epoch 2-3
  - 样本量先做 2k-5k 就够验证
    这和 TUNE_DOC.md 给的方向基本一致。

  5. 样本配比
     建议：

  - 50% 时间问答
  - 20% “有证据必须回答”
  - 15% “无证据必须拒答”
  - 15% 输出稳定性样本（短答、去重复、禁止复述问题）

  如果只做一个最小可行版本，我建议你们先别急着训练，先把这三件事改掉，否则微调效果会被链路噪声吞掉：

  1. 去掉 /C:/Projects/VibeBrainDance/BrainDance/app/lib/pages/recall.dart:464 里“检索失败就塞最近 3 条”的 fallback。
  2. 给本地/服务端检索都加上时间字段参与排序，尤其是“最近/昨天/上周”。
  3. 回答 prompt 改成结构化输入，并加硬规则：有命中不得说没有。

  最后给你一个适合你们项目的落地判断标准。微调是否成功，不看主观感觉，看 4 个指标：

  - 时间问答准确率：最近/昨天/上周 是否答对
  - False-No-Answer Rate：有检索命中却回答“没有”的比例
  - Repetition Rate：重复句/循环输出比例
  - Evidence Utilization Rate：回答里是否实际用了返回的 description/objects/tags/created_at

## Response:
我先到你们 BrainDance 代码里核对这些判断，重点看 recall 链路、local RAG、服务端时间解析和训练文档是否真的对得上。

我已经确认了一部分：你贴的四个“根因”里，至少前三个方向是和代码现状高度一致的，尤其是 recall 里的 fallback 和生成后补丁。再补一下 local\_rag\_index、服务端搜索和训练文档的细节。

已思考 36s

已收到应用响应

我改用仓库直读了。现在能更稳地确认两件事：一是这些建议有没有被代码支持，二是哪些点需要改成“先改链路，再谈微调”。

已收到应用响应

总体判断：**这份建议大方向是可取的，而且和你当前 BrainDance 代码现状基本对得上。**  
但我会把结论再收紧一点：

**不是“先微调，再看看能不能变好”，而是“先把检索-证据-回答链路收紧，再做一个很窄的微调”。**  
否则你现在链路里的噪声，会把微调收益吃掉。

我核对代码后的判断如下。

你说的第 1 个根因是成立的。`_buildMemoryContext()` 里本地检索失败后，会直接退回到 `_models/_allModels` 的前 3 条，而 `_fetchModels()` 又是按 `created_at desc` 拉全量模型，所以这里确实把“检索命中失败”和“最近内容兜底”混在了一起。这样会让模型把“最近三条”误当成“与问题相关的证据”，污染推理与训练分布。这个判断和代码一致[](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

[recall](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

[

recall

](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

第 2 个根因也成立。`local_rag_index.dart` 的本地索引只把 `scene_id/description/tags/objects/meta_info` 拼进 `searchable_text`，检索分数也是 `cosine + lexical`，排序只按 `similarity`。`created_at` 仅出现在 fingerprint 里用于缓存失效，不参与检索文本、不参与打分、不参与排序。因此“最近/昨天/上周”这类问题，在本地 RAG 侧目前确实没有时间语义保障[](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)

[local\_rag\_index](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)

[

local\_rag\_index

](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)[

local\_rag\_index

](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)[

local\_rag\_index

](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)

第 3 个根因也基本成立。服务端 `search-models` 已经在做意图解析，明确要求提取 `search_text / start_time / end_time`，示例里就包含“上周拍的红色杯子”；后面也把 `filter_start/filter_end` 传给 `match_memory_poses`。而 SQL 层确实支持基于 `created_at` 的时间过滤。但这条能力现在只存在于 cloud search API，不在本地问答 `_askLocalQuestion()` 的强制证据链里，所以会出现“服务端能按时间搜，端侧回答却不一定按时间答”的割裂[](https://github.com/tianxingleo/BrainDance/blob/main/supabase/functions/search-models/index.ts)

[index](https://github.com/tianxingleo/BrainDance/blob/main/supabase/functions/search-models/index.ts)

[

index

](https://github.com/tianxingleo/BrainDance/blob/main/supabase/functions/search-models/index.ts)[

index

](https://github.com/tianxingleo/BrainDance/blob/main/supabase/functions/search-models/index.ts)[

20260306123456\_match\_memory\_pos…

](https://github.com/tianxingleo/BrainDance/blob/main/supabase/migrations/20260306123456_match_memory_poses.sql)[

20260306123456\_match\_memory\_pos…

](https://github.com/tianxingleo/BrainDance/blob/main/supabase/migrations/20260306123456_match_memory_poses.sql)

第 4 个根因也成立。你前端现在确实是在做“生成后/生成中修补”：prompt 里写了“没有明确答案再回答不知道、不要重复题目”，同时又加了 `stopSequences`、`_sanitizeLocalAnswer()`、`_shouldLockAnswer()` 去截断和去重。这些都能缓解，但它们本质上是后处理，不是从生成目标或输出协议上根治[](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

[recall](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

[

recall

](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

所以，**你的方案不是不可取，相反是方向对的；只是还需要更明确地区分“现在该改链路”和“之后再微调模型”。**

我给你的判断是：

### 一、这份建议里最对的部分

**1\. “只微调能力，不微调事实”是对的。**  
对 BrainDance 这种个人时空记忆产品，用户最近拍了什么、在哪里拍的，这些都应该留在可更新的检索层，不该塞进参数。参数更适合学“如何看证据并回答”。

**2\. “Retrieval-first Answer Agent” 也对，但要更窄。**  
你这里其实不需要一个很自由的 agent，更像一个固定四段式流水线：

*   问题分类 / 槽位提取
*   按时间和语义检索
*   证据规整
*   受约束回答

这比“让一个小模型自己决定要不要信 RAG”稳定得多。

**3\. “问题 + 检索结果 + 标准回答”的训练格式是对的。**  
因为你要训练的不是世界知识，而是证据使用习惯。

**4\. “加入负样本监督”非常有必要。**  
尤其是：

*   有证据却说没有
*   重复输出 / 循环输出

这两个就是你当前链路最痛的 failure mode。

* * *

### 二、我会修改这份建议的地方

**1\. 先别急着上 agent 这个词，先做“硬规则问答器”。**  
你现在最缺的不是 agent 能力，而是**证据约束**。  
第一阶段可以根本不做复杂 agent，只做一个固定函数：

`question -> structured intent -> retrieval -> evidence list -> answer renderer`

先把这条打通，很多“模型乱答”问题会先消掉一半。

**2\. “最近类问题优先按 created\_at desc 取 top-k，向量只做辅助手段”这个判断，我赞成。**  
但建议再具体一点：

*   如果意图是 `recent_capture / yesterday / last_week`，**时间过滤和时间排序应当成为主路径**
*   语义相似度只用来在同一时间窗内重排
*   没时间词的普通问答，才走“语义优先”

也就是你要做的是**query plan 路由**，不是统一用一个召回公式。

**3\. 你现在的“无证据时允许说不知道”，要改成“看 hit\_count，不看模型主观判断”。**  
这一点建议里说得对，但还不够硬。  
我建议从 prompt 级别提升到代码协议级别：

*   检索阶段返回 `hit_count`
*   生成器拿到的是结构化 evidence
*   当 `hit_count == 0` 时，才允许模板化拒答
*   当 `hit_count > 0` 时，直接禁用“我不知道/没有记录”类输出

这不要只靠提示词，最好在后处理里再兜一次。

* * *

### 三、按你当前项目，最值得先改的 3 件事

#### 1）删掉本地问答里的 fallback recent-3

这个优先级最高。  
因为它会直接污染“证据是否命中”的语义边界。现在 `_buildMemoryContext()` 在空检索时塞最近三条，这会让模型经常在错误证据上硬答[](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

[recall](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

更合理的做法是：

*   本地检索空命中 → 返回空 evidence
*   UI 显示“暂无相关记忆”
*   不再偷偷补最近三条

#### 2）给本地 RAG 加时间语义

当前本地索引没有时间参与搜索，只能搜内容，不能搜“最近”[](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)

[local\_rag\_index](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)

[

local\_rag\_index

](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)

最小改法不是先上复杂 temporal embedding，而是：

*   在 payload 中显式保存 `created_at_epoch`
*   搜索前先做时间解析（最近/昨天/上周）
*   若命中时间意图：
    *   先按时间窗过滤
    *   再在窗内做相似度排序
    *   “最近”则按 `created_at desc` 主排
*   `_formatMemorySnippet()` 里把时间也显式给模型，不要只给描述/标签/对象

#### 3）把回答输入改成结构化 evidence，而不是自由文本片段

你现在喂给模型的是“片段1/场景/描述/标签/对象/摘要”的串接文本。这比原始 JSON 好，但还不够稳[](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

[recall](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)

建议改成类似：

```
{
  "question_type": "recent_capture",
  "time_range": {"start":"...", "end":"..."},
  "hit_count": 3,
  "evidence": [
    {
      "created_at": "...",
      "scene_id": "...",
      "description": "...",
      "objects": ["..."],
      "tags": ["..."]
    }
  ]
}
```

然后 prompt 明确限定：

*   只根据 evidence 回答
*   命中时不得拒答
*   最多列 3 条
*   按时间倒序
*   不复述问题
*   不解释规则

* * *

### 四、关于微调：可取，但要放在第二阶段

你的 LoRA SFT 思路我基本认同，尤其适合端侧小模型。  
不过我建议分两阶段，而不是直接一把梭：

#### 阶段 A：先不训练，做链路 A/B

先比较：

*   旧链路：自由问答 + fallback recent3
*   新链路：结构化 evidence + hit\_count 硬规则 + 时间路由

只要新链路已经把 False-No-Answer 和重复率压下去，说明方向对。

#### 阶段 B：再做小规模 LoRA

训练目标只放在这四件事：

*   读懂结构化 evidence
*   按时间排序
*   有证据就答
*   短答稳定停住

这个时候 2k-5k 样本的 SFT 才有意义。  
不然你训练到的会是“如何适应一条脏链路”。

* * *

### 五、我对训练配方的具体建议

你提的 `Qwen3-1.7B / 3B + LoRA SFT` 是合理的。  
结合你现在端侧场景，我会这样定：

**更推荐起点：Qwen3-1.7B**

*   先验证链路和数据格式
*   端侧内存、延迟更友好
*   足够承担“证据转自然语言”这种窄任务

**LoRA 先保守**

*   rank 8 可以
*   lr 2e-5 可以
*   epoch 2-3 可以
*   但我会先看验证集上的 repetition / false-no-answer，再决定要不要继续

**DPO/ORPO 值得做，但别太早**

*   先把 SFT 跑通
*   再对“有证据却拒答”和“重复输出”做偏好优化
*   否则你会在不稳定输入分布上做 preference，对齐目标也会飘

* * *

### 六、我会补充的训练样本类型

除了你列的四类，我建议再加三类：

**1\. 时间冲突样本**  
比如 evidence 里既有上周也有更早内容，问题问“上周拍了什么”。  
训练模型只取时间窗内结果，不被相似但越界的内容干扰。

**2\. 证据部分覆盖样本**  
比如用户问“我最近拍的红杯子和猫”，evidence 只有红杯子。  
模型要回答“最近有红杯子的记录，未见猫相关记录”，而不是全拒答。

**3\. 格式稳定样本**  
专门训练：

*   不复述问题
*   不输出“根据记忆片段”
*   不输出规则说明
*   句长控制在 1-3 句

* * *

### 七、你的成功指标也对，但我会再补一个

你列的四个指标都很对。  
我建议加一个：

**Temporal Ranking Consistency**

*   对“最近/昨天/上周”类问题
*   返回结果是否严格按时间规则排序

因为你现在最大的问题，不只是“答不答”，还有“时间上答得对不对”。

* * *

### 八、最终结论

我的结论是：

**这份建议是可取的，且与当前 BrainDance 代码问题高度匹配。**  
但它最该落地的顺序应该是：

**先修链路，再做窄模型，再谈 agent。**

更具体地说：

1.  先去掉 `_buildMemoryContext()` 的 recent-3 fallback[](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)
    [recall](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)
2.  给本地检索补时间解析、时间过滤和时间排序[](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)
    [local\_rag\_index](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)
    [
    local\_rag\_index
    ](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)
3.  把回答器改成“结构化 evidence -> 受约束生成”
4.  然后再用小样本 LoRA 去微调“证据使用能力”
5.  最后才考虑是否需要更完整的 agent 编排

顺手补一个小发现：你提到的 `TUNE_DOC.md`，我这次在当前仓库里没直接定位到，所以**P-RAG 那部分我没法基于仓库文件再确认一遍**；但就你贴出的方案本身看，方向是对的。

下一步如果你愿意，我可以直接按你这个项目现状，给你出一版 **“最小可行改造方案”**：包括本地时间检索路由、结构化 evidence schema、以及新的回答 prompt/后处理规则。

## Prompt:
请你给我详细讲讲，微调每一步应该做什么，以及可能会遇到哪些坑

## Response:
我结合你仓库里的问答链路来讲，不只说“训练怎么做”，也会把每一步为什么要这样做、最容易翻车的点一起拆开。先补查一下仓库里和训练、问答相关的文件，确保建议贴着你项目现状。

已思考 12s

可以。我按**真正落地微调的顺序**来讲，尽量贴着 BrainDance 这种“本地个人时空记忆问答”的场景说，而不是泛泛讲 LoRA 教程。

先给你一个总图：  
对你这个项目，微调不是第一步，而是第 **4** 步。

正确顺序是：

1.  先定义任务边界
2.  再把推理链路改成“证据先行”
3.  再产训练数据
4.  再做 SFT 微调
5.  再做偏好优化或小规模强化
6.  最后上线评估和迭代

如果顺序反了，最常见的结果就是：**模型训了，但线上还是乱答、漏答、重复答。**

* * *

一、先定清楚：你到底在微调什么
===============

你这个场景里，微调目标不是“让模型记住用户拍过什么”。

而是让模型学会下面这 5 件事：

1.  看懂结构化检索结果
2.  根据时间语义组织答案
3.  有证据就回答，没证据才拒答
4.  答案短、稳、不复读
5.  不越权脑补检索里没有的事实

这是最重要的一步。很多项目一开始就失败，是因为目标写成了：

*   “提升模型智能”
*   “让本地模型更懂用户”
*   “让小模型更像 ChatGPT”

这些都太虚了，最后数据集会越做越散。

你应该把目标写成更像这样：

> 给定用户问题和结构化 evidence，输出一段 1 到 3 句的中文回答；若 hit\_count>0，禁止说“没有记录”；若 hit\_count=0，输出“暂无相关记录”类模板；对 recent/yesterday/last\_week 类问题按 created\_at 倒序回答。

* * *

二、先改推理链路，不然数据会脏
===============

这一步你一定不能跳。

因为微调学到的是**你现在链路里的分布**。  
如果链路本身把“没查到”和“拿最近内容凑上下文”混在一起，那模型只会学坏。

你项目里最需要先收紧的 3 个点，就是我上条说的那几个：

*   检索失败时不要再偷塞 recent 3 条
*   本地检索要显式处理时间语义
*   回答输入要变成结构化 evidence，而不是一大段松散文本

这一步具体做什么
--------

### 1）定义统一的输入协议

先不要急着训，先把线上/离线共用的样本格式定下来。

建议统一成这种结构：

```
{
  "question": "我最近拍了什么？",
  "intent": {
    "question_type": "recent_capture",
    "time_range": {
      "start": "2026-03-14T00:00:00Z",
      "end": "2026-03-21T23:59:59Z"
    },
    "need": "list"
  },
  "retrieval": {
    "hit_count": 3,
    "evidence": [
      {
        "created_at": "2026-03-20T18:22:31Z",
        "scene_id": "scene_001",
        "description": "桌上有一杯咖啡和一本笔记本",
        "objects": ["咖啡杯", "笔记本"],
        "tags": ["桌面", "办公"]
      },
      {
        "created_at": "2026-03-19T09:11:02Z",
        "scene_id": "scene_002",
        "description": "窗边有一盆绿植和白色马克杯",
        "objects": ["绿植", "马克杯"],
        "tags": ["窗边", "室内"]
      }
    ]
  },
  "answer_style": {
    "max_items": 3,
    "language": "zh-CN",
    "tone": "brief"
  }
}
```

### 2）定义统一的输出协议

训练前先定好输出长什么样。

比如：

```
{
  "answer": "你最近拍到的内容包括：3月20日的咖啡杯和笔记本，3月19日的绿植和白色马克杯。",
  "used_evidence_ids": ["scene_001", "scene_002"],
  "answer_type": "direct"
}
```

如果你只训练纯文本输出，也行，但我更建议中间先保留一个结构化 target，方便排错。

### 3）把“拒答条件”写死成代码逻辑

不要让模型自己判断“该不该拒答”。

应该是：

*   `hit_count == 0` 才能走拒答模板
*   `hit_count > 0` 必须回答
*   `recent` 类必须先按时间排序后再给模型

这一步最常见的坑
--------

### 坑 1：边训练边改输入格式

这是最常见的灾难。  
今天喂“片段1/片段2”，明天改成 JSON，后天又加 summary，模型最后会学得很散。

**原则：训练前先把输入 schema 冻住。**

### 坑 2：把检索错误也当成模型错误

比如其实是没召回到，但你以为是模型“不会答”。  
这样会错误地把问题归因到微调。

**原则：训练集里要区分**

*   retrieval miss
*   retrieval hit but answer wrong
*   retrieval hit but answer incomplete

### 坑 3：把事实写进参数

比如你拿某个用户最近 2000 条记忆去训，希望模型以后直接记得。  
这会很快过时，而且会造成“旧记忆残留”。

**原则：事实放检索层，能力放参数层。**

* * *

三、数据设计：这是成败核心
=============

微调成不成，70% 看数据，不是看超参。

你这个场景，数据不是普通聊天对话，而是：

**问题 + 意图 + 证据 + 标准回答**

第一步：定义样本类型
----------

至少要有下面 6 类。

### 1）单条命中

例子：

*   “我昨天拍了什么？”
*   evidence 只有 1 条

训练目标：

*   直接答
*   不说“可能”
*   不复述问题

### 2）多条命中

例子：

*   “我最近拍了什么？”
*   evidence 有 3 到 10 条

训练目标：

*   按时间倒序
*   列 1 到 3 条
*   抽取关键信息，不要全量抄 description

### 3）部分覆盖

例子：

*   “我最近拍的红杯子和猫有哪些？”
*   evidence 只有红杯子，没有猫

训练目标：

*   只回答被证据覆盖的部分
*   对未命中的部分明确说明未见相关记录
*   不能整句拒答

### 4）零命中

例子：

*   “我上周拍到过钢琴吗？”
*   evidence 为空

训练目标：

*   输出统一拒答模板
*   不脑补
*   不说“也许你拍过”

### 5）时间冲突

例子：

*   问“昨天拍了什么”，但 evidence 同时有昨天、上周、上个月内容

训练目标：

*   只用时间窗内的记录
*   时间优先级高于语义相似

### 6）输出稳定性

例子：

*   多轮中重复的 prompt
*   边界输入
*   evidence 描述相似但不完全相同

训练目标：

*   不循环
*   不重复句子
*   不复述规则
*   不把 evidence 原样复制一大段

* * *

第二步：数据来源怎么做
-----------

你有三种做法。

### 做法 A：从真实日志回放生成

最推荐。

方法：

*   从真实用户问题里抽样
*   跑当前检索器拿 evidence
*   人工修正正确答案

优点：

*   贴近真实流量
*   失败模式真实

缺点：

*   标注成本高

### 做法 B：从记忆库模板合成

也很有用。

方法：

*   从现有 model\_assets/memory\_poses 里采样若干记录
*   自动构造问句：
    *   最近拍了什么
    *   昨天拍到什么
    *   上周有没有咖啡杯
    *   最近有哪些室内场景
*   再自动生成草稿答案，人工抽检

优点：

*   起量快
*   容易平衡样本分布

缺点：

*   容易过于模板化

### 做法 C：混合方案

最实际。

建议：

*   60% 合成样本
*   40% 真实日志修订样本

这样既能起量，也不会太假。

* * *

第三步：标注标准要写死
-----------

你必须先写一个小型标注规范，不然不同人写出来的答案风格会散。

至少要明确这些规则：

*   命中时必须回答
*   无命中才拒答
*   最多列 3 条
*   优先说对象和场景核心信息
*   时间表达统一格式
*   不写“根据你提供的信息”
*   不写“从记忆片段看”
*   不复述用户问题
*   不输出列表符号，还是输出短自然句，必须统一

数据阶段常见坑
-------

### 坑 1：正样本很多，反例太少

结果模型学会“凡事都答”，但不会在空证据时稳住。

你必须专门配：

*   零命中
*   部分覆盖
*   时间冲突
*   有证据却容易误拒答

### 坑 2：答案写得太长

训练集如果大量是“解释型回答”，小模型会学会啰嗦。  
而你线上需要的是短答。

### 坑 3：标注员偷偷补脑

比如 evidence 里只写“桌上有咖啡杯”，标注答案写成“你昨天在办公室喝咖啡”。  
这就是凭常识脑补。

### 坑 4：把 retrieval miss 的样本也标成拒答正确

如果本应召回却没召回，这类样本不应直接算“模型应该拒答”，否则会把错误行为教进去。

* * *

四、数据清洗与切分：很多人忽略，但非常关键
=====================

应该做什么
-----

### 1）去重

同一个 question + evidence 组合，别重复太多次。  
否则模型会记模板。

### 2）控制分布

建议一开始接近这样：

*   40% 时间问答
*   20% 有证据必须回答
*   15% 无证据拒答
*   15% 部分覆盖
*   10% 输出稳定性/抗重复

### 3）训练/验证/测试切分

不要随机切得太随意。  
要按“场景”和“问法”一起防泄漏。

更稳的切法是：

*   train：大部分 scene\_id
*   val：未见过的一部分 scene\_id + 已见问法变体
*   test：未见 scene\_id + 未见问法组合

这样你才能知道模型学到的是“使用 evidence”，不是“背模板”。

常见坑
---

### 坑 1：同一条 evidence 出现在训练和测试

模型看起来指标很高，其实只是记住了。

### 坑 2：验证集太小

最后你会被偶然波动误导。

### 坑 3：线上问法和训练问法差很多

比如训练里全是“我最近拍了什么”，线上却常出现：

*   “这两天我都扫了啥”
*   “前几天拍过哪些东西”
*   “我是不是最近见过杯子”

所以 paraphrase 样本一定要有。

* * *

五、SFT 微调：真正开始训练时，每一步做什么
=======================

这是你最关心的部分，我按工程顺序讲。

Step 1：先定基座模型
-------------

你当前场景更推荐从小模型开始：

*   端侧优先：Qwen3-1.7B
*   稍高质量：Qwen3-3B

为什么先不更大：

*   你任务很窄，不是开放域聊天
*   端侧延迟和内存是硬约束
*   小模型更适合先验证链路与数据

### 坑

*   一上来追大模型，成本和集成复杂度飙升
*   量化后效果掉太多，反而不如小模型 + 好数据

* * *

Step 2：把样本转成训练格式
----------------

SFT 常见两种：

### 方案 A：chat 格式

适合直接训“给 JSON 输入，回自然语言答案”。

例如：

```
{
  "messages": [
    {
      "role": "system",
      "content": "你是个人记忆问答助手。你只能根据给定证据回答。"
    },
    {
      "role": "user",
      "content": "{...结构化输入...}"
    },
    {
      "role": "assistant",
      "content": "你最近拍到的内容包括..."
    }
  ]
}
```

### 方案 B：instruction 格式

更适合你这种窄任务。

```
{
  "instruction": "根据给定证据回答问题。命中时必须回答，无命中时输出暂无相关记录。",
  "input": "{...结构化输入...}",
  "output": "你最近拍到的内容包括..."
}
```

我更建议你先用 **instruction 格式**，更稳、更容易控。

### 坑

*   system prompt 在线上和训练时差太多
*   训练时喂 JSON，线上却喂自由文本
*   训练 output 很短，线上却要求多段解释

* * *

Step 3：设置 LoRA 训练参数
-------------------

你前面提的初始值可以作为起点：

*   rank = 8
*   lr = 2e-5
*   epoch = 2 到 3
*   batch 看显存
*   sequence length 按输入 JSON 长度来定

更关键的是别只盯着这些数。  
你这个任务上，**数据格式和目标函数**比 rank 更重要。

建议从小实验开始：

*   先用 500 到 1000 条做 smoke test
*   看模型有没有学到“命中必答、零命中拒答、短答不循环”
*   再扩到 2k 到 5k

### 坑

*   一开始就上全量数据，结果排错很难
*   学习率太大，模型开始胡说或风格崩
*   epoch 太多，小数据过拟合严重
*   输入太长，真正有效的 evidence 被截断

* * *

Step 4：训练时要盯哪些指标
----------------

不要只看 loss。  
loss 下降不代表你的产品可用。

你应该在验证集上每轮都测这几个业务指标：

1.  时间问答准确率
2.  False-No-Answer Rate
3.  Repetition Rate
4.  Evidence Utilization Rate
5.  Temporal Ranking Consistency

### 怎么理解

**False-No-Answer Rate**  
有 evidence，却输出“没有记录/不知道”的比例。

**Evidence Utilization Rate**  
答案里有没有实际使用 evidence 中的对象、时间、描述核心点。

**Temporal Ranking Consistency**  
最近类问题是否按时间顺序答。

### 坑

*   只看 perplexity/loss
*   人工感觉“更自然了”，但业务指标变差
*   验证样本不包含真实难例，导致误判

* * *

六、SFT 后的偏好优化：什么时候做，怎么做
======================

当你已经有一个“基本能答”的模型后，再做这一层才有意义。

适合做什么
-----

你这里最适合拿偏好优化解决两类问题：

1.  有证据却拒答
2.  重复输出 / 啰嗦 / 复述问题

数据怎么做
-----

同一个输入，准备：

*   rejected：错误答案
*   chosen：正确答案

例如：

### 输入

问题：“我最近拍了什么？”  
evidence：3 条 recent 记录

### rejected

“没有明确答案。”

### chosen

“你最近拍到的内容包括咖啡杯、笔记本和窗边的绿植。”

再比如：

### rejected

“你最近拍了什么？你最近拍了什么？你最近拍了什么？”

### chosen

“你最近拍到的内容包括咖啡杯和笔记本。”

用什么方法
-----

你提到 DPO / ORPO，这方向是对的。  
工程上我会这样建议：

*   先 SFT
*   再小规模 DPO 或 ORPO
*   数据量不需要太大，但偏好对要精准

### 坑

*   SFT 还没稳定就上 DPO
*   chosen/rejected 质量差，甚至只是措辞不同，不是真正业务优劣
*   偏好数据只覆盖“文风”，没覆盖“证据使用”

* * *

七、上线前一定要做的离线评估
==============

上线前别只抽几个例子看效果。

建议做一个固定 benchmark 集，分桶评测：

桶 1：时间类
-------

*   最近拍了什么
*   昨天拍了什么
*   上周拍过什么
*   最近有没有杯子

桶 2：对象类
-------

*   我拍过猫吗
*   最近见过键盘吗

桶 3：组合类
-------

*   最近拍过红色杯子和绿植吗

桶 4：零命中类
--------

*   上周拍过钢琴吗

桶 5：抗重复类
--------

*   同一问题重复问
*   证据很像但不完全相同

每一桶都算指标，不然你会只看到平均值好看。

### 坑

*   平均分高，但某个关键桶非常差
*   只测“答对率”，不测“拒答率”和“重复率”
*   不测长尾问法

* * *

八、上线集成：微调后还要改什么
===============

微调不是替代工程约束，而是增强工程约束。

上线时你仍然应该保留这些机制：

1.  结构化 evidence 输入
2.  hit\_count 驱动的硬规则
3.  时间路由
4.  输出长度限制
5.  轻量后处理去重

但注意，后处理应当是**薄薄一层兜底**，不是主要控制手段。

### 坑

*   训练后把 prompt 和 schema 改掉
*   线上改成另一种检索格式
*   训练时按时间排序，线上却没排序就送模型

* * *

九、你这个项目最可能遇到的 12 个坑
===================

我把最重要的单独列出来。

1\. 先训后修链路
----------

结果：模型学会适应脏 evidence，而不是学会正确答题。

2\. 输入 schema 不稳定
-----------------

结果：模型对格式敏感，线上一改就掉点。

3\. 训练样本太像模板
------------

结果：线上一换说法就不会了。

4\. 零命中样本太少
-----------

结果：模型凡事都答，开始脑补。

5\. 有证据必须回答的负样本太少
-----------------

结果：模型动不动就说“不知道”。

6\. 时间语义没有在检索层解决
----------------

结果：模型只能瞎猜“最近”。

7\. 标注答案太长太解释型
--------------

结果：小模型又啰嗦又容易循环。

8\. 训练/测试泄漏
-----------

结果：离线分数漂亮，线上翻车。

9\. 过拟合少量用户记忆
-------------

结果：模型对旧场景有残留幻觉。

10\. 量化后分布变化
------------

结果：训练后未量化模型很好，端上 int4/int8 一落地就变差。  
所以评估一定要测**最终部署形态**。

11\. 只盯 loss，不盯业务指标
-------------------

结果：看起来训练很成功，实际上 False-No-Answer 还很高。

12\. 把 agent 做太自由
-----------------

结果：本来只是“证据转自然语言”的任务，最后被做成一个会自行决策、自己改检索策略、自己扩写解释的系统，失控。

* * *

十、给你一个实际可执行的路线
==============

如果是我来推进，我会按这个节奏做。

第 1 周：先修链路
----------

目标不是训练，而是把训练输入定型。

要完成：

*   去掉 recent3 fallback
*   本地检索接入时间过滤/排序
*   evidence schema 定稿
*   输出模板和拒答规则定稿

交付物：

*   一份固定 JSON schema
*   一份线上推理协议文档
*   100 条真实问题的基线表现

第 2 周：做 500 到 1000 条小数据
-----------------------

要完成：

*   合成 + 真实日志混合采样
*   标注规范
*   验证集和测试集切分
*   先跑小样本 smoke test

交付物：

*   第一版训练集
*   第一版评估脚本
*   基线模型 vs SFT 模型对比

第 3 周：扩到 2k 到 5k 条 SFT
----------------------

要完成：

*   扩样本分布
*   加入更多部分覆盖/时间冲突/零命中
*   跑正式 LoRA

交付物：

*   SFT 模型
*   五项业务指标报告

第 4 周：补偏好优化
-----------

只在这时做：

*   False-No-Answer 定向修正
*   Repetition 定向修正

交付物：

*   小规模 DPO/ORPO 版本
*   和 SFT-only 的 AB 对比

* * *

十一、你现在就可以准备的文件清单
================

为了开始微调，你至少要准备这几类东西。

1\. schema 文档
-------------

定义：

*   question
*   intent
*   retrieval.hit\_count
*   evidence 字段
*   output 规则

2\. 标注规范
--------

定义：

*   命中/零命中/部分覆盖如何写答案
*   时间表达格式
*   长度要求
*   禁止脑补

3\. 数据导出脚本
----------

把真实问题和检索结果导出来，生成训练样本草稿。

4\. 评估脚本
--------

自动算：

*   时间准确率
*   False-No-Answer
*   Repetition
*   Evidence Utilization
*   Temporal Ranking

5\. 失败样本库
---------

专门收集：

*   有证据却拒答
*   重复输出
*   时间答错
*   部分覆盖处理错误

这个失败样本库后面会非常有价值，既能做 DPO，也能做回归测试。

* * *

十二、最后给你一句最实用的判断标准
=================

对 BrainDance 这种任务，微调成功不该表现为：

*   “模型更聪明了”
*   “回答更像 ChatGPT 了”

而应该表现为这三件具体的事：

1.  **该答的时候更少漏答**
2.  **不该答的时候更少脑补**
3.  **答的时候更短、更稳、更按时间来**

只要这三点没明显提升，哪怕 loss 很漂亮，也不算真正成功。

这次我没从仓库里再搜到你前面提到的 `TUNE_DOC.md`，所以这里没有基于那份文档逐段展开；我现在这套讲法主要是结合你现有 BrainDance 问答链路和你描述的目标来给的。

下一条我可以直接继续给你一份更实操的东西：  
**“BrainDance 微调数据集该怎么设计”**，包括样本 JSON 模板、正负例写法、以及一套可直接开做的字段规范。

## Prompt:
请你告诉我具体应该如何从零开始训练，每一步做什么，比如用什么软件用什么算法

## Response:
我直接给你一条“能开工”的路线：从环境、工具、数据格式、训练算法到部署，都按 BrainDance 这种本地记忆问答来落。重点不是炫技，而是让你知道每一步该装什么、跑什么、产出什么。

已思考 16s

可以。下面我按**从零开始真的做一遍**的方式讲，尽量具体到：

*   每一步要做什么
*   用什么软件
*   用什么算法
*   产出什么结果
*   最容易踩什么坑

我会先给你一条**最推荐路线**，然后再给你两个备选路线。

对你这个 BrainDance 项目，我最推荐的是：

**检索链路先修好 + 小模型 LoRA SFT + 后续少量 DPO**

也就是：

1.  先修本地 RAG 和回答协议
2.  再做数据集
3.  再用 LoRA 做监督微调
4.  最后针对“有证据却说没有”和“重复输出”做偏好优化

不是一上来全参训练，也不是一上来上 agent。

* * *

一、先定你要训练成什么样
============

先别装软件，第一步是写一页纸目标。

你要训练的不是“一个通用聊天模型”，而是一个非常窄的模型：

> 输入：用户问题 + 结构化检索证据  
> 输出：一段简短中文回答  
> 规则：有证据必须答，没证据才拒答，时间类问题按 created\_at 处理，不重复，不脑补

这一步产出一份文档就够，内容包括：

### 1\. 输入长什么样

例如：

```
{
  "question": "我最近拍了什么？",
  "intent": {
    "question_type": "recent_capture",
    "time_range": {
      "start": "2026-03-14T00:00:00Z",
      "end": "2026-03-21T23:59:59Z"
    }
  },
  "retrieval": {
    "hit_count": 2,
    "evidence": [
      {
        "created_at": "2026-03-20T18:22:31Z",
        "scene_id": "scene_001",
        "description": "桌上有一杯咖啡和一本笔记本",
        "objects": ["咖啡杯", "笔记本"],
        "tags": ["桌面", "办公"]
      }
    ]
  }
}
```

### 2\. 输出长什么样

例如：

```
{
  "answer": "你最近拍到的内容包括咖啡杯和笔记本，时间是 3 月 20 日。",
  "answer_type": "direct"
}
```

### 3\. 业务规则

例如：

*   `hit_count > 0` 不能回答“没有记录”
*   `hit_count = 0` 才能拒答
*   `recent/yesterday/last_week` 按时间规则回答
*   最多回答 1 到 3 条
*   不复述问题
*   不解释规则
*   不说“根据你提供的信息”

这一步为什么重要：  
因为后面所有软件、数据、训练，都是围着这个目标转。目标一散，训练就散。

* * *

二、先把推理链路修到可训练
=============

你现在最不适合直接训练的原因，是链路里有噪声。

你仓库里至少有这几个点会污染训练：

*   recall 里检索不到时会 fallback 到最近内容，这会把“没命中”和“随便塞点上下文”混在一起[](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)
    [recall](https://github.com/tianxingleo/BrainDance/blob/104e00ae3e5c5f8622cdb896f8fe57a72c6d73ab/app/lib/pages/recall.dart)
*   本地 RAG 没把时间语义真正纳入检索排序，`created_at` 只用于 fingerprint，不参与搜索排序[](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)
    [local\_rag\_index](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)
    [
    local\_rag\_index
    ](https://github.com/tianxingleo/BrainDance/blob/main/app/lib/services/local_rag_index.dart)
*   服务端有时间解析和时间过滤，但本地问答链路没强制用它作为“证据层”[](https://github.com/tianxingleo/BrainDance/blob/main/supabase/functions/search-models/index.ts)
    [index](https://github.com/tianxingleo/BrainDance/blob/main/supabase/functions/search-models/index.ts)
    [
    20260306123456\_match\_memory\_pos…
    ](https://github.com/tianxingleo/BrainDance/blob/main/supabase/migrations/20260306123456_match_memory_poses.sql)

所以训练前你要先改 3 件事。

第一步要改什么
-------

### 1\. 去掉“检索失败就塞 recent 3 条”

检索为空就返回空 evidence，不要偷偷补最近内容。

### 2\. 本地检索加时间路由

规则建议写成：

*   问“最近/昨天/上周”  
    先做时间解析  
    再做时间过滤  
    再排序
*   只有普通对象搜索，才语义相似度优先

### 3\. 给模型喂结构化 evidence

不要继续喂一大段杂乱文本。

训练和线上都统一成：

*   question
*   intent
*   retrieval.hit\_count
*   evidence\[\]

这样后面训练才稳。

* * *

三、从零开始需要装什么软件
=============

下面给你一个**最实用的软件栈**。

方案 A：最推荐，简单稳妥
-------------

适合你这种第一次把项目训练跑起来。

### 系统环境

*   Ubuntu 22.04
*   Python 3.10 或 3.11
*   CUDA 12.x
*   Git

### 训练框架

*   PyTorch
*   Transformers
*   PEFT
*   TRL
*   Datasets
*   bitsandbytes
*   accelerate

### 更省事的封装

二选一即可：

*   **LLaMA-Factory**
*   **Unsloth**

我的建议是：

### 如果你想最快跑通

选 **LLaMA-Factory**

优点：

*   配置化
*   支持 SFT、LoRA、DPO
*   做实验很快
*   不用你自己拼太多训练脚本

### 如果你偏向单卡极致省显存

选 **Unsloth**

优点：

*   单卡训练很方便
*   LoRA 跑得快
*   对小模型实验很友好

### 如果你想完全掌控

直接用：

*   Transformers + PEFT + TRL

优点：

*   灵活
*   可控
*   后期便于做定制

缺点：

*   你要自己写更多代码

* * *

四、硬件怎么选
=======

你这个任务不是开放域大模型训练，所以硬件没必要特别夸张。

最小可行
----

*   1 张 24GB 显卡
*   比如 RTX 4090 / 5090 级别，或云上 24GB 卡

可以做：

*   1.7B 到 3B 模型
*   QLoRA / LoRA
*   2k 到 5k 样本训练

更舒服
---

*   1 张 48GB
*   或 2 张 24GB

可以做：

*   更大 batch
*   更长输入
*   更稳的验证

不推荐
---

*   直接全参微调
*   一上来训 7B 以上并且想端侧部署

因为你这个任务没必要。

* * *

五、模型怎么选，算法怎么选
=============

模型基座
----

第一版建议：

*   **Qwen3-1.7B**：端侧优先
*   **Qwen3-3B**：效果优先但还能接受

你这个任务上，先别追求很大。  
因为你不是让模型“懂世界”，而是让它“会根据证据说话”。

训练算法
----

从零开始我建议这样：

### 阶段 1：LoRA SFT

这是主算法。

SFT = Supervised Fine-Tuning  
也就是监督微调。

训练目标：

*   给定 question + evidence
*   输出标准答案

LoRA 的作用：

*   不改全部参数
*   只训练小量可学习矩阵
*   显存便宜
*   收敛快
*   适合你这种窄任务

### 阶段 2：DPO 或 ORPO

等 SFT 跑通后再做。

目标是专门修：

*   有证据却拒答
*   重复输出
*   啰嗦
*   不按时间答

不推荐的起手式
-------

*   全参微调：太贵，不必要
*   RLHF 起手：太重
*   一上来训练 agent：你现在先需要稳定答题器

* * *

六、数据集从零怎么做
==========

这是最核心的部分。

数据格式怎么定
-------

推荐用 instruction 风格，最容易控。

每条样本长这样：

```
{
  "instruction": "你是个人记忆问答助手。只能根据给定证据回答。若 hit_count>0 必须回答；若 hit_count=0 才能说暂无记录。回答简短，不复述问题。",
  "input": {
    "question": "我最近拍了什么？",
    "intent": {
      "question_type": "recent_capture",
      "time_range": {
        "start": "2026-03-14T00:00:00Z",
        "end": "2026-03-21T23:59:59Z"
      }
    },
    "retrieval": {
      "hit_count": 2,
      "evidence": [
        {
          "created_at": "2026-03-20T18:22:31Z",
          "scene_id": "scene_001",
          "description": "桌上有一杯咖啡和一本笔记本",
          "objects": ["咖啡杯", "笔记本"],
          "tags": ["桌面", "办公"]
        }
      ]
    }
  },
  "output": "你最近拍到的内容包括咖啡杯和笔记本，时间是 3 月 20 日。"
}
```

样本从哪里来
------

### 方法 1：从真实日志做

最有价值。

做法：

*   导出真实用户问句
*   用你修好的检索链路生成 evidence
*   人工标注答案

### 方法 2：从数据库合成

起量最快。

做法：

*   从 model\_assets / memory\_poses 采样
*   自动生成问题模板
*   再自动生成草稿答案
*   人工抽查修正

### 方法 3：混合

推荐你用这个。

*   60% 合成
*   40% 真实修订

样本类型至少要有这些
----------

### 1\. 单命中

“我昨天拍了什么？”

### 2\. 多命中

“我最近拍了什么？”

### 3\. 部分覆盖

“我最近拍到杯子和猫了吗？”  
只有杯子，没有猫

### 4\. 零命中

“我上周拍过钢琴吗？”

### 5\. 时间冲突

evidence 有昨天和上月内容，但问题只问昨天

### 6\. 输出稳定性

防重复、防复读、防啰嗦

样本量怎么起步
-------

### 烟雾测试

先 500 到 1000 条  
目的是看训练链路是否正常

### 第一版可用

2000 到 5000 条  
足够验证方向

### 再往后

不是越多越好，先看错误分布

* * *

七、数据清洗应该怎么做
===========

训练前一定要做这几件事。

1\. 去重
------

相同 question + evidence 别重复太多次。

2\. 统一风格
--------

答案长度、日期表达、语气要统一。

3\. 做 train/val/test 切分
-----------------------

建议：

*   train：80%
*   val：10%
*   test：10%

但不要只随机切。  
更稳的是按 `scene_id` 或时间段做部分隔离，防止泄漏。

4\. 做失败样本单独桶
------------

专门留出：

*   有证据却说没有
*   重复输出
*   时间答错
*   部分覆盖处理错

这批数据后面做 DPO 非常有用。

* * *

八、训练软件怎么用，具体流程是什么
=================

下面我给你一条最容易开工的具体路线。

* * *

路线 1：LLaMA-Factory 跑 LoRA SFT
-----------------------------

### 第 1 步：准备环境

安装：

*   Python
*   CUDA
*   PyTorch
*   LLaMA-Factory

你会得到一个训练目录，里面放：

*   数据集 jsonl
*   配置文件 yaml
*   输出目录

### 第 2 步：整理数据

把样本转成 json/jsonl。

例如每行一条：

```
{"instruction":"你是个人记忆问答助手。只能根据证据回答。","input":"{\"question\":\"我最近拍了什么？\",\"retrieval\":{\"hit_count\":1,\"evidence\":[{\"created_at\":\"2026-03-20T18:22:31Z\",\"description\":\"桌上有一杯咖啡和一本笔记本\",\"objects\":[\"咖啡杯\",\"笔记本\"],\"tags\":[\"桌面\",\"办公\"]}]}}","output":"你最近拍到的内容包括咖啡杯和笔记本，时间是 3 月 20 日。"}
```

### 第 3 步：写训练配置

核心配置包括：

*   base model
*   stage = sft
*   finetuning\_type = lora
*   learning rate
*   epoch
*   batch size
*   max length
*   validation

第一版建议：

*   lr = 2e-5
*   epoch = 2 到 3
*   lora rank = 8
*   max length 按你 evidence 长度定，别太短

### 第 4 步：启动训练

训练时你主要盯：

*   train loss
*   val loss
*   业务验证集表现

### 第 5 步：导出 adapter

训练完先不要急着合并权重。  
先保留 LoRA adapter，便于回滚和做对比。

* * *

路线 2：Unsloth 跑 QLoRA
--------------------

如果你更想省显存。

### 软件

*   Python
*   PyTorch
*   Unsloth
*   Transformers
*   PEFT

### 算法

*   4bit 加载基座
*   QLoRA 微调

适合：

*   单张 24GB 卡
*   1.7B 到 3B 模型
*   快速验证

缺点：

*   有时你后续做定制评估脚本，还是要自己补

* * *

路线 3：自己写 Transformers + PEFT + TRL
----------------------------------

适合你后面要完全深度定制。

核心组件：

*   `transformers`：模型与 tokenizer
*   `datasets`：加载数据
*   `peft`：LoRA
*   `trl`：SFTTrainer / DPOTrainer
*   `accelerate`：多卡与训练管理

这条路线最灵活，但最费工程时间。

* * *

九、SFT 时每一步到底做什么
===============

这部分我拆得更细一点。

Step 1：加载 tokenizer
-------------------

作用：

*   把 instruction/input/output 转成 token

你要注意：

*   输入 JSON 不要过长
*   evidence 字段不要塞太多无用信息
*   日期格式统一，否则 tokenizer 学得很碎

Step 2：构造 prompt
----------------

训练时 prompt 要和线上尽量一致。

例如训练时就是：

*   system/instruction
*   input(JSON)
*   target(output)

不要训练时喂 JSON，线上却喂自然语言片段，这样会掉点。

Step 3：挂 LoRA
-------------

你要指定注入哪些层。  
一般用默认主干注意力层就够，不要一上来乱改。

Step 4：设置训练参数
-------------

第一版保守就行：

*   LoRA rank 8
*   alpha 16 或 32
*   dropout 0.05 左右
*   lr 2e-5
*   epoch 2~3
*   warmup 少量
*   weight decay 适中

Step 5：训练
---------

训练目标很简单：让模型生成你的标准答案。

Step 6：每个 epoch 做业务评估
---------------------

不要只看 loss。  
要跑固定题集，看这些：

*   时间问答准确率
*   False-No-Answer Rate
*   Repetition Rate
*   Evidence Utilization Rate
*   Temporal Ranking Consistency

* * *

十、训练完成后怎么评估
===========

你不能只抽 10 条看着顺眼就上线。

建议固定一个 benchmark，分桶测：

桶 1：最近类
-------

*   我最近拍了什么
*   这几天我拍了什么

桶 2：昨天/上周类
----------

*   我昨天拍了什么
*   上周拍过杯子吗

桶 3：对象类
-------

*   我最近拍过猫吗

桶 4：部分覆盖类
---------

*   我最近拍过杯子和猫吗

桶 5：零命中类
--------

*   我最近拍过钢琴吗

桶 6：抗重复类
--------

*   重复问同一问题
*   evidence 很像但不同

### 每桶都统计

*   命中时回答率
*   零命中拒答率
*   时间正确率
*   重复率

* * *

十一、第二阶段怎么做 DPO / ORPO
=====================

等 SFT 模型已经“基本会答”之后，再做这一层。

它解决什么问题
-------

最适合修这些：

*   有 evidence 却答“没有”
*   回答啰嗦
*   回答重复
*   先说废话再答
*   顺序不按时间

训练数据怎么做
-------

每条输入配一对答案：

*   chosen：正确答案
*   rejected：错误答案

例如：

### 输入

“我最近拍了什么？” + 两条 evidence

### rejected

“没有相关记录。”

### chosen

“你最近拍到的内容包括咖啡杯和笔记本，时间是 3 月 20 日。”

再比如：

### rejected

“你最近拍了什么？你最近拍了什么？你最近拍了什么？”

### chosen

“你最近拍到的内容包括咖啡杯和笔记本。”

软件
--

*   TRL
*   或 LLaMA-Factory 里的 DPO 支持

注意
--

DPO 不该替代 SFT。  
一定是：

**SFT 先把任务学会，DPO 再把偏好拉正。**

* * *

十二、训练后怎么部署到本地
=============

你最终要的是本地 AI 问答，不是训练报告。

所以训练完后还要做三件事。

1\. 导出模型
--------

你有两种方式：

### 方式 A：保留基座 + LoRA adapter

优点：

*   灵活
*   容易切回旧版本
*   实验方便

### 方式 B：合并权重后再量化

优点：

*   部署更直接

2\. 量化
------

本地端侧几乎一定要量化。

常见目标：

*   4bit
*   8bit

但这里一定要记住：  
**量化后效果会变。**

所以评估必须测：

*   未量化模型
*   量化后的最终部署模型

3\. 接到你的 BrainDance 问答链路
------------------------

线上推理应该是：

1.  用户问题
2.  本地时间解析 / 意图解析
3.  本地检索
4.  生成结构化 evidence
5.  把 evidence 送给微调模型
6.  输出短答案
7.  轻量去重后处理

模型只是第 5 步，不是整条链路的替代品。

* * *

十三、你最该避免的坑
==========

我给你列最关键的 10 个。

1\. 先训练，后修检索
------------

这是最大的坑。  
你会把错误 evidence 分布教给模型。

2\. 把事实写进参数
-----------

模型会很快过时，还会残留旧记忆幻觉。

3\. 训练数据太模板化
------------

线下很好，线上一换问法就不行。

4\. 零命中样本太少
-----------

模型开始到处脑补。

5\. 有证据必须回答的样本太少
----------------

模型动不动说“没有记录”。

6\. 时间语义只交给模型理解
---------------

正确做法是：时间先在检索层解决，模型只负责表述。

7\. 训练时 schema 和线上 schema 不一致
-----------------------------

这是掉点重灾区。

8\. 只看 loss 不看业务指标
------------------

loss 漂亮，不代表产品可用。

9\. 忽略量化后回归
-----------

量化后经常会变短、变呆、变容易拒答。

10\. DPO 太早做
------------

SFT 都没学会就做 DPO，只会更乱。

* * *

十四、给你一条最实用的开工方案
===============

如果你现在就要开始，我建议按这个最小路线做。

第 0 步：先准备环境
-----------

软件：

*   Ubuntu
*   Python 3.11
*   CUDA
*   PyTorch
*   LLaMA-Factory

第 1 步：先修代码链路
------------

做完这 3 个改动再训练：

*   去掉 recall 里的 recent3 fallback
*   本地 RAG 加时间过滤和排序
*   evidence schema 统一

第 2 步：导出第一批数据
-------------

做 800 条左右：

*   300 条 recent/yesterday/last\_week
*   150 条 有证据必须回答
*   150 条 零命中
*   100 条 部分覆盖
*   100 条 抗重复

第 3 步：用 Qwen3-1.7B 做 LoRA SFT
-----------------------------

参数起点：

*   rank 8
*   lr 2e-5
*   epoch 2
*   先跑 smoke test

第 4 步：看 5 个业务指标
---------------

不是看主观感觉，而是看：

*   时间准确率
*   False-No-Answer
*   重复率
*   Evidence Utilization
*   排序一致性

第 5 步：扩到 2k-5k 样本
-----------------

只有第 3 步结果靠谱，才扩。

第 6 步：再做少量 DPO
--------------

只修两类：

*   有证据却拒答
*   重复输出

第 7 步：量化后部署再测一轮
---------------

一定测最终端侧形态。

* * *

十五、如果你问我“最推荐的软件+算法组合是什么”
========================

我会给你一个非常明确的答案：

最推荐组合
-----

**LLaMA-Factory + Qwen3-1.7B + LoRA SFT + 后续少量 DPO**

原因：

*   上手快
*   显存压力小
*   适合你的窄任务
*   容易做多轮实验
*   便于本地部署

如果你只有一张 24GB 卡
--------------

**Unsloth + QLoRA** 也很好。

如果你后面要深度定制
----------

再转到：  
**Transformers + PEFT + TRL**

* * *

十六、你现在下一步最该做什么
==============

不是立刻训练，而是先把下面 4 个文件做出来：

1.  `schema.md`  
    定义输入输出协议
2.  `annotation_guide.md`  
    定义怎么标注答案、什么时候拒答、时间怎么写
3.  `build_dataset.py`  
    把真实数据和检索结果导成训练样本
4.  `eval.py`  
    自动统计 5 个业务指标

这 4 个东西一旦有了，后面的训练就会非常顺。

如果你愿意，我下一条可以直接给你一套**可执行的工程模板**：  
包括目录结构、`jsonl` 样本格式、`LLaMA-Factory` 训练配置思路，以及一版适合 BrainDance 的 `build_dataset.py` 设计。

## Prompt:
be90efc0-adb6-414f-abd6-025add05a685.md

文件

357ea2e9-a371-46e6-adbd-39b707b941f6.pdf

PDF

继续，以及这是之前提到的两个文档

## Response:
已思考若干秒

我继续，并把你刚上传的两份文档也纳进来一起校正。

先说结论：  
这两份文档对你有帮助，但**要分开看**。

*   你那份整理版方案，作为项目落地提纲是可用的，尤其“训练能力而不是事实”“LoRA 一次微调 + 动态检索”的主线是对的
    be90efc0-adb6-414f-abd6-025add0…
*   那篇 P-RAG 论文可以提供一些启发，但**不能直接照搬成 BrainDance 的实现蓝图**，因为它研究的是 biomedical / multi-hop QA，不是“个人时空记忆问答”；而且它的 P-RAG 里“参数知识 + 检索知识”的关系，比你这个场景更偏“领域知识补强”，不等于你应该把用户记忆再塞进参数里
    357ea2e9-a371-46e6-adbd-39b707b…

所以我下面给你的，是一版**结合你项目现状 + 这两份文档后，更能直接开工的训练路线**。

* * *

一、先用这两份文档校正方向
=============

1\. 你整理文档里哪些是对的
---------------

这几条我认可：

*   训练“如何利用检索结果回答”，而不是训练“记住用户拍过什么”
    be90efc0-adb6-414f-abd6-025add0…
*   数据格式做成“问题 + 相关记忆 + 输出”是对的，本质是在训 evidence use pattern，而不是普通闲聊
    be90efc0-adb6-414f-abd6-025add0…
*   LoRA 是适合你当前阶段的，成本低，容易在小模型上验证
    be90efc0-adb6-414f-abd6-025add0…
*   单跳问答不要迷信 CoT，这点你的文档后半段也意识到了，这和论文里的发现基本一致：CoT 对多跳帮助明显，但对简单单跳未必有益，甚至可能有副作用
    be90efc0-adb6-414f-abd6-025add0…
    357ea2e9-a371-46e6-adbd-39b707b…

2\. 需要修正的地方
-----------

### 第一处：别把“P-RAG”理解成“把用户知识参数化”

论文里的 P-RAG 是“参数知识 + 检索知识”混合，用 LoRA 让模型更适应某个领域，再结合检索做回答

357ea2e9-a371-46e6-adbd-39b707b…

这在医学 QA 里成立，因为“医学知识”本来就是一种相对稳定、可以参数化吸收的领域知识。

但你这里是 BrainDance 的**个人动态记忆**：

*   今天拍了荷花
*   明天拍了咖啡杯
*   下周又新增几十条

这种事实变化太快，不应该参数化。  
所以你应该借鉴的是：

*   LoRA 学“如何读证据和组织回答”
*   用户事实永远留在检索层

不是借鉴：

*   把用户最近记录也变成 parametric knowledge

### 第二处：不要把“检索片段文本”当最终格式

你文档里示例是：

*   2025-03-20: 在西湖拍了荷花
*   2025-03-18: 在植物园拍了樱花

这适合概念展示，但真到工程里，最好不要长期用这种纯自然语言片段做训练输入。  
更稳的是：

*   结构化 JSON evidence
*   再转成统一 prompt

原因很简单：  
结构化更适合你后面做评估、排错、扩字段和自动生成训练集。

### 第三处：top-k 不能机械固定

论文实验里 top-k 检索是标准 RAG 配置的一部分，能给你一个参考，但不代表 BrainDance 永远 k=3 最优

357ea2e9-a371-46e6-adbd-39b707b…

你这里更适合按意图分：

*   `recent_capture`：优先 top 3
*   `count` 类问题：可能要 top 5 或 top 10
*   `detail` 类问题：top 1 到 3 就够

* * *

二、我给你的最终训练路线
============

现在进入你真正关心的部分：  
**从零开始具体怎么训练，每一步做什么，用什么软件、什么算法。**

我给你一条最适合现在开工的路线：

训练总路线
-----

1.  定任务协议
2.  修线上推理链路
3.  造训练数据
4.  训练第一版 LoRA SFT
5.  跑业务评估
6.  补 DPO/ORPO
7.  量化部署回归测试

* * *

三、第 0 步：先准备环境
=============

你需要的软件
------

### 开发环境

*   Ubuntu 22.04
*   Python 3.10 或 3.11
*   Git
*   CUDA 12.x
*   NVIDIA 驱动

### Python 训练栈

*   PyTorch
*   transformers
*   datasets
*   peft
*   trl
*   accelerate
*   bitsandbytes

### 训练前端工具，二选一

*   **LLaMA-Factory**
*   **ms-swift**

你前面的文档里提到 `ms-swift` 或 `LLaMA-Factory`，这两个都能用

be90efc0-adb6-414f-abd6-025add0…

但如果你问我更推荐哪个：

我的建议
----

### 第一版优先选：LLaMA-Factory

原因：

*   社区资料多
*   SFT / DPO 支持成熟
*   配置直观
*   更适合你先把第一版跑起来

### 如果你更熟阿里系工具链

可以用 ms-swift  
也没问题，只是第一版我觉得 LLaMA-Factory 更容易少踩坑。

* * *

四、第 1 步：先修链路，不修别训
=================

这一点比软件重要。

你现在必须先把线上问答改成一种**可训练的稳定协议**。  
否则你训出来的模型只是在适应噪声。

这一步要完成什么
--------

### 1\. 去掉 recall 的 recent fallback

检索空命中就空命中，不许再偷偷喂最近 3 条。  
因为这会让“无证据”和“伪证据”混在一起。

### 2\. 做一个轻量意图解析器

别一开始就让大模型来做。

你可以先用**规则 + 小型映射**：

识别这些类型：

*   `recent_capture`
*   `time_query`
*   `object_query`
*   `count_query`
*   `detail_query`

以及槽位：

*   `time_range`
*   `target_object`
*   `need=list/count/detail`

### 3\. 本地检索改成双通道

不是只有向量召回。

建议：

#### 时间驱动类问题

例如：

*   最近拍了什么
*   昨天拍了什么
*   上周拍过杯子吗

处理逻辑：

1.  解析时间范围
2.  先按时间过滤
3.  再按相似度重排
4.  最终按业务规则排序

#### 普通对象语义类问题

例如：

*   我拍过猫吗
*   我之前是不是见过红杯子

处理逻辑：

1.  语义召回
2.  结合 tags/objects 做 lexical boost
3.  必要时再按时间辅助排序

### 4\. 统一 evidence schema

这一步特别重要。

我建议先冻结成这个格式：

```
{
  "question": "我最近拍了什么？",
  "intent": {
    "question_type": "recent_capture",
    "time_range": {
      "start": "2026-03-14T00:00:00Z",
      "end": "2026-03-21T23:59:59Z"
    },
    "target_object": null,
    "need": "list"
  },
  "retrieval": {
    "hit_count": 2,
    "evidence": [
      {
        "scene_id": "scene_001",
        "created_at": "2026-03-20T18:22:31Z",
        "display_name": "",
        "description": "桌上有一杯咖啡和一本笔记本",
        "objects": ["咖啡杯", "笔记本"],
        "tags": ["桌面", "办公"]
      },
      {
        "scene_id": "scene_002",
        "created_at": "2026-03-19T09:11:02Z",
        "display_name": "",
        "description": "窗边有一盆绿植和白色马克杯",
        "objects": ["绿植", "马克杯"],
        "tags": ["窗边", "室内"]
      }
    ]
  }
}
```

* * *

五、第 2 步：做数据集
============

这一步决定 70% 成败。

你要做的不是聊天数据，而是 evidence-to-answer 数据
-----------------------------------

你前面文档这一点方向是对的：  
用“问题 + 相关记忆 + 输出”去训练

be90efc0-adb6-414f-abd6-025add0…

我建议升级成：

*   instruction
*   input(JSON schema)
*   output(标准回答)

推荐数据格式
------

```
{
  "instruction": "你是个人记忆问答助手。只能依据给定证据回答。若 hit_count>0 必须回答；若 hit_count=0 才能回答暂无相关记录。回答简短，按时间规则组织，不复述问题。",
  "input": {
    "question": "我最近拍了什么？",
    "intent": {
      "question_type": "recent_capture",
      "time_range": {
        "start": "2026-03-14T00:00:00Z",
        "end": "2026-03-21T23:59:59Z"
      },
      "target_object": null,
      "need": "list"
    },
    "retrieval": {
      "hit_count": 2,
      "evidence": [
        {
          "scene_id": "scene_001",
          "created_at": "2026-03-20T18:22:31Z",
          "description": "桌上有一杯咖啡和一本笔记本",
          "objects": ["咖啡杯", "笔记本"],
          "tags": ["桌面", "办公"]
        }
      ]
    }
  },
  "output": "你最近拍到的内容包括咖啡杯和笔记本，时间是 3 月 20 日。"
}
```

数据从哪来
-----

### 路线 A：真实日志回放

最有价值。

做法：

*   导出用户真实问题
*   用修好的检索器生成 evidence
*   人工写正确答案

### 路线 B：从数据库自动合成

最省时间。

做法：

*   从 `model_assets` / `memory_poses` 采样
*   自动拼问题模板
*   自动生成草稿答案
*   人工抽样校正

### 最推荐：混合

*   60% 自动合成
*   40% 真实问题修订

* * *

六、第 3 步：样本类型怎么配
===============

你那份文档里已经提到 0 条、1 条、多条、各种时间表达，这个方向对

be90efc0-adb6-414f-abd6-025add0…

但我建议你配得更细。

第一版数据分布建议
---------

### 40% 时间问答

例如：

*   最近拍了什么
*   昨天拍了什么
*   上周拍了什么
*   3 月 20 日拍了什么

### 20% 有证据必须回答

例如：

*   我最近拍过杯子吗
*   昨天有没有植物

### 15% 无证据拒答

例如：

*   我上周拍过钢琴吗

### 15% 部分覆盖

例如：

*   我最近拍过杯子和猫吗  
    实际 evidence 只有杯子

### 10% 输出稳定性

例如：

*   同一个问题不同 phrasing
*   多条相似 evidence
*   防复读、防长答

* * *

七、第 4 步：先做小规模数据，不要一上来全量
=======================

第一轮
---

先做 500 到 1000 条，做 smoke test。

目的不是追高分，而是验证 4 件事：

1.  模型有没有学会有证据就答
2.  零命中会不会稳拒答
3.  时间排序会不会变好
4.  会不会减少重复输出

如果这 4 件事没明显改善，别扩到 5k。

第二轮
---

通过 smoke test 后，再扩到 2000 到 5000 条。

你文档里写 1000 到 2000 条作为初始集，我觉得合理；如果第一轮有效，再到 2k-5k 更稳

be90efc0-adb6-414f-abd6-025add0…

* * *

八、第 5 步：开始第一版训练
===============

模型怎么选
-----

### 首选

*   Qwen3-1.7B

理由：

*   够小
*   适合端侧
*   足以做“证据转自然语言”

### 第二选择

*   Qwen3-3B

理由：

*   表达更稳
*   但部署代价更大

算法怎么选
-----

### 第一阶段：LoRA SFT

这是主方法。

原因：

*   便宜
*   快
*   不容易灾难性遗忘
*   适合窄任务

这一点和你文档、论文方向是一致的：LoRA 通过冻结骨干参数，只训练小适配层，更适合资源受限环境，也更适合垂直适配

be90efc0-adb6-414f-abd6-025add0…

357ea2e9-a371-46e6-adbd-39b707b…

参数怎么起步
------

你文档里建议：

*   rank 8
*   alpha 32
*   lr 2e-5
*   epoch 3
    be90efc0-adb6-414f-abd6-025add0…

论文里是：

*   rank 2
*   alpha 32
*   dropout 0.05
*   1 epoch
    357ea2e9-a371-46e6-adbd-39b707b…

### 对 BrainDance 的建议

我会取中间偏稳的版本：

*   `lora_rank = 8`
*   `lora_alpha = 16 或 32`
*   `dropout = 0.05`
*   `learning_rate = 2e-5`
*   `num_train_epochs = 2`
*   先别上 3，除非验证集还在涨
*   `batch_size` 看显存，24GB 卡先从 2 或 4 开始

为什么不直接照论文 rank=2？  
因为论文任务是特定 biomedical QA，且基座是 1B，答案形式也比较规整；你这里涉及：

*   时间表达
*   列表组织
*   部分覆盖
*   拒答控制
*   输出稳定性

rank=8 更保险。

* * *

九、第 6 步：训练时用什么工具最顺
==================

用 LLaMA-Factory 时，你要做的事
-----------------------

### 1\. 准备数据

把数据做成 json/jsonl。

### 2\. 配数据集描述

告诉工具：

*   input 字段是什么
*   output 字段是什么
*   instruction 怎么拼

### 3\. 配训练参数

核心就是：

*   base model
*   sft
*   lora
*   rank
*   lr
*   epoch
*   max length

### 4\. 跑训练

保留：

*   adapter 权重
*   训练日志
*   验证集输出样本

### 5\. 每个 epoch 跑业务评估

不是只看 loss。

* * *

十、第 7 步：你必须看的业务指标
=================

这部分你前面已经提到了，我现在补成可执行版。

1\. 时间问答准确率
-----------

问题问“昨天/最近/上周”，答案是否按正确时间窗回答。

2\. False-No-Answer Rate
------------------------

有 evidence，却回答“没有记录/不知道”的比例。

这个对你最关键。

3\. Repetition Rate
-------------------

重复句、循环输出比例。

4\. Evidence Utilization Rate
-----------------------------

回答里有没有真正用到 evidence 中的：

*   description
*   objects
*   tags
*   created\_at

5\. Temporal Ranking Consistency
--------------------------------

多条结果时，是否按时间要求排序。

* * *

十一、第 8 步：什么时候做 DPO / ORPO
=========================

这里也结合两份文档校正一下。

你文档里虽然没展开 DPO 的具体工程步骤，但你前面和我对话里提到“同一输入做错误/正确偏好对”，这是很对的。  
我建议：

先决条件
----

只有当 SFT 模型已经做到：

*   基本会答
*   基本能拒答
*   基本不太乱

才做 DPO / ORPO。

DPO 用来修什么
---------

只修两类最值：

### 1\. 有证据却拒答

输入一样：

*   rejected: “没有相关记录”
*   chosen: “你最近拍到的内容包括咖啡杯和笔记本……”

### 2\. 重复输出

输入一样：

*   rejected: “你最近拍了什么？你最近拍了什么？……”
*   chosen: “你最近拍到的内容包括咖啡杯和笔记本。”

不建议 DPO 去解决什么
-------------

不要拿 DPO 去修：

*   检索 miss
*   时间解析错误
*   schema 不一致

这些是工程链路问题，不是偏好问题。

* * *

十二、第 9 步：量化与部署
==============

你最终要的是本地 AI 问答，所以训练后必须过部署这关。

推荐部署流程
------

### 1\. 先保留 adapter

便于快速回滚、对比。

### 2\. 再做量化

4bit 或 8bit 都可能是你的端侧目标。

### 3\. 用最终部署形态再测一遍

这一步很容易被忽略。

很多时候：

*   未量化模型很好
*   量化后更容易短路拒答、漏细节、变呆

所以**最终评估一定要在量化后的模型上做**。

* * *

十三、结合论文，我对 CoT 的最终建议
====================

这点要特别说，因为你上传的论文里一个重要发现就是：

*   CoT 对 multi-hop 帮助很大
*   对 single-hop biomedical QA 可能反而轻微变差
    357ea2e9-a371-46e6-adbd-39b707b…

这对你非常有启发。

对 BrainDance 的建议
----------------

### 默认不要 CoT

对于：

*   我最近拍了什么
*   我昨天拍了什么
*   我最近拍过杯子吗

这些都属于偏单跳、证据明确的问题。  
不建议上 CoT。

### 只在复杂问题时考虑简化 CoT

例如：

*   我上周拍的花和这周拍的花哪个更多
*   我最近在室内拍到的杯子和植物分别有哪些

这类涉及比较、汇总、过滤叠加的问题，才可以考虑：

*   先在 prompt 里加一层很轻的“先筛选再回答”
*   或者做专门复杂意图路由

但这应该是第二阶段，不是第一阶段。

* * *

十四、我给你一版最实际的开工清单
================

第 1 周
-----

完成：

*   去掉 recent fallback
*   完成意图 schema
*   完成 evidence schema
*   导出 100 条样本人工检查

第 2 周
-----

完成：

*   生成 800 条第一版数据
*   写标注规范
*   写评估脚本
*   跑第一版 smoke test

第 3 周
-----

完成：

*   扩到 2k-3k 数据
*   用 Qwen3-1.7B 做 LoRA SFT
*   跑五项业务指标

第 4 周
-----

完成：

*   收集失败样本
*   做小规模 DPO/ORPO
*   量化部署回归测试

* * *

十五、现在这两份文档对你的“最终提炼”
===================

我把它们压缩成一句最关键的话：

**论文告诉你：LoRA + 检索增强 + 选择性 CoT 是有价值的；你的项目文档告诉你：应该训练“如何根据记忆回答”；而你项目本身要求你再补上一条——用户事实绝不能参数化，必须永远留在检索层。**

be90efc0-adb6-414f-abd6-025add0…

357ea2e9-a371-46e6-adbd-39b707b…

所以 BrainDance 的最佳路线不是“复刻 P-RAG”，而是：

**把 P-RAG 的“参数适配思想”裁剪成一个更窄的本地记忆问答器。**

* * *

下一条我可以直接给你最实用的一部分：  
**BrainDance 训练集 JSONL 规范 + 样本模板 + `build_dataset.py` 设计草图**。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)