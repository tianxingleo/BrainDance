# Qwen3-1.7B 微调实践记录 Part 18

## Part 18：体验层与 formatter 打磨

### 本 part 目标

- 不改 LoRA
- 不重写 retrieval 主干
- 重点收掉当前真实链路里仍然偏 debug 风格的问题：
  - `recent_hit` 风格统一
  - `must_answer` focus 更稳
  - inventory 回答更人话
  - abstract semantic summary 更像正常产品回答

### 本 part 的结论

Part 18 最终没有继续开新一轮训练，也没有回去继续抠 `object_lookup`，而是把回答层收敛成了 4 条稳定 formatter 路由：

- `recent_answer_formatter`
- `must_answer_focus_formatter`
- `inventory_formatter`
- `semantic_summary_formatter`

在真实链路的 `17` 条体验验证集上，最终结果为：

- `formatter_answer_rate = 1.0`
- `natural_style_rate = 1.0`
- `recent_style_rate = 1.0`
- `must_answer_focus_rate = 1.0`
- `inventory_humanized_rate = 1.0`
- `semantic_summary_readability = 1.0`

这说明当前主线已经从“能答出来”推进到了“回答更像产品，而不是 debug 系统”。

---

## 一、问题确认

Part 17 收口后，`object_lookup` 检索已经形成有效闭环，但真实体验层还有 4 个明显问题：

### 1. `recent_hit` 仍有风格波动

表现为：

- 有时像列举碎片
- 有时短语直接硬拼
- 有时虽然答对，但不像一个自然短答

### 2. `must_answer` 还有 focus 不稳尾巴

典型问法包括：

- `最近拍到过什么办公桌上的东西？`
- `最近拍到过什么地球仪相关画面？`
- `最近拍到过什么书架相关画面？`

主要问题不是“没检到”，而是：

- 主目标没有稳定出现在第一分句
- 陪衬物体有时比主目标更显眼

### 3. inventory 回答仍偏工程视角

虽然 inventory special-case 已经稳定，但旧回答更像：

- debug 摘要
- 内部命名回显
- 简单列条目

缺少“最近生成过几个模型，主要包括什么”的产品表达。

### 4. abstract semantic query 还没有被真正单独收口

第一次实现后，真实链路里暴露出 3 个残留问题：

- 某些抽象语义问法会被误吸进 `inventory`
- 某些抽象语义词没有真正参与 lexical / formatter 路径
- parser 偶尔把 `search_text` 吃空，导致后续链路拿不到 semantic term

这说明 abstract semantic 仍然需要代码层显式兜底，而不是继续依赖 LoRA 自由发挥。

---

## 二、代码改动

### 1. 在 `run_real_chain_debug.py` 中补齐 4 条 formatter 主路由

本轮在 [run_real_chain_debug.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py) 内新增并接入：

- `recent_answer_formatter`
- `must_answer_focus_formatter`
- `inventory_formatter`
- `semantic_summary_formatter`

对应实现包括：

- `build_recent_answer()`
- `build_must_answer_focus_answer()`
- `build_model_inventory_answer()` 人话化升级
- `build_semantic_lookup_answer()` 可读性升级

同时把 `answer_route` 显式扩展为：

- `recent_answer_formatter`
- `must_answer_focus_formatter`
- `inventory_formatter`
- `semantic_summary_formatter`
- `lora_generation`

这样 interactive / debug / eval 三条链路都能直接观测 formatter 实际接管比例。

### 2. recent formatter 收敛成固定短答风格

本轮对 `recent_hit` 的策略是：

- 统一输出 1 句
- 优先按 `created_at` 从近到远取最近 1 到 2 个 evidence
- 每个 evidence 只抽 1 到 2 个高辨识度对象
- 短语级连接改成更自然的 `以及`

同时补了一个细节：

- 如果问题里有显式 focus term，但 evidence 里没有直接命中该词，也会保留 `xxx相关内容里` 这个主框架

这样 `书架相关画面` 这类问法不会再退化成只列陪衬物体。

### 3. must-answer formatter 改成“主目标优先”

`must_answer_focus_formatter` 的核心规则是：

- 先抽主目标
- 再补 1 到 2 个陪衬对象
- 第一分句必须围绕 focus term 组织

对容器类 focus，例如：

- `办公桌`
- `书架`
- `地球仪`

统一改成：

- `最近拍到过 xxx 相关内容，能看到 ...`

这样可以避免回答虽然自然，但主目标不稳地被边角物件盖过去。

### 4. inventory formatter 改成人话表达

`build_model_inventory_answer()` 做了两点变化：

- 从“直接列内部条目”改成“最近生成过 N 个模型，主要包括 ...”
- 保留可选 `include_scene_ids` 参数，便于后续 debug 态扩展，但默认用户态不暴露 scene id

最终 inventory 回答保持：

- 用户态短答
- debug 能力仍可保留扩展空间

### 5. abstract semantic query 补了显式兜底

这部分是 Part 18 后半段真正解决的问题。

本轮新增：

- `collect_semantic_lookup_terms()`
- `infer_semantic_terms_from_question()`
- `is_abstract_semantic_query()`

具体策略是：

1. 若抽象语义词存在，就把 query 显式识别成 semantic formatter 目标，而不是普通 object lookup
2. semantic expansion 词真正参与 retrieval 与 lexical fallback
3. 若 parser 没给出 `search_text`，就从原问题中回填 semantic term
4. 对 `偏理工一点的模型` 这类问法，禁止误走 inventory special-case
5. `build_semantic_lookup_answer()` 允许从 description 中回收语义词，不只盯 objects/tags

这样 abstract semantic 的稳定性不再依赖 parser 必须完美输出。

---

## 三、测试与验证

### 1. 新增测试

新增测试文件：

- [test_part18_formatters.py](/ltx-data/BrainDance/tests/test_part18_formatters.py)

覆盖点包括：

- recent formatter 的自然短答风格
- must-answer formatter 的 focus 优先
- inventory formatter 的人话表达
- semantic formatter 的可读性
- semantic description fallback
- abstract semantic 问法不误走 inventory
- semantic term question-level backfill

同时保留 Part 17 的回归测试：

- [test_part17_object_lookup.py](/ltx-data/BrainDance/tests/test_part17_object_lookup.py)

本轮测试命令：

```bash
pytest -q tests/test_part17_object_lookup.py tests/test_part18_formatters.py
python -m py_compile ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py ai_engine/finetune_qwen3/scripts/evaluate_experience_part18.py
```

结果：

- `13 passed`
- `py_compile` 通过

### 2. 新增 Part 18 体验验证集

新增文件：

- [experience_eval_cases_part18.json](/ltx-data/BrainDance/ai_engine/finetune_qwen3/data/experience_eval_cases_part18.json)

共 `17` 条，覆盖 4 组：

- `recent_hit = 4`
- `must_answer = 5`
- `inventory = 4`
- `abstract_semantic = 4`

### 3. 新增体验评测脚本

新增脚本：

- [evaluate_experience_part18.py](/ltx-data/BrainDance/ai_engine/finetune_qwen3/scripts/evaluate_experience_part18.py)

评测维度不再只盯 hit/bad，而是直接看体验层指标：

- `formatter_answer_rate`
- `natural_style_rate`
- `recent_style_rate`
- `must_answer_focus_rate`
- `inventory_humanized_rate`
- `semantic_summary_readability`

评测命令：

```bash
python ai_engine/finetune_qwen3/scripts/evaluate_experience_part18.py \
  --cases_file ai_engine/finetune_qwen3/data/experience_eval_cases_part18.json \
  --output_file ai_engine/finetune_qwen3/logs/experience_eval_part18.jsonl \
  --summary_file ai_engine/finetune_qwen3/logs/experience_eval_part18_summary.json \
  --hard_cases_file ai_engine/finetune_qwen3/logs/experience_eval_part18_hard_cases.json
```

最终结果：

- `case_count = 17`
- `answer_route_counts`
  - `recent_answer_formatter = 8`
  - `must_answer_focus_formatter = 1`
  - `inventory_formatter = 4`
  - `semantic_summary_formatter = 4`
- `formatter_answer_rate = 1.0`
- `natural_style_rate = 1.0`
- `recent_style_rate = 1.0`
- `must_answer_focus_rate = 1.0`
- `inventory_humanized_rate = 1.0`
- `semantic_summary_readability = 1.0`

#### 按组结果

- `recent_hit`
  - `count = 4`
  - `formatter_answer_rate = 1.0`
  - `natural_style_rate = 1.0`
  - `recent_style_rate = 1.0`
- `must_answer`
  - `count = 5`
  - `formatter_answer_rate = 1.0`
  - `natural_style_rate = 1.0`
  - `must_answer_focus_rate = 1.0`
- `inventory`
  - `count = 4`
  - `formatter_answer_rate = 1.0`
  - `natural_style_rate = 1.0`
  - `inventory_humanized_rate = 1.0`
- `abstract_semantic`
  - `count = 4`
  - `formatter_answer_rate = 1.0`
  - `natural_style_rate = 1.0`
  - `semantic_summary_readability = 1.0`

---

## 四、过程中发现的问题与解决

### 1. recent formatter 首版短语拼接仍然偏硬

首轮单测里出现：

- `触控笔和机械键盘和书架和地球仪`

问题本质：

- 对象级 `和`
- 短语级 `和`

被混在了一起。

解决：

- 单独增加短语级连接函数
- recent 主句统一用 `以及`

### 2. abstract semantic 首版被误吸进普通 object / recent / inventory 路由

首轮真实链路评测暴露：

- `偏理工一点的模型` 被误当 inventory
- `计算机科学相关内容` 被误当 must-answer
- `学术一点的内容` 被 recent formatter 吞掉

解决：

- 把 abstract semantic 提升成显式 formatter 路由
- semantic expansion 词参与 retrieval
- inventory special-case 增加 semantic guard

### 3. parser 偶尔把 semantic search_text 吃空

典型 case：

- `有没有偏理工一点的模型？`

解析结果一度是：

- `search_text = ""`

导致：

- retrieval 拿不到 semantic term
- formatter 无法工作

解决：

- 新增 `infer_semantic_terms_from_question()`
- 在 retrieval 前做 question-level semantic backfill

### 4. semantic 匹配首版只盯 objects/tags，不足以覆盖 description

首版里有些 evidence：

- objects 里只有泛化物体
- 但 description 里明确提到 `教材 / 词典 / 算法导论`

解决：

- `build_semantic_lookup_answer()` 增加 description 命中兜底

---

## 五、本 part 结论与下一步

Part 18 已经完成了预定主线：

- 没有继续开新 LoRA
- 没有回去深挖 `object_lookup`
- 也没有重写 retrieval 主骨架

而是通过 formatter 与回答策略层，把当前系统从“检索能找对，但回答偏 debug”推进到了“回答更像产品”。

当前可以认为已经完成的能力包括：

- `recent_hit` 的统一自然短答
- `must_answer` 的 focus 稳定性
- inventory 的人话表达
- abstract semantic 的单独 formatter 化

如果继续往下做，下一步更值得看的就不是“再不要再训一轮”，而是：

- 扩大 Part 18 体验验证集规模
- 把 interactive session 的真实用户反馈继续纳入同类指标
- 再看是否存在新的长尾 query class 需要 formatter 化

就当前阶段而言，Part 18 已经可以正式收口。
