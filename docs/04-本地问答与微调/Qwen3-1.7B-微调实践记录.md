# Qwen3-1.7B 微调实践记录

## 项目目标

- 参考仓库内早期本地问答微调建议稿（现已转为本地归档，不再作为 Git 表层文档）
- 从零开始实践 BrainDance 本地问答场景的 `Qwen3-1.7B` 微调
- 明确要求：
  - 只使用 `gpu1`
  - 选择最快可落地路线
  - 每一部分记录做了什么、结果如何、遇到什么问题

---

## Part 0：方案筛选与环境检查

### 本部分目标

- 读取现有微调建议文档
- 判断当前仓库是否已有可复用训练脚本
- 判断 `Braindance` conda 环境能否直接用于微调
- 选定最快的工程路线

### 已完成内容

- 阅读了早期本地问答微调建议稿，并据此整理当前实验路径
- 检查了仓库内是否已有 LoRA / SFT / Qwen 微调脚本
- 检查了 GPU 使用情况
- 检查了 `Braindance` conda 环境中的 Python、PyTorch 与 Hugging Face 相关依赖状态

### 当前观察结果

- `gpu0` 正在高占用中，不可用
- `gpu1` 当前空闲，可作为本次微调设备
- 文档里推荐的最快路线是：
  - `LLaMA-Factory + Qwen3-1.7B + LoRA SFT`
- 仓库当前没有现成的 Qwen3 LoRA 微调脚本可直接复用

### Braindance 环境结论

- `Braindance` 环境不适合直接作为本次微调环境
- 原因 1：运行 `conda run -n Braindance python` 时，`sys.path` 会引入 `~/.local/lib/python3.10/site-packages`
- 原因 2：这导致环境会混用用户级安装包，存在导包污染
- 原因 3：关闭用户级 site-packages 后，环境里只有 `torch`，缺少 `transformers`、`trl`、`bitsandbytes` 等关键训练依赖
- 原因 4：当前 `pip` 也优先指向 `~/.local/bin/pip`，不利于稳定管理训练依赖

### 本部分遇到的问题

- `Braindance` 环境存在用户级 Python 包污染
- 训练依赖不完整
- 如果继续在该环境上修补，容易污染现有项目环境，也不利于排障

### 当前决策

- 不直接复用 `Braindance` 环境
- 新建一个独立的微调 conda 环境
- 全程固定使用：
  - `CUDA_VISIBLE_DEVICES=1`
- 初始打算采用：
  - `LLaMA-Factory + Qwen3-1.7B + LoRA SFT`
- 但该路线还需要在下一部分验证 Python 版本兼容性

### 下一步

- 创建独立微调环境
- 安装最小可运行训练依赖
- 拉起 Qwen3-1.7B 基座
- 开始整理 BrainDance 问答微调数据格式

---

## Part 1：独立微调环境与训练路线修正

### 本部分目标

- 创建干净的训练环境
- 只绑定 `gpu1`
- 让 PyTorch 在新环境中正常识别 CUDA
- 按“最快可落地”原则确认最终训练软件栈

### 已完成内容

- 创建新环境：
  - `conda create -n qwen3_ft python=3.10`
- 为该环境固定配置：
  - `PYTHONNOUSERSITE=1`
  - `HF_ENDPOINT=https://hf-mirror.com`
  - `TOKENIZERS_PARALLELISM=false`
- 创建训练工作目录：
  - `ai_engine/finetune_qwen3/`
- 安装 CUDA 版 PyTorch：
  - `torch 2.10.0+cu126`
  - `torchvision 0.25.0+cu126`
  - `torchaudio 2.10.0+cu126`
- 验证 `gpu1` 可用：
  - `CUDA_VISIBLE_DEVICES=1`
  - `torch.cuda.is_available() == True`
  - 设备识别为 `NVIDIA L20`
  - `bf16` 可用

### 关键问题

- 我按文档优先尝试了 `LLaMA-Factory`
- 但当前最新版 `LLaMA-Factory` 的 `pyproject.toml` 要求：
  - `Python >= 3.11`
- 而当前已经完成 CUDA PyTorch 安装的环境是：
  - `Python 3.10`

### 路线调整原因

- 如果为了 `LLaMA-Factory` 再重建 `Python 3.11` 环境，需要重新下载和安装整套 CUDA PyTorch
- 这会直接拉长总耗时，不符合“最快完成”的要求
- 因此本次实践改为更直接的最小工程路线：
  - `transformers + peft + trl + accelerate`
  - 自写最小 LoRA SFT 训练脚本

### 当前结论

- 最终采用的实际路线为：
  - `Qwen3-1.7B + LoRA SFT + transformers/peft/trl`
- 放弃本次使用最新版 `LLaMA-Factory`
- 原因不是方案错误，而是当前主机上“重新建 3.11 环境”的边际成本更高

### 本部分遇到的问题

- `LLaMA-Factory` 与 `Python 3.10` 不兼容
- 训练工具路线需要根据当前环境即时调整，不能机械照搬文档建议

### 下一步

- 安装 `transformers`、`datasets`、`peft`、`trl`、`accelerate`
- 整理 BrainDance 的训练样本结构
- 编写最小 LoRA SFT 训练脚本
- 开始做第一轮 smoke test

---

## Part 2：数据集与最小训练脚本落地

### 本部分目标

- 先做一版可以真正开跑的 BrainDance 风格 SFT 数据
- 让训练输入贴近“问题 + 检索结果 + 标准回答”
- 准备最小可控的 LoRA SFT 训练与 smoke-eval 脚本

### 已完成内容

- 安装完成训练依赖：
  - `transformers 4.57.1`
  - `datasets 4.0.0`
  - `peft 0.18.1`
  - `accelerate 1.11.0`
  - `trl 0.24.0`
- 新建数据生成脚本：
  - `ai_engine/finetune_qwen3/scripts/build_sft_dataset.py`
- 新建训练脚本：
  - `ai_engine/finetune_qwen3/scripts/train_lora_sft.py`
- 新建 smoke-eval 脚本：
  - `ai_engine/finetune_qwen3/scripts/run_smoke_eval.py`
- 新建 GPU1 启动脚本：
  - `ai_engine/finetune_qwen3/scripts/run_train_gpu1.sh`
  - `ai_engine/finetune_qwen3/scripts/run_smoke_eval_gpu1.sh`

### 数据来源

- `ai_engine/demo/rag/data/output_analyzed/frame_*.json`
  - 作为“触控笔桌面场景”真实描述来源
- `ai_engine/3dgs/test_search_data.py`
  - 作为书房 / 卧室 / 厨房 / 客厅 / 办公室等结构化场景来源

### 当前数据集设计

- 样本格式：
  - `messages = [system, user, assistant]`
- `user` 内容是 JSON：
  - `question`
  - `retrieval.intent`
  - `retrieval.hit_count`
  - `retrieval.evidence[]`
- `evidence` 字段统一为：
  - `scene_id`
  - `display_name`
  - `description`
  - `objects`
  - `tags`
  - `created_at`

### 当前数据集结果

- 生成清单文件：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_manifest.json`
- 生成训练集：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl`
- 生成验证集：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_val.jsonl`
- 当前规模：
  - `record_count = 21`
  - `example_count = 55`
  - `train_count = 47`
  - `val_count = 8`

### 已覆盖样本类型

- `recent_list`
- `time_qa`
- `must_answer`
- `no_hit`
- `partial_coverage`
- `stability`

### 训练脚本设计要点

- 基座模型：
  - `Qwen/Qwen3-1.7B`
- 单卡：
  - `CUDA_VISIBLE_DEVICES=1`
- 精度：
  - `bf16`
- 训练方式：
  - `LoRA SFT`
- LoRA 参数：
  - `rank=8`
  - `alpha=16`
  - `dropout=0.05`
- 只对 assistant 部分计算 loss
  - 通过 chat template 分别构造 prompt 与 full conversation
  - 将 prompt token 对应 label 置为 `-100`

### 本部分遇到的问题

- 当前第一版数据集是“可运行 smoke test 数据”，不是最终 2k-5k 规模正式集
- 之所以先做小数据，是为了先验证：
  - 模型能否稳定读取结构化证据
  - 训练脚本与 GPU1 能否跑通
  - 输出是否先朝正确方向收敛

### 下一步

- 下载 `Qwen/Qwen3-1.7B`
- 在 `gpu1` 上跑第一轮 LoRA SFT smoke test
- 跑 smoke-eval，对比基座与 LoRA adapter 的输出差异
- 再决定是否把数据扩到更大规模

---

## Part 3：模型下载、基线评估与两轮 smoke training

### 本部分目标

- 拉起 `Qwen/Qwen3-1.7B`
- 先看基座在 BrainDance 风格问答上的原始表现
- 在 `gpu1` 上完成至少一轮真实 LoRA SFT
- 根据 probe case 结果，快速做第二轮数据迭代并复训

### 已完成内容

- 下载并缓存 `Qwen/Qwen3-1.7B`
- 跑通基座 smoke-eval：
  - `ai_engine/finetune_qwen3/logs/base_smoke_eval.log`
- 完成第一轮 LoRA SFT：
  - 输出目录：`ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_smoke`
- 根据第一轮结果扩充数据集到 `220` 条
- 完成第二轮 LoRA SFT：
  - 输出目录：`ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_smoke_v2`
- 跑通第二轮 adapter smoke-eval：
  - `ai_engine/finetune_qwen3/logs/adapter_smoke_eval_v2.log`

### 基座模型基线观察

基座 smoke-eval 结果：

- `recent_hit`
  - 输出：`触控笔桌面采集 01`
  - 问题：过短，只报了一个结果，没有真正利用多条证据
- `no_hit`
  - 输出：`暂无相关记录`
  - 结果：符合预期
- `partial_hit`
  - 输出：`触控笔和冰箱都有记录。`
  - 问题：明显 hallucination，错误地把无证据的 `冰箱` 也答成命中

### 第一轮训练结果

- 数据规模：
  - `47 train / 8 val`
- 训练耗时：
  - 约 `31.6s`
- 验证集损失：
  - `eval_loss = 2.2485`

第一轮结论：

- 训练链路已完整跑通
- 但 probe case 改善不明显
- 原因判断：
  - 数据量太小
  - `partial_coverage` 样本过少
  - 对关键错误模式约束不够密

### 第二轮数据扩充

- 将数据扩到：
  - `198 train / 22 val`
  - 总计 `220` 条
- 重点扩充：
  - `partial_coverage`
  - `must_answer`
  - 同一证据下的多种问法

### 第二轮训练结果

- 训练耗时：
  - 约 `121.5s`
- 验证集损失：
  - `eval_loss = 2.1500`
- 相比第一轮：
  - `2.2485 -> 2.1500`
  - 有小幅改善

### 第二轮 smoke-eval 观察

- `recent_hit`
  - 输出：`{"answer":"最近拍的两张照片分别是触控笔桌面采集 01 和书房场景。"}`
  - 结果：比基座更能利用两条证据，但输出格式开始出现 JSON 化倾向
- `no_hit`
  - 输出：`暂无相关记录`
  - 结果：维持正确
- `partial_hit`
  - 输出：`有记录。`
  - 结果：虽然还不理想，但已经不再像基座那样直接 hallucinate 出“冰箱也有记录”

### 当前判断

- 从“工程可执行性”角度，这次从零微调实践已经成立：
  - 新环境可用
  - GPU1 跑通
  - 数据生成、训练、评估链路都已打通
  - 已产出可复用 LoRA adapter
- 从“业务效果”角度，这仍然只是第一版 smoke result：
  - `partial_coverage` 还不够稳
  - `recent_hit` 的输出格式还需进一步规整

### 本部分遇到的问题

- Qwen3 基座对这类结构化证据问答有明显先验干扰
- 小规模数据很难一次性把“部分命中但不能补全未命中对象”学稳
- 第二轮虽然压住了部分 hallucination，但回答还偏保守
- 说明后续还需要：
  - 更强的 `partial_coverage` 数据
  - 更多“有命中必须答具体内容”的样本
  - 必要时加入 DPO/偏好对

### 当前产物

- 文档记录：
  - `docs/04-本地问答与微调/Qwen3-1.7B-微调实践记录.md`
- 数据集：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl`
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_val.jsonl`
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_sft_manifest.json`
- 训练脚本：
  - `ai_engine/finetune_qwen3/scripts/train_lora_sft.py`
- 推理脚本：
  - `ai_engine/finetune_qwen3/scripts/run_smoke_eval.py`
- 启动脚本：
  - `ai_engine/finetune_qwen3/scripts/run_train_gpu1.sh`
  - `ai_engine/finetune_qwen3/scripts/run_smoke_eval_gpu1.sh`
- 第一轮 adapter：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_smoke`
- 第二轮 adapter：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_smoke_v2`

### 下一步

- 把 `partial_coverage` 和 `must_answer` 再扩一轮到数百到上千条
- 增加更严格的输出格式样本，避免回答 JSON 化
- 再做一轮 SFT
- 只有当 SFT 已经稳定后，再考虑 DPO / ORPO

---

## Part 4：第三轮 SFT 执行计划（最短闭环）

### 当前原则

- 不上 DPO
- 不立刻接回 BrainDance 产品链路
- 先只解决两个核心问题：
  - `partial_coverage` 幻觉
  - 输出 JSON 化

### 第三轮目标

- `partial_coverage` 不再把未命中对象答成命中
- `must_answer` 不再只答“有记录”
- `recent_hit` 输出恢复成自然语言短答，不再出现 `{"answer": ...}`

### 第三轮只做三件事

#### 1. 扩训练数据到 800~1500

重点配比：

- 35% `partial_coverage`
- 30% `must_answer`
- 20% `recent_list/time_qa`
- 15% `no_hit + stability`

重点约束：

- `partial_coverage` 只答证据覆盖部分
- `must_answer` 必须答具体内容
- `anti-json` 样本统一用自然语言短句

#### 2. 先补固定 benchmark

目标规模：

- 60~100 条

分组：

- `recent_hit`
- `no_hit`
- `partial_coverage`
- `must_answer`
- `stability`

重点指标：

- `False-No-Answer Rate`
- `Partial Hallucination Rate`
- `Natural Output Rate`
- `Evidence Utilization Rate`
- `Partial-Hit Precision`

#### 3. 跑第三轮 SFT

保持：

- `Qwen3-1.7B`
- `LoRA rank=8`
- `alpha=16`
- `dropout=0.05`
- 只用 `gpu1`

只做小改：

- system 里明确禁止 JSON 输出
- 增加“错误格式矫正”型样本

### 当前执行顺序

1. 先扩第三轮数据
2. 再补 benchmark
3. 再跑第三轮 SFT
4. 最后只做 debug / 灰度验证，不直接接产品链路

### 当前不做的事情

- 不做 DPO / ORPO
- 不直接扩到 5k 正式集
- 不直接全量接回 BrainDance 问答链路

---

## Part 5：固定 benchmark 落地与评估口径修正

### 本部分目标

- 按第三轮计划先补固定 benchmark
- 让 benchmark 能稳定衡量：
  - `false_no_answer`
  - `partial_hallucination`
  - `partial_hit_precision`
  - `natural_output`
  - `evidence_utilization`
- 避免把“部分命中 + 明确否定未命中对象”的正确回答误判成拒答

### 已完成内容

- 在 `ai_engine/finetune_qwen3/scripts/build_sft_dataset.py` 中生成固定 benchmark：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl`
- benchmark 规模固定为：
  - `80` 条
- 分组固定为：
  - `recent_hit = 16`
  - `no_hit = 16`
  - `partial_coverage = 16`
  - `must_answer = 16`
  - `stability = 16`
- 新增 benchmark 评估脚本：
  - `ai_engine/finetune_qwen3/scripts/evaluate_benchmark.py`
- 在 `gpu1` 上重跑了基座 benchmark：
  - `ai_engine/finetune_qwen3/logs/benchmark_base.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_base.metrics.log`

### 评估口径修正

- 初版 `false_no_answer` 规则过于粗糙
- 问题在于：
  - 只要回答里出现“暂无相关记录”之类短语，就可能被计成拒答
  - 但 `partial_coverage` 的正确回答本来就应该对未命中对象明确说“暂无相关记录”或“未见相关记录”
- 因此修正为：
  - 只有当样本本身 `hit_count > 0`
  - 且回答出现拒答短语
  - 且没有真正利用 evidence
  - 且没有对 supported object 作出正向命中表达
  - 才计为 `false_no_answer`

### 修正后基座 benchmark 结果

- `false_no_answer_rate = 0.0312`
- `partial_hallucination_rate = 0.3125`
- `natural_output_rate = 1.0`
- `evidence_utilization_rate = 0.9688`
- `partial_hit_precision = 0.75`
- `must_answer_specific_rate = 1.0`

### 本部分结论

- benchmark 已经具备“第三轮前后可对比”的最小硬评估能力
- 当前最关键的业务指标仍然是：
  - `partial_hallucination_rate`
  - `partial_hit_precision`
- 基座在 `partial_coverage` 上仍然有明显 hallucination，适合作为第三轮对照基线

---

## Part 6：第三轮 benchmark 与 smoke check

### 本部分目标

- 在 `gpu1` 上完成第三轮 adapter 的固定 benchmark
- 再补一轮 smoke check，确认：
  - `partial_coverage` 幻觉是否明显压住
  - `must_answer` 是否还会只答“有记录”
  - 输出是否还会出现 JSON 化

### 已完成内容

- 第三轮训练已完成：
  - adapter：`ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3`
  - 训练日志：`ai_engine/finetune_qwen3/logs/train_round3.log`
- 在 `gpu1` 上完成第三轮 benchmark：
  - `ai_engine/finetune_qwen3/logs/benchmark_round3.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_round3.metrics.log`
- 在 `gpu1` 上完成第三轮 smoke-eval：
  - `ai_engine/finetune_qwen3/logs/adapter_smoke_eval_round3.log`

### 第三轮训练结果

- 训练集：
  - `900`
- 验证集：
  - `100`
- 估算步数：
  - `113 step / epoch`
- 训练时长：
  - `516.334s`
- 最终验证损失：
  - `eval_loss = 0.5051568150520325`

### 第三轮 benchmark 结果

- `false_no_answer_rate = 0.0`
- `partial_hallucination_rate = 0.0`
- `natural_output_rate = 1.0`
- `evidence_utilization_rate = 1.0`
- `partial_hit_precision = 1.0`
- `must_answer_specific_rate = 1.0`

### 与基座对比

- `false_no_answer_rate`
  - `0.0312 -> 0.0`
- `partial_hallucination_rate`
  - `0.3125 -> 0.0`
- `evidence_utilization_rate`
  - `0.9688 -> 1.0`
- `partial_hit_precision`
  - `0.75 -> 1.0`

### smoke check 结果

- `recent_hit`
  - 输出：`最近拍到的内容包括：触控笔桌面采集 01；书房场景。`
- `no_hit`
  - 输出：`暂无相关记录。`
- `partial_hit`
  - 输出：`最近拍到过触控笔，冰箱暂无相关记录。`

### 本部分结论

- 第三轮已经达成当前最短闭环的三个验收目标：
  - `partial_coverage` 不再把未命中对象答成命中
  - `must_answer` 不再只答“有记录”
  - `recent_hit` 不再输出 JSON
- 当前 round3 adapter 已经比基座更适合作为 BrainDance 本地问答的下一步 debug 候选

### 当前仍保留的细小问题

- 虽然没有 JSON 化，但部分回答仍有一点“半结构化标点”倾向
  - 例如 `包括：...；...`
- 这不影响当前第三轮闭环结论
- 但如果后续接真实链路 debug，可以再额外补几类更严格的纯自然语言 style 样本

---

## Part 7：GPU1 利用率观察与下一轮吞吐建议

### 本部分目标

- 回答“为什么 `gpu1` 看起来没有打满”
- 判断当前训练是否存在明显吞吐浪费
- 给出下一轮更快的批量设置建议

### 当前观察

- 第三轮训练脚本当前配置为：
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 8`
  - `cutoff_len = 1536`
- 这种配置的优点是稳
- 但缺点也很明显：
  - 单步显存占用偏低
  - GPU 利用率更容易波动
  - 总训练时间不一定最优

### 显存探测结果

在同样 `Qwen3-1.7B + LoRA + bf16 + seq_len=1536` 条件下，`gpu1` 显存峰值大致为：

- `batch_size = 1` 时约 `6.47 GB`
- `batch_size = 2` 时约 `9.69 GB`
- `batch_size = 4` 时约 `16.13 GB`
- `batch_size = 8` 时约 `29.00 GB`
- `batch_size = 10` 时约 `35.43 GB`
- `batch_size = 12` 时约 `41.88 GB`

### 当前结论

- 这张 `L20 46GB` 卡对当前任务来说还有明显余量
- 第三轮没继续改 batch，原因不是显存不够，而是：
  - 当前目标是先把最短闭环做完
  - 不再引入新的训练变量，避免影响 round3 可解释性

### 下一轮建议

- 如果进入下一轮 SFT，可以优先尝试：
  - `per_device_train_batch_size = 4`
  - `gradient_accumulation_steps = 2`
- 先保持等效全局 batch 不变，再看：
  - 训练时长是否缩短
  - loss 曲线是否稳定
- 如果仍然稳定，再考虑继续升到：
  - `batch_size = 8`
  - `gradient_accumulation_steps = 1`

### 当前停止点

- 到 2026-03-21 这一步为止，最短闭环已经完成
- 当前不继续开第四轮训练
- 下一步更合理的动作不是立刻继续训，而是：
  - 先把 round3 adapter 接到本地 debug 模式做真实链路对比
  - 如果真实链路再暴露新 failure mode，再决定是否开第四轮

---

## Part 8：round3 接回本地 debug 链路

### 本部分目标

- 不开第四轮训练
- 不改正式产品链路
- 先把 `qwen3_1p7b_lora_sft_round3` 接到本地 debug-only 问答链路
- 只替换最后一步回答生成，保留真实检索链路
- 在真实链路下对比：
  - `off`
  - `base`
  - `lora_round3`
  - `compare`

### 当前约束

- 只使用 `gpu1`
- 继续保持训练时的 structured evidence 输入风格：
  - `question`
  - `retrieval.intent`
  - `retrieval.hit_count`
  - `retrieval.evidence[]`
- 当前阶段不做：
  - `round4`
  - `DPO / ORPO`
  - 正式链路替换

### 路线选择

这一步没有直接把 LoRA 接进 Supabase Edge Function / Deno 运行时，而是先走宿主机侧最短闭环：

- 用 Python 写一个真实链路 debug runner
- 复用当前 BrainDance 检索侧真实能力：
  - DashScope 意图解析
  - DashScope embedding
  - Supabase RPC：`match_memory_poses`
  - `recent/time` 在代码里按 `created_at` 排序
- 只在最后一步切换生成器：
  - `off`
  - `base`
  - `lora_round3`
  - `compare`

这样做的原因是：

- 本机当前没有可直接复现的 Supabase 本地运行环境
- `supabase` CLI 和 `deno` 当前都不在本机可用路径内
- 本地 `127.0.0.1:54321` 未启动
- 直接改 Edge Function 接 GPU1 会增加额外联调变量，不利于当前最短闭环

### 已完成内容

- 新增真实链路 debug runner：
  - `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`
- 新增 `gpu1` 启动脚本：
  - `ai_engine/finetune_qwen3/scripts/run_real_chain_debug_gpu1.sh`
- 固化 Part 8 probe case：
  - `ai_engine/finetune_qwen3/data/real_chain_debug_cases_part8.json`
- 真实链路日志输出位置固定为：
  - `ai_engine/finetune_qwen3/logs/real_chain_debug_cases.jsonl`
  - `ai_engine/finetune_qwen3/logs/real_chain_debug_summary.json`

### 当前脚本能力

`run_real_chain_debug.py` 当前支持：

- 真实检索链路：
  - `recent_capture / time_qa`
  - `object_lookup`
  - `partial_coverage`
- 生成模式切换：
  - `off`
  - `base`
  - `lora_round3`
  - `compare`
- 日志字段包含：
  - `question`
  - `intent`
  - `hit_count`
  - `evidence`
  - `support_map`
  - `base_answer`
  - `lora_answer`
  - `retrieval_latency_sec`
- 新增可配置能力：
  - `--cases_file`
  - `--overwrite_output`
- 追加了运行标识与生成耗时字段：
  - `run_id`
  - `base_generation_latency_sec`
  - `lora_generation_latency_sec`

### 当前固定 probe case

- `recent_hit`
  - `我最近拍了什么？`
- `no_hit`
  - `我最近拍过钢琴吗？`
- `partial_coverage`
  - `我最近拍过笔记本电脑和钢琴吗？`
- `must_answer`
  - `最近拍到过什么笔记本电脑相关画面？`

### 已完成验证

先做了 retrieval-only 验证：

- 运行命令：
  - `bash ai_engine/finetune_qwen3/scripts/run_real_chain_debug_gpu1.sh off --cases_file ai_engine/finetune_qwen3/data/real_chain_debug_cases_part8.json --overwrite_output`
- 最新 retrieval-only 运行：
  - `run_id = real_chain_debug_20260321T044903Z`

retrieval-only 结果说明：

- `recent_hit`
  - `intent = recent_capture`
  - `hit_count = 3`
- `no_hit`
  - `intent = no_hit`
  - `hit_count = 0`
- `partial_coverage`
  - `intent = partial_coverage`
  - `hit_count = 1`
  - `support_map = {"笔记本电脑": true, "钢琴": false}`
- `must_answer`
  - 先后试了：
    - `最近拍到过什么桌面物体？`
    - `最近拍到过什么办公桌上的东西？`
    - `最近和键盘有关的画面有哪些？`
    - `我最近拍过什么办公场景？`
    - `最近拍到过什么笔记本电脑相关画面？`
  - 最终选定：
    - `最近拍到过什么笔记本电脑相关画面？`
  - 原因：
    - 检索结果最稳定地回到真实桌面 / 办公画面
    - 比“办公场景”这类泛问法更适合作为 must-answer probe

### compare 验证

- 运行命令：
  - `bash ai_engine/finetune_qwen3/scripts/run_real_chain_debug_gpu1.sh compare --cases_file ai_engine/finetune_qwen3/data/real_chain_debug_cases_part8.json`
- 最新 compare 运行：
  - `run_id = real_chain_debug_20260321T045448Z`
- 输出文件：
  - `ai_engine/finetune_qwen3/logs/real_chain_debug_cases.jsonl`
  - `ai_engine/finetune_qwen3/logs/real_chain_debug_summary.json`

### compare 结果

- `recent_hit`
  - `base`：`我最近拍了洛天依毛绒玩偶和蓝色地球仪。`
  - `lora_round3`：`最近拍到过洛天依毛绒玩偶，还有蓝色地球仪（带银色金属支架）。`
  - 结论：
    - 两者都为自然语言
    - `lora_round3` 更贴近 evidence，细节更具体

- `no_hit`
  - `base`：`暂无相关记录。`
  - `lora_round3`：`目前没有拍到钢琴的相关记录。`
  - 结论：
    - 两者都稳定拒答
    - `lora_round3` 保持了 no-hit 的自然语言短答

- `partial_coverage`
  - evidence：
    - `support_map = {"笔记本电脑": true, "钢琴": false}`
  - `base`：`我最近拍过HONOR笔记本电脑和AOC显示器。`
  - `lora_round3`：`目前只找到过AOC显示器、HONOR笔记本电脑、定制渐变色机械键盘等记录。`
  - 结论：
    - `lora_round3` 没有把“钢琴” hallucinate 成命中
    - 但这一次真实链路输出里，没有显式补一句“钢琴暂无相关记录”
    - 因此当前判断是：
      - `partial hallucination` 已压住
      - 但“显式否定未命中对象”的风格仍有继续收紧空间

- `must_answer`
  - question：
    - `最近拍到过什么笔记本电脑相关画面？`
  - `base`：`近期拍到AOC显示器、HONOR笔记本电脑、定制渐变色机械键盘、Elaina婚纱手办、《算法导论》中英文版、《高等数学》教材等画面。`
  - `lora_round3`：`3月18日拍到过AOC显示器、HONOR笔记本电脑、定制渐变色机械键盘等。`
  - 结论：
    - `lora_round3` 能具体回答
    - 没有退化成“有记录”
    - 还利用了 `created_at`

### 本部分结论

- round3 已经以本地 debug-only 方式接回真实链路
- 当前 debug 开关形态不是产品代码里的布尔开关，而是宿主机侧 runner 模式切换：
  - `off`
  - `base`
  - `lora_round3`
  - `compare`
- 在当前四组真实链路 case 下：
  - 没有回到 JSON 输出
  - `no_hit` 保持稳定
  - `must_answer` 能具体回答
  - `partial_coverage` 未再出现“钢琴也命中”的 hallucination

### 当前保留问题

- `partial_coverage` 在真实链路里虽然不再 hallucinate
- 但还没有每次都显式说出“未命中对象暂无相关记录”
- 这说明：
  - 离线 benchmark 已绿
  - 但真实链路里仍存在轻微的 style / behavior gap

### 当前停止点

- Part 8 的最短闭环已经完成：
  - `round3 adapter -> 真实检索链路 -> 本地 debug compare`
- 当前不继续开：
  - `round4`
  - `DPO`
  - 正式产品替换
- 下一步如果继续推进，应该优先做：
  - 收集真实链路失败样本
  - 重点补 `partial_coverage` 的“显式否定未命中对象”样本
  - 然后再决定是否开 `round4`

---

## Part 9：真实链路失败样本沉淀（round4 seed）

### 本部分目标

- 不直接开 `round4`
- 先从真实链路里收集“值得回灌”的失败样本
- 只为后续 patch round 做 seed，不扩大训练范围

### 本部分输入

- Part 8 已完成的本地 debug runner：
  - `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`
- 本次新增的 probe case 文件：
  - `ai_engine/finetune_qwen3/data/real_chain_probe_cases_part9.json`

### probe 设计

本次只围绕 3 类 gap 继续做真实链路 compare：

- `partial_missing_negation`
- `must_answer_too_broad`
- `style_not_natural`

probe case 共 `8` 条，覆盖：

- `recent_hit`
  - `这几天我拍了什么？`
- `no_hit`
  - `我最近拍过钢琴吗？`
- `partial_coverage`
  - `我最近拍过笔记本电脑和钢琴吗？`
  - `我最近拍过地球仪和钢琴吗？`
  - `我最近拍过显示器和钢琴吗？`
- `must_answer`
  - `最近拍到过什么笔记本电脑相关画面？`
  - `最近拍到过什么显示器相关画面？`
  - `最近拍到过什么办公桌上的东西？`

### 已完成验证

- 运行命令：
  - `bash ai_engine/finetune_qwen3/scripts/run_real_chain_debug_gpu1.sh compare --cases_file ai_engine/finetune_qwen3/data/real_chain_probe_cases_part9.json --output_file ai_engine/finetune_qwen3/logs/real_chain_probe_part9_cases.jsonl --summary_file ai_engine/finetune_qwen3/logs/real_chain_probe_part9_summary.json --overwrite_output`
- 运行环境：
  - 仅使用 `gpu1`
- 本次 compare 运行：
  - `run_id = real_chain_debug_20260321T052221Z`

### probe 观察结果

- `recent_hit`
  - `lora_round3` 在问法变成“这几天我拍了什么？”时，开始出现：
    - 分号
    - 长描述片段
    - 列举偏散
  - 说明真实链路下仍有 `style_not_natural`

- `partial_coverage`
  - `我最近拍过笔记本电脑和钢琴吗？`
    - `support_map = {"笔记本电脑": true, "钢琴": false}`
    - `lora_round3` 只答了命中部分，没有显式补“钢琴暂无相关记录”
  - `我最近拍过显示器和钢琴吗？`
    - `lora_round3` 输出：
      - `目前只找到显示器相关内容，未见钢琴相关记录。`
    - 这条是符合预期的
  - `我最近拍过地球仪和钢琴吗？`
    - `support_map = {"地球仪": true, "钢琴": false}`
    - 但 `lora_round3` 回答成：
      - `地球仪和钢琴没有相关记录。`
    - 这里暴露出一个比“缺少显式否定”更严重的小问题：
      - 支持对象被错否定

- `must_answer`
  - `最近拍到过什么笔记本电脑相关画面？`
    - `lora_round3` 结果可接受，偏简洁
  - `最近拍到过什么显示器相关画面？`
    - `lora_round3` 结果可接受
  - `最近拍到过什么办公桌上的东西？`
    - `lora_round3` 输出成：
      - `最近拍到过白色婚纱、紫色玫瑰花束、白色办公桌等。`
    - 问题是：
      - 没先概括主目标
      - 直接散列局部物体
      - 不适合作为高质量 must-answer 风格

### seed 文件落地

- 新增 round4 seed failure 文件：
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_seed.jsonl`

字段包含：

- `question`
- `intent`
- `hit_count`
- `evidence`
- `support_map`
- `base_answer`
- `lora_answer`
- `failure_type`
- `target_answer`
- `failure_notes`

### 当前 seed 规模

- `4` 条

按 `failure_type` 分布为：

- `style_not_natural = 1`
- `partial_missing_negation = 1`
- `must_answer_too_broad = 1`
- `partial_false_negative = 1`

### 本部分结论

- 当前最值得进入 patch round4 的，不是再扩大通用数据集
- 而是先把真实链路 failure seed 固化下来
- 目前已经至少拿到 3 个你关心的主类别样本：
  - `partial_missing_negation`
  - `must_answer_too_broad`
  - `style_not_natural`
- 同时还额外暴露出一个更值得注意的真实链路问题：
  - `partial_false_negative`

### 当前停止点

- 到 Part 9 为止，已经完成：
  - 真实链路 probe
  - 失败样本分型
  - round4 seed 文件落地
- 当前仍然不做：
  - 大规模扩数据
  - 直接开正式 `round4`
  - `DPO / ORPO`

### 下一步

- 如果继续推进，最合理的是：
  - 以 `real_chain_failures_round4_seed.jsonl` 为核心
  - 扩成 `50 ~ 150` 条高质量修错样本
  - 再做一个只修行为 gap 的 patch round4

---

## Part 10：round4 patch 数据扩充与小修复训练

### 本部分目标

- 不扩大通用训练集
- 只围绕 `real_chain_failures_round4_seed.jsonl` 做 patch 扩充
- 优先修复：
  - `partial_false_negative`
  - `partial_missing_negation`
  - `must_answer_too_broad`
  - `style_not_natural`
- 以 `round3` adapter 为基础做一轮小型 patch training
- 跑完后立刻回真实链路 compare，不直接替换产品链路

### 已完成内容

- 新增 round4 patch 数据构建脚本：
  - `ai_engine/finetune_qwen3/scripts/build_round4_patch_dataset.py`
- 新增 round4 训练入口：
  - `ai_engine/finetune_qwen3/scripts/run_train_round4_patch_gpu1.sh`
- 更新训练脚本，支持从已有 adapter 继续训练：
  - `ai_engine/finetune_qwen3/scripts/train_lora_sft.py`
  - 新增参数：
    - `--adapter_path`
- 更新 benchmark 评估脚本，补充 patch round4 关注指标：
  - `ai_engine/finetune_qwen3/scripts/evaluate_benchmark.py`
  - 新增指标：
    - `partial_false_negative_rate`
    - `partial_missing_negation_rate`
    - `must_answer_focus_rate`
    - `natural_style_rate`
- 更新真实链路 debug runner 的分析逻辑：
  - `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`
  - 新增：
    - 分句级否定判定
    - must-answer focus 分析
    - summary 中的 `lora_metrics`

### patch 数据结果

- 以 Part 9 的 `4` 条 seed 为核心，扩成 `80` 条 patch 样本
- 实际配比为：
  - `partial_false_negative = 30`
  - `partial_missing_negation = 20`
  - `must_answer_too_broad = 20`
  - `style_not_natural = 10`
- 新增 patch 文件：
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_patch.jsonl`
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_patch_train.jsonl`
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_patch_val.jsonl`
- patch split 结果：
  - `patch_train_count = 72`
  - `patch_val_count = 8`
- 与 round3 主体数据合并后得到：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_round4_train.jsonl`
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_round4_val.jsonl`
- merged 数据规模变为：
  - `train = 972`
  - `val = 108`
- 对应 manifest：
  - `ai_engine/finetune_qwen3/data/braindance_qwen3_round4_manifest.json`

### round4 训练执行

- 训练日期：
  - `2026-03-21`
- 训练方式：
  - 以 `qwen3_1p7b_lora_sft_round3` 为初始化 adapter
  - 保持 LoRA 配置不变
  - 只用 `gpu1`
  - 只跑 `1 epoch`
- 运行命令：
  - `bash ai_engine/finetune_qwen3/scripts/run_train_round4_patch_gpu1.sh ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_patch 1`
- 输出目录：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_patch`
- 训练日志：
  - `ai_engine/finetune_qwen3/logs/train_round4_patch.log`
- 训练结果：
  - `train_runtime = 310.6s`
  - `eval_loss = 0.3859`

### 离线 benchmark 对比

为避免口径漂移，这里使用了更新后的 benchmark 脚本分别重跑：

- round3：
  - `ai_engine/finetune_qwen3/logs/benchmark_round3_part10.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_round3_part10.metrics.log`
- round4 patch：
  - `ai_engine/finetune_qwen3/logs/benchmark_round4_patch.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_round4_patch.metrics.log`

关键指标对比：

- `natural_style_rate`
  - `0.8375 -> 0.9000`
- `must_answer_focus_rate`
  - `0.6250 -> 0.8750`
- `partial_false_negative_rate`
  - `0.0000 -> 0.0625`
- `partial_missing_negation_rate`
  - `0.0000 -> 0.0000`
- `partial_hallucination_rate`
  - `0.0000 -> 0.0625`

当前离线 benchmark 结论：

- 风格和 must-answer focus 明显变好
- 但 `partial_coverage` 出现了一个新的回归点：
  - `partial_coverage_006`
  - 输出：
    - `目前只找到写字台相关内容，没有看到椅子。`
  - 正确支持对象本应是：
    - `椅子`
  - 说明 patch round4 虽然压住了真实链路上的 `地球仪 -> 整句全否`
  - 但在固定 benchmark 上引入了一条“支持/未支持对象反转”的小回归

### 真实链路 compare 对比

本部分没有只看主观样例，而是对 Part 9 probe 做了前后对比。

为了保证前后口径一致，补充生成了两份重打分 summary：

- round3 baseline：
  - `ai_engine/finetune_qwen3/logs/real_chain_probe_part10_round3_rescored_summary.json`
- round4 patch：
  - `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_patch_rescored_summary.json`

同时保留了 round4 的真实链路原始输出：

- `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_patch_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_patch_summary.json`

真实链路关键指标对比：

- `natural_style_rate`
  - `0.7500 -> 0.8750`
- `partial_false_negative_rate`
  - `0.3333 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.3333 -> 0.3333`
- `must_answer_focus_rate`
  - `0.3333 -> 0.6667`

逐项观察：

- `style_not_natural`
  - round3：
    - `最近拍到过洛天依毛绒玩偶、蓝色展台底座等；最近拍到过白色多层书架角落的特写……`
  - round4：
    - `最近拍到过洛天依毛绒玩偶，还有白色多层书架角落的特写。`
  - 结论：
    - 分号和散列感明显减轻

- `partial_false_negative`
  - round3：
    - `地球仪和钢琴没有相关记录。`
  - round4：
    - `目前只找到地球仪相关内容，未见钢琴。`
  - 结论：
    - 最关键问题已在真实链路下被修正

- `must_answer_too_broad`
  - round3：
    - `最近拍到过白色婚纱、紫色玫瑰花束、白色办公桌等。`
  - round4：
    - `最近拍到过办公桌上银发蓝眼的Elaina手办，还有紫色玫瑰花束。`
  - 结论：
    - 已能先围绕主目标组织答案

### 当前残留问题

- `partial_missing_negation` 还没完全收敛
  - 真实链路中的：
    - `我最近拍过笔记本电脑和钢琴吗？`
  - round4 仍然输出：
    - `目前只找到过AOC显示器、HONOR笔记本电脑、机械键盘等。`
  - 仍缺一句：
    - `钢琴暂无相关记录`

- `must_answer` 的多命中回答仍有半结构化倾向
  - 例如：
    - `3月18日拍到过AOC显示器、HONOR笔记本电脑；3月4日拍到过联想笔记本电脑。`
  - 这条在真实链路里仍被判成 `natural_style = false`

- 固定 benchmark 上出现一条新的 `partial_coverage` 小回归
  - 说明这一轮 patch 已经有效
  - 但还不适合直接当成“彻底收敛”

### 本部分结论

- 这轮 round4 patch 是有效的
- 最重要的真实链路目标已经达到：
  - `partial_false_negative` 从 `0.3333` 压到 `0`
- `style_not_natural` 和 `must_answer_too_broad` 也都有改善
- 但 `partial_missing_negation` 还没压下去
- 同时固定 benchmark 出现了轻微回归

### 下一步

- 不直接替换正式产品链路
- 继续补一小轮更窄的 patch 样本，重点只打两类：
  - `partial_missing_negation`
  - `multi-hit must_answer` 的去分号、去半结构化列举
- 可以再补 `10 ~ 20` 条办公桌 / 笔记本电脑 / 显示器相关 patch
- 然后基于 `qwen3_1p7b_lora_sft_round4_patch` 再跑 `0.5 ~ 1` epoch 小修复

## Part 11：round4.1 超小修复轮

### 本部分目标

- 不扩大通用训练集
- 只修两类行为 gap：
  - `partial_missing_negation`
  - `multi-hit must_answer` 的去分号、去半结构化列举
- 继续基于 `qwen3_1p7b_lora_sft_round4_patch` 做超小 patch
- 同时回看 fixed benchmark 上那条 `partial_coverage` 回归是否消失

### 数据落地

- 新增 round4.1 patch 构建脚本：
  - `ai_engine/finetune_qwen3/scripts/build_round4_1_patch_dataset.py`
- 新增 round4.1 训练入口：
  - `ai_engine/finetune_qwen3/scripts/run_train_round4_1_patch_gpu1.sh`
- 新增超小 patch 文件：
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch.jsonl`
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch_train.jsonl`
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch_val.jsonl`
  - `ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch_manifest.json`

本轮新增 patch 共 `20` 条：

- `partial_missing_negation = 12`
- `multi_hit_must_answer_style = 8`

具体来源拆分为：

- `round4_seed_partial_missing_negation = 6`
- `study_room_partial_guard = 4`
- `round4_seed_partial_false_negative_guard = 2`
- `real_chain_multi_hit_notebook = 8`

其中：

- `study_room_partial_guard` 用于补一小层“命中对象先答命中，再补否定”的稳定性保护
- `partial_false_negative_guard` 虽然仍归在 `partial_missing_negation` 思路下，但本质上是防止 `地球仪 -> 整句全否` 反弹
- `real_chain_multi_hit_notebook` 直接围绕真实链路里的 `笔记本电脑` 多命中半结构化回答问题来做

另外，为了避免“只训新增 20 条”导致 round4 已压住的行为遗忘，本轮脚本额外生成了 patch-only combined 文件：

- `ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch_plus_round4_train.jsonl`
- `ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch_plus_round4_val.jsonl`

combined 后的数据规模为：

- `train = 88`
- `val = 12`

### benchmark 观察口径补充

- fixed benchmark 的 `partial_coverage_006` 存在一个标签 caveat：
  - evidence 里确实出现了 `写字台`
  - 但 benchmark metadata 把它标成了 unsupported
- 本轮没有直接改 benchmark 文件
- 仍然把这条保留为离线稳定性观察点，目的是避免 round4 的“支持/未支持对象反转”继续扩散

### round4.1 第一次尝试：只训新增 20 条 patch

训练方式：

- 初始化 adapter：
  - `qwen3_1p7b_lora_sft_round4_patch`
- 数据：
  - 只用新增的 `20` 条 round4.1 patch
- 训练轮数：
  - `0.75 epoch`
- 输出目录：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch`
- 训练日志：
  - `ai_engine/finetune_qwen3/logs/train_round4_1_patch.log`
- 训练结果：
  - `eval_loss = 0.6586`

离线 benchmark：

- 输出：
  - `ai_engine/finetune_qwen3/logs/benchmark_round4_1_patch.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_round4_1_patch.metrics.log`
- 关键指标：
  - `natural_style_rate = 0.8250`
  - `must_answer_focus_rate = 0.7500`
  - `partial_false_negative_rate = 0.0000`
  - `partial_missing_negation_rate = 0.0000`

真实链路 probe：

- 输出：
  - `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_1_patch_cases.jsonl`
  - `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_1_patch_summary.json`
- 关键指标：
  - `natural_style_rate = 0.7500`
  - `partial_false_negative_rate = 0.3333`
  - `partial_missing_negation_rate = 0.0000`
  - `must_answer_focus_rate = 0.6667`

这次试跑的结论很明确：

- `partial_missing_negation` 的确被压到了 `0`
- 但 `partial_false_negative` 在真实链路里反弹回了 `0.3333`
- 说明“只拿新增 20 条单独接训”过窄，会把 round4 已有的 guard 行为冲掉
- 因此这版 adapter 不保留为 round4.1 最终候选

### round4.1 最终版：round4 patch + round4.1 patch 的 patch-only combined 续训

训练方式：

- 初始化 adapter：
  - `qwen3_1p7b_lora_sft_round4_patch`
- 数据：
  - `round4 patch + round4.1 patch`
  - 仅使用 patch-only combined 数据，不回混主训练集
- 训练轮数：
  - `0.5 epoch`
- 输出目录：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`
- 训练日志：
  - `ai_engine/finetune_qwen3/logs/train_round4_1_patch_mixed.log`
- 训练结果：
  - `eval_loss = 1.2365`

离线 benchmark：

- 输出：
  - `ai_engine/finetune_qwen3/logs/benchmark_round4_1_patch_mixed.json`
  - `ai_engine/finetune_qwen3/logs/benchmark_round4_1_patch_mixed.metrics.log`

相对 round4 patch 的关键指标变化：

- `natural_style_rate`
  - `0.9000 -> 0.8500`
- `must_answer_focus_rate`
  - `0.8750 -> 0.8125`
- `partial_false_negative_rate`
  - `0.0625 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.0000 -> 0.0000`
- `partial_hallucination_rate`
  - `0.0625 -> 0.0000`

固定 benchmark 上那条回归点 `partial_coverage_006` 的结果：

- round4 patch：
  - `目前只找到写字台相关内容，没有看到椅子。`
- round4.1 mixed：
  - `目前只找到椅子相关内容，没有写字台。`

说明：

- 这条 round4 引入的“支持 / 未支持对象反转”回归已经消失
- 同时没有新增新的 `partial_false_negative` 或 `partial_missing_negation`

真实链路 probe：

- 输出：
  - `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_1_patch_mixed_cases.jsonl`
  - `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_1_patch_mixed_summary.json`

相对 round4 patch 的关键指标变化：

- `natural_style_rate`
  - `0.8750 -> 1.0000`
- `partial_false_negative_rate`
  - `0.0000 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.3333 -> 0.0000`
- `must_answer_focus_rate`
  - `0.6667 -> 0.6667`

关键 case 观察：

- `我最近拍过笔记本电脑和钢琴吗？`
  - round4 patch：
    - `目前只找到过AOC显示器、HONOR笔记本电脑、机械键盘等。`
  - round4.1 mixed：
    - `目前只找到过笔记本电脑，没有钢琴相关记录。`
  - 结论：
    - `partial_missing_negation` 已补齐

- `我最近拍过地球仪和钢琴吗？`
  - round4 patch：
    - `目前只找到地球仪相关内容，未见钢琴。`
  - round4.1 mixed：
    - `地球仪相关内容已收录，钢琴未见相关记录。`
  - 结论：
    - `partial_false_negative` 没有反弹

- `最近拍到过什么笔记本电脑相关画面？`
  - round4 patch：
    - `3月18日拍到过AOC显示器、HONOR笔记本电脑；3月4日拍到过联想笔记本电脑。`
  - round4.1 mixed：
    - `3月18日拍到过HONOR笔记本电脑，3月4日拍到过联想笔记本电脑。`
  - 结论：
    - 多命中回答已经去掉分号和半结构化散列感

### 当前残留问题

- fixed benchmark 上的离线风格指标仍有轻微回撤
  - `natural_style_rate = 0.85`
  - 仍低于 round4 patch 的 `0.90`
- 离线回撤主要集中在非本轮目标组：
  - `recent_hit`
  - `stability`
  - 部分旧的 `must_answer` 长描述 case
- 真实链路里的 `must_answer_focus_rate` 没继续上升
  - 仍是 `0.6667`
  - 当前主要卡在 `最近拍到过什么办公桌上的东西？`
  - 输出虽然自然，但 focus term 仍不够稳定地落在 `Elaina 手办`

### 本部分结论

- round4.1 的 patch-only 第一版不够稳，已明确否决
- round4.1 的 mixed 版是本轮更合理的结果：
  - `partial_missing_negation` 在真实链路下降到 `0`
  - `multi-hit must_answer` 的分号 / 半结构化问题明显改善
  - round4 最关键成果 `partial_false_negative = 0` 没有丢
  - fixed benchmark 上 round4 新增的 `partial_coverage` 回归也已消失
- 但 fixed benchmark 的整体风格指标没有比 round4 patch 更好

### 下一步建议

- 暂不继续扩大训练轮次
- 进入 `debug-only` 观察阶段更合适
- 如果后续真实使用里没有暴露新的稳定 failure mode：
  - 先停训练
  - 继续拿 `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 做本地 compare / debug
- 如果后续要推进到更正式的产品开关：
  - 再单独处理 `recent_hit / stability` 的离线风格回撤
  - 但那已经是 round4.2 以后的问题，不属于这轮超小修复的范围

## Part 12：debug-only 观察期

### 本部分目标

- 暂停继续训练
- 以 `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 作为当前 best debug adapter
- 扩大真实链路 compare 样本面
- 观察是否存在新的稳定 failure mode

### 当前结论

- 现在先不继续训练
- 当前最合理的 debug 候选已经明确：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`
- 这版 adapter 在真实链路上，比 `qwen3_1p7b_lora_sft_round4_patch` 更符合本轮目标
- fixed benchmark 虽然仍有轻微风格回撤，但没有破坏关键能力
- 因此当前主线从“继续训一轮”切换为“真实链路 debug-only 观察”

### 当前 best 版本与保留对照

当前 best debug adapter：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`

作为基线保留的对照版本：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3`
- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_patch`

保留这三版的目的很直接：

- 后续任何真实链路回归，都可以明确和哪一版对比
- 可以快速判断问题是 round4 就存在，还是 round4.1 mixed 新引入

### 进入观察期的依据

真实链路相对 round4 patch 的关键变化：

- `natural_style_rate`
  - `0.8750 -> 1.0000`
- `partial_false_negative_rate`
  - `0.0000 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.3333 -> 0.0000`
- `must_answer_focus_rate`
  - `0.6667 -> 0.6667`

fixed benchmark 相对 round4 patch 的关键变化：

- `natural_style_rate`
  - `0.9000 -> 0.8500`
- `must_answer_focus_rate`
  - `0.8750 -> 0.8125`
- `partial_false_negative_rate`
  - `0.0625 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.0000 -> 0.0000`
- `partial_hallucination_rate`
  - `0.0625 -> 0.0000`

当前判断是：

- round4.1 mixed 已经达到一个更合理的平衡点
- 它修掉了本轮最重要的真实链路 gap
- 离线的小幅回撤主要集中在非本轮目标组
- 因此下一步最有价值的动作不是 round4.2，而是扩大真实链路观察面

### 当前原则

- 不开 `round4.2`
- 不做 `DPO / ORPO`
- 不直接替换正式产品链路
- 优先以真实链路表现决定后续是否继续训练
- 零星不完美先接受，不为轻微离线分数回撤继续开新轮次

### 观察期执行方案

#### 1. 扩大真实链路 compare 样本面

- 不再只依赖固定 `8` 条 probe
- 扩到 `20 ~ 40` 条真实 debug 问题
- 仍按 failure type 分组观察，不做无结构混测

推荐分组：

- `recent_hit`
- `no_hit`
- `partial_coverage`
- `must_answer`
- `multi_hit_must_answer`
- `stability`

本轮优先补两类真实问法：

- 口语化 recent
  - 例如：
    - `这两天我都拍了啥？`
    - `最近又扫到什么了？`
    - `最近有哪些桌面上的东西？`
- 多目标 partial
  - 例如：
    - `我最近拍过笔记本电脑、地球仪和钢琴吗？`
    - `最近拍到过显示器和小提琴吗？`

重点不是继续证明二元 partial 已修，而是看三元组合和口语化问法会不会暴露新 failure mode。

#### 2. 日志里增加 failure triage 字段

建议在真实链路 compare 日志中增加人工回填字段：

- `triage_label`

推荐枚举：

- `ok`
- `style_minor`
- `focus_minor`
- `partial_regression`
- `retrieval_issue`
- `new_failure_mode`

增加这个字段的目的是把问题快速分层：

- 是模型行为还要继续修
- 还是检索侧证据本身有问题
- 还是只是风格轻微瑕疵，不值得继续训练

#### 3. 设定停止训练阈值

在以下情况出现前，不再继续训练：

- 真实链路出现新的稳定复现 failure mode
- 同一类 failure 在 `20 ~ 40` 条 debug 问题里多次出现
- 问题已经影响是否推进产品开关

换句话说：

- 如果只是零星不完美，先接受
- 如果只是固定 benchmark 上的轻微风格波动，先不训
- 只有真实链路里出现稳定、可复现、会影响产品判断的问题，才值得开 round4.2

### 当前不做的事

- 不立刻开 `round4.2`
- 不因为 fixed benchmark 上 `natural_style_rate` 的轻微回撤推翻 round4.1 mixed
- 不把当前问题升级成偏好优化问题，因此暂不考虑 `DPO / ORPO`

原因是：

- 当前最关键的 gap 已经不是“大偏差”
- 而是少量行为边界和真实链路覆盖不足
- 在这个阶段继续训练，过度优化的风险已经高于预期收益

### 什么时候才值得开 round4.2

只有满足以下任一条件，才值得继续训：

- 条件 A：真实链路出现新稳定问题
  - 例如三元 partial 组合反转、口语化 recent 明显退化、多命中 must-answer 又回到半结构化
- 条件 B：准备推进到更正式的产品开关
  - 这时才值得专门修 `recent_hit`、`stability` 和更广泛的 style 一致性
- 条件 C：真实使用证明某类 focus 问法非常重要
  - 当前最典型的是：
    - `最近拍到过什么办公桌上的东西？`
  - 如果这类问法后续频繁出现，再考虑把它单独做成 round4.2 目标

### 当前验收标准

- `partial_false_negative` 不反弹
- `partial_missing_negation` 保持为 `0`
- `multi-hit must_answer` 保持自然语言表达
- 如果 `20 ~ 40` 条真实 debug 问题中没有新的稳定 failure mode，则继续停训，进入更长期的 debug 使用

### 本部分结论

- 训练主线先暂停
- `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 作为当前 best debug adapter 保留
- 下一阶段重点不是继续做 round4.2，而是扩大真实链路 compare 样本面
- 只有在观察期里发现新的稳定 failure mode，或者产品开关需要更高一致性时，才值得继续开新训练轮次

### 本部分实际落地

#### 1. 新增 Part 12 真实 debug 题集

新增题集文件：

- `ai_engine/finetune_qwen3/data/real_chain_debug_cases_part12.json`

本轮实际落地为 `26` 条真实 debug 问题，分组如下：

- `recent_hit = 4`
- `no_hit = 4`
- `partial_coverage = 8`
- `must_answer = 4`
- `multi_hit_must_answer = 3`
- `stability = 3`

相比 Part 9 的 `8` 条 probe，这一轮覆盖面明显扩大，并且补进了：

- 口语化 recent
  - 例如：
    - `这两天我都拍了啥？`
    - `最近又扫到什么了？`
    - `最近有哪些桌面上的东西？`
- 三元 partial
  - 例如：
    - `我最近拍过笔记本电脑、地球仪和钢琴吗？`
    - `最近拍到过显示器、笔记本电脑和小提琴吗？`
- multi-hit must-answer
  - 例如：
    - `最近拍到过什么手办相关画面？`
    - `最近拍到过什么书架相关画面？`
    - `最近拍到过什么书籍相关内容？`

#### 2. 扩充 real-chain debug 日志字段与分组统计

更新脚本：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增或增强的能力：

- 支持 case 级唯一标识：
  - `case_id`
- 支持人工回填字段：
  - `triage_label`
- 将 `multi_hit_must_answer` 纳入 must-answer 统计
- summary 中新增：
  - `group_counts`
  - `triage_counts`
  - `metrics_by_group`

这样做的目的很直接：

- 后续可以按 case 级别持续补人工 triage
- 也可以直接看各组而不是只看 overall summary

#### 3. 新增 Part 12 一键 compare 入口

新增脚本：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_part12_compare_gpu1.sh`

这个脚本会顺序跑三版 adapter：

- `qwen3_1p7b_lora_sft_round3`
- `qwen3_1p7b_lora_sft_round4_patch`
- `qwen3_1p7b_lora_sft_round4_1_patch_mixed`

统一使用：

- `ai_engine/finetune_qwen3/data/real_chain_debug_cases_part12.json`

对应输出文件：

- `ai_engine/finetune_qwen3/logs/real_chain_part12_round3_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round3_summary.json`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_patch_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_patch_summary.json`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_1_patch_mixed_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_1_patch_mixed_summary.json`

#### 4. 观察期脚本稳定性修复

在实际跑 Part 12 compare 时，暴露了两个非模型层问题：

- `model_assets.created_at` 存在格式不完全规整的时间字符串
  - 会导致 `parse_datetime()` 在 lexical fallback 过滤时中断整轮评估
- DashScope embedding 接口存在偶发 `SSL EOF` 抖动
  - 会让整轮 debug 因单次网络请求失败而终止

因此在 `run_real_chain_debug.py` 里额外补了两类稳定性修复：

- 更稳健的时间解析与过滤
- 对 DashScope / Supabase 请求增加轻量重试

这两项修复不改变模型行为，但能显著提升真实链路批量观察的稳定性。

### Part 12 实际运行结果

#### 1. 三版 adapter 的真实链路整体对比

在同一份 `26` 条 Part 12 题集上，三版 adapter 的关键指标如下：

- round3
  - `natural_style_rate = 0.7308`
  - `partial_false_negative_rate = 0.1250`
  - `partial_missing_negation_rate = 0.3750`
  - `must_answer_focus_rate = 0.2857`

- round4 patch
  - `natural_style_rate = 0.7692`
  - `partial_false_negative_rate = 0.1250`
  - `partial_missing_negation_rate = 0.2500`
  - `must_answer_focus_rate = 0.4286`

- round4.1 patch mixed
  - `natural_style_rate = 0.8462`
  - `partial_false_negative_rate = 0.0000`
  - `partial_missing_negation_rate = 0.0000`
  - `must_answer_focus_rate = 0.5714`

从这轮更大样本面的真实链路结果看：

- `round4.1 mixed` 仍然是当前最合理的 debug adapter
- `partial_false_negative` 和 `partial_missing_negation` 都没有反弹
- `must_answer_focus_rate` 虽然仍未完全理想，但已经继续优于 `round3 / round4 patch`

#### 2. 分组结果

`round4.1 mixed` 的 group-level 指标如下：

- `recent_hit`
  - `natural_style_rate = 0.5000`
- `no_hit`
  - `natural_style_rate = 1.0000`
- `partial_coverage`
  - `natural_style_rate = 0.8750`
  - `partial_false_negative_rate = 0.0000`
  - `partial_missing_negation_rate = 0.0000`
- `must_answer`
  - `natural_style_rate = 0.7500`
  - `must_answer_focus_rate = 0.5000`
- `multi_hit_must_answer`
  - `natural_style_rate = 1.0000`
  - `must_answer_focus_rate = 0.6667`
- `stability`
  - `natural_style_rate = 1.0000`

这组结果说明：

- Part 12 新增的三元 partial 没有打出新的 stable partial regression
- `stability` 组整体是稳的
- 当前残留问题主要集中在：
  - `recent_hit`
  - 少量 `must_answer / multi_hit_must_answer`

#### 3. 当前残留 failure triage

`round4.1 mixed` 在 `26` 条题集里，没有出现新的 stable `partial_regression`。
当前可见的残留问题共 `6` 条，主要都是轻量问题：

- `triage_counts`
  - `ok = 20`
  - `style_minor = 3`
  - `focus_minor = 3`

- `style_minor`
  - `part12_recent_002`
    - `这两天我都拍了啥？`
  - `part12_recent_004`
    - `最近有哪些桌面上的东西？`
  - `part12_partial_006`
    - `最近拍到过显示器、笔记本电脑和小提琴吗？`

- `focus_minor`
  - `part12_must_003`
    - `最近拍到过什么办公桌上的东西？`
  - `part12_must_004`
    - `最近拍到过什么地球仪相关画面？`
  - `part12_multi_002`
    - `最近拍到过什么书架相关画面？`

当前最值得记住的三个 observation：

- `最近拍到过什么办公桌上的东西？`
  - `round4.1 mixed` 已经比 round3 更聚焦
  - 但 focus term 还不够稳定地落在 `Elaina手办`

- `最近拍到过什么地球仪相关画面？`
  - 三版都存在“答到地球仪，但顺带拉出过多周边上下文”的问题
  - 这更像旧的 focus/style 边界问题，不是 round4.1 mixed 新引入

- `最近拍到过什么书架相关画面？`
  - 当前回答会偏向书架上的具体物品
  - 但不够稳定地直接点出 `书架`
  - 这一类更适合先记为 `focus_minor` 观察，而不是立刻开训练轮次

### 本部分最终判断

- Part 12 已经按计划完成：
  - 题集扩到 `26` 条
  - 三版 real-chain compare 已跑通
  - `triage_label` 字段已补到日志结构
  - 观察期脚本稳定性问题也已顺手修掉

- 当前没有看到足以立即开启 `round4.2` 的新稳定 failure mode

- `round4.1 mixed` 在更大真实样本面上继续保持：
  - `partial_false_negative = 0`
  - `partial_missing_negation = 0`

- 当前残留问题仍然主要是：
  - `recent_hit` 的风格波动
  - 少量 `must_answer / multi_hit_must_answer` 的 focus 不够稳

因此：

- 继续停训
- 继续拿 `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 做真实链路 debug-only 观察
- 只有在后续继续积累 case 时，确认这些 focus/style 问题已经形成稳定复现模式，才值得进入 `round4.2`

---

## Part 13：交互式 Debug 程序与 real-chain 检索兜底修复（2026-03-21）

### 本部分目标

- 做一个真正可交互的本地 debug 程序
- 让人工用户可以直接提问、立即看到回答、并把每轮问答记录下来
- 自己复现实例问题，确认当前 real-chain 是“没联网”还是“召回策略有缺陷”
- 修复 `object_lookup` 类常见问法在真实数据上 `hit_count=0` 的问题

### 发现的问题

在第一次交互式试跑中，用户输入：

- `我有没有生成什么模型`
- `帮我找一下洛天依模型`

程序都返回：

- `intent = no_hit`
- `hit_count = 0`
- `evidence = []`

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_001.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_001.summary.json`

### 首轮排查结论

不是 RAG 完全没连上，而是“检索链路联通正常，但当前 object 类召回策略不稳”。

已经确认的事实：

- DashScope key 能正常读取
- Supabase REST `model_assets` 能正常返回数据
- 远端库里确实存在“洛天依”相关记录

例如通过直接请求 `model_assets` 可见：

- `scene_20260319_893174`
- `scene_20260319_893174_3dgs`
- `test_scene_sharp_1768839315`

这些记录的 `description / objects / tags` 中都明确包含：

- `洛天依`
- `洛天依毛绒玩偶`
- `洛天依手办`

### 根因定位

根因不是单点，而是两个问题叠加：

#### 1. 交互启动器使用 `conda run`

最初交互脚本通过：

- `conda run -n qwen3_ft python ...`

启动。

这导致 `input()` 在某些终端环境下直接读到 EOF，表现为：

- 刚打印 `你>` 就自动退出

这不是模型问题，而是交互 stdin 被启动器吞掉。

#### 2. `object_lookup` 路径过度依赖向量 RPC，且后置过滤过严

在 `run_real_chain_debug.py` 里，`object_lookup` 的主路径是：

- DashScope 解析意图
- DashScope embedding
- Supabase RPC：`match_memory_poses`
- 再用目标词做字符串过滤

但实际排查发现：

- `match_memory_poses` 对 `洛天依`、`洛天依模型`、`模型` 在默认 `match_threshold = 0.5` 下都返回 `0` 条
- 即使把阈值降到 `0.1`，也主要召回无关项
- 对 `洛天依模型` 这种问法，解析器会给出：
  - `search_text = "洛天依模型"`
  - `target_objects = ["洛天依模型"]`
- 后续过滤要求结果里直接出现 `洛天依模型` 这个完整短语
- 但真实数据里通常是：
  - `洛天依`
  - `洛天依毛绒玩偶`
  - `洛天依手办`

所以会出现：

- 真实库里明明有相关记录
- 向量召回不稳定
- 词面过滤又太死
- 最终 `raw_rows` 被压成空数组

### 本次修复内容

#### 1. 新增交互式 debug 程序

新增文件：

- `ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py`

功能：

- 真实检索链路
- 本地 LoRA 生成
- 单轮即时回答
- 可选显示 evidence
- 每轮可输入人工反馈
- 自动写 JSONL 日志与 summary

新增启动脚本：

- `ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh`
- `ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu1.sh`

#### 2. 修复交互 stdin 问题

把交互启动脚本从：

- `conda run -n qwen3_ft python ...`

改成：

- 初始化 conda shell hook
- `conda activate qwen3_ft`
- `exec python ...`

这样 stdin 会直接交给 Python 进程，交互式 `input()` 不再提前读到 EOF。

#### 3. 给 `object_lookup / partial_coverage` 增加词面兜底检索

在 `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py` 中新增：

- `normalize_lookup_terms()`
- `row_matches_lookup_terms()`
- `lexical_fallback_model_assets()`

修复策略：

- 对 `洛天依模型` 这类词做去泛化清洗
  - 例如去掉后缀：
    - `模型`
    - `场景`
    - `内容`
    - `记录`
    - `画面`
- 当向量 RPC 返回空结果，或返回结果被目标词过滤为空时：
  - 回退到 `model_assets` 做词面匹配
- 在 `partial_coverage` 中，如果单个 target 没通过向量召回：
  - 也允许走词面兜底补一个 matched row

#### 4. 给“模型清单类问法”增加窄范围 special-case

继续排查发现，下面这类问题：

- `我有没有生成什么模型`

在当前意图解析中可能被解析成：

- `question_type = object_lookup`
- `search_text = ""`
- `target_objects = []`

这时既不会命中实体召回，也不会进入 recent list 路径。

因此新增：

- `is_model_inventory_query()`
- `build_model_inventory_answer()`

处理逻辑：

- 如果原始问题本质上是在问“最近生成了哪些模型”
- 则直接把检索切到 `model_assets` 最近记录
- 并用一个 deterministic 的短句组织回答

这样做的原因很直接：

- 这类问法更像“资产清单查询”
- 不是 LoRA 当前主要训练的“证据问答”类型
- 直接交给模型自由发挥，容易回到“没有找到生成模型内容”这类歧义输出

### 自测与验证

#### 1. 代表性问题集

这次实际自测的问题集为：

- `我有没有生成什么模型`
- `帮我找一下洛天依模型`
- `找一下洛天依`
- `我最近拍过钢琴吗？`
- `我最近拍过显示器和钢琴吗？`
- `我最近拍了什么？`

临时 cases 文件：

- `/tmp/manual_debug_cases_stage2.json`

运行命令：

```bash
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
eval "$(conda shell.bash hook)"
conda activate qwen3_ft

python ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py \
  --mode lora_round3 \
  --adapter_path ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed \
  --cases_file /tmp/manual_debug_cases_stage2.json \
  --output_file ai_engine/finetune_qwen3/logs/manual_debug_stage2_cases.jsonl \
  --summary_file ai_engine/finetune_qwen3/logs/manual_debug_stage2_summary.json \
  --overwrite_output
```

输出结果：

- `ai_engine/finetune_qwen3/logs/manual_debug_stage2_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/manual_debug_stage2_summary.json`

#### 2. 修复后的关键结果

修复后：

- `我有没有生成什么模型`
  - `hit_count = 3`
  - 回答：`最近生成过这些模型：scene_20260319_893174、scene_20260319_893174_3dgs、scene_20260318_893381。`

- `帮我找一下洛天依模型`
  - `hit_count = 3`
  - 回答：`目前找到的主要是洛天依毛绒玩偶、洛天依手办等。`

- `找一下洛天依`
  - `hit_count = 3`
  - 回答：`最近和洛天依相关的主要是商业展台装置，还有动漫手办特写照片。`

- `我最近拍过钢琴吗？`
  - `hit_count = 0`
  - 回答仍保持正确 no-hit

- `我最近拍过显示器和钢琴吗？`
  - `hit_count = 1`
  - 回答：`目前只找到显示器相关内容，没有钢琴相关记录。`

- `我最近拍了什么？`
  - `hit_count = 3`
  - 回答：`最近拍到过洛天依毛绒玩偶，还有蓝色地球仪（带银色金属支架）。`

这说明：

- 当前问题不是 LoRA 生成层坏掉
- 而是检索前后处理需要更稳健的兜底
- 修复后，常见实体类 object 查询已经恢复到可用状态

#### 3. 交互脚本本身验证

为确认交互程序路径也正常，额外做了一次自动输入验证：

```bash
printf '帮我找一下洛天依模型\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name auto_verify_luotianyi
```

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_luotianyi.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_luotianyi.summary.json`

结果确认：

- 交互程序不再自动 EOF 退出
- `hit_count = 3`
- evidence 中正确出现：
  - `scene_20260319_893174`
  - `scene_20260319_893174_3dgs`
  - `test_scene_sharp_1768839315`

对原始失败问法也追加了一次自动验证：

```bash
printf '我有没有生成什么模型\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name auto_verify_model_inventory
```

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_model_inventory.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_model_inventory.summary.json`

结果确认：

- `intent = recent_capture`
- `hit_count = 3`
- 交互脚本直接输出：
  - `最近生成过这些模型：scene_20260319_893174、scene_20260319_893174_3dgs、scene_20260318_893381。`

### 当前结论

- 交互式 debug 程序已经可用
- 当前 real-chain 并不是“没联网”
- 真正的问题是：
  - 向量 RPC 对部分常见实体词召回不稳定
  - `object_lookup` 的后置过滤过于字面化
- “模型清单类问法”不适合完全依赖当前 LoRA 自由生成
- 通过给 `model_assets` 增加词面兜底后：
  - `洛天依模型` 这类常见用户问法已经恢复正常
- 通过增加窄范围 inventory special-case 后：
  - `我有没有生成什么模型` 这类原始失败问法也已恢复到可用状态

### 下一步建议

- 继续用交互程序积累 `20 ~ 40` 条真实用户问答
- 在日志中重点观察以下类型：
  - 二次元实体名
  - 手办 / 展台 / 摆件 / 地球仪 / 显示器这类具体物品
  - `partial_coverage`
  - recent 口语化问法
- 如果后续暴露更多“向量召回弱、词面兜底才救回来”的 case：
  - 再考虑专门优化检索层，而不是直接继续训 LoRA

## Part 14：检索路由可观测性补强（2026-03-21）

### 本部分目标

- 把 Part 13 里已经出现的重要检索分支显式记录到日志
- 区分“这是模型回答的，还是检索兜底救回来的”
- 给 inventory special-case 一个明确的 query class，避免它继续作为隐式布尔分支散落在 `object_lookup` 流程中
- 让 batch debug 和 interactive debug 都能直接统计 retrieval 路由分布

### 本次修改

#### 1. 给 inventory 问法增加显式 `query_class`

在 `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py` 中：

- 保留 `is_model_inventory_query()` 的判断逻辑
- 但不再只用 `model_inventory_query: bool` 隐式控制分支
- 改为显式生成：
  - `query_class = inventory`
  - 非 inventory 问法则保持原本的 `question_type`

这样后续日志里可以直接看出：

- 这是普通 `object_lookup`
- 还是 inventory 类型问题

而不用再从代码分支里倒推。

#### 2. 新增检索链路元信息

在 real-chain 检索结果里新增两个核心字段：

- `retrieval_route`
- `fallback_trigger_reason`

当前实际落地的路由类型包括：

- `inventory_special_case`
- `lexical_fallback`
- `vector_plus_filter`
- `vector_only`
- `recent_list`

当前触发原因字段包括：

- `inventory_query`
- `rpc_empty`
- `post_filter_empty`

其中：

- `inventory_special_case + inventory_query`
  - 表示这轮不是向量召回，而是显式走了“最近模型清单”路径
- `lexical_fallback + rpc_empty`
  - 表示向量 RPC 没召回到可用结果，最后由词面兜底救回
- `lexical_fallback + post_filter_empty`
  - 表示向量结果有返回，但被后置过滤压空，最后由词面兜底救回

#### 3. batch / interactive 两条日志链都接入新字段

已同步更新：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`
- `ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py`

现在以下输出都会直接包含：

- `query_class`
- `retrieval_route`
- `fallback_trigger_reason`

并且 summary 中新增分布统计：

- `retrieval_route_counts`
- `fallback_reason_counts`

这意味着后面继续积累真实问答时，可以直接回答：

- 当前多少 case 走的是向量主路径
- 多少 case 是 lexical fallback 救回来的
- inventory 特判到底占了多大比例

### 自测与验证

#### 1. 静态校验

执行：

```bash
python -m py_compile \
  ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py \
  ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py
```

结果：

- 通过，无语法错误

#### 2. batch real-chain smoke

构造两条代表性 case：

- `我有没有生成什么模型`
- `帮我找一下洛天依模型`

执行：

```bash
bash ai_engine/finetune_qwen3/scripts/run_real_chain_debug_gpu1.sh off \
  --cases_file /tmp/retrieval_route_smoke_cases.json \
  --output_file /tmp/retrieval_route_smoke.jsonl \
  --summary_file /tmp/retrieval_route_smoke.summary.json \
  --overwrite_output
```

结果确认：

- `我有没有生成什么模型`
  - `query_class = inventory`
  - `retrieval_route = inventory_special_case`
  - `fallback_trigger_reason = inventory_query`

- `帮我找一下洛天依模型`
  - `query_class = object_lookup`
  - `retrieval_route = lexical_fallback`
  - `fallback_trigger_reason = rpc_empty`

summary 中也已正确聚合出：

- `retrieval_route_counts`
  - `inventory_special_case = 1`
  - `lexical_fallback = 1`
- `fallback_reason_counts`
  - `inventory_query = 1`
  - `rpc_empty = 1`

#### 3. interactive smoke

执行：

```bash
printf '我有没有生成什么模型\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --skip_feedback \
    --session_name retrieval_route_interactive_smoke \
    --log_file /tmp/retrieval_route_interactive.jsonl \
    --summary_file /tmp/retrieval_route_interactive.summary.json
```

结果确认：

- 交互日志已写入：
  - `query_class = inventory`
  - `retrieval_route = inventory_special_case`
  - `fallback_trigger_reason = inventory_query`
- interactive summary 已新增：
  - `retrieval_route_counts`
  - `fallback_reason_counts`

### 当前结论

- Part 13 中已经变得重要的检索分支，现在都能被日志直接观测到
- inventory 问法不再只是隐式特判，而是有了显式 `query_class`
- 后续继续积累 `20 ~ 40` 条真实问答时，可以直接量化：
  - 向量主路径占比
  - lexical fallback 依赖度
  - inventory special-case 命中比例

### 下一步建议

- 继续跑交互式 debug，优先积累：
  - 二次元实体名
  - 手办 / 展台 / 摆件 / 地球仪 / 显示器
  - 口语化 recent
  - `partial_coverage`
- 如果后续统计显示：
  - `lexical_fallback` 占比持续偏高
  - 且大量原因是 `rpc_empty` 或 `post_filter_empty`
- 那下一阶段主线应明确切到：
  - 检索层专项优化
  - 而不是继续开新的 LoRA 训练轮次

---

## Part 14：人工交互复盘（`manual_debug_next`）与当前微调质量判断

### 本部分目标

- 复盘一轮真实人工交互
- 判断当前 `round4_1_patch_mixed` 的真实体验是否达到“可用”
- 区分问题究竟来自：
  - LoRA 微调本身
  - 检索路由
  - 规则提示过强
  - answer 后处理缺失
- 给出下一步最值得做的改进项

### 本次观察对象

- 交互命令：
  - `bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh --show_evidence --session_name manual_debug_next`
- 会话 summary：
  - `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_next.summary.json`
- 会话日志：
  - `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_next.jsonl`
- 当前 adapter：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`

### 本次人工交互结果

- 共 `5` 轮
- `feedback_count = 4`
- `error_count = 0`
- 平均检索耗时：
  - `5.321s`
- 平均生成耗时：
  - `0.973s`

逐轮现象如下：

- `你好`
  - 回答：`现在没有找到和你的提问相关的内容。`
  - 问题：把寒暄误当成检索问答，缺少最基本的 assistant persona / 开场白

- `你是谁`
  - 回答：`目前没有找到与“你是谁”相关的记录。`
  - 问题：完全没有身份认识，说明当前 system prompt 和训练分布把模型压成了“只能检索，不会对话”

- `最近有什么模型`
  - 回答：`最近生成过这些模型：scene_20260319_893174、scene_20260319_893174_3dgs、scene_20260318_893381。`
  - 问题：功能层面算“命中了”，但表达层面很差，直接报 `scene_id / display_name`，不像用户语言

- `帮我查一下有没有电脑相关的模型`
  - 回答：`目前只找到过联想笔记本电脑，未见电脑相关模型。`
  - 问题：这里其实检索命中了 `联想笔记本电脑` 和 `HONOR笔记本电脑`
  - 但回答把“电脑相关内容”错误收缩成“不是模型”，表明“模型”一词触发了过强的 inventory / object 混合语义

- `帮我找一下理工相关的 内容，或者计算机科学相关`
  - 回答：`目前只找到大连理工大学购物袋相关内容。`
  - 问题：这是最典型的“词面命中但语义失败”
  - 实际 evidence 里明明还有：
    - `《算法导论》`
    - `《高等数学》`
    - `词典`
    - `地球仪`
    - `办公/学习桌面`
  - 但模型和检索都没有把“理工 / 计算机科学”映射到这些证据

### 当前判断：这轮微调“有进步，但还不能算对话可用”

从这轮人工测试看，`round4_1_patch_mixed` 已经明显优于更早的版本：

- “模型库存类问法”已经不再完全 no-hit
- 常见物体类问法能命中并输出短句
- 输出基本不会再泄漏 JSON

但距离“可直接给人使用”的程度还有明显差距，主要问题有 3 类。

### 问题 1：模型被训得过于检索机，不会处理闲聊和身份问题

这不是单次偶然，而是训练目标本身导致的。

当前推理 system prompt 是：

- 只能根据 retrieval 提供的证据回答
- `hit_count == 0` 时只能回答“暂无相关记录”
- 不解释规则
- 最多两句

这套约束对 RAG 事实问答有效，但副作用很明显：

- `你好`
- `你是谁`
- `你能做什么`

这类问题全部会被压成 no-hit。

结论：

- 当前模型不是“不聪明”
- 而是“被规则和训练分布刻意压窄了”

### 问题 2：模型库存查询虽然通了，但答案模板不自然

这一点基本不是 LoRA 自由生成导致，而是代码里专门的 special-case 导致。

当前 real-chain 在识别到 model inventory query 时，会直接走：

- `build_model_inventory_answer()`

并返回固定模板：

- `最近生成过这些模型：{display_name1}、{display_name2}...`

因此回答天然会偏向：

- `scene_20260319_893174`
- `scene_20260319_893174_3dgs`

这种内部标识符，而不是：

- `最近生成过一个洛天依主题展台场景`
- `还有一个书架/收藏角场景`

结论：

- 这不是 LoRA 退化
- 主要是 answer formatter 过于工程内部视角

### 问题 3：抽象语义类查询能力仍然偏弱

`理工相关 / 计算机科学相关` 这个 case 很有代表性。

当前检索和训练更擅长：

- 明确物体词
  - 电脑
  - 显示器
  - 手办
  - 地球仪
- 明确 recent 问法
- 明确 partial coverage 问法

但不擅长：

- 上位概念
  - 理工
  - 学术
  - 计算机科学
  - 二次元相关
  - 学习相关

原因大概率有两个：

- 检索 query parser 只抽取了 `理工 计算机科学` 这样的词串，没有映射到“算法导论 / 高数 / 显示器 / 笔记本 / 办公桌 / 白板”等具体可召回对象
- 训练集里这类“抽象标签 -> 具体证据概括”的样本太少

### 与历史 benchmark 对照后的判断

从已有日志看，这轮人工交互表现和历史 probe 是一致的，不是偶发波动。

已有 summary：

- `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_1_patch_summary.json`
  - `natural_style_rate = 0.75`
  - `partial_false_negative_rate = 0.3333`
  - `must_answer_focus_rate = 0.6667`

- `ai_engine/finetune_qwen3/logs/real_chain_probe_round4_patch_rescored_summary.json`
  - `natural_style_rate = 0.875`
  - `partial_false_negative_rate = 0.0`
  - `partial_missing_negation_rate = 0.3333`
  - `must_answer_focus_rate = 0.6667`

这说明当前模型的主要提升集中在：

- 不乱输出结构化格式
- 对命中证据能说出更像自然语言的短句
- partial coverage 比之前更稳

但仍未覆盖：

- assistant persona
- 抽象概念归纳
- inventory 结果的人类可读化

### 当前结论

如果只评价“微调是否有效”，答案是：

- 有效

因为它已经解决了一部分最核心的问题：

- 命中时不再频繁输出空泛句
- no-hit 和 partial coverage 的基本纪律已经建立
- 真实链路里可以回答一部分用户问题

但如果评价“当前效果是否足够好”，答案是：

- 还不够

目前更像：

- 一个已经能工作的检索问答原型

而不是：

- 一个对用户友好的本地记忆助手

### 改进优先级建议

我建议不要马上再盲目加大训练轮数，而是按下面顺序改。

#### 优先级 1：先补推理侧兜底，不先继续训

最值得立刻改的是 inference policy，而不是继续堆 LoRA 数据。

建议加 3 个推理侧分支：

- 闲聊 / 寒暄白名单
  - `你好`
  - `嗨`
  - `hello`
  - 输出固定欢迎语

- 身份类白名单
  - `你是谁`
  - `你能做什么`
  - 输出固定 persona 说明

- model inventory answer formatter
  - 不要直接输出 `scene_id`
  - 优先从 evidence 中提取：
    - 主题对象
    - 场景摘要
    - 时间

这一层改完，用户感知会立刻提升，且风险小、见效最快。

#### 优先级 2：补一小轮“抽象概念归纳”训练样本

专门补 `50 ~ 150` 条这类数据：

- 理工相关
- 计算机科学相关
- 学习相关
- 办公相关
- 二次元相关
- 动漫相关
- 桌搭相关
- 收藏相关

训练目标不是让模型胡乱联想，而是学会：

- 从多个具体 object / tag / description 中
- 概括成一个上位类别回答

例如：

- evidence 有 `《算法导论》`、`《高等数学》`、`笔记本电脑`、`显示器`
- 回答可以是：
  - `有，这类内容主要出现在办公学习桌面场景里，能看到《算法导论》、高数教材、笔记本电脑和显示器。`

#### 优先级 3：修 query intent 到 retrieval term 的映射

这个问题如果只靠 LoRA 很难彻底解决。

建议在 query parser 或 retrieval 前处理里补一个轻量同义映射层，例如：

- `理工`
  - 扩展到：`算法 数学 教材 显示器 笔记本电脑 白板 学习 办公`

- `计算机科学`
  - 扩展到：`算法导论 电脑 笔记本 机械键盘 显示器 办公桌`

这类做法会比单纯继续微调更稳定。

#### 优先级 4：再做一轮小 patch 训练

等前面三项落地后，再补一轮小规模 patch：

- persona / greeting 样本
- inventory humanization 样本
- abstract semantic lookup 样本

否则现在直接继续训，很容易把现有“短句纪律”打坏，或者把回答训得更模板化。

### 下一步执行建议

建议下一步不要直接做 `round5 full retrain`，而是先做一个最小闭环：

1. 改推理侧兜底
2. 追加一批抽象语义样本
3. 重跑人工交互
4. 再决定是否值得继续训

更具体地说：

- 第一阶段先只动代码，不动模型权重
  - 目标：把 `你好 / 你是谁 / 最近有什么模型` 这三类体验先修正

- 第二阶段再补小数据集
  - 目标：修 `理工 / 计算机科学 / 学习相关` 这类抽象问法

- 第三阶段再复测这组人工 case
  - 重点看：
    - 是否仍然输出内部 scene id
    - 是否还能维持 no-hit/partial coverage 纪律
    - 抽象概括是否开始可用

---

## Part 15：直接修推理链，不等下一轮训练

### 本部分目标

- 不等下一轮 LoRA
- 直接把 Part 14 暴露出的高频体验问题在推理链里修掉
- 做完真实交互回归，确认没有把已有 guardrails 打坏

### 本次直接修改的内容

修改文件：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

本次没有改训练权重，只改推理链逻辑。

#### 1. 新增非检索类问题兜底

新增了轻量规则识别：

- 寒暄类：
  - `你好`
  - `您好`
  - `hi`
  - `hello`
- 身份类：
  - `你是谁`
  - `你能做什么`
  - `你会做什么`

命中后不再走 retrieval + no-hit，而是直接返回固定 persona 文案。

这样修复后：

- `你好`
  - 不再回答“没有相关记录”
- `你是谁`
  - 不再回答“未找到与你是谁相关的记录”

#### 2. 改写模型清单类 special-case 的输出

之前 `build_model_inventory_answer()` 会直接输出：

- `scene_20260319_893174`
- `scene_20260319_893174_3dgs`

这种内部资产 ID。

现在改成：

- 优先从 evidence 的 `description` 提取首段摘要
- 做重复去重
- 生成更像用户语言的短句

例如现在会输出：

- `最近生成过的模型主要有以虚拟歌手洛天依（Luotianyi）为主题的商业展台装置和白色多层书架角落的特写。`

虽然还不算完美，但已经明显优于直接报内部 ID。

#### 3. 给抽象语义查询增加扩展词和确定性摘要

新增了一层抽象概念扩展：

- `理工`
- `理工科`
- `计算机科学`
- `计算机`
- `学习相关`
- `学术`

扩展到更容易命中的检索词，例如：

- `算法导论`
- `高等数学`
- `教材`
- `词典`
- `笔记本电脑`
- `显示器`
- `白板`

同时新增了 semantic query 的 special answer：

- 如果 query 属于抽象概念问法
- 并且 evidence 已经命中
- 则优先从 objects / tags 中抽取最相关项
- 按优先级组织成短句

这样避免把这种问题完全交给当前 LoRA 自由发挥。

#### 4. 给语义扩展结果增加 merge + rerank

对于抽象概念问法，不再只信单一路径向量召回。

现在会：

- 先跑原本 vector retrieval
- 再跑 lexical fallback
- 合并结果
- 按匹配词覆盖数 + 时间做 rerank

这个改动的意义是：

- 避免只因为一个弱词面命中
  - 例如“大连理工大学购物袋”
- 就把真正更相关的：
  - `《算法导论》`
  - `《高等数学》`
  - `笔记本电脑`
  - `显示器`
  - `白板`
  压下去

### 本次验证

#### 1. 语法检查

已执行：

```bash
python -m py_compile \
  ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py \
  ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py
```

结果：

- 通过

#### 2. 真实交互回归

已执行：

```bash
printf '你好\n\n你是谁\n\n最近有什么模型\n\n帮我找一下理工相关的内容，或者计算机科学相关\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name manual_debug_next_fix_v2
```

以及：

```bash
printf '最近有什么模型\n\n帮我找一下理工相关的内容，或者计算机科学相关\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name manual_debug_next_fix_v3
```

还执行了 guardrail 回归：

```bash
printf '我最近拍过钢琴吗？\n\n我最近拍过显示器和钢琴吗？\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name manual_debug_guardrails_v1
```

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_next_fix_v2.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_next_fix_v3.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_guardrails_v1.jsonl`

### 关键结果

修复后：

- `你好`
  - 回答：`我是 BrainDance 的本地记忆问答助手，可以帮你查最近拍到的内容、物体线索和生成过的模型。`

- `你是谁`
  - 回答：`我是 BrainDance 的本地记忆问答助手，主要帮你根据本地记录查询拍摄内容、物体线索和模型资产。`

- `最近有什么模型`
  - 不再输出 `scene_id`
  - 现在回答：`最近生成过的模型主要有以虚拟歌手洛天依（Luotianyi）为主题的商业展台装置和白色多层书架角落的特写。`

- `帮我找一下理工相关的内容，或者计算机科学相关`
  - 现在可返回更合理的抽象摘要
  - 例如：`有，相关内容里能看到《算法导论》、《高等数学》教材、HONOR笔记本电脑、白色笔记本电脑。`

- `我最近拍过钢琴吗？`
  - 仍保持：`目前没有钢琴相关记录。`

- `我最近拍过显示器和钢琴吗？`
  - 仍保持：`最近拍到过显示器，但没有拍到钢琴。`

### 当前结论

这次修改说明：

- 这批问题不需要靠继续加训才能修
- 推理链前后处理就能明显提升用户体验
- 当前 LoRA 适合做“证据转短句”
- 但 persona、闲聊、inventory humanization、抽象概念归纳，仍然更适合代码侧兜底

### 当前遗留问题

还没完全做到理想状态的点：

- `最近有什么模型`
  - 现在虽然不再报内部 ID
  - 但表达仍偏书面，后续可以再压缩成更短、更口语化

- `理工 / 计算机科学`
  - 已经能命中更合适的内容
  - 但摘要项还不够精炼，后续可继续做对象归一化
  - 例如把：
    - `HONOR笔记本电脑`
    - `白色笔记本电脑`
    统一成：
    - `笔记本电脑`

### 下一步建议

- 先继续积累真实交互日志
- 下一轮再考虑补一个很小的 patch：
  - `abstract semantic summary`
  - `inventory humanization`
  - `object normalization`
- 在这之前，不建议贸然继续大规模重训

## Part 12：debug-only 观察期

### 本部分目标

- 暂停继续训练
- 以 `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 作为当前 best debug adapter
- 扩大真实链路 compare 样本面
- 观察是否存在新的稳定 failure mode

### 当前结论

- 现在先不继续训练
- 当前最合理的 debug 候选已经明确：
  - `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`
- 这版 adapter 在真实链路上，比 `qwen3_1p7b_lora_sft_round4_patch` 更符合本轮目标
- fixed benchmark 虽然仍有轻微风格回撤，但没有破坏关键能力
- 因此当前主线从“继续训一轮”切换为“真实链路 debug-only 观察”

### 当前 best 版本与保留对照

当前 best debug adapter：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed`

作为基线保留的对照版本：

- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3`
- `ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_patch`

保留这三版的目的很直接：

- 后续任何真实链路回归，都可以明确和哪一版对比
- 可以快速判断问题是 round4 就存在，还是 round4.1 mixed 新引入

### 进入观察期的依据

真实链路相对 round4 patch 的关键变化：

- `natural_style_rate`
  - `0.8750 -> 1.0000`
- `partial_false_negative_rate`
  - `0.0000 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.3333 -> 0.0000`
- `must_answer_focus_rate`
  - `0.6667 -> 0.6667`

fixed benchmark 相对 round4 patch 的关键变化：

- `natural_style_rate`
  - `0.9000 -> 0.8500`
- `must_answer_focus_rate`
  - `0.8750 -> 0.8125`
- `partial_false_negative_rate`
  - `0.0625 -> 0.0000`
- `partial_missing_negation_rate`
  - `0.0000 -> 0.0000`
- `partial_hallucination_rate`
  - `0.0625 -> 0.0000`

当前判断是：

- round4.1 mixed 已经达到一个更合理的平衡点
- 它修掉了本轮最重要的真实链路 gap
- 离线的小幅回撤主要集中在非本轮目标组
- 因此下一步最有价值的动作不是 round4.2，而是扩大真实链路观察面

### 当前原则

- 不开 `round4.2`
- 不做 `DPO / ORPO`
- 不直接替换正式产品链路
- 优先以真实链路表现决定后续是否继续训练
- 零星不完美先接受，不为轻微离线分数回撤继续开新轮次

### 观察期执行方案

#### 1. 扩大真实链路 compare 样本面

- 不再只依赖固定 `8` 条 probe
- 扩到 `20 ~ 40` 条真实 debug 问题
- 仍按 failure type 分组观察，不做无结构混测

推荐分组：

- `recent_hit`
- `no_hit`
- `partial_coverage`
- `must_answer`
- `multi_hit_must_answer`
- `stability`

本轮优先补两类真实问法：

- 口语化 recent
  - 例如：
    - `这两天我都拍了啥？`
    - `最近又扫到什么了？`
    - `最近有哪些桌面上的东西？`
- 多目标 partial
  - 例如：
    - `我最近拍过笔记本电脑、地球仪和钢琴吗？`
    - `最近拍到过显示器和小提琴吗？`

重点不是继续证明二元 partial 已修，而是看三元组合和口语化问法会不会暴露新 failure mode。

#### 2. 日志里增加 failure triage 字段

建议在真实链路 compare 日志中增加人工回填字段：

- `triage_label`

推荐枚举：

- `ok`
- `style_minor`
- `focus_minor`
- `partial_regression`
- `retrieval_issue`
- `new_failure_mode`

增加这个字段的目的是把问题快速分层：

- 是模型行为还要继续修
- 还是检索侧证据本身有问题
- 还是只是风格轻微瑕疵，不值得继续训练

#### 3. 设定停止训练阈值

在以下情况出现前，不再继续训练：

- 真实链路出现新的稳定复现 failure mode
- 同一类 failure 在 `20 ~ 40` 条 debug 问题里多次出现
- 问题已经影响是否推进产品开关

换句话说：

- 如果只是零星不完美，先接受
- 如果只是固定 benchmark 上的轻微风格波动，先不训
- 只有真实链路里出现稳定、可复现、会影响产品判断的问题，才值得开 round4.2

### 当前不做的事

- 不立刻开 `round4.2`
- 不因为 fixed benchmark 上 `natural_style_rate` 的轻微回撤推翻 round4.1 mixed
- 不把当前问题升级成偏好优化问题，因此暂不考虑 `DPO / ORPO`

原因是：

- 当前最关键的 gap 已经不是“大偏差”
- 而是少量行为边界和真实链路覆盖不足
- 在这个阶段继续训练，过度优化的风险已经高于预期收益

### 什么时候才值得开 round4.2

只有满足以下任一条件，才值得继续训：

- 条件 A：真实链路出现新稳定问题
  - 例如三元 partial 组合反转、口语化 recent 明显退化、多命中 must-answer 又回到半结构化
- 条件 B：准备推进到更正式的产品开关
  - 这时才值得专门修 `recent_hit`、`stability` 和更广泛的 style 一致性
- 条件 C：真实使用证明某类 focus 问法非常重要
  - 当前最典型的是：
    - `最近拍到过什么办公桌上的东西？`
  - 如果这类问法后续频繁出现，再考虑把它单独做成 round4.2 目标

### 当前验收标准

- `partial_false_negative` 不反弹
- `partial_missing_negation` 保持为 `0`
- `multi-hit must_answer` 保持自然语言表达
- 如果 `20 ~ 40` 条真实 debug 问题中没有新的稳定 failure mode，则继续停训，进入更长期的 debug 使用

### 本部分结论

- 训练主线先暂停
- `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 作为当前 best debug adapter 保留
- 下一阶段重点不是继续做 round4.2，而是扩大真实链路 compare 样本面
- 只有在观察期里发现新的稳定 failure mode，或者产品开关需要更高一致性时，才值得继续开新训练轮次

### 本部分实际落地

#### 1. 新增 Part 12 真实 debug 题集

新增题集文件：

- `ai_engine/finetune_qwen3/data/real_chain_debug_cases_part12.json`

本轮实际落地为 `26` 条真实 debug 问题，分组如下：

- `recent_hit = 4`
- `no_hit = 4`
- `partial_coverage = 8`
- `must_answer = 4`
- `multi_hit_must_answer = 3`
- `stability = 3`

相比 Part 9 的 `8` 条 probe，这一轮覆盖面明显扩大，并且补进了：

- 口语化 recent
  - 例如：
    - `这两天我都拍了啥？`
    - `最近又扫到什么了？`
    - `最近有哪些桌面上的东西？`
- 三元 partial
  - 例如：
    - `我最近拍过笔记本电脑、地球仪和钢琴吗？`
    - `最近拍到过显示器、笔记本电脑和小提琴吗？`
- multi-hit must-answer
  - 例如：
    - `最近拍到过什么手办相关画面？`
    - `最近拍到过什么书架相关画面？`
    - `最近拍到过什么书籍相关内容？`

#### 2. 扩充 real-chain debug 日志字段与分组统计

更新脚本：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py`

新增或增强的能力：

- 支持 case 级唯一标识：
  - `case_id`
- 支持人工回填字段：
  - `triage_label`
- 将 `multi_hit_must_answer` 纳入 must-answer 统计
- summary 中新增：
  - `group_counts`
  - `triage_counts`
  - `metrics_by_group`

这样做的目的很直接：

- 后续可以按 case 级别持续补人工 triage
- 也可以直接看各组而不是只看 overall summary

#### 3. 新增 Part 12 一键 compare 入口

新增脚本：

- `ai_engine/finetune_qwen3/scripts/run_real_chain_part12_compare_gpu1.sh`

这个脚本会顺序跑三版 adapter：

- `qwen3_1p7b_lora_sft_round3`
- `qwen3_1p7b_lora_sft_round4_patch`
- `qwen3_1p7b_lora_sft_round4_1_patch_mixed`

统一使用：

- `ai_engine/finetune_qwen3/data/real_chain_debug_cases_part12.json`

对应输出文件：

- `ai_engine/finetune_qwen3/logs/real_chain_part12_round3_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round3_summary.json`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_patch_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_patch_summary.json`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_1_patch_mixed_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/real_chain_part12_round4_1_patch_mixed_summary.json`

#### 4. 观察期脚本稳定性修复

在实际跑 Part 12 compare 时，暴露了两个非模型层问题：

- `model_assets.created_at` 存在格式不完全规整的时间字符串
  - 会导致 `parse_datetime()` 在 lexical fallback 过滤时中断整轮评估
- DashScope embedding 接口存在偶发 `SSL EOF` 抖动
  - 会让整轮 debug 因单次网络请求失败而终止

因此在 `run_real_chain_debug.py` 里额外补了两类稳定性修复：

- 更稳健的时间解析与过滤
- 对 DashScope / Supabase 请求增加轻量重试

这两项修复不改变模型行为，但能显著提升真实链路批量观察的稳定性。

### Part 12 实际运行结果

#### 1. 三版 adapter 的真实链路整体对比

在同一份 `26` 条 Part 12 题集上，三版 adapter 的关键指标如下：

- round3
  - `natural_style_rate = 0.7308`
  - `partial_false_negative_rate = 0.1250`
  - `partial_missing_negation_rate = 0.3750`
  - `must_answer_focus_rate = 0.2857`

- round4 patch
  - `natural_style_rate = 0.7692`
  - `partial_false_negative_rate = 0.1250`
  - `partial_missing_negation_rate = 0.2500`
  - `must_answer_focus_rate = 0.4286`

- round4.1 patch mixed
  - `natural_style_rate = 0.8462`
  - `partial_false_negative_rate = 0.0000`
  - `partial_missing_negation_rate = 0.0000`
  - `must_answer_focus_rate = 0.5714`

从这轮更大样本面的真实链路结果看：

- `round4.1 mixed` 仍然是当前最合理的 debug adapter
- `partial_false_negative` 和 `partial_missing_negation` 都没有反弹
- `must_answer_focus_rate` 虽然仍未完全理想，但已经继续优于 `round3 / round4 patch`

#### 2. 分组结果

`round4.1 mixed` 的 group-level 指标如下：

- `recent_hit`
  - `natural_style_rate = 0.5000`
- `no_hit`
  - `natural_style_rate = 1.0000`
- `partial_coverage`
  - `natural_style_rate = 0.8750`
  - `partial_false_negative_rate = 0.0000`
  - `partial_missing_negation_rate = 0.0000`
- `must_answer`
  - `natural_style_rate = 0.7500`
  - `must_answer_focus_rate = 0.5000`
- `multi_hit_must_answer`
  - `natural_style_rate = 1.0000`
  - `must_answer_focus_rate = 0.6667`
- `stability`
  - `natural_style_rate = 1.0000`

这组结果说明：

- Part 12 新增的三元 partial 没有打出新的 stable partial regression
- `stability` 组整体是稳的
- 当前残留问题主要集中在：
  - `recent_hit`
  - 少量 `must_answer / multi_hit_must_answer`

#### 3. 当前残留 failure triage

`round4.1 mixed` 在 `26` 条题集里，没有出现新的 stable `partial_regression`。
当前可见的残留问题共 `6` 条，主要都是轻量问题：

- `triage_counts`
  - `ok = 20`
  - `style_minor = 3`
  - `focus_minor = 3`

- `style_minor`
  - `part12_recent_002`
    - `这两天我都拍了啥？`
  - `part12_recent_004`
    - `最近有哪些桌面上的东西？`
  - `part12_partial_006`
    - `最近拍到过显示器、笔记本电脑和小提琴吗？`

- `focus_minor`
  - `part12_must_003`
    - `最近拍到过什么办公桌上的东西？`
  - `part12_must_004`
    - `最近拍到过什么地球仪相关画面？`
  - `part12_multi_002`
    - `最近拍到过什么书架相关画面？`

当前最值得记住的三个 observation：

- `最近拍到过什么办公桌上的东西？`
  - `round4.1 mixed` 已经比 round3 更聚焦
  - 但 focus term 还不够稳定地落在 `Elaina手办`

- `最近拍到过什么地球仪相关画面？`
  - 三版都存在“答到地球仪，但顺带拉出过多周边上下文”的问题
  - 这更像旧的 focus/style 边界问题，不是 round4.1 mixed 新引入

- `最近拍到过什么书架相关画面？`
  - 当前回答会偏向书架上的具体物品
  - 但不够稳定地直接点出 `书架`
  - 这一类更适合先记为 `focus_minor` 观察，而不是立刻开训练轮次

### 本部分最终判断

- Part 12 已经按计划完成：
  - 题集扩到 `26` 条
  - 三版 real-chain compare 已跑通
  - `triage_label` 字段已补到日志结构
  - 观察期脚本稳定性问题也已顺手修掉

- 当前没有看到足以立即开启 `round4.2` 的新稳定 failure mode

- `round4.1 mixed` 在更大真实样本面上继续保持：
  - `partial_false_negative = 0`
  - `partial_missing_negation = 0`

- 当前残留问题仍然主要是：
  - `recent_hit` 的风格波动
  - 少量 `must_answer / multi_hit_must_answer` 的 focus 不够稳

因此：

- 继续停训
- 继续拿 `qwen3_1p7b_lora_sft_round4_1_patch_mixed` 做真实链路 debug-only 观察
- 只有在后续继续积累 case 时，确认这些 focus/style 问题已经形成稳定复现模式，才值得进入 `round4.2`

---

## Part 13：交互式 Debug 程序与 real-chain 检索兜底修复（2026-03-21）

### 本部分目标

- 做一个真正可交互的本地 debug 程序
- 让人工用户可以直接提问、立即看到回答、并把每轮问答记录下来
- 自己复现实例问题，确认当前 real-chain 是“没联网”还是“召回策略有缺陷”
- 修复 `object_lookup` 类常见问法在真实数据上 `hit_count=0` 的问题

### 发现的问题

在第一次交互式试跑中，用户输入：

- `我有没有生成什么模型`
- `帮我找一下洛天依模型`

程序都返回：

- `intent = no_hit`
- `hit_count = 0`
- `evidence = []`

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_001.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/manual_debug_001.summary.json`

### 首轮排查结论

不是 RAG 完全没连上，而是“检索链路联通正常，但当前 object 类召回策略不稳”。

已经确认的事实：

- DashScope key 能正常读取
- Supabase REST `model_assets` 能正常返回数据
- 远端库里确实存在“洛天依”相关记录

例如通过直接请求 `model_assets` 可见：

- `scene_20260319_893174`
- `scene_20260319_893174_3dgs`
- `test_scene_sharp_1768839315`

这些记录的 `description / objects / tags` 中都明确包含：

- `洛天依`
- `洛天依毛绒玩偶`
- `洛天依手办`

### 根因定位

根因不是单点，而是两个问题叠加：

#### 1. 交互启动器使用 `conda run`

最初交互脚本通过：

- `conda run -n qwen3_ft python ...`

启动。

这导致 `input()` 在某些终端环境下直接读到 EOF，表现为：

- 刚打印 `你>` 就自动退出

这不是模型问题，而是交互 stdin 被启动器吞掉。

#### 2. `object_lookup` 路径过度依赖向量 RPC，且后置过滤过严

在 `run_real_chain_debug.py` 里，`object_lookup` 的主路径是：

- DashScope 解析意图
- DashScope embedding
- Supabase RPC：`match_memory_poses`
- 再用目标词做字符串过滤

但实际排查发现：

- `match_memory_poses` 对 `洛天依`、`洛天依模型`、`模型` 在默认 `match_threshold = 0.5` 下都返回 `0` 条
- 即使把阈值降到 `0.1`，也主要召回无关项
- 对 `洛天依模型` 这种问法，解析器会给出：
  - `search_text = "洛天依模型"`
  - `target_objects = ["洛天依模型"]`
- 后续过滤要求结果里直接出现 `洛天依模型` 这个完整短语
- 但真实数据里通常是：
  - `洛天依`
  - `洛天依毛绒玩偶`
  - `洛天依手办`

所以会出现：

- 真实库里明明有相关记录
- 向量召回不稳定
- 词面过滤又太死
- 最终 `raw_rows` 被压成空数组

### 本次修复内容

#### 1. 新增交互式 debug 程序

新增文件：

- `ai_engine/finetune_qwen3/scripts/interactive_debug_chat.py`

功能：

- 真实检索链路
- 本地 LoRA 生成
- 单轮即时回答
- 可选显示 evidence
- 每轮可输入人工反馈
- 自动写 JSONL 日志与 summary

新增启动脚本：

- `ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh`
- `ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu1.sh`

#### 2. 修复交互 stdin 问题

把交互启动脚本从：

- `conda run -n qwen3_ft python ...`

改成：

- 初始化 conda shell hook
- `conda activate qwen3_ft`
- `exec python ...`

这样 stdin 会直接交给 Python 进程，交互式 `input()` 不再提前读到 EOF。

#### 3. 给 `object_lookup / partial_coverage` 增加词面兜底检索

在 `ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py` 中新增：

- `normalize_lookup_terms()`
- `row_matches_lookup_terms()`
- `lexical_fallback_model_assets()`

修复策略：

- 对 `洛天依模型` 这类词做去泛化清洗
  - 例如去掉后缀：
    - `模型`
    - `场景`
    - `内容`
    - `记录`
    - `画面`
- 当向量 RPC 返回空结果，或返回结果被目标词过滤为空时：
  - 回退到 `model_assets` 做词面匹配
- 在 `partial_coverage` 中，如果单个 target 没通过向量召回：
  - 也允许走词面兜底补一个 matched row

#### 4. 给“模型清单类问法”增加窄范围 special-case

继续排查发现，下面这类问题：

- `我有没有生成什么模型`

在当前意图解析中可能被解析成：

- `question_type = object_lookup`
- `search_text = ""`
- `target_objects = []`

这时既不会命中实体召回，也不会进入 recent list 路径。

因此新增：

- `is_model_inventory_query()`
- `build_model_inventory_answer()`

处理逻辑：

- 如果原始问题本质上是在问“最近生成了哪些模型”
- 则直接把检索切到 `model_assets` 最近记录
- 并用一个 deterministic 的短句组织回答

这样做的原因很直接：

- 这类问法更像“资产清单查询”
- 不是 LoRA 当前主要训练的“证据问答”类型
- 直接交给模型自由发挥，容易回到“没有找到生成模型内容”这类歧义输出

### 自测与验证

#### 1. 代表性问题集

这次实际自测的问题集为：

- `我有没有生成什么模型`
- `帮我找一下洛天依模型`
- `找一下洛天依`
- `我最近拍过钢琴吗？`
- `我最近拍过显示器和钢琴吗？`
- `我最近拍了什么？`

临时 cases 文件：

- `/tmp/manual_debug_cases_stage2.json`

运行命令：

```bash
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
eval "$(conda shell.bash hook)"
conda activate qwen3_ft

python ai_engine/finetune_qwen3/scripts/run_real_chain_debug.py \
  --mode lora_round3 \
  --adapter_path ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed \
  --cases_file /tmp/manual_debug_cases_stage2.json \
  --output_file ai_engine/finetune_qwen3/logs/manual_debug_stage2_cases.jsonl \
  --summary_file ai_engine/finetune_qwen3/logs/manual_debug_stage2_summary.json \
  --overwrite_output
```

输出结果：

- `ai_engine/finetune_qwen3/logs/manual_debug_stage2_cases.jsonl`
- `ai_engine/finetune_qwen3/logs/manual_debug_stage2_summary.json`

#### 2. 修复后的关键结果

修复后：

- `我有没有生成什么模型`
  - `hit_count = 3`
  - 回答：`最近生成过这些模型：scene_20260319_893174、scene_20260319_893174_3dgs、scene_20260318_893381。`

- `帮我找一下洛天依模型`
  - `hit_count = 3`
  - 回答：`目前找到的主要是洛天依毛绒玩偶、洛天依手办等。`

- `找一下洛天依`
  - `hit_count = 3`
  - 回答：`最近和洛天依相关的主要是商业展台装置，还有动漫手办特写照片。`

- `我最近拍过钢琴吗？`
  - `hit_count = 0`
  - 回答仍保持正确 no-hit

- `我最近拍过显示器和钢琴吗？`
  - `hit_count = 1`
  - 回答：`目前只找到显示器相关内容，没有钢琴相关记录。`

- `我最近拍了什么？`
  - `hit_count = 3`
  - 回答：`最近拍到过洛天依毛绒玩偶，还有蓝色地球仪（带银色金属支架）。`

这说明：

- 当前问题不是 LoRA 生成层坏掉
- 而是检索前后处理需要更稳健的兜底
- 修复后，常见实体类 object 查询已经恢复到可用状态

#### 3. 交互脚本本身验证

为确认交互程序路径也正常，额外做了一次自动输入验证：

```bash
printf '帮我找一下洛天依模型\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name auto_verify_luotianyi
```

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_luotianyi.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_luotianyi.summary.json`

结果确认：

- 交互程序不再自动 EOF 退出
- `hit_count = 3`
- evidence 中正确出现：
  - `scene_20260319_893174`
  - `scene_20260319_893174_3dgs`
  - `test_scene_sharp_1768839315`

对原始失败问法也追加了一次自动验证：

```bash
printf '我有没有生成什么模型\n\n/quit\n' | \
  bash ai_engine/finetune_qwen3/scripts/run_interactive_debug_gpu0.sh \
    --show_evidence \
    --session_name auto_verify_model_inventory
```

对应日志：

- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_model_inventory.jsonl`
- `ai_engine/finetune_qwen3/logs/interactive_sessions/auto_verify_model_inventory.summary.json`

结果确认：

- `intent = recent_capture`
- `hit_count = 3`
- 交互脚本直接输出：
  - `最近生成过这些模型：scene_20260319_893174、scene_20260319_893174_3dgs、scene_20260318_893381。`

### 当前结论

- 交互式 debug 程序已经可用
- 当前 real-chain 并不是“没联网”
- 真正的问题是：
  - 向量 RPC 对部分常见实体词召回不稳定
  - `object_lookup` 的后置过滤过于字面化
- “模型清单类问法”不适合完全依赖当前 LoRA 自由生成
- 通过给 `model_assets` 增加词面兜底后：
  - `洛天依模型` 这类常见用户问法已经恢复正常
- 通过增加窄范围 inventory special-case 后：
  - `我有没有生成什么模型` 这类原始失败问法也已恢复到可用状态

### 下一步建议

- 继续用交互程序积累 `20 ~ 40` 条真实用户问答
- 在日志中重点观察以下类型：
  - 二次元实体名
  - 手办 / 展台 / 摆件 / 地球仪 / 显示器这类具体物品
  - `partial_coverage`
  - recent 口语化问法
- 如果后续暴露更多“向量召回弱、词面兜底才救回来”的 case：
  - 再考虑专门优化检索层，而不是直接继续训 LoRA
