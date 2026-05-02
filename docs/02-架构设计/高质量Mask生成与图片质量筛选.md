# 高质量 Mask 生成与图片质量筛选

本文档详细记录 Python 后端 3DGS Pipeline 中两个关键环节的实现：**高质量 Mask 生成**和**图片质量筛选**。

---

## 一、高质量 Mask 生成

### 1.1 整体流水线

```
输入图片集
  │
  ├─ Step 1: Qwen-VL 分析中心物体 → 生成检测提示词
  │
  ├─ Step 2: YOLO World 目标检测 → 获取 BBox
  │
  ├─ Step 3: SAM2 精确分割 → 生成 Mask
  │
  ├─ Step 4: Mask 清洗与质检 → 剔除不合格结果
  │
  └─ Step 5: 输出透明 PNG + 更新 transforms.json
```

**涉及文件：**
| 文件 | 职责 |
|---|---|
| `ai_engine/3dgs/src/modules/ai_segmentor.py` | 主流水线编排 |
| `ai_engine/3dgs/src/utils/cv_algorithms.py` | Mask 清洗算法 |

### 1.2 Step 1: 中心物体识别（Qwen-VL）

**函数:** `get_central_object_prompt(images_dir, sample_count=3)`

**目的：** 自动分析视频抽帧中始终位于画面中心的主体物体，生成适合 YOLO 检测的英文提示词。

**实现逻辑：**

1. **均匀采样**：用 `np.linspace` 从全部图片中等间距抽取 3 张，覆盖物体的不同角度
2. **多模态调用**：将 3 张图 + 文本 Prompt 发送给阿里云 Qwen-VL-Plus（`qwen-plus` 模型）
3. **Prompt 策略**：要求模型优先描述**视觉特征**（颜色、材质、形状）而非功能名称。例如：
   - 不说 "electric shaver"，说 "gray metal object"
   - 不说 "portable charger"，说 "white rectangular box"
4. **清洗输出**：去除标点符号和引号，防止干扰 YOLO 检测

**输出示例：** `"gray metal object"`、`"white rectangular box"`

### 1.3 Step 2: YOLO World 目标检测

**模型：** `yolov8s-worldv2.pt`（存放在共享模型目录 `shared_model_dir`）

**实现逻辑：**

1. 加载 YOLO World 模型，通过 `det_model.set_classes([text_prompt])` 设置检测目标
2. 对每张图片执行 `det_model.predict(img_path, conf=0.05)`，置信度阈值极低（0.05），确保不漏检
3. 如果检测到多个 BBox，调用 `_pick_center_box()` 筛选**最接近画面中心**的那个

**中心框选择算法 (`_pick_center_box`)：**
```
计算每个 BBox 的中心点坐标
计算中心点到画面中心 (w/2, h/2) 的欧氏距离
选择距离最小的 BBox
```

4. 如果 YOLO 完全没检测到物体，退化为**中心点模式**：在画面正中心生成一个 10×10 像素的小框，交给 SAM 处理

### 1.4 Step 3: SAM2 精确分割

**模型：** `sam2.1_l.pt`（存放在共享模型目录 `shared_model_dir`）

**实现逻辑：**

1. 将 YOLO 检测到的 BBox（或中心点退化框）作为 Prompt 输入 SAM
2. SAM 输出像素级分割 Mask
3. 如果 SAM 输出了多个 Mask（如物体有多个部分），用 `np.any(..., axis=0)` **合并为一个完整 Mask**

### 1.5 Step 4: Mask 清洗与质检（核心算法）

**函数:** `clean_and_verify_mask(mask, img_name="")`

这是保证 Mask 质量的关键环节，分为**清洗**和**质检**两个阶段。

#### 阶段 A：连通域清洗

| 步骤 | 算法 | 说明 |
|---|---|---|
| 1 | `cv2.connectedComponentsWithStats(connectivity=8)` | 8-连通域分析，识别所有独立区域 |
| 2 | 遍历所有标签，找最大面积块 | 主体通常是最大连通域 |
| 3 | 面积阈值检查 | 最大块面积 < 全图 0.5% → 判定为噪点，剔除 |
| 4 | 重构 Mask | 只保留最大连通域标签，其余全部置 0（去除飞溅噪点） |

#### 阶段 B：形状质检

| 检查项 | 算法 | 阈值 | 目的 |
|---|---|---|---|
| 实心度 (Solidity) | `contourArea / convexHullArea` | ≥ 0.88 | 过滤边缘毛糙或粘连阴影的 Mask |
| 长宽比 (Aspect Ratio) | `boundingRect.w / boundingRect.h` | ≤ 4.5 | 过滤长条形误检（如桌面缝隙、墙角线） |

**实心度原理：** 凸包（Convex Hull）是用橡皮筋包住物体的最小凸形状。正常物体的实心度接近 1.0；如果 Mask 边缘不规则或粘连了背景阴影，实心度会明显降低。

#### 阶段 C：边缘腐蚀

```
kernel = 3×3 全 1 矩阵
cleaned_mask = cv2.erode(mask, kernel, iterations=1)
```

**效果：** 白色区域向内收缩约 1 像素，去除物体边缘可能残留的"光晕"或背景杂色。

### 1.6 Step 5: 输出透明 PNG

**函数:** `_save_transparent_png(img_path, mask)`

1. 对 Mask 做 `cv2.GaussianBlur(5×5)` 平滑，使边缘过渡自然
2. 将模糊后的 Mask 归一化为 `[0.0, 1.0]` 作为 Alpha 通道
3. 将原图 BGR 通道 × Alpha 合成为 BGRA 四通道图像
4. 保存为 `.png` 格式，删除原始 `.jpg`
5. 更新 `transforms.json` 中的 `file_path` 指向新 PNG

### 1.7 失败处理

| 情况 | 处理 |
|---|---|
| Mask 为空 / 主体过小 | 删除该图片，从 transforms.json 移除 |
| 实心度 < 0.88 | 删除该图片 |
| 长宽比 > 4.5 | 删除该图片 |
| 全部图片被剔除 | 返回 `False`，流水线终止 |
| YOLO 无检测结果 | 退化为中心点模式 |

---

## 二、图片质量筛选

图片质量筛选分为两个独立阶段：**传统 CV 模糊过滤**（快速、本地）和 **AI 质量评分**（精准、云端）。

### 2.1 阶段一：传统 CV 模糊过滤

**文件:** `ai_engine/3dgs/src/modules/image_proc.py`
**函数:** `ImageProcessor.smart_filter_blurry_images(image_folder, keep_ratio=0.85)`

#### 算法流程

```
全部图片
  │
  ├─ 对每张图片计算清晰度得分（3×3 网格拉普拉斯方差）
  │
  ├─ 按得分排序，丢弃最低 (1 - keep_ratio)% 的图片
  │
  └─ 如果剩余图片 > max_images，均匀降采样
```

#### 清晰度评分算法

1. 图片转灰度
2. 将图片等分为 **3×3 网格**（9 个区域）
3. 对每个区域计算 `cv2.Laplacian(roi, CV_64F).var()`（拉普拉斯方差）
4. 取 9 个区域中的**最大值**作为该图片的清晰度得分

**为什么取最大值而非均值？** 因为视频中物体可能只占画面的一部分，取最大值能确保物体所在的清晰区域不被画面其他部分的模糊拉低。

#### 拉普拉斯方差原理详解

**核心结论：拉普拉斯方差越低，图片越模糊。** 这是一个无参考（no-reference）的清晰度度量，不需要"标准清晰图"来对比，直接对单张图片就能判断。

**拉普拉斯算子**是一个二阶微分算子，检测图像中**灰度变化剧烈的地方**（即边缘）。它的卷积核为：

```
 0  -1   0
-1   4  -1
 0  -1   0
```

对一个像素，它的计算逻辑是：**自身灰度 × 4 − 上下左右四个邻居的灰度之和**。

- 在平坦区域（如纯色墙面），每个像素和邻居几乎一样 → 拉普拉斯值接近 **0**
- 在边缘处（如物体轮廓），像素和邻居差异大 → 拉普拉斯值**很大**

然后对整张图（或某个区域）的拉普拉斯结果计算**方差（variance）**：

- 方差高 → 图像中存在大量剧烈变化的边缘 → **清晰**
- 方差低 → 整图灰度变化平缓 → **模糊**（运动模糊、失焦等）

对应代码（`image_proc.py:44`）：

```python
score = cv2.Laplacian(roi, cv2.CV_64F).var()
```

**效果示例：**

| 场景 | 拉普拉斯方差 | 结果 |
|---|---|---|
| 静止清晰的物体 | 高（~500+） | 保留 |
| 运动模糊的帧 | 低（~50 以下） | 剔除 |
| 失焦的帧 | 低（~30 以下） | 剔除 |
| 纯色墙壁（无纹理） | 极低（~5） | 剔除 |

**局限性：** 无法区分"本身就缺少纹理的物体"和"模糊"，所以后面还有 AI 质评（Qwen-VL）作为第二层保障。此外，阈值是动态的（基于百分位数），不是固定值，会随数据集自适应。

#### 过滤策略

| 参数 | 默认值 | 说明 |
|---|---|---|
| `keep_ratio` | 0.85 | 保留得分最高的 85% 图片 |
| `max_images` | 300 (配置) | 最终保留的图片数量上限 |
| 阈值计算方式 | `np.percentile(scores, (1-0.85)*100)` | 动态阈值，第 15 百分位 |

#### 降采样策略

当过滤后图片数量仍超过 `max_images` 时：
- 使用 `np.linspace` 在图片序列中均匀抽取 `max_images` 张
- 多余图片移入 `trash_smart` 目录

### 2.2 阶段二：AI 质量评分

**文件:** `ai_engine/3dgs/src/modules/scene_analyzer.py`
**类:** `SceneAnalyzer`

#### `run()` 方法 —— 场景级质量评估

**目的：** 对整个图片集进行综合评估，判断是否适合 3DGS 重建。

**流程：**

1. **随机采样**：从全部图片中随机抽取最多 6 张
2. **图片编码**：将图片转为 base64 data URI，超过 10MB 自动压缩
3. **AI 分析**：调用 Qwen-VL-Plus 模型进行评估
4. **解析结果**：从 JSON 响应中提取 score、reason、tags、description、objects

**AI 评估标准（通过 Prompt 约定）：**

| 字段 | 说明 |
|---|---|
| `score` (0-100) | 整体质量分数，综合光照、清晰度、纹理丰富度 |
| `reason` | 评分理由 |
| `tags` | 5-10 个关键词标签（如"室内"、"红色"、"马克杯"） |
| `description` | 详细自然语言场景描述 |
| `objects` | 图中主要物体列表 |

**判定逻辑：**
```python
passed = (score >= self.cfg.min_quality_score)  # 默认阈值: 40 分
```

#### 图片自动压缩机制

当图片超过 9MB（`TARGET_DATA_URI_BYTES = 10MB × 0.9`）时自动压缩：

1. **缩放候选序列：** `[1.0, 0.85, 0.7, 0.55, 0.4]`
2. **质量候选序列：** `[90, 80, 70, 60, 50, 40]`
3. 依次尝试不同缩放比例 × JPEG 质量组合，直到 data URI < 9MB
4. 使用 `PIL.Image.Resampling.LANCZOS` 高质量缩放

#### 安全兜底

| 异常场景 | 处理 |
|---|---|
| 无 API Key | 跳过质检，默认通过 (score=60) |
| API 调用失败 | 默认通过 (score=60) |
| 模型返回"未识别到图片" | 跳过 AI 质检，默认通过 |
| 图片少于 5 张 | 直接拒绝 (score=0) |

### 2.3 辅助分析接口

`SceneAnalyzer` 还提供以下辅助功能：

| 方法 | 用途 |
|---|---|
| `analyze_single_image(path)` | 单张图片分析，返回 score/tags/description |
| `select_best_preview(frames, images_dir)` | 从候选帧中选出最佳封面图 |
| `select_best_image(image_paths)` | 从候选图中选出最适合快速重建的图 |
| `classify_scene_or_object(image_path)` | 判断输入是"场景"还是"单物体" |

---

## 三、配置参数汇总

所有参数通过 `ai_engine/3dgs/src/config.py` 的 `PipelineConfig` 管理，支持环境变量和 TOML 配置文件。

### Mask 生成相关

| 参数 | 默认值 | 说明 |
|---|---|---|
| `shared_model_dir` | `../../models` | 共享模型目录（YOLO、SAM 权重存放位置） |
| `enable_ai` | `False` | 是否启用 AI 分割 |

### 图片质量筛选相关

| 参数 | 环境变量 | TOML 路径 | 默认值 | 说明 |
|---|---|---|---|---|
| `max_images` | `MAX_IMAGES` | `training.max_images` | 300 | 单次处理最大图片数 |
| `min_quality_score` | `MIN_QUALITY_SCORE` | `training.min_quality_score` | 40 | AI 质检最低及格分 |
| `keep_ratio` | — (硬编码) | — | 0.85 | CV 模糊过滤保留比例 |
| `dashscope_api_key` | `DASHSCOPE_API_KEY` | `api.dashscope_api_key` | — | 阿里云 API Key |
| `dashscope_vl_model` | `DASHSCOPE_VL_MODEL` | `api.dashscope_vl_model` | `qwen3-vl-plus` | 视觉语言模型 |
| `dashscope_timeout_seconds` | `DASHSCOPE_TIMEOUT_SECONDS` | `api.dashscope_timeout_seconds` | 45.0 | API 超时（秒） |

### 硬编码阈值（在算法中）

| 阈值 | 位置 | 值 | 说明 |
|---|---|---|---|
| 连通域面积下限 | `cv_algorithms.py:49` | `h * w * 0.005` (0.5%) | 低于此值为噪点 |
| 实心度下限 | `cv_algorithms.py:78` | 0.88 | 低于此值边缘毛糙 |
| 长宽比上限 | `cv_algorithms.py:85` | 4.5 | 超过此值为异常形状 |
| 腐蚀核大小 | `cv_algorithms.py:92` | 3×3 | 边缘收缩约 1 像素 |
| YOLO 置信度 | `ai_segmentor.py:159` | 0.05 | 极低阈值，不漏检 |
| Qwen 采样数 | `ai_segmentor.py:31` | 3 | 均匀采样图片数量 |
| 场景分析采样数 | `scene_analyzer.py:208` | 6 | 随机采样图片数量 |
| data URI 大小上限 | `scene_analyzer.py:22` | 10 MB | 超过则自动压缩 |

---

## 四、数据流全景图

```
视频文件
  │
  ▼
视频抽帧 ──→ 全部帧图片
  │
  ▼
┌─────────────────────────────┐
│  阶段一：CV 模糊过滤         │
│  (image_proc.py)            │
│  3×3 网格拉普拉斯方差        │
│  保留 top 85% + 限 300 张    │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  阶段二：AI 质量评分         │
│  (scene_analyzer.py)        │
│  Qwen-VL-Plus 打分          │
│  score ≥ 40 才通过           │
│  提取 tags/description 等   │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  阶段三：AI Mask 分割        │
│  (ai_segmentor.py)          │
│  Qwen-VL → 提示词           │
│  YOLO World → BBox          │
│  SAM2 → 像素 Mask           │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  阶段四：Mask 清洗质检       │
│  (cv_algorithms.py)         │
│  连通域分析 → 保留最大块     │
│  实心度 ≥ 0.88              │
│  长宽比 ≤ 4.5               │
│  3×3 腐蚀收缩边缘           │
└─────────────────────────────┘
  │
  ▼
透明 PNG Mask + 更新 transforms.json
  │
  ▼
送入 3DGS 训练
```
