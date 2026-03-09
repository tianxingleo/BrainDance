# Time Peeling V1 实施计划与技术设计

> 文档目标：完整记录 Time Peeling V1 的实施范围、架构设计、接口契约、算法细节、容错策略、测试方案与上线步骤，作为后续开发、联调与答辩材料。

---

## 1. 背景与目标

### 1.1 背景
BrainDance 已具备以下能力：
- 3D 模型异步生成（Worker + Supabase）
- 空间语义检索（`search-models`）
- 单模型查看与视角跳转（WebGL Viewer）

但“时光剥离（Time Peeling）”在 V1 前缺少闭环，主要缺口是：
- 同一物理空间的多次扫描缺乏统一逻辑空间抽象
- 历史切片通常被新结果覆盖，不便回看
- Viewer 仅单模型渲染，无法双切片叠加

### 1.2 V1 目标
在同一物理空间内，支持两个时间切片的对齐叠加：
- 用户可选择“当前切片 + 历史切片”进入 Viewer
- 支持透明度滑动与显示模式切换（仅当前 / 仅历史 / 叠加）
- 保持旧链路可用（原 Recall / 搜索 / 单模型查看不破坏）

### 1.3 成功标准
- 数据层：同空间多次扫描形成连续 `space_captures`，历史不覆盖
- 服务层：`time-peeling-view` 可返回双模型渲染配置
- 算法层：对齐流程产出 `alignment_matrix` 与 `alignment_score`
- 客户端：时间轴入口可达，双模型可叠加渲染且可交互控制

### 1.4 明确不做（V1 边界）
- 不做多层（>2）并发渲染
- 不做自动变化区域分割高亮
- 不做手动对齐编辑器（仅返回 `needs_manual_align` 状态）

---

## 2. 术语与核心概念

- `memory_spaces`：逻辑空间，表示“同一物理地点”的抽象容器
- `space_captures`：空间中的时间切片，每次采集/建模生成一条记录
- `model_assets`：模型资产，V1 增加 `space_id/capture_id/captured_at` 维度
- `alignment_matrix`：4x4 变换矩阵（右切片对齐到基准切片坐标系）
- `alignment_score`：对齐质量分数（当前实现基于 ICP fitness）
- `needs_manual_align`：对齐可信度不足，需要人工校准（V2）

---

## 3. 总体架构设计

### 3.1 分层
1. 数据层（PostgreSQL / Supabase）
- 新增空间与切片表
- 扩展资产表，建立“空间->切片->模型”链路

2. 计算层（Python Worker）
- 任务处理时创建/绑定空间与切片
- 建模完成后执行“粗配准 + 精配准”，回写矩阵/分数/状态

3. 服务层（Edge Functions + RPC）
- `get_space_captures(space_id)`：时间轴数据
- `time-peeling-view`：双切片渲染配置编排
- `search-models`：返回空间与切片维度用于跳转

4. 客户端层（Flutter + WebGL）
- Recall 增加“时光剥离”入口
- 新增 TimePeelingPage 选择空间与切片
- Viewer 增加双模型桥接、透明度与模式切换

### 3.2 时序（主链路）
1. 用户提交任务（可带 `space_id`）
2. Worker 锁定任务，校验/创建 `memory_spaces`
3. Worker 创建 `space_captures(status=processing)`
4. 建模完成，上传模型到 storage
5. Worker 执行对齐，计算 `alignment_matrix/alignment_score`
6. Worker 回写 `model_assets` 与 `space_captures` 状态
7. 用户在 TimePeelingPage 选择两个切片
8. 调用 `time-peeling-view` 返回渲染 payload
9. WebGL 双模型加载并叠加显示

---

## 4. 数据模型设计（V1）

### 4.1 新增表：`memory_spaces`
字段：
- `id uuid pk`
- `user_id text not null`
- `title text`
- `created_at timestamptz`
- `updated_at timestamptz`

语义：
- 一个 `memory_space` 对应一个用户下的“同一物理空间”

### 4.2 新增表：`space_captures`
字段：
- `id uuid pk`
- `space_id uuid fk -> memory_spaces.id`
- `user_id text`
- `scene_id text`
- `captured_at timestamptz`
- `status text`（`processing/completed/needs_manual_align/failed`）
- `align_to_capture_id uuid`（指向基准切片）
- `alignment_matrix jsonb`（默认单位矩阵）
- `alignment_score double precision`
- `created_at timestamptz`

语义：
- 每次采集生成一个 capture，不覆盖历史

### 4.3 扩展表：`model_assets`
新增字段：
- `space_id uuid`
- `capture_id uuid`
- `captured_at timestamptz`

保留：
- 现有 `scene_id` 仍保留，保证旧链路兼容

### 4.4 扩展表：`processing_tasks`
新增字段：
- `space_id uuid`

用途：
- 上游可指定写入已有空间；不传则 Worker 自动创建

### 4.5 索引设计
- `space_captures(space_id, captured_at desc)`：时间轴读取
- `model_assets(capture_id)`：切片->模型映射
- `model_assets(space_id, captured_at desc)`：空间资产列表

### 4.6 RLS 设计
原则：全部基于 `auth.uid()::text = user_id`
- `memory_spaces`：CURD 全部限制本用户
- `space_captures`：CURD 全部限制本用户
- 跨表查询（RPC/Edge）仅返回本用户数据

---

## 5. API 与接口契约

### 5.1 RPC：`get_space_captures(p_space_id uuid)`
返回字段：
- `capture_id`
- `scene_id`
- `captured_at`
- `status`
- `model_url`（当前实现返回 `ply_path`）
- `alignment_matrix`
- `alignment_score`

用途：
- TimePeelingPage 渲染时间轴切片列表

### 5.2 Edge Function：`time-peeling-view`
路径：`POST /functions/v1/time-peeling-view`

入参：
```json
{
  "space_id": "uuid",
  "left_capture_id": "uuid",
  "right_capture_id": "uuid"
}
```

出参：
```json
{
  "success": true,
  "space_id": "...",
  "base_capture_id": "...",
  "overlay_capture_id": "...",
  "base_model": "https://...",
  "overlay_model": "https://...",
  "overlay_alignment_matrix": [16 numbers],
  "alignment_score": 0.78,
  "default_alpha": 0.5,
  "initial_pose": [16 numbers] | null
}
```

校验逻辑：
- 必须登录（Bearer JWT）
- `space_id` 必须属于当前用户
- 两个 capture 必须同属该空间且属于当前用户
- 两个 capture 都必须有可渲染模型路径

### 5.3 搜索接口字段增强
`match_memory_poses` 已扩展返回：
- `space_id`
- `capture_id`
- `captured_at`

目的：
- 从搜索结果可直接跳转时光剥离页并定位到切片

---

## 6. Worker 任务链路设计

### 6.1 空间与切片生命周期
在 `CloudWorker._process_task` 中新增：
1. `_ensure_space(task, user_id, scene_id)`
- 若任务自带有效 `space_id`：直接复用
- 否则自动创建 `memory_spaces` 并回填 `processing_tasks.space_id`

2. `_create_capture(...)`
- 创建 `space_captures(status=processing)`
- 记录本次 `captured_at`

### 6.2 对齐流程
入口：`_compute_alignment(...)`

基准选择：
- 当前空间内最近完成的历史 capture（排除当前 `scene_id`）

流程：
1. 读取当前模型与基准模型
2. 粗配准：FPFH + RANSAC（Open3D）
3. 精配准：ICP Point-to-Plane（Open3D）
4. 产出：`alignment_matrix` + `alignment_score`

降级规则：
- 无历史切片：单位矩阵，`score=1.0`，`status=completed`
- 不是 `.ply`：跳过配准，`status=needs_manual_align`
- Open3D 不可用或异常：单位矩阵，`score=0`，`needs_manual_align`

阈值：
- `TIME_PEELING_ALIGNMENT_THRESHOLD`（默认 `0.6`）
- `score >= threshold` => `completed`
- 否则 => `needs_manual_align`

### 6.3 回写策略
- `model_assets` 回写：`space_id/capture_id/captured_at`
- `model_assets.meta_info` 增补：
  - `alignment_matrix`
  - `alignment_score`
  - `align_to_capture_id`
- `space_captures` 回写：
  - `status`
  - `align_to_capture_id`
  - `alignment_matrix`
  - `alignment_score`

### 6.4 兼容性说明
当前实现仍使用：
- `model_assets.upsert(..., on_conflict="scene_id")`

影响：
- 若完全复用同一 `scene_id`，仍会覆盖同 `scene_id` 资产行
- V1 通过“新任务默认新 `scene_id`”与 `space_captures` 保留时间序列
- V2 建议：引入 `capture_id` 唯一资产键，彻底去除 `scene_id` 覆盖语义

---

## 7. 客户端与渲染设计

### 7.1 Flutter 数据模型
- `SpaceCapture`
- `TimePeelingPayload`

作用：
- 统一解析 RPC/Function 返回，降低页面层杂糅逻辑

### 7.2 TimePeelingPage
功能：
1. 加载用户 `memory_spaces`
2. 调 `get_space_captures` 拉取时间切片
3. 选择基准/叠加切片
4. 调 `time-peeling-view` 获取渲染 payload
5. 打开 WebGLViewerPage 并传入 `timePeelingPayload`

### 7.3 Recall 入口
- 顶栏新增“时光剥离”按钮
- 搜索结果若含 `space_id`，可直接按该空间打开时光剥离页

### 7.4 Viewer 桥接协议
新增：
- `window.loadTimePeelFromFlutter(payload)`
- `window.setTimePeelAlpha(alpha)`
- `window.setTimePeelMode(mode)`

模式：
- `blend`
- `base`
- `overlay`

### 7.5 WebGL 实现策略
#### 当前实现（V1 可落地）
受限于本机构建环境，采用运行时 patch：
- `app/assets/webgl/time_peeling_patch.js`
- 在 `index.html` 注入脚本
- 复用现有 `window.viewer`，先加载 base，再加载 overlay
- overlay 应用 `alignment_matrix` 分解后的 TRS

#### 源码实现（后续可正式构建替换）
已在 `GaussianViewer.vue` 增加对应接口与逻辑，待 Node20+ 环境统一构建后可切换到源码版

---

## 8. 容错与降级设计

### 8.1 对齐失败
- capture 标记 `needs_manual_align`
- UI 仍可进入查看，但建议提示“需手动校准（V2）”

### 8.2 模型缺失
- `time-peeling-view` 返回 404 + 明确错误信息

### 8.3 权限失败
- `time-peeling-view` 对空间归属与切片归属双重校验

### 8.4 运行环境限制
- 若无法重建 Vue bundle，运行时 patch 保证功能可用
- 后续在标准 Node20+ 环境切换为源码构建产物

---

## 9. 性能设计与注意事项

### 9.1 渲染
- V1 限制双切片，控制显存与排序压力
- 叠加模式采用 opacity 调整，避免频繁重建场景

### 9.2 对齐计算
- 采用降采样（voxel down sample）
- 将配准控制在可接受时延，不阻塞主流程

### 9.3 网络
- `time-peeling-view` 返回可直接访问的 public URL
- Flutter WebView 通过本地代理转发 HTTPS，兼容证书场景

---

## 10. 测试设计

### 10.1 数据与权限
- 迁移执行通过：表、字段、索引、函数存在
- RLS 验证：跨用户不可读写

### 10.2 Worker
- 无历史切片：应生成单位矩阵并 `completed`
- 历史切片可用且模型为 PLY：产出非空矩阵与分数
- 分数不足：状态为 `needs_manual_align`

### 10.3 API
- `get_space_captures` 返回按时间降序
- `time-peeling-view` 参数合法返回 payload，不合法返回明确错误

### 10.4 客户端
- TimePeelingPage 列表加载、切片选择、跳转 Viewer
- alpha 滑条、模式切换即时生效
- 搜索结果可带 `space_id/capture_id` 跳转

### 10.5 回归
- 原 Recall 单模型浏览不受影响
- 原 `search-models` 兼容旧客户端解析
- 原 Worker 流程（视频/单图）可继续跑通

---

## 11. 上线与回滚策略

### 11.1 上线步骤
1. 先部署数据库迁移
2. 部署 `time-peeling-view` Function
3. 部署 Worker（含新对齐模块）
4. 部署 Flutter 客户端
5. 观察日志与数据质量

### 11.2 回滚策略
- 服务层回滚：回退 Function 即可禁用新入口
- 客户端回滚：隐藏 TimePeeling 入口
- 数据层回滚：不建议直接删表，采用“停用入口 + 保留数据”

---

## 12. 已知风险与 V2 方向

### 12.1 已知风险
1. 资产 upsert 仍依赖 `scene_id`，存在覆盖语义
2. 非 PLY（splat/ksplat）对齐能力有限，常走降级路径
3. 运行时 patch 与源码 bundle 双轨，需后续统一

### 12.2 V2 建议
1. 引入 `capture_id` 作为资产唯一主锚，彻底去覆盖化
2. 提供手动校准工具（锚点式）
3. 增加变化区域检测/高亮
4. 支持多切片时间播放与关键帧插值

---

## 13. 对应代码清单（便于定位）

- 数据迁移：`supabase/migrations/20260309223000_time_peeling_v1.sql`
- Edge Function：`supabase/functions/time-peeling-view/index.ts`
- Worker：`ai_engine/3dgs/src/core/worker.py`
- 对齐模块：`ai_engine/3dgs/src/modules/time_peeling_aligner.py`
- 记忆模块扩展：
  - `ai_engine/3dgs/src/modules/knowledge_base.py`
  - `ai_engine/3dgs/src/modules/rag_memory.py`
- Flutter：
  - `app/lib/models/time_peeling_models.dart`
  - `app/lib/pages/time_peeling.dart`
  - `app/lib/pages/recall.dart`
  - `app/lib/pages/webgl_viewer.dart`
- WebGL 运行时 patch：
  - `app/assets/webgl/time_peeling_patch.js`
  - `app/assets/webgl/index.html`
- WebGL 源码扩展：
  - `3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue`

---

## 14. 结论

Time Peeling V1 已形成“数据结构 -> 任务生产 -> 对齐计算 -> 渲染编排 -> 客户端交互”的最小闭环，并保持对旧链路的兼容。V1 重点解决了“可用性与闭环”，V2 再提升“精细度与编辑能力”。

