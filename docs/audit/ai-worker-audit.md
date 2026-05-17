# AI Worker / 3DGS Pipeline 审计报告

## 审计概要

- 审计时间：2026-04-30
- 审计范围：`ai_engine/` 全部 Python 文件（约 120 个 .py 文件，核心业务约 25 个）
- 发现问题数：22 个
- P0: 4 个 | P1: 5 个 | P2: 7 个 | P3: 6 个

## Pipeline 流程图

```
用户提交任务 (Flutter)
    |
    v
Supabase processing_tasks 表 (status=pending)
    |
    v
WorkerSupervisor 轮询 worker_nodes 表
    |  (desired_state=run → 拉起子进程)
    v
CloudWorker.start() 主循环
    |
    v
_tick() 轮询 processing_tasks (status=pending, limit=1)
    |
    v
_process_task()
    |
    +-- 阶段 A: 锁定任务 → status=processing
    |
    +-- 阶段 B: 下载资源 (Supabase Storage / HTTP URL)
    |     |
    |     +-- single_image_sam3d / single_image_sharp: 多候选路径探测 + 直接 URL 下载
    |     +-- video_3dgs / da3_feed_forward: 下载 video.mp4
    |     +-- sparse2dgs: 尝试 images.zip → 回退 video.mp4
    |
    +-- 阶段 C: 执行 Pipeline (通过 PipelineFactory)
    |     |
    |     +-- video_dual_chain: 快链(SHARP/SAM3D) + 慢链(video_3dgs/da3) 并行
    |     +-- video_3dgs: FFmpeg抽帧 → AI质检 → GLOMAP/DA3解算 → AI分割 → Nerfstudio训练 → PLY导出
    |     +-- single_image_sharp: SHARP推理 → 旋转修正 → RAG分析
    |     +-- single_image_sam3d: SAM3D推理 → 旋转修正 → RAG分析
    |     +-- da3_feed_forward_3dgs: 抽帧 → DA3解算 → 直接反投影导出
    |     +-- sparse2dgs: 解压/抽帧 → COLMAP → Sparse2DGS训练
    |     +-- da3_sugar: 抽帧 → DA3解算 → SuGaR脚本
    |     +-- da3_2dgs: 抽帧 → DA3解算 → 2DGS训练
    |
    +-- 阶段 D: 上传结果 → Supabase Storage + model_assets 表 + 向量化入库
    |
    +-- 阶段 E: status=completed / failed
    |
    +-- finally: 清理临时文件, 重置日志, 恢复 idle 状态
```

---

## P0 级问题

### P0-01: 任务抢单存在竞态条件，多 Worker 可重复处理同一任务

- 严重程度：P0 (数据完整性 / 资源浪费)
- 文件：`ai_engine/3dgs/src/core/worker.py`，第 452-459 行、第 780-783 行

**描述**：CloudWorker 通过 `select("*").eq("status", "pending").limit(1)` 查询待处理任务，随后用 `update({"status": "processing"})` 锁定。这两个操作之间没有任何原子性保证。当多个 Worker 同时轮询时，可能同时读到同一个 pending 任务，导致：

1. 两个 Worker 同时下载同一视频、同时训练，浪费 GPU 资源
2. 两个 Worker 同时写入 Supabase Storage，后写覆盖先写，结果不可控
3. 最终只有一个 Worker 能标记 completed，另一个 Worker 的结果可能丢失

```python
# 第 452-459 行：非原子查询
response = self.supabase.table(self.TABLE_NAME)\
    .select("*").eq("status", "pending").limit(1).execute()
if response.data:
    self._process_task(response.data[0])

# 第 780-783 行：无乐观锁的更新
self.supabase.table(self.TABLE_NAME).update({
    "status": "processing",
    "logs": []
}).eq("id", task_id).execute()
```

**建议**：
- 使用 Supabase RPC 调用一个原子性的"抢单"函数，内部用 `UPDATE ... WHERE status = 'pending' RETURNING *` 实现
- 或在 `processing_tasks` 表增加 `worker_id` + `claimed_at` 字段，结合乐观锁（`WHERE status = 'pending' AND claimed_at IS NULL`）
- 考虑使用 PostgreSQL `SKIP LOCKED` 语义（通过 RPC 封装）

---

### P0-02: mask_guided.py 是未实现的空壳，但已注册到 PipelineFactory

- 严重程度：P0 (运行时崩溃)
- 文件：`ai_engine/3dgs/src/pipelines/mask_guided.py`，第 1-19 行
- 文件：`ai_engine/3dgs/src/core/factory.py`，第 3 行（引用 MultiImagePipeline）

**描述**：`mask_guided.py` 的 `run()` 方法体几乎为空，最后一行引用了未定义变量 `final_ply_path` 和 `metadata`：

```python
class MultiImagePipeline(BasePipeline):
    def run(self, input_path: str, params: Dict[str, Any]):
        self.log("启动多图重建流水线...")
        use_mask = params.get('use_mask', False)
        if use_mask:
            self.log("检测到 Mask 需求，正在运行 SAM 分割...")
            # 调用 src/modules/ai_segmentor.py

        # 运行 Colmap/Glomap
        # 运行 Nerfstudio

        return final_ply_path, metadata  # NameError: final_ply_path 未定义
```

`PipelineFactory` 中注册了 `"multi_image": MultiImagePipeline`，如果用户提交 `task_type="multi_image"` 的任务，必然抛出 `NameError`。

**建议**：要么实现完整逻辑，要么从 `PipelineFactory` 中移除注册并返回"不支持"错误。

---

### P0-03: da3_2dgs_pipeline.py 使用 pty 模块，Windows 上无法运行

- 严重程度：P0 (平台兼容性 / 部署阻断)
- 文件：`ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py`，第 2 行

**描述**：该文件在模块顶层 `import pty`，`pty` 是 Unix-only 模块。当前开发环境为 Windows 11（根据项目上下文），如果此模块被 import（通过 `factory.py` 的延迟导入链），会直接抛出 `ModuleNotFoundError`，可能导致 Worker 启动失败。

```python
import pty    # Unix-only, Windows 上不存在
```

`PipelineFactory` 在模块加载时会 import 所有 Pipeline 类（第 1-8 行），这意味着 `DA3TwoDGSPipeline` 会被 import，从而触发 `import pty` 失败。

**建议**：
- 改用 `subprocess.Popen` + `select` 实现跨平台的实时输出读取
- 或将 `import pty` 移到 `_run_cmd` 方法内部，仅在 Unix 环境下使用

---

### P0-04: knowledge_base.py 顶层硬性 import openai/supabase，缺失依赖时整个 3dgs 包无法加载

- 严重程度：P0 (模块加载失败)
- 文件：`ai_engine/3dgs/src/modules/knowledge_base.py`，第 7-8 行

**描述**：`knowledge_base.py` 在顶层无保护地导入：

```python
from openai import OpenAI      # 如果 openai 未安装，直接 ImportError
from supabase import Client     # 如果 supabase 未安装，直接 ImportError
```

`worker.py` 在初始化时导入 `KnowledgeBase`（第 27 行），这意味着如果运行环境中缺少 `openai` 包，整个 CloudWorker 无法启动。

对比 `rag_memory.py`（第 8-11 行）和 `scene_analyzer.py`（第 15-18 行）都做了 try-catch 保护，但 `knowledge_base.py` 遗漏了。

**建议**：改为 try-catch 导入模式，与其他模块保持一致。

---

## P1 级问题

### P1-01: 多处裸 except 吞掉所有异常，掩盖真实错误

- 严重程度：P1 (可调试性 / 可靠性)
- 涉及文件与行号：
  - `ai_engine/3dgs/src/modules/ai_segmentor.py`，第 218 行：`_get_prompt` 裸 except
  - `ai_engine/3dgs/src/utils/geometry.py`，第 74 行：`analyze_and_calculate_adaptive_collider` 裸 except
  - `ai_engine/3dgs/src/core/worker.py`，第 483-484 行：`_parse_task_params` 中 `task_params = {}` 静默吞错
  - `ai_engine/3dgs/src/core/worker.py`，第 926-927 行：`except: pass` 吞掉日志记录失败

**描述**：裸 `except:` 会捕获 `SystemExit`、`KeyboardInterrupt` 等不应被捕获的异常，且不记录任何信息。

```python
# ai_segmentor.py:218
def _get_prompt(self):
    try:
        prompt = get_central_object_prompt(self.images_dir)
        return prompt if prompt else "central object"
    except:                              # 裸 except，吞掉一切
        return "central object"

# geometry.py:74
    except:                              # 裸 except，几何分析失败时返回空
        return [], "unknown"

# worker.py:926-927
    except:                              # 裸 except
        pass
```

当 `geometry.py` 的碰撞器分析失败时，Nerfstudio 会使用空参数启动，可能导致训练几何错误但不会有任何提示。当 `task_params` 解析失败时，所有任务参数将丢失。

**建议**：所有 `except:` 改为 `except Exception as e:` 并记录日志。

---

### P1-02: NerfstudioEngine.train / export 无超时限制，GPU 任务可能无限挂起

- 严重程度：P1 (资源泄漏 / 任务死锁)
- 文件：`ai_engine/3dgs/src/modules/nerf_engine.py`，第 139 行、第 154 行
- 文件：`ai_engine/3dgs/src/modules/glomap_runner.py`，第 278 行
- 文件：`ai_engine/3dgs/src/modules/sharp_engine.py`，第 57 行

**描述**：所有外部训练进程调用均未设置 `timeout` 参数：

```python
# nerf_engine.py:139
subprocess.run(cmd, check=True, env=self.env)  # 无 timeout

# nerf_engine.py:154
subprocess.run(["ns-export", "gaussian-splat", ...], check=True, env=self.env)  # 无 timeout

# glomap_runner.py:278
process = subprocess.Popen(cmd, ...)  # 无 timeout
```

如果 CUDA OOM 或 Nerfstudio 内部死锁，训练进程会永远挂起。Worker 将永远卡在这个任务上，无法处理后续任务。虽然 Supervisor 有 interrupt 机制，但依赖人工干预。

**建议**：
- 为每个 subprocess 调用设置合理的 timeout（如训练 4 小时、导出 30 分钟）
- 超时后自动终止进程并标记任务 failed
- 在 `task_params` 中允许用户自定义超时时间

---

### P1-03: NerfstudioEngine.export 将结果写死复制到模块目录下 results/，与 Worker 上传逻辑脱节

- 严重程度：P1 (输出路径不一致)
- 文件：`ai_engine/3dgs/src/modules/nerf_engine.py`，第 194-199 行

**描述**：

```python
# nerf_engine.py:194-199
results_dir = Path(__file__).parent / "results"   # src/modules/results/
results_dir.mkdir(exist_ok=True)
target_path = results_dir / f"{self.cfg.project_name}.ply"
shutil.copy2(str(final_ply), str(target_path))
return target_path
```

导出方法返回的路径是 `src/modules/results/{project_name}.ply`，但 Worker 的 `_run_pipeline_once`（worker.py:636）检查的是 pipeline 返回的路径是否存在。如果 `shutil.copy2` 因磁盘满或其他原因失败，路径不存在会直接导致任务失败。

更重要的是，这个 `results/` 目录是全局共享的，多个任务并行时文件名可能冲突（虽然当前是单 Worker 模式，但 Supervisor 架构允许未来多 Worker）。

**建议**：返回工作目录内的原始 PLY 路径即可，不需要额外复制。

---

### P1-04: PipelineConfig.__post_init__ 将 API Key 写入环境变量，存在泄漏风险

- 严重程度：P1 (安全)
- 文件：`ai_engine/3dgs/src/config.py`，第 362-365 行

**描述**：

```python
# config.py:362-365
if self.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)
if self.dashscope_api_key:
    os.environ.setdefault("DASHSCOPE_API_KEY", self.dashscope_api_key)
```

每次创建 `PipelineConfig` 实例时，都会将 API Key 写入进程环境变量。这些环境变量可以通过 `/proc/{pid}/environ`（Linux）被其他进程读取，也可能会被日志框架意外打印。

此外，`PipelineConfig()` 在多个地方被无参数调用（如 `scene_analyzer.py:104`、`pipeline.py:54`），每次都会触发 `__post_init__`，重复写入环境变量。

**建议**：API Key 仅在需要时从 Config 对象读取，不要写入全局环境变量。

---

### P1-05: demo/mysupabase/worker.py 使用错误的环境变量名且包含硬编码凭证

- 严重程度：P1 (安全 / 功能失效)
- 文件：`ai_engine/demo/mysupabase/worker.py`，第 9-10 行

**描述**：

```python
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # 错误的 key 名
BUCKET_NAME = "braindance-assets"                   # 硬编码
```

该 Demo Worker 使用 `SUPABASE_SERVICE_KEY` 而非 `SUPABASE_KEY`，与主 Worker（worker.py:66 使用 `cfg.supabase_key`）不一致。如果有人按此 demo 配置，将无法正常连接 Supabase。

此外，demo 中的 `dummy_result = b"Ply model data header..."` 会将假数据上传到正式 Storage bucket，污染生产数据。

**建议**：demo 文件应标注为示例代码，使用明确的示例环境变量名，或在文档中说明与生产代码的差异。

---

## P2 级问题

### P2-01: worker.py 中 CACHE_DIR 硬编码为 ~/braindance_workspace，不支持多实例隔离

- 严重程度：P2 (可扩展性)
- 文件：`ai_engine/3dgs/src/core/worker.py`，第 102-103 行

**描述**：

```python
self.CACHE_DIR = Path.home() / "braindance_workspace"
self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

所有 Worker 实例共享同一个缓存目录。虽然当前场景下 scene_id 不同会使用不同子目录，但 `_dual_chain_frames`、`fast_chain`、`slow_chain` 等中间目录如果 scene_id 意外相同（如重复提交），会产生冲突。

**建议**：使用临时目录或在路径中加入 worker_id 隔离。

---

### P2-02: GlomapRunner GLOMAP 环境隔离白名单过于激进，可能丢失关键环境变量

- 严重程度：P2 (环境兼容性)
- 文件：`ai_engine/3dgs/src/modules/glomap_runner.py`，第 246-266 行

**描述**：

```python
if is_glomap:
    clean_env = {
        "PATH": ...,
        "LD_LIBRARY_PATH": ...,
        "HOME": ...,
        "USER": ...,
        "LANG": ...,
        "SHELL": ...,
        "TERM": ...
    }
```

GLOMAP 的环境白名单只保留了极少数变量。如果用户通过 `.env` 设置了 `HTTP_PROXY`、`NO_PROXY`、`CUDA_VISIBLE_DEVICES`（非 CUDA/NVIDIA 前缀）等变量，GLOMAP 将丢失这些配置，可能无法正确下载模型或访问 GPU。

**建议**：基于黑名单（排除已知冲突变量如 `PYTHONPATH`、`LD_PRELOAD`、`CONDA_*`）而非白名单。

---

### P2-03: Supervisor 不向 worker_nodes 表推送自身状态

- 严重程度：P2 (运维可观测性)
- 文件：`ai_engine/3dgs/src/core/supervisor.py`

**描述**：`WorkerSupervisor` 定期从 `worker_nodes` 表读取 `desired_state`，但从不写入自身状态（如 "supervisor_running"、"child_started"）。如果 Supervisor 进程崩溃或被 kill，Dashboard/运维人员无法区分"Supervisor 离线"和"Supervisor 在线但子 Worker 离线"。

**建议**：Supervisor 也应向 `worker_nodes` 写入心跳，或使用独立的 `supervisor_nodes` 表。

---

### P2-04: _run_cmd 中 FFmpeg stderr 被 DEVNULL 吞掉，抽帧失败时无法诊断原因

- 严重程度：P2 (可调试性)
- 涉及文件：
  - `ai_engine/3dgs/src/pipelines/video_3dgs.py`，第 128-134 行
  - `ai_engine/3dgs/src/pipelines/da3_feed_forward_pipeline.py`，第 74-80 行
  - `ai_engine/3dgs/src/pipelines/da3_sugar_pipeline.py`，第 59-76 行
  - `ai_engine/3dgs/src/core/local_runner.py`，第 28-33 行

**描述**：所有 FFmpeg 抽帧调用都使用 `stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL`，如果 FFmpeg 失败，只能看到 `"FFmpeg 抽帧失败: Command ..."` 的错误码，无法知道是视频损坏、编解码器缺失还是磁盘满。

```python
subprocess.run([
    "ffmpeg", "-y", "-i", str(dest_video_path),
    ...
    str(temp_dir / "frame_%05d.jpg")
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
```

对比 `pipeline.py`（旧版）的实现（第 76-90 行）会捕获 stderr 并打印，但新版 pipeline 丢失了这个改进。

**建议**：捕获 stderr，在失败时打印 FFmpeg 的错误输出。

---

### P2-05: SceneAnalyzer.select_best_image 每次创建新的 OpenAI 客户端，未复用

- 严重程度：P2 (性能 / 资源管理)
- 文件：`ai_engine/3dgs/src/modules/scene_analyzer.py`，第 509 行

**描述**：

```python
# scene_analyzer.py:509
client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # 每次新建客户端
```

`select_best_image` 方法（快链帧选择）每次调用都创建一个新的 `OpenAI` 客户端实例，而同类方法 `run`（第 249 行）和 `analyze_single_image`（第 326 行）使用了 `self._get_client()` 复用实例。新建客户端会创建新的 HTTP 连接池，在高并发或多次调用场景下浪费资源。

**建议**：统一使用 `self._get_client()` 获取客户端实例。

---

### P2-06: worker.py _tick 中 no-error 和 no-task 分支均 sleep 但不区分，可能导致任务饥饿

- 严重程度：P2 (调度效率)
- 文件：`ai_engine/3dgs/src/core/worker.py`，第 462-466 行

**描述**：

```python
else:
    self._set_worker_state(status="idle")
    time.sleep(3)     # 无任务时固定 3 秒
    print(".", end="", flush=True)
```

无任务时固定等待 3 秒。如果任务队列中积压了多个 pending 任务，每个任务之间至少有 3 秒空闲期。虽然单 Worker 模式下影响不大，但如果未来支持多任务调度，这个固定间隔会成为瓶颈。

**建议**：采用退避策略——连续空轮询时逐步增加 sleep 时间（如 1s → 2s → 4s → 最大 10s），发现任务时重置。

---

### P2-07: 重复 import os 语句

- 严重程度：P2 (代码质量)
- 涉及文件：
  - `ai_engine/3dgs/src/modules/ai_segmentor.py`，第 6-7 行：连续两次 `import os`
  - `ai_engine/3dgs/src/core/pipeline_base.py`，第 1 行和第 6 行：连续两次 `import os`

**描述**：同一文件中重复导入 `os` 模块，虽然 Python 不会重复加载模块，但暴露了代码缺少 lint 规范。

**建议**：删除重复导入，配置 Ruff/flake8 规则自动检测。

---

## P3 级问题

### P3-01: worker.py 本地缓存文件名使用 scene_id 直接拼接，存在命名冲突风险

- 严重程度：P3 (边界条件)
- 文件：`ai_engine/3dgs/src/core/worker.py`，第 793 行、第 818 行

**描述**：

```python
input_path = self.CACHE_DIR / f"{scene_id}.png"     # 单图
input_path = self.CACHE_DIR / f"{scene_id}.mp4"      # 视频
```

如果两个不同用户的 scene_id 相同（虽然概率低），会在同一路径产生文件冲突。

**建议**：使用 `{user_id}/{scene_id}` 子目录隔离。

---

### P3-02: _detect_total_vram_gb 仅检查 GPU 0，不支持多 GPU 环境

- 严重程度：P3 (多 GPU 限制)
- 文件：`ai_engine/3dgs/src/core/worker.py`，第 645-651 行
- 文件：`ai_engine/3dgs/src/core/local_runner.py`，第 14-22 行

**描述**：

```python
def _detect_total_vram_gb(self) -> float:
    import torch
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory  # 硬编码 device 0
```

始终检查 GPU 0，但 `CUDA_VISIBLE_DEVICES` 可能已将物理 GPU 3 映射为逻辑 GPU 0。虽然大多数场景不会有问题，但当 `CUDA_VISIBLE_DEVICES` 为空且 `GPU_INDEX` 指向非 0 的 GPU 时，检测结果不准确。

---

### P3-03: compress_ply_to_splat 缺少对 Sparse2DGS 二维尺度字段的完备校验

- 严重程度：P3 (格式兼容性)
- 文件：`ai_engine/3dgs/src/utils/ply_utils.py`，第 91-107 行

**描述**：代码对只有 2 个 scale 字段的 Sparse2DGS PLY 做了特殊处理（补第三个轴），但如果恰好有 `scale_0`、`scale_2` 而缺少 `scale_1`（脏数据），排序后的 `scale_fields` 可能产生错误的索引匹配。

---

### P3-04: da3_2dgs_pipeline.py 中 pty 文件描述符泄漏风险

- 严重程度：P3 (资源泄漏)
- 文件：`ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py`，第 343-386 行

**描述**：使用 `pty.openpty()` 创建伪终端，如果 `subprocess.Popen` 抛出异常，`master_fd` 和 `slave_fd` 可能未被正确关闭。虽然 finally 块关闭了 `master_fd`，但 `slave_fd` 在 Popen 成功时才关闭（第 353 行 `os.close(slave_fd)`），Popen 失败时泄漏。

---

### P3-05: image_proc.py 的智能过滤结果计数方式不精确

- 严重程度：P3 (日志准确性)
- 文件：`ai_engine/3dgs/src/modules/image_proc.py`，第 75 行

**描述**：

```python
self.log_callback(f"清洗结束，剩余 {len(list(image_dir.glob('*')))} 张。")
```

`glob('*')` 会匹配所有文件（包括非图片文件如 `.txt`、`.DS_Store`），导致显示的"剩余"数量可能比实际图片数量多。

---

### P3-06: spatial_anchor.py 的 webgl_poses 保存时未同步更新 output_data 中的 frames

- 严重程度：P3 (数据一致性)
- 文件：`ai_engine/3dgs/src/modules/spatial_anchor.py`，第 160-171 行

**描述**：

```python
for frame in webgl_poses:
    frame['image_url'] = f"{user_id}/{scene_id}/output/images/{frame['id']}"

# 重新保存 webgl_poses.json
with open(webgl_poses_path, 'w') as f:
    json.dump(output_data, f, indent=4)  # 使用的是旧的 output_data，frames 已更新但 ...
```

`output_data` 是在第 64-72 行创建的字典，其 `frames` 字段引用了 `webgl_poses` 列表。由于 Python 的引用语义，修改 `webgl_poses` 中的元素会自动反映到 `output_data["frames"]` 中，所以目前实际不会出错。但这种隐式依赖很脆弱，如果有人重构为 `output_data = { ... "frames": list(webgl_poses) ... }`（浅拷贝），就会产生 bug。

---

## 关键路径分析

### 演示最可能失败的路径

1. **GLOMAP/COLMAP 位姿解算失败** (概率最高)
   - 原因：GPU 驱动版本不匹配、CUDA 库路径被 Conda 污染、视频内容不适合（纯色/低纹理/快速运动）
   - 影响：任务直接 failed，用户只看到"解算失败"，无详细信息
   - 缓解：已有 GPU→CPU 降级和 GLOMAP→COLMAP mapper 回退，但回退路径的日志不够充分

2. **Qwen-VL API 调用超时或返回非 JSON 格式** (概率较高)
   - 原因：网络抖动、API 限流、模型输出不稳定
   - 影响：`scene_analyzer.run()` 返回 `(True, 60, "Analysis Error (Default Pass)")`，跳过质检继续；但如果 `select_best_image` 失败，快链可能选择质量差的帧
   - 缓解：大部分路径有容错，但超时 45 秒可能让整个 pipeline 卡住较长时间

3. **Supabase Storage 上传大文件超时** (概率中等)
   - 原因：PLY 文件可能达到数百 MB，网络带宽有限
   - 影响：Worker 虽然有默认 300 秒超时，但 `_upload_and_upsert_assets` 没有重试机制（对比 `upload_and_record` 有 3 次重试）
   - 缓解：上传失败后会尝试写入 model_assets 失败日志，但任务仍标记为 completed，导致用户看到"完成"但无模型

4. **Nerfstudio ns-train 进程 OOM 崩溃** (概率中等)
   - 原因：300 张 1920px 图片 + 30000 迭代 + 高分辨率纹理，GPU 显存不足
   - 影响：subprocess.run 抛 CalledProcessError，但错误信息只有 exit code
   - 缓解：无自动重试或降级（如降低分辨率后重试）

5. **Sparse2DGS 的 CLMVSNet checkpoint 不存在** (首次部署必遇)
   - 原因：checkpoint 路径硬编码了 `/ltx-data/Sparse2DGS/model_clmvsnet.ckpt` 和 `/home/ltx/projects/Sparse2DGS/model_clmvsnet.ckpt`
   - 影响：FileNotFoundError 且错误信息清晰，但需要人工干预
   - 缓解：已提供详细的搜索路径列表

---

## 建议新建 Issue 清单

| 编号 | 标题 | 优先级 | 标签 |
|------|------|--------|------|
| 1 | [P0] 实现原子性任务抢单，防止多 Worker 竞态 | 高 | backend, bug |
| 2 | [P0] 移除或实现 mask_guided.py / MultiImagePipeline | 高 | backend, cleanup |
| 3 | [P0] da3_2dgs_pipeline.py 替换 Unix-only pty 为跨平台方案 | 高 | backend, platform |
| 4 | [P0] knowledge_base.py 添加 try-catch 保护 openai/supabase 导入 | 高 | backend, reliability |
| 5 | [P1] 消除所有裸 except，统一使用 except Exception as e + 日志 | 中 | backend, quality |
| 6 | [P1] 为所有 subprocess 调用添加 timeout 参数 | 中 | backend, reliability |
| 7 | [P1] 修复 NerfstudioEngine.export 不必要的 results/ 复制 | 中 | backend, refactor |
| 8 | [P1] 移除 PipelineConfig.__post_init__ 中的环境变量注入 | 中 | backend, security |
| 9 | [P1] 清理 demo worker 代码，修正环境变量名 | 低 | demo, cleanup |
| 10 | [P2] FFmpeg 调用捕获 stderr 用于诊断 | 低 | backend, debug |
| 11 | [P2] 统一 OpenAI 客户端初始化，复用 _get_client() | 低 | backend, performance |
| 12 | [P2] Supervisor 添加状态上报到 worker_nodes 表 | 低 | backend, ops |
| 13 | [P3] 清理重复 import 语句，配置 linter 规则 | 低 | quality |
| 14 | [P3] VRAM 检测支持 CUDA_VISIBLE_DEVICES 映射 | 低 | backend, multi-gpu |

---

## 附录：审计覆盖文件清单

### 核心业务（已完整阅读）
- `ai_engine/3dgs/main.py` - 入口
- `ai_engine/3dgs/src/config.py` - 配置管理
- `ai_engine/3dgs/src/core/supervisor.py` - Supervisor 进程
- `ai_engine/3dgs/src/core/worker.py` - CloudWorker
- `ai_engine/3dgs/src/core/pipeline_base.py` - Pipeline 基类
- `ai_engine/3dgs/src/core/pipeline.py` - 旧版 Pipeline
- `ai_engine/3dgs/src/core/factory.py` - Pipeline 工厂
- `ai_engine/3dgs/src/core/local_runner.py` - 本地运行模式
- `ai_engine/3dgs/src/pipelines/video_3dgs.py` - 视频 3DGS Pipeline
- `ai_engine/3dgs/src/pipelines/single_image_sharp.py` - SHARP 单图
- `ai_engine/3dgs/src/pipelines/single_image_sam3d.py` - SAM3D 单图
- `ai_engine/3dgs/src/pipelines/sparse2dgs.py` - Sparse2DGS 多图
- `ai_engine/3dgs/src/pipelines/da3_feed_forward_pipeline.py` - DA3 前馈
- `ai_engine/3dgs/src/pipelines/da3_sugar_pipeline.py` - DA3+SuGaR
- `ai_engine/3dgs/src/pipelines/da3_2dgs_pipeline.py` - DA3+2DGS
- `ai_engine/3dgs/src/pipelines/mask_guided.py` - Mask 引导（空壳）
- `ai_engine/3dgs/src/modules/nerf_engine.py` - Nerfstudio 引擎
- `ai_engine/3dgs/src/modules/glomap_runner.py` - GLOMAP 解算
- `ai_engine/3dgs/src/modules/da3_runner.py` - DA3 解算
- `ai_engine/3dgs/src/modules/scene_analyzer.py` - AI 场景分析
- `ai_engine/3dgs/src/modules/ai_segmentor.py` - AI 语义分割
- `ai_engine/3dgs/src/modules/sharp_engine.py` - SHARP 推理
- `ai_engine/3dgs/src/modules/rag_memory.py` - RAG 记忆
- `ai_engine/3dgs/src/modules/knowledge_base.py` - 知识库
- `ai_engine/3dgs/src/modules/spatial_anchor.py` - 空间语义锚点
- `ai_engine/3dgs/src/modules/image_proc.py` - 图像预处理
- `ai_engine/3dgs/src/utils/ply_utils.py` - PLY 工具
- `ai_engine/3dgs/src/utils/geometry.py` - 几何算法
- `ai_engine/3dgs/src/utils/common.py` - 通用工具

### Demo/辅助（已审阅）
- `ai_engine/demo/mysupabase/worker.py` - Demo Worker

### 未深入审计（脚本工具，非核心链路）
- `ai_engine/finetune_qwen3/scripts/` - 约 40 个微调/评测脚本
- `ai_engine/demo/` - 演示脚本
- `ai_engine/3dgs/src/libs/` - 第三方库（ml-sharp、sam-3d-objects）
- `ai_engine/3dgs/tests/` - 测试文件
