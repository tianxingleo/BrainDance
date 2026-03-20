# src/core/worker.py
# 功能：实现云端工作者逻辑，监听任务队列并处理3D重建任务
# 实现：通过Supabase轮询任务，下载资源，执行pipeline，上传结果
# 逻辑：1. 轮询Supabase任务 2. 锁定任务 3. 下载资源 4. 执行pipeline 5. 上传结果 6. 清理资源
# 包含：CloudWorker类、任务监听逻辑、资源管理、日志同步、RAG集成
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 引入 Supabase 客户端库，用于连接云数据库和存储
from supabase import create_client, Client
# 引入 python-dotenv 库，用于加载本地 .env 文件中的环境变量（保护密钥安全）
from dotenv import load_dotenv

# 引入项目内部配置类和核心管线函数
from src.core.factory import PipelineFactory  # 🟢 引入工厂
from src.modules.rag_memory import RagMemory # 🟢 引入新模块
from src.modules.knowledge_base import KnowledgeBase # 🟢 引入
from src.utils.ply_utils import compress_model_for_delivery

# [初始化] 加载当前目录下的 .env 文件
# 这一步必须在所有 os.getenv 调用之前执行，否则读不到变量
load_dotenv()

_UNSET = object()

class CloudWorker:
    """
    ☁️ CloudWorker (云端工人)
    
    职责：
    1. 持续监听 Supabase 数据库中的任务表。
    2. 抢单：发现状态为 'pending' 的任务并锁定。
    3. 执行：下载视频 -> 调用 3DGS 核心引擎 -> 生成模型。
    4. 汇报：实时同步日志到数据库，并将最终结果上传回云存储。
    """

    def __init__(self):
        """
        初始化 Worker：连接 Supabase，准备本地缓存目录。
        """
        # --- 1. 读取环境变量配置 ---
        # 使用 os.getenv 读取 .env 文件中的配置，第二个参数是默认值
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        if self.SUPABASE_URL and not self.SUPABASE_URL.endswith('/'):
            self.SUPABASE_URL += '/'
            
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        self.BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "braindance-assets")  # 存储桶名称
        self.TABLE_NAME = os.getenv("SUPABASE_TABLE", "processing_tasks")    # 任务表名称
        self.WORKER_TABLE = os.getenv("SUPABASE_WORKER_TABLE", "worker_nodes")
        
        # --- 2. 防御性检查 ---
        # 如果关键配置缺失，直接报错停止，避免后续出现莫名其妙的连接错误
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("❌ 初始化失败：未找到 Supabase 配置！请检查 .env 文件是否存在且填写正确。")

        # --- 3. 建立连接 ---
        # 修复 HTTPX 对 no_proxy 的 CIDR 解析不兼容的问题，强制把目标 IP 塞入 no_proxy
        import urllib.parse
        if self.SUPABASE_URL:
            parsed = urllib.parse.urlparse(self.SUPABASE_URL)
            if parsed.hostname:
                no_proxy = os.environ.get("no_proxy", "")
                if parsed.hostname not in no_proxy:
                    os.environ["no_proxy"] = f"{no_proxy},{parsed.hostname}" if no_proxy else parsed.hostname

        # 创建 Supabase 客户端实例，后续所有数据库/存储操作都通过它进行
        self.supabase: Client = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        
        # 🟢 初始化记忆模块
        self.memory = RagMemory(self.supabase)
        self.kb = KnowledgeBase(self.supabase) # 🟢 实例化知识库

        # --- 4. 准备本地工作区 ---
        # 🟢 [修改后] 找回原来的 "braindance_workspace"，实现路径归一化
        # 这样所有的任务数据和模型都会存放在 /home/ltx/braindance_workspace
        self.CACHE_DIR = Path.home() / "braindance_workspace"
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 🟢 [新增] 定义一个公共模型目录，用于存放 SAM 权重等，实现多任务共享
        self.MODELS_DIR = self.CACHE_DIR / "models"
        self.MODELS_DIR.mkdir(exist_ok=True)
        
        # --- 5. 初始化日志缓冲区 ---
        # [关键设计] 用于解决“读写冲突”问题。
        # 我们不再每次 Select 数据库，而是将当前任务的所有日志保存在这个内存列表里。
        # 每次有新日志，append 进这里，然后把整个列表覆盖上传到云端。
        self.current_task_logs = []
        self.worker_id = os.getenv("WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.heartbeat_interval = max(3, int(os.getenv("WORKER_HEARTBEAT_INTERVAL", "10")))
        self.online_timeout_seconds = max(
            self.heartbeat_interval * 2,
            int(os.getenv("WORKER_ONLINE_TIMEOUT_SECONDS", str(self.heartbeat_interval * 3))),
        )
        self._state_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._status = "starting"
        self._current_task_id = None
        self._current_scene_id = None
        self._desired_state = "run"
        self._stop_requested = False
        self._stop_reason = None

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _set_worker_state(self, *, status=None, current_task_id=_UNSET, current_scene_id=_UNSET, stop_requested=None, stop_reason=None):
        with self._state_lock:
            if status is not None:
                self._status = status
            if current_task_id is not _UNSET:
                self._current_task_id = current_task_id
            if current_scene_id is not _UNSET:
                self._current_scene_id = current_scene_id
            if stop_requested is not None:
                self._stop_requested = stop_requested
            if stop_reason is not None:
                self._stop_reason = stop_reason

    def _worker_row_payload(self, *, status_override=None, stopped=False):
        with self._state_lock:
            status = status_override or self._status
            current_task_id = self._current_task_id
            current_scene_id = self._current_scene_id
            stop_reason = self._stop_reason
            desired_state = self._desired_state

        payload = {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "status": status,
            "current_task_id": current_task_id,
            "current_scene_id": current_scene_id,
            "last_heartbeat": self._now_iso(),
            "metadata": {
                "online_timeout_seconds": self.online_timeout_seconds,
                "stop_reason": stop_reason,
            },
        }

        if desired_state != "run":
            payload["metadata"]["desired_state_seen"] = desired_state

        if stopped:
            payload["stopped_at"] = self._now_iso()
            payload["current_task_id"] = None
            payload["current_scene_id"] = None

        return payload

    def _push_worker_state(self, *, status_override=None, stopped=False):
        try:
            self.supabase.table(self.WORKER_TABLE).upsert(
                self._worker_row_payload(status_override=status_override, stopped=stopped),
                on_conflict="worker_id",
            ).execute()
        except Exception as e:
            print(f"⚠️ [WorkerRegistry] 状态同步失败: {e}")

    def _fetch_desired_state(self):
        try:
            response = self.supabase.table(self.WORKER_TABLE)\
                .select("desired_state, control_note, control_requested_at")\
                .eq("worker_id", self.worker_id)\
                .limit(1)\
                .execute()
            row = response.data[0] if response.data else None
            if not row:
                return "run", None
            return (row.get("desired_state") or "run").lower(), row.get("control_note")
        except Exception as e:
            print(f"⚠️ [WorkerRegistry] 控制指令读取失败: {e}")
            return "run", None

    def _apply_desired_state(self, desired_state, control_note=None):
        desired_state = (desired_state or "run").lower()
        with self._state_lock:
            self._desired_state = desired_state
            busy = self._current_task_id is not None
            current_status = self._status
        if desired_state in ("pause", "interrupt"):
            self._set_worker_state(
                status="stopping",
                stop_requested=True,
                stop_reason=control_note or f"{desired_state} requested",
            )
        elif desired_state == "run":
            next_status = "busy" if busy else "idle"
            if current_status == "offline":
                next_status = "offline"
            self._set_worker_state(status=next_status, stop_requested=False, stop_reason=None)

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            desired_state, control_note = self._fetch_desired_state()
            self._apply_desired_state(desired_state, control_note)
            self._push_worker_state()

    def _start_heartbeat_loop(self):
        self._push_worker_state()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="worker-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat_loop(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self.heartbeat_interval + 1)

    def _should_stop(self):
        with self._state_lock:
            return self._stop_requested

    def _finalize_worker_shutdown(self, status="offline"):
        self._set_worker_state(status=status, current_task_id=None, current_scene_id=None)
        self._push_worker_state(status_override=status, stopped=True)

    def _sync_log(self, task_id):
        """
        🔄 [内部方法] 日志同步器
        
        功能：将内存中的日志缓冲区 (self.current_task_logs) 全量推送到 Supabase。
        设计哲学：采用“内存为王，覆盖更新”策略，避免多线程下的数据覆盖问题。
        """
        try:
            # 直接调用 Update 接口，将 logs 字段更新为当前的内存列表
            self.supabase.table(self.TABLE_NAME).update({
                "logs": self.current_task_logs
            }).eq("id", task_id).execute()
        except Exception as e:
            # ⚠️ 注意：日志同步失败属于“非致命错误”。
            # 如果网络抖动导致日志没发出去，不应该中断核心训练任务，所以这里只打印不抛出异常。
            print(f"⚠️ [网络抖动] 日志同步跳过: {e}")

    def start(self):
        """
        🚀 [主入口] 启动监听循环
        这是外部调用的唯一入口，启动后会进入死循环，直到被手动停止。
        """
        print(f"🚀 [CloudWorker] 启动成功! Worker={self.worker_id} 正在监听任务表: [{self.TABLE_NAME}]")
        self._set_worker_state(status="idle")
        self._start_heartbeat_loop()
        try:
            while not self._should_stop():
                # 执行一次“心跳”检测
                self._tick()
        except KeyboardInterrupt:
            # 捕获 Ctrl+C 中断信号，优雅退出
            print("\n🛑 [CloudWorker] 接收到停止信号，服务已关闭。")
            self._set_worker_state(stop_requested=True, stop_reason="keyboard interrupt")
        finally:
            self._stop_heartbeat_loop()
            self._finalize_worker_shutdown()

    def _tick(self):
        """
        💓 [心跳函数] 单次轮询逻辑
        """
        try:
            desired_state, control_note = self._fetch_desired_state()
            self._apply_desired_state(desired_state, control_note)
            if self._should_stop():
                print("\n🛑 [CloudWorker] 收到远程暂停指令，准备退出。")
                return

            # --- 1. 轮询数据库 ---
            # 查询条件：状态(status)必须是 'pending' (待处理)
            # limit(1)：每次只取 1 个任务，避免贪多嚼不烂
            response = self.supabase.table(self.TABLE_NAME)\
                .select("*").eq("status", "pending").limit(1).execute()
            
            # --- 2. 判断是否有任务 ---
            if response.data:
                # 🎯 发现任务！立即处理
                # response.data 是一个列表，我们取第一个元素
                self._process_task(response.data[0])
            else:
                self._set_worker_state(status="idle")
                # 💤 没有任务，休眠 3 秒
                # 避免死循环空转导致 CPU 占用率过高，同时也减少数据库压力
                time.sleep(3)
                # 打印一个小点，证明程序还活着 (心跳显示)
                print(".", end="", flush=True)
                
        except Exception as e:
            # 🛡️ 容错处理
            # 比如数据库断连、查询超时等，打印错误并强制休息 5 秒，防止错误刷屏
            print(f"\n⚠️ 轮询错误: {e}")
            time.sleep(5)

    def _process_task(self, task):
        """
        ⚙️ [核心逻辑] 处理单个任务
        """
        # --- 1. 解包任务信息 ---
        task_id = task['id']            # 任务唯一ID
        scene_id = task['scene_id']     # 场景/项目ID (作为文件名)
        # 获取用户ID，如果数据库里没存 user_id 字段，就用默认值 'default_user'
        user_id = task.get('user_id', 'default_user') 
        self._set_worker_state(status="busy", current_task_id=task_id, current_scene_id=scene_id)
        self._push_worker_state()
        
        print(f"\n📥 [接收任务] 场景ID: {scene_id} | 任务ID: {task_id}")

        # --- 2. 重置日志缓冲区 ---
        # [重要] 开始新任务前，必须清空上一条任务的残留日志，防止串台
        self.current_task_logs = []

        # --- 3. 定义回调函数 (闭包) ---
        # 这个函数会传给 pipeline.py，让核心引擎在深层代码里也能发日志
        def on_pipeline_log(message):
            # A. 构造标准日志对象 (时间戳 + 消息)
            log_entry = {
                "ts": int(time.time()), # 当前秒级时间戳
                "msg": message
            }
            # B. 写入本地内存 (操作极快，绝对不丢数据)
            self.current_task_logs.append(log_entry)
            
            # C. 触发云端同步
            self._sync_log(task_id)
            
            # D. 输出到终端，方便本地排查问题
            print(f"[{scene_id}] {message}")

        try:
            # =================== 阶段 A: 锁定任务 ===================
            # 将状态改为 'processing'，告诉其他 Worker 这个任务我接了，别抢
            # 同时清空 logs 字段，准备开始新纪录
            self.supabase.table(self.TABLE_NAME).update({
                "status": "processing",
                "logs": []
            }).eq("id", task_id).execute()

            # =================== 阶段 B: 下载资源 ===================
            on_pipeline_log("正在从云端下载资源...")
            
            # 获取任务类型：直接使用 task_type 字段，默认 video_3dgs
            task_type = task.get('task_type', 'video_3dgs')
            print(f"🔧 检测到任务类型: {task_type}")
            
            # 根据任务类型确定下载路径
            if task_type in ('single_image_sam3d', 'single_image_sharp'):
                input_path = self.CACHE_DIR / f"{scene_id}.png"
                storage_path = f"{user_id}/{scene_id}/raw/image.png"
                on_pipeline_log("下载单张图片...")
            elif task_type in ('sparse2dgs',):
                input_path = self.CACHE_DIR / f"{scene_id}_images.zip"
                storage_path = f"{user_id}/{scene_id}/raw/images.zip"
                on_pipeline_log("下载多图压缩包 (images.zip)...")
            else:
                # video_3dgs 默认路径
                input_path = self.CACHE_DIR / f"{scene_id}.mp4"
                storage_path = f"{user_id}/{scene_id}/raw/video.mp4"
                on_pipeline_log("下载视频...")
            
            # 下载文件流并写入本地
            try:
                with open(input_path, 'wb') as f:
                    res = self.supabase.storage.from_(self.BUCKET_NAME).download(storage_path)
                    f.write(res)
            except Exception as e:
                raise RuntimeError(f"资源下载失败 (路径: {storage_path}): {e}") from e

            # =================== 阶段 C: 执行引擎 ===================
            # 1. 获取任务类型和参数
            task_type = task.get('task_type', 'video_3dgs') if isinstance(task, dict) else 'video_3dgs'
            
            # task_params 可能是 JSON 字符串，需要解析
            task_params_raw = task.get('task_params') if isinstance(task, dict) else None
            
            try:
                import json
                if isinstance(task_params_raw, str) and task_params_raw:
                    task_params = json.loads(task_params_raw)
                elif isinstance(task_params_raw, dict):
                    task_params = task_params_raw
                else:
                    task_params = {}
            except Exception:
                task_params = {}
            
            # 确保 task_params 始终是一个字典，不是 None
            if task_params is None:
                task_params = {}

            # 🟢 [新增] 支持从 Supabase 任务表顶层字段直接读取 mapper_type
            if isinstance(task, dict) and task.get('mapper_type'):
                task_params['mapper_type'] = task['mapper_type']
            
            # 准备输出目录: 修改为 user_id/scene_id/output 格式
            task_output_dir = self.CACHE_DIR / user_id / scene_id / "output"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 准备上下文 (把通用的东西打包)
            context = {
                "task_id": task_id,
                "scene_id": scene_id,
                "user_id": user_id,
                "work_root": task_output_dir,
                "log_callback": on_pipeline_log,
                "shared_model_dir": self.MODELS_DIR,
                "supabase": self.supabase
            }

            # 3. [核心修改] 通过工厂实例化 Pipeline
            on_pipeline_log(f"正在加载流水线: {task_type}")
            pipeline = PipelineFactory.get_pipeline(task_type, context)
            
            # 4. [核心修改] 执行多态的 run 方法
            # input_path 可能是视频路径，也可能是图片路径，根据 task_type 决定下载逻辑
            
            final_model_path, metadata = pipeline.run(str(input_path), task_params)

            # 🟢 [修改点 2] 立即同步 AI 分析结果到数据库
            # 不管训练是否成功，只要有分析结果，都应该存下来
            if metadata:
                update_data = {}
                
                # 1. 同步分数
                if "ai_score" in metadata:
                    update_data["quality_score"] = metadata["ai_score"]
                
                # 2. 同步标签
                if "ai_tags" in metadata:
                    update_data["tags"] = metadata["ai_tags"]
                
                # 3. 同步评价原因 (新!)
                if "ai_reason" in metadata:
                    update_data["quality_reason"] = metadata["ai_reason"]
                
                # 执行更新
                if update_data:
                    self.supabase.table(self.TABLE_NAME)\
                        .update(update_data)\
                        .eq("id", task_id)\
                        .execute()
                    on_pipeline_log(f"✅ AI 评分已同步: {metadata.get('ai_score')}分")
                
                # 🟢 2. [新增] 存入 RAG 向量库 (只有当包含描述信息时才存)
                if "ai_description" in metadata:
                    on_pipeline_log("🧠 正在生成场景记忆 (Embedding)...")
                    self.memory.save_to_knowledge_base(
                        task_data=task,
                        description=metadata["ai_description"],
                        objects=metadata.get("ai_objects", [])
                    )

            # 校验结果：如果 pipeline 返回 None 或者文件不存在，说明训练挂了
            if not final_model_path or not Path(final_model_path).exists():
                raise RuntimeError("Pipeline 执行结束，但未生成有效的模型文件，请检查训练日志。")

            model_path_obj = Path(final_model_path)
            delivery_format = str(
                task_params.get("delivery_format") or os.getenv("MODEL_DELIVERY_FORMAT", "splat")
            ).lower()
            opacity_threshold = float(
                task_params.get("compression_opacity_threshold", os.getenv("COMPRESSION_OPACITY_THRESHOLD", "0.05"))
            )
            ksplat_alpha_threshold = int(
                task_params.get("ksplat_alpha_threshold", os.getenv("KSPLAT_ALPHA_THRESHOLD", "1"))
            )

            if model_path_obj.suffix.lower() == ".ply" and delivery_format in ("splat", "ksplat"):
                try:
                    compressed_path = compress_model_for_delivery(
                        model_path=str(model_path_obj),
                        output_format=delivery_format,
                        opacity_threshold=opacity_threshold,
                        ksplat_script_path=os.getenv("KSPLAT_SCRIPT_PATH", ""),
                        alpha_removal_threshold=ksplat_alpha_threshold,
                        log_callback=on_pipeline_log,
                    )
                    model_path_obj = Path(compressed_path)
                    on_pipeline_log(f"✅ 模型压缩完成: {model_path_obj.name}")
                except Exception as e:
                    on_pipeline_log(f"⚠️ 模型压缩失败，回退原始文件上传: {e}")

            # =================== 阶段 D: 上传结果 ===================
            on_pipeline_log("训练完成，正在上传结果到云端...")
            
            # 1. 上传模型文件（支持 .ply / .splat / .ksplat）
            model_suffix = model_path_obj.suffix.lower() or ".ply"
            upload_ply_key = f"{user_id}/{scene_id}/output/point_cloud{model_suffix}"
            with open(model_path_obj, 'rb') as f:
                self.supabase.storage.from_(self.BUCKET_NAME).upload(
                    path=upload_ply_key, 
                    file=f, 
                    # x-upsert=true 和 upsert=true 表示如果文件已存在则覆盖
                    file_options={"content-type": "application/octet-stream", "x-upsert": "true", "upsert": "true"}
                )

            # 2. 上传 transforms.json (用于网页预览)
            # 假设该文件在 PLY 同级目录或配置指定的目录
            transforms_file = task_output_dir / "data" / "transforms.json"
            if transforms_file.exists():
                upload_json_key = f"{user_id}/{scene_id}/output/transforms.json"
                with open(transforms_file, 'rb') as f:
                    self.supabase.storage.from_(self.BUCKET_NAME).upload(
                        path=upload_json_key,
                        file=f,
                        file_options={"content-type": "application/json", "x-upsert": "true", "upsert": "true"}
                    )
                on_pipeline_log("上传 transforms.json 成功")

            # 3. 上传预览图 preview_img_path (如果存在)
            if metadata and metadata.get("preview_img_path") and Path(metadata["preview_img_path"]).exists():
                upload_img_key = f"{user_id}/{scene_id}/output/preview.jpg" # 假设是 jpg
                with open(metadata["preview_img_path"], 'rb') as f:
                    self.supabase.storage.from_(self.BUCKET_NAME).upload(
                        path=upload_img_key,
                        file=f,
                        file_options={"content-type": "image/jpeg", "x-upsert": "true", "upsert": "true"}
                    )
                # 替换为远程 URL
                remote_url = self.supabase.storage.from_(self.BUCKET_NAME).get_public_url(upload_img_key)
                metadata["preview_img_path"] = remote_url
                on_pipeline_log("上传预览图成功")

            # 3.5 强制同步 model_assets 的模型路径，保证客户端总能拿到最新后缀
            try:
                model_assets_row = {
                    "scene_id": scene_id,
                    "user_id": user_id,
                    "ply_path": upload_ply_key,
                }
                if metadata:
                    if "ai_description" in metadata:
                        model_assets_row["description"] = metadata.get("ai_description", "")
                    if "ai_tags" in metadata:
                        model_assets_row["tags"] = metadata.get("ai_tags", [])
                    if "ai_objects" in metadata:
                        model_assets_row["objects"] = metadata.get("ai_objects", [])
                    if "preview_img_path" in metadata:
                        model_assets_row["preview_img_path"] = metadata.get("preview_img_path", "")
                self.supabase.table("model_assets").upsert(
                    model_assets_row,
                    on_conflict="scene_id"
                ).execute()
            except Exception as e:
                on_pipeline_log(f"⚠️ 同步 model_assets 路径失败: {e}")

            # =================== 🟢 [RAG 集成] 资产入库 ===================
            # 只有当 AI 成功生成了描述，才存入知识库
            if metadata and "ai_description" in metadata:
                on_pipeline_log("📚 正在将资产存入知识库...")
                self.kb.add_asset(
                    task_data=task,
                    metadata=metadata,
                    ply_path=upload_ply_key  # 记录云端路径
                )
            # =============================================================

            # =================== 阶段 E: 完结撒花 ===================
            # 更新状态为 'completed'
            self.supabase.table(self.TABLE_NAME).update({
                "status": "completed"
            }).eq("id", task_id).execute()
            
            on_pipeline_log("✅ 任务全部完成")
            print("✅ 任务完成")

        except Exception as e:
            # =================== 异常处理 ===================
            print(f"❌ 任务处理失败: {e}")
            
            # 1. 尝试将错误信息记录到云端日志，让用户知道死在哪一步了
            try:
                self.current_task_logs.append({"ts": int(time.time()), "msg": f"❌ 严重错误: {str(e)}"})
                self._sync_log(task_id)
            except:
                pass # 如果这时候连网都断了，就放弃写日志
            
            # 2. 将任务状态标记为 'failed'，避免死循环重试
            self.supabase.table(self.TABLE_NAME).update({"status": "failed"}).eq("id", task_id).execute()
        
        finally:
            self._set_worker_state(status="idle", current_task_id=None, current_scene_id=None)
            desired_state, control_note = self._fetch_desired_state()
            self._apply_desired_state(desired_state, control_note)
            self._push_worker_state()

            # =================== 🧹 清理工作 (新增逻辑) ===================
            import shutil # 确保引入 shutil

            # 1. 删除源文件 (视频或图片)
            if 'input_path' in locals() and input_path.exists():
                try:
                    os.remove(input_path)
                    print(f"🗑️ 已删除临时文件: {input_path.name}")
                except Exception as e:
                    print(f"⚠️ 删除文件失败: {e}")
            
            # 2. 删除任务输出目录 (包含图片、COLMAP数据、PLY等所有中间产物)
            # ⚠️ 警告：如果你还没有修改 ai_segmentor.py 让模型下载到公共目录，
            # 这里的删除操作会把下载在里面的 AI 模型也删掉！请务必先做"模型搬家"。
            if 'task_output_dir' in locals() and task_output_dir.exists():
                try:
                    shutil.rmtree(task_output_dir)
                    print(f"🗑️ 已清空任务工作区: {task_output_dir.name}")
                except Exception as e:
                    print(f"⚠️ 清理工作区失败: {e}")

            # 3. 重置日志
            self.current_task_logs = []
