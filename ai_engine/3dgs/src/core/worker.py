# src/core/worker.py
# 功能：实现云端工作者逻辑，监听任务队列并处理3D重建任务
# 实现：通过Supabase轮询任务，下载资源，执行pipeline，上传结果
# 逻辑：1. 轮询Supabase任务 2. 锁定任务 3. 下载资源 4. 执行pipeline 5. 上传结果 6. 清理资源
# 包含：CloudWorker类、任务监听逻辑、资源管理、日志同步、RAG集成
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse, unquote
from urllib.request import urlopen, Request

from supabase import Client, ClientOptions, create_client

from src.config import PipelineConfig, ensure_no_proxy_for_url
from src.core.factory import PipelineFactory
from src.modules.knowledge_base import KnowledgeBase
from src.modules.rag_memory import RagMemory
from src.modules.scene_analyzer import SceneAnalyzer
from src.utils.ply_utils import compress_model_for_delivery

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)

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

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化 Worker：连接 Supabase，准备本地缓存目录。
        """
        self.cfg = config or PipelineConfig()
        self.SUPABASE_URL = self.cfg.supabase_url
        self.SUPABASE_KEY = self.cfg.supabase_key
        self.BUCKET_NAME = self.cfg.supabase_bucket
        self.TABLE_NAME = self.cfg.supabase_table
        self.WORKER_TABLE = os.getenv("SUPABASE_WORKER_TABLE", "worker_nodes")
        self.postgrest_timeout = max(5, int(os.getenv("SUPABASE_POSTGREST_TIMEOUT_SECONDS", "15")))
        self.storage_timeout = max(30, int(os.getenv("SUPABASE_STORAGE_TIMEOUT_SECONDS", "300")))
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("❌ 初始化失败：未找到 Supabase 配置！请检查 .env 文件是否存在且填写正确。")

        # --- 3. 建立连接 ---
        # 修复 HTTPX 对 no_proxy 的 CIDR 解析不兼容的问题，强制把目标 IP 塞入 no_proxy
        if self.SUPABASE_URL:
            ensure_no_proxy_for_url(self.SUPABASE_URL)

        # 创建 Supabase 客户端实例，后续所有数据库/存储操作都通过它进行
        self.supabase: Client = create_client(
            self.SUPABASE_URL,
            self.SUPABASE_KEY,
            options=ClientOptions(
                postgrest_client_timeout=self.postgrest_timeout,
                storage_client_timeout=self.storage_timeout,
            ),
        )
        
        # 🟢 初始化记忆模块
        self.memory = RagMemory(self.supabase, self.cfg)
        self.kb = KnowledgeBase(self.supabase, self.cfg)

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

    @staticmethod
    def _contains_emoji(message: str) -> bool:
        """仅将带 emoji 的日志同步到云端，普通进度日志只留在本地终端。"""
        return bool(EMOJI_PATTERN.search(message or ""))

    def _record_cloud_log(self, task_id: str, message: str) -> None:
        """按规则写入云端日志缓冲区。"""
        if not self._contains_emoji(message):
            return

        self.current_task_logs.append({
            "ts": int(time.time()),
            "msg": message,
        })
        self._sync_log(task_id)

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

    def _normalize_storage_key(self, raw_value: object) -> Optional[str]:
        """把 task 中各种可能的路径/URL 统一转成 bucket 内对象 key。"""
        if raw_value is None:
            return None
        value = str(raw_value).strip().strip('"').strip("'")
        if not value:
            return None

        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            value = unquote(parsed.path or "").lstrip("/")
            bucket = self.BUCKET_NAME
            url_prefixes = [
                f"storage/v1/object/public/{bucket}/",
                f"storage/v1/object/sign/{bucket}/",
                f"storage/v1/object/authenticated/{bucket}/",
                f"storage/v1/render/image/public/{bucket}/",
                f"storage/v1/render/image/authenticated/{bucket}/",
            ]
            for prefix in url_prefixes:
                if value.startswith(prefix):
                    value = value[len(prefix):]
                    break
        else:
            value = value.lstrip("/")

        if value.startswith(f"{self.BUCKET_NAME}/"):
            value = value[len(self.BUCKET_NAME) + 1:]

        value = value.strip("/")
        return value or None

    def _build_single_image_candidates(self, task: dict, task_params: dict, user_id: str, scene_id: str) -> list[str]:
        image_exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic"]
        candidates: list[str] = []
        direct_urls: list[str] = []

        def add(raw: object) -> None:
            if raw is None:
                return
            raw_str = str(raw).strip()
            parsed = urlparse(raw_str)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                direct_urls.append(raw_str)
            key = self._normalize_storage_key(raw_str)
            if key:
                candidates.append(key)

        # 1) 明确字段（task_params + 任务顶层）
        direct_keys = [
            "storage_path",
            "input_storage_path",
            "image_path",
            "image_storage_path",
            "input_path",
            "file_path",
            "source_path",
            "path",
            "image_url",
            "input_url",
            "url",
            "public_url",
            "asset_url",
        ]
        for key in direct_keys:
            add(task_params.get(key))
            add(task.get(key))

        # 2) 常见嵌套结构
        for container_key in ("input", "image", "source", "asset"):
            payload = task_params.get(container_key)
            if isinstance(payload, dict):
                for key in direct_keys:
                    add(payload.get(key))
        for list_key in ("images", "files", "assets", "inputs"):
            payload = task_params.get(list_key)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        for key in direct_keys:
                            add(item.get(key))
                    else:
                        add(item)

        # 3) 标准与兼容目录模式
        for ext in image_exts:
            candidates.extend(
                [
                    f"{user_id}/{scene_id}/raw/image{ext}",
                    f"{user_id}/{scene_id}/raw/{scene_id}{ext}",
                    f"{user_id}/{scene_id}/image{ext}",
                    f"{user_id}/{scene_id}/{scene_id}{ext}",
                    f"{scene_id}/raw/image{ext}",
                    f"{scene_id}/raw/{scene_id}{ext}",
                ]
            )

        # 4) 目录探测：自动补充真实存在的对象名（服务端 key 可列目录）
        def list_image_keys(prefix: str, max_depth: int = 2) -> list[str]:
            found: list[str] = []
            queue: list[tuple[str, int]] = [(prefix.strip("/"), 0)]
            while queue:
                current, depth = queue.pop(0)
                if not current:
                    continue
                try:
                    rows = self.supabase.storage.from_(self.BUCKET_NAME).list(current)
                except Exception:
                    continue
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    name = str(row.get("name", "")).strip()
                    if not name:
                        continue
                    suffix = Path(name).suffix.lower()
                    child_key = f"{current}/{name}".strip("/")
                    if suffix in image_exts:
                        found.append(child_key)
                    elif depth < max_depth:
                        # list() 返回目录时通常没有后缀，继续向下探测
                        queue.append((child_key, depth + 1))
            return found

        try:
            probe_dirs = [
                f"{user_id}/{scene_id}/raw",
                f"{user_id}/{scene_id}",
                f"{scene_id}/raw",
                scene_id,
            ]
            for prefix in probe_dirs:
                candidates.extend(list_image_keys(prefix, max_depth=2))
        except Exception:
            pass

        # 去重并保持顺序
        return list(dict.fromkeys(direct_urls + candidates))

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

    def _parse_task_params(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_params_raw = task.get('task_params') if isinstance(task, dict) else None
        try:
            if isinstance(task_params_raw, str) and task_params_raw:
                task_params = json.loads(task_params_raw)
            elif isinstance(task_params_raw, dict):
                task_params = task_params_raw
            else:
                task_params = {}
        except Exception:
            task_params = {}
        if task_params is None:
            task_params = {}
        if isinstance(task, dict) and task.get('mapper_type'):
            task_params['mapper_type'] = task['mapper_type']
        return task_params

    def _sync_task_metadata(self, task_id: str, task: Dict[str, Any], metadata: Dict[str, Any], on_pipeline_log):
        if not metadata:
            return
        update_data = {}
        if "ai_score" in metadata:
            update_data["quality_score"] = metadata["ai_score"]
        if "ai_tags" in metadata:
            update_data["tags"] = metadata["ai_tags"]
        if "ai_reason" in metadata:
            update_data["quality_reason"] = metadata["ai_reason"]
        if update_data:
            self.supabase.table(self.TABLE_NAME).update(update_data).eq("id", task_id).execute()
            on_pipeline_log(f"✅ AI 评分已同步: {metadata.get('ai_score')}分")
        if "ai_description" in metadata:
            on_pipeline_log("🧠 正在生成场景记忆 (Embedding)...")
            self.memory.save_to_knowledge_base(
                task_data=task,
                description=metadata["ai_description"],
                objects=metadata.get("ai_objects", [])
            )

    def _compress_if_needed(self, model_path_obj: Path, task_params: Dict[str, Any], on_pipeline_log) -> Path:
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
        return model_path_obj

    def _upload_and_upsert_assets(
        self,
        task: Dict[str, Any],
        model_path_obj: Path,
        metadata: Dict[str, Any],
        user_id: str,
        scene_id: str,
        task_output_dir: Path,
        on_pipeline_log,
    ) -> str:
        on_pipeline_log("训练完成，正在上传结果到云端...")
        model_suffix = model_path_obj.suffix.lower() or ".ply"
        upload_ply_key = f"{user_id}/{scene_id}/output/point_cloud{model_suffix}"
        with open(model_path_obj, 'rb') as f:
            self.supabase.storage.from_(self.BUCKET_NAME).upload(
                path=upload_ply_key,
                file=f,
                file_options={"content-type": "application/octet-stream", "x-upsert": "true", "upsert": "true"}
            )

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

        if metadata and metadata.get("preview_img_path") and Path(metadata["preview_img_path"]).exists():
            upload_img_key = f"{user_id}/{scene_id}/output/preview.jpg"
            with open(metadata["preview_img_path"], 'rb') as f:
                self.supabase.storage.from_(self.BUCKET_NAME).upload(
                    path=upload_img_key,
                    file=f,
                    file_options={"content-type": "image/jpeg", "x-upsert": "true", "upsert": "true"}
                )
            remote_url = self.supabase.storage.from_(self.BUCKET_NAME).get_public_url(upload_img_key)
            metadata["preview_img_path"] = remote_url
            on_pipeline_log("上传预览图成功")

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

        if metadata and "ai_description" in metadata:
            on_pipeline_log("📚 正在将资产存入知识库...")
            self.kb.add_asset(task_data=task, metadata=metadata, ply_path=upload_ply_key)
        return upload_ply_key

    def _run_pipeline_once(
        self,
        task: Dict[str, Any],
        task_type: str,
        input_path: Path,
        task_params: Dict[str, Any],
        work_dir: Path,
        on_pipeline_log,
    ) -> Tuple[Path, Dict[str, Any]]:
        task_id = task['id']
        scene_id = task['scene_id']
        user_id = task.get('user_id', 'default_user')

        context = {
            "task_id": task_id,
            "scene_id": scene_id,
            "user_id": user_id,
            "work_root": work_dir,
            "log_callback": on_pipeline_log,
            "shared_model_dir": self.MODELS_DIR,
            "supabase": self.supabase
        }

        on_pipeline_log(f"正在加载流水线: {task_type}")
        pipeline = PipelineFactory.get_pipeline(task_type, context)
        final_model_path, metadata = pipeline.run(str(input_path), task_params)
        self._sync_task_metadata(task_id, task, metadata, on_pipeline_log)

        if not final_model_path or not Path(final_model_path).exists():
            raise RuntimeError("Pipeline 执行结束，但未生成有效的模型文件，请检查训练日志。")

        model_path_obj = self._compress_if_needed(Path(final_model_path), task_params, on_pipeline_log)
        self._upload_and_upsert_assets(task, model_path_obj, metadata, user_id, scene_id, work_dir, on_pipeline_log)
        return model_path_obj, metadata

    def _detect_total_vram_gb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                return float(total_mem) / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def _extract_candidate_frames(self, video_path: Path, out_dir: Path, sample_count: int, on_pipeline_log) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        on_pipeline_log("🖼️ 正在提取候选帧用于快链...")
        frame_pattern = out_dir / "frame_%05d.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "fps=5,scale=1280:1280:force_original_aspect_ratio=decrease:flags=lanczos",
            "-q:v", "2",
            str(frame_pattern),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        frames = sorted(out_dir.glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("未能从视频提取候选帧")
        if sample_count <= 1:
            return [frames[0]]
        if len(frames) <= sample_count:
            return frames
        step = (len(frames) - 1) / float(sample_count - 1)
        picked = [frames[int(round(i * step))] for i in range(sample_count)]
        return sorted(set(picked))

    def _run_video_dual_chain(self, task: Dict[str, Any], video_path: Path, task_params: Dict[str, Any], task_output_dir: Path, on_pipeline_log):
        scene_id = task['scene_id']
        sample_count = int(task_params.get("best_frame_sample_count", 8))
        slow_pipeline = str(task_params.get("slow_pipeline", "video_3dgs")).strip() or "video_3dgs"
        if slow_pipeline not in ("video_3dgs", "da3_feed_forward_3dgs"):
            slow_pipeline = "video_3dgs"
        vram_threshold = float(task_params.get("sam3d_vram_threshold_gb", 25))

        frames_dir = task_output_dir / "_dual_chain_frames"
        fast_work_dir = task_output_dir / "fast_chain"
        slow_work_dir = task_output_dir / "slow_chain"
        fast_work_dir.mkdir(parents=True, exist_ok=True)
        slow_work_dir.mkdir(parents=True, exist_ok=True)

        cfg = PipelineConfig()
        analyzer = SceneAnalyzer(cfg)
        fast_ok = False
        slow_ok = False
        fast_error = None
        slow_error = None

        candidate_frames = self._extract_candidate_frames(video_path, frames_dir, sample_count, on_pipeline_log)
        best_idx, best_reason = analyzer.select_best_image([str(p) for p in candidate_frames], log_callback=on_pipeline_log)
        best_image = candidate_frames[max(0, min(best_idx, len(candidate_frames) - 1))]
        on_pipeline_log(f"✅ 最佳帧已选定: {best_image.name}（{best_reason}）")

        classify_label, classify_reason = analyzer.classify_scene_or_object(str(best_image), log_callback=on_pipeline_log)
        on_pipeline_log(f"🔍 快链目标判定: {classify_label}（{classify_reason}）")

        fast_params = dict(task_params)
        fast_params["scene_id"] = scene_id
        fast_task_type = "single_image_sharp"
        if classify_label == "object":
            total_vram_gb = self._detect_total_vram_gb()
            if total_vram_gb >= vram_threshold:
                fast_task_type = "single_image_sam3d"
                on_pipeline_log(f"🧠 显存 {total_vram_gb:.1f}GB >= {vram_threshold}GB，快链使用 SAM3D")
            else:
                on_pipeline_log(f"⚠️ 显存 {total_vram_gb:.1f}GB < {vram_threshold}GB，快链降级为 SHARP")
        else:
            on_pipeline_log("🏞️ 判定为场景，快链使用 SHARP")

        try:
            self._run_pipeline_once(
                task=task,
                task_type=fast_task_type,
                input_path=best_image,
                task_params=fast_params,
                work_dir=fast_work_dir,
                on_pipeline_log=on_pipeline_log,
            )
            fast_ok = True
            on_pipeline_log("⚡ 快链完成")
        except Exception as e:
            fast_error = e
            on_pipeline_log(f"⚠️ 快链失败，将继续执行慢链: {e}")

        slow_params = dict(task_params)
        slow_params["scene_id"] = scene_id
        try:
            self._run_pipeline_once(
                task=task,
                task_type=slow_pipeline,
                input_path=video_path,
                task_params=slow_params,
                work_dir=slow_work_dir,
                on_pipeline_log=on_pipeline_log,
            )
            slow_ok = True
            on_pipeline_log("🐢 慢链完成")
        except Exception as e:
            slow_error = e
            on_pipeline_log(f"⚠️ 慢链失败: {e}")

        if not fast_ok and not slow_ok:
            raise RuntimeError(f"快慢双链均失败 | fast={fast_error} | slow={slow_error}")

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
            # 只把带 emoji 的关键信息同步到云端，终端仍然打印完整日志。
            self._record_cloud_log(task_id, message)
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

            task_type = task.get('task_type', 'video_3dgs')
            task_params = self._parse_task_params(task)
            print(f"🔧 检测到任务类型: {task_type}")
            storage_path: Optional[str] = None
            if task_type in ('single_image_sam3d', 'single_image_sharp'):
                input_path = self.CACHE_DIR / f"{scene_id}.png"
                single_image_candidates = self._build_single_image_candidates(
                    task=task,
                    task_params=task_params,
                    user_id=user_id,
                    scene_id=scene_id,
                )
                on_pipeline_log("下载单张图片...")
            elif task_type in ('sparse2dgs',):
                sparse2dgs_tmpdir = tempfile.TemporaryDirectory(prefix=f"{scene_id}_sparse2dgs_")
                sparse2dgs_tmpdir_path = Path(sparse2dgs_tmpdir.name)
                sparse2dgs_candidates = [
                    (
                        sparse2dgs_tmpdir_path / "images.zip",
                        f"{user_id}/{scene_id}/raw/images.zip",
                        "下载多图压缩包 (images.zip)...",
                    ),
                    (
                        sparse2dgs_tmpdir_path / "video.mp4",
                        f"{user_id}/{scene_id}/raw/video.mp4",
                        "未找到 images.zip，回退下载视频 (video.mp4)...",
                    ),
                ]
                storage_path = sparse2dgs_candidates[0][1]
            else:
                input_path = self.CACHE_DIR / f"{scene_id}.mp4"
                storage_path = f"{user_id}/{scene_id}/raw/video.mp4"
                on_pipeline_log("下载视频...")

            try:
                if task_type in ('sparse2dgs',):
                    last_error = None
                    for candidate_input_path, candidate_storage_path, download_message in sparse2dgs_candidates:
                        on_pipeline_log(download_message)
                        try:
                            with open(candidate_input_path, 'wb') as f:
                                res = self.supabase.storage.from_(self.BUCKET_NAME).download(candidate_storage_path)
                                f.write(res)
                            input_path = candidate_input_path
                            storage_path = candidate_storage_path
                            break
                        except Exception as e:
                            last_error = e
                            if candidate_input_path.exists():
                                try:
                                    candidate_input_path.unlink()
                                except OSError:
                                    pass
                    else:
                        raise RuntimeError(
                            f"既找不到 {user_id}/{scene_id}/raw/images.zip，也找不到 {user_id}/{scene_id}/raw/video.mp4: {last_error}"
                        )
                else:
                    if task_type in ('single_image_sam3d', 'single_image_sharp'):
                        last_error = None
                        for candidate_storage_path in single_image_candidates:
                            storage_path = candidate_storage_path
                            parsed = urlparse(candidate_storage_path)
                            if parsed.scheme in {"http", "https"} and parsed.netloc:
                                candidate_suffix = Path(unquote(parsed.path)).suffix.lower()
                            else:
                                candidate_suffix = Path(candidate_storage_path).suffix.lower()
                            if candidate_suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                                candidate_suffix = ".png"
                            candidate_input_path = self.CACHE_DIR / f"{scene_id}{candidate_suffix}"
                            try:
                                if parsed.scheme in {"http", "https"} and parsed.netloc:
                                    req = Request(candidate_storage_path, headers={"User-Agent": "BrainDance-Worker/1.0"})
                                    with urlopen(req, timeout=30) as resp:
                                        content = resp.read()
                                    with open(candidate_input_path, 'wb') as f:
                                        f.write(content)
                                else:
                                    with open(candidate_input_path, 'wb') as f:
                                        res = self.supabase.storage.from_(self.BUCKET_NAME).download(candidate_storage_path)
                                        f.write(res)
                                input_path = candidate_input_path
                                storage_path = candidate_storage_path
                                break
                            except Exception as e:
                                last_error = e
                                if candidate_input_path.exists():
                                    try:
                                        candidate_input_path.unlink()
                                    except OSError:
                                        pass
                        else:
                            raise RuntimeError(
                                f"单图素材下载失败，已尝试路径: {single_image_candidates}，最后错误: {last_error}"
                            )
                    else:
                        with open(input_path, 'wb') as f:
                            res = self.supabase.storage.from_(self.BUCKET_NAME).download(storage_path)
                            f.write(res)
            except Exception as e:
                failed_path = storage_path or "unknown"
                raise RuntimeError(f"资源下载失败 (路径: {failed_path}): {e}") from e
            task_output_dir = self.CACHE_DIR / user_id / scene_id / "output"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            if task_type == "video_dual_chain":
                self._run_video_dual_chain(
                    task=task,
                    video_path=input_path,
                    task_params=task_params,
                    task_output_dir=task_output_dir,
                    on_pipeline_log=on_pipeline_log,
                )
            else:
                self._run_pipeline_once(
                    task=task,
                    task_type=task_type,
                    input_path=input_path,
                    task_params=task_params,
                    work_dir=task_output_dir,
                    on_pipeline_log=on_pipeline_log,
                )

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
                self._record_cloud_log(task_id, f"❌ 严重错误: {str(e)}")
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

            if 'sparse2dgs_tmpdir' in locals():
                try:
                    sparse2dgs_tmpdir.cleanup()
                except Exception as e:
                    print(f"⚠️ 清理 sparse2dgs 临时下载目录失败: {e}")

            # 3. 重置日志
            self.current_task_logs = []
