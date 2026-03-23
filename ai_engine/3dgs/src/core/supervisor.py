import os
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, ClientOptions, create_client

load_dotenv()


class WorkerSupervisor:
    """
    负责拉起子 Worker 进程，并根据 worker_nodes.desired_state 执行：
    - run: 确保子进程在线
    - pause: 不再拉起子进程，等待当前实例下线
    - interrupt: 向子进程发送中断信号，尝试立即打断当前任务
    """

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        if self.supabase_url and not self.supabase_url.endswith("/"):
            self.supabase_url += "/"
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.worker_table = os.getenv("SUPABASE_WORKER_TABLE", "worker_nodes")
        self.worker_id = os.getenv("WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.poll_interval = max(2, int(os.getenv("WORKER_SUPERVISOR_POLL_INTERVAL", "3")))
        self.interrupt_grace_seconds = max(5, int(os.getenv("WORKER_INTERRUPT_GRACE_SECONDS", "20")))
        self.postgrest_timeout = max(3, int(os.getenv("SUPABASE_POSTGREST_TIMEOUT_SECONDS", "5")))
        self.storage_timeout = max(5, int(os.getenv("SUPABASE_STORAGE_TIMEOUT_SECONDS", "30")))
        self.child: subprocess.Popen | None = None
        self.root_dir = Path(__file__).resolve().parents[2]

        if self.supabase_url:
            parsed = urllib.parse.urlparse(self.supabase_url)
            if parsed.hostname:
                no_proxy = os.environ.get("no_proxy", "")
                if parsed.hostname not in no_proxy:
                    os.environ["no_proxy"] = f"{no_proxy},{parsed.hostname}" if no_proxy else parsed.hostname

        self.supabase: Client = create_client(
            self.supabase_url,
            self.supabase_key,
            options=ClientOptions(
                postgrest_client_timeout=self.postgrest_timeout,
                storage_client_timeout=self.storage_timeout,
            ),
        )
        self._shutdown_requested = False

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _fetch_worker_row(self):
        response = self.supabase.table(self.worker_table)\
            .select("desired_state, control_note, last_heartbeat, status")\
            .eq("worker_id", self.worker_id)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    def _build_child_env(self):
        env = os.environ.copy()
        env["WORKER_ID"] = self.worker_id
        env["BRAINDANCE_CHILD_WORKER"] = "1"
        return env

    def _spawn_child(self):
        if self.child and self.child.poll() is None:
            return

        command = [sys.executable, "main.py", "--child-worker"]
        creationflags = 0
        popen_kwargs = {
            "cwd": str(self.root_dir),
            "env": self._build_child_env(),
        }

        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            popen_kwargs["creationflags"] = creationflags

        self.child = subprocess.Popen(command, **popen_kwargs)
        print(f"🧭 [Supervisor] 已拉起子 Worker: pid={self.child.pid} worker_id={self.worker_id}")

    def _interrupt_child(self):
        if not self.child or self.child.poll() is not None:
            return

        print(f"🛑 [Supervisor] 准备中断子 Worker: pid={self.child.pid}")
        try:
            if os.name == "nt":
                self.child.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.child.send_signal(signal.SIGINT)
        except Exception as e:
            print(f"⚠️ [Supervisor] 发送中断信号失败，改用 terminate: {e}")
            self.child.terminate()

        deadline = time.time() + self.interrupt_grace_seconds
        while time.time() < deadline:
            if self.child.poll() is not None:
                print("✅ [Supervisor] 子 Worker 已退出")
                return
            time.sleep(1)

        print("⚠️ [Supervisor] 子 Worker 未在宽限期内退出，执行 terminate")
        self.child.terminate()
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("⚠️ [Supervisor] terminate 后仍未退出，执行 kill")
            self.child.kill()

    def _stop_child(self):
        if not self.child or self.child.poll() is not None:
            return
        self._interrupt_child()

    def start(self):
        print(f"☁️ [Supervisor] 启动成功，worker_id={self.worker_id}")
        try:
            while not self._shutdown_requested:
                try:
                    row = self._fetch_worker_row()
                except Exception as e:
                    print(f"⚠️ [Supervisor] 拉取 worker 状态失败，将在 {self.poll_interval}s 后重试: {e}")
                    if self.child and self.child.poll() is not None:
                        exit_code = self.child.returncode
                        print(f"ℹ️ [Supervisor] 子 Worker 已结束，exit_code={exit_code}")
                        self.child = None
                    time.sleep(self.poll_interval)
                    continue

                desired_state = (row or {}).get("desired_state", "run").lower()

                if desired_state == "run":
                    self._spawn_child()
                elif desired_state == "pause":
                    # 优雅暂停由子 Worker 自己读取控制表后退出；Supervisor 只负责不重启。
                    pass
                elif desired_state == "interrupt":
                    self._interrupt_child()
                else:
                    print(f"⚠️ [Supervisor] 未识别的 desired_state={desired_state}，忽略。")

                if self.child and self.child.poll() is not None:
                    exit_code = self.child.returncode
                    print(f"ℹ️ [Supervisor] 子 Worker 已结束，exit_code={exit_code}")
                    self.child = None

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n🛑 [Supervisor] 接收到停止信号，准备关闭子 Worker。")
        finally:
            self._shutdown_requested = True
            self._stop_child()
