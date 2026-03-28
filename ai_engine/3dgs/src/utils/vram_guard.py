import atexit
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


_ENV_CHILD_FLAG = "BRAINDANCE_VRAM_GUARD_CHILD"


@dataclass(frozen=True)
class VramGuardSettings:
    enabled: bool
    reserve_gb: float
    min_free_gb: float
    chunk_gb: float
    poll_interval_seconds: float
    startup_timeout_seconds: float

    @classmethod
    def from_config(cls, config) -> "VramGuardSettings":
        return cls(
            enabled=bool(config.vram_guard_enabled),
            reserve_gb=max(0.0, float(config.vram_guard_reserve_gb)),
            min_free_gb=max(0.0, float(config.vram_guard_min_free_gb)),
            chunk_gb=max(0.25, float(config.vram_guard_chunk_gb)),
            poll_interval_seconds=max(0.2, float(config.vram_guard_poll_interval_seconds)),
            startup_timeout_seconds=max(1.0, float(config.vram_guard_startup_timeout_seconds)),
        )

    def should_start(self) -> bool:
        return self.enabled and self.reserve_gb > 0


def compute_target_reserve_bytes(
    free_bytes: int,
    reserve_target_bytes: int,
    min_free_bytes: int,
) -> int:
    if reserve_target_bytes <= 0 or free_bytes <= 0:
        return 0
    allowed = max(0, int(free_bytes) - int(min_free_bytes))
    return min(int(reserve_target_bytes), allowed)


class VramGuardProcess:
    def __init__(self, settings: VramGuardSettings):
        self.settings = settings
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if not self.settings.should_start():
            return
        if os.getenv(_ENV_CHILD_FLAG) == "1":
            return
        if self.process and self.process.poll() is None:
            return

        env = os.environ.copy()
        env[_ENV_CHILD_FLAG] = "1"
        command = [sys.executable, "-m", "src.utils.vram_guard", "--serve"]
        self.process = subprocess.Popen(command, env=env, cwd=str(Path(__file__).resolve().parents[2]))
        atexit.register(self.stop)
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        if not self.process:
            return
        deadline = time.time() + self.settings.startup_timeout_seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                print(f"⚠️ [VRAMGuard] 守护进程启动失败，退出码={self.process.returncode}")
                self.process = None
                return
            time.sleep(0.2)
        if self.process and self.process.poll() is None:
            return
        print("⚠️ [VRAMGuard] 等待守护进程启动超时，将继续执行主流程。")

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None


@contextmanager
def managed_vram_guard(config):
    settings = VramGuardSettings.from_config(config)
    guard = VramGuardProcess(settings)
    guard.start()
    try:
        yield guard
    finally:
        guard.stop()


class _Allocator:
    def __init__(self, device_index: int, chunk_bytes: int):
        self.device_index = device_index
        self.chunk_bytes = max(1, int(chunk_bytes))
        self._buffers = []

    @property
    def reserved_bytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size() for buffer in self._buffers)

    def reserve_until(self, target_bytes: int, torch_module) -> None:
        while self.reserved_bytes < target_bytes:
            next_size = min(self.chunk_bytes, target_bytes - self.reserved_bytes)
            tensor = torch_module.empty(next_size, dtype=torch_module.uint8, device=f"cuda:{self.device_index}")
            self._buffers.append(tensor)

    def release_until(self, target_bytes: int, torch_module) -> None:
        while self.reserved_bytes > target_bytes and self._buffers:
            self._buffers.pop()
        torch_module.cuda.empty_cache()
        if hasattr(torch_module.cuda, "ipc_collect"):
            torch_module.cuda.ipc_collect()

    def release_all(self, torch_module) -> None:
        self._buffers.clear()
        torch_module.cuda.empty_cache()
        if hasattr(torch_module.cuda, "ipc_collect"):
            torch_module.cuda.ipc_collect()


def _run_guard_loop() -> int:
    try:
        from src.config import PipelineConfig
    except Exception as exc:
        print(f"⚠️ [VRAMGuard] 加载配置失败: {exc}")
        return 1

    settings = VramGuardSettings.from_config(PipelineConfig())
    if not settings.should_start():
        print("ℹ️ [VRAMGuard] 未启用或未配置预留容量，直接退出。")
        return 0

    try:
        import torch
    except Exception as exc:
        print(f"⚠️ [VRAMGuard] 未能导入 torch，跳过显存守护: {exc}")
        return 0

    if not torch.cuda.is_available():
        print("⚠️ [VRAMGuard] 当前环境不可用 CUDA，跳过显存守护。")
        return 0

    device_index = 0
    torch.cuda.set_device(device_index)

    allocator = _Allocator(
        device_index=device_index,
        chunk_bytes=int(settings.chunk_gb * (1024 ** 3)),
    )
    reserve_target_bytes = int(settings.reserve_gb * (1024 ** 3))
    min_free_bytes = int(settings.min_free_gb * (1024 ** 3))
    running = True

    def _stop(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(
        "🛡️ [VRAMGuard] 已启动: "
        f"reserve={settings.reserve_gb:.1f}GB "
        f"min_free={settings.min_free_gb:.1f}GB "
        f"chunk={settings.chunk_gb:.2f}GB"
    )

    while running:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            del total_bytes
            target_bytes = compute_target_reserve_bytes(
                free_bytes=free_bytes,
                reserve_target_bytes=reserve_target_bytes,
                min_free_bytes=min_free_bytes,
            )
            if allocator.reserved_bytes > target_bytes:
                allocator.release_until(target_bytes, torch)
            elif allocator.reserved_bytes < target_bytes:
                allocator.reserve_until(target_bytes, torch)
        except RuntimeError as exc:
            print(f"⚠️ [VRAMGuard] 显存调整失败，先释放后重试: {exc}")
            allocator.release_all(torch)
        except Exception as exc:
            print(f"⚠️ [VRAMGuard] 运行异常: {exc}")
        time.sleep(settings.poll_interval_seconds)

    allocator.release_all(torch)
    print("🧹 [VRAMGuard] 已停止并释放预留显存。")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if "--serve" not in argv:
        print("Usage: python -m src.utils.vram_guard --serve")
        return 1
    return _run_guard_loop()


if __name__ == "__main__":
    raise SystemExit(main())
