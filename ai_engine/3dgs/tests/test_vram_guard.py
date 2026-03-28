import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.utils.vram_guard import VramGuardSettings, compute_target_reserve_bytes


class _ConfigStub:
    vram_guard_enabled = True
    vram_guard_reserve_gb = 24
    vram_guard_min_free_gb = 8
    vram_guard_chunk_gb = 1
    vram_guard_poll_interval_seconds = 2
    vram_guard_startup_timeout_seconds = 15


def test_compute_target_reserve_bytes_keeps_owner_free_budget():
    reserve = compute_target_reserve_bytes(
        free_bytes=40 * 1024**3,
        reserve_target_bytes=24 * 1024**3,
        min_free_bytes=8 * 1024**3,
    )
    assert reserve == 24 * 1024**3


def test_compute_target_reserve_bytes_shrinks_when_free_memory_drops():
    reserve = compute_target_reserve_bytes(
        free_bytes=10 * 1024**3,
        reserve_target_bytes=24 * 1024**3,
        min_free_bytes=8 * 1024**3,
    )
    assert reserve == 2 * 1024**3


def test_vram_guard_settings_from_config_normalizes_ranges():
    cfg = _ConfigStub()
    settings = VramGuardSettings.from_config(cfg)

    assert settings.should_start() is True
    assert settings.reserve_gb == 24.0
    assert settings.min_free_gb == 8.0
    assert settings.chunk_gb == 1.0
