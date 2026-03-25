import sys
from pathlib import Path

import pytest


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.pipelines.da3_2dgs_pipeline import DA3TwoDGSPipeline


def make_pipeline(tmp_path: Path) -> DA3TwoDGSPipeline:
    context = {
        "task_id": "test_task_da3_2dgs",
        "scene_id": "test_scene_da3_2dgs",
        "work_root": str(tmp_path),
        "log_callback": lambda msg: None,
    }
    return DA3TwoDGSPipeline(context)


def test_resolve_viewer_port_defaults_to_ephemeral(tmp_path: Path):
    pipeline = make_pipeline(tmp_path)
    assert pipeline._resolve_viewer_port(None) == 0
    assert pipeline._resolve_viewer_port("") == 0


def test_resolve_viewer_port_rejects_occupied_port(tmp_path: Path):
    import socket

    pipeline = make_pipeline(tmp_path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen()

    try:
        with pytest.raises(RuntimeError, match="已被占用"):
            pipeline._resolve_viewer_port(port)
    finally:
        sock.close()


def test_run_cmd_streams_newline_logs(tmp_path: Path):
    logs = []
    pipeline = DA3TwoDGSPipeline(
        {
            "task_id": "test_task_da3_2dgs",
            "scene_id": "test_scene_da3_2dgs",
            "work_root": str(tmp_path),
            "log_callback": logs.append,
        }
    )

    pipeline._run_cmd(
        [
            sys.executable,
            "-c",
            "print('hello'); print('world')",
        ],
        cwd=tmp_path,
        env={},
        desc="test",
    )

    assert any("hello" in log for log in logs)
    assert any("world" in log for log in logs)


def test_run_cmd_streams_carriage_return_progress(tmp_path: Path):
    logs = []
    pipeline = DA3TwoDGSPipeline(
        {
            "task_id": "test_task_da3_2dgs",
            "scene_id": "test_scene_da3_2dgs",
            "work_root": str(tmp_path),
            "log_callback": logs.append,
        }
    )

    script = (
        "import sys, time\n"
        "for i in range(3):\n"
        "    sys.stdout.write(f'progress {i}/3\\r')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.02)\n"
        "print('done')\n"
    )

    pipeline._run_cmd(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={},
        desc="test-progress",
    )

    assert any("progress 0/3" in log for log in logs)
    assert any("progress 1/3" in log for log in logs)
    assert any("progress 2/3" in log for log in logs)
    assert any("done" in log for log in logs)
