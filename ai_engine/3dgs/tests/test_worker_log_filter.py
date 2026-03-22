import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.core.worker import CloudWorker


def test_contains_emoji_only_matches_emoji_logs():
    assert CloudWorker._contains_emoji("🚀 启动流水线")
    assert CloudWorker._contains_emoji("❌ 严重错误: DA3 解算失败")
    assert not CloudWorker._contains_emoji("正在从云端下载资源...")
    assert not CloudWorker._contains_emoji("下载视频...")


def test_record_cloud_log_keeps_only_emoji_logs():
    worker = CloudWorker.__new__(CloudWorker)
    worker.current_task_logs = []

    sync_calls = []

    def fake_sync(task_id):
        sync_calls.append(task_id)

    worker._sync_log = fake_sync

    worker._record_cloud_log("task_1", "正在从云端下载资源...")
    worker._record_cloud_log("task_1", "🚀 启动流水线")
    worker._record_cloud_log("task_1", "✅ 任务全部完成")

    assert len(worker.current_task_logs) == 2
    assert [entry["msg"] for entry in worker.current_task_logs] == [
        "🚀 启动流水线",
        "✅ 任务全部完成",
    ]
    assert sync_calls == ["task_1", "task_1"]
