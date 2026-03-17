from types import SimpleNamespace
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.modules.glomap_runner import GlomapRunner


def make_runner():
    cfg = SimpleNamespace(
        colmap_use_gpu=True,
        colmap_gpu_index="0",
        colmap_bin="",
        glomap_bin="",
    )
    runner = GlomapRunner.__new__(GlomapRunner)
    runner.cfg = cfg
    runner.colmap_use_gpu = cfg.colmap_use_gpu
    runner.colmap_gpu_index = cfg.colmap_gpu_index
    runner.colmap_exe = "/usr/local/bin/colmap"
    runner.glomap_exe = "/usr/local/bin/glomap"
    runner.env = {}
    return runner


def test_build_cpu_fallback_cmd_only_rewrites_current_command():
    cmd = [
        "/usr/local/bin/colmap", "sequential_matcher",
        "--database_path", "/tmp/database.db",
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "2",
    ]

    new_cmd, found_gpu_flag = GlomapRunner._build_cpu_fallback_cmd(cmd)

    assert found_gpu_flag is True
    assert new_cmd != cmd
    assert cmd[5] == "1"
    assert cmd[7] == "2"
    assert new_cmd[5] == "0"
    assert new_cmd[7] == "-1"


def test_command_uses_gpu_reads_effective_command_state():
    assert GlomapRunner._command_uses_gpu(["colmap", "--FeatureMatching.use_gpu", "1"]) is True
    assert GlomapRunner._command_uses_gpu(["colmap", "--FeatureMatching.use_gpu", "0"]) is False


def test_retry_cpu_does_not_disable_future_gpu_usage(monkeypatch):
    runner = make_runner()
    attempts = []

    def fake_popen(cmd, stdout, stderr, text, env):
        attempts.append((list(cmd), dict(env)))

        class FakeStdout:
            def __iter__(self):
                return iter(())

        class FakeProcess:
            def __init__(self, returncode):
                self.stdout = FakeStdout()
                self.returncode = returncode

            def wait(self):
                return self.returncode

        return FakeProcess(returncode=1 if len(attempts) == 1 else 0)

    monkeypatch.setattr("src.modules.glomap_runner.subprocess.Popen", fake_popen)

    runner._run_cmd(
        [
            "/usr/local/bin/colmap", "sequential_matcher",
            "--database_path", "/tmp/database.db",
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.gpu_index", "0",
        ],
        "Step 2: 顺序匹配 (COLMAP GPU)",
        retry_cpu=True,
    )

    assert runner.colmap_use_gpu is True
    assert len(attempts) == 2
    assert attempts[0][0][5] == "1"
    assert attempts[0][0][7] == "0"
    assert attempts[1][0][5] == "0"
    assert attempts[1][0][7] == "-1"
