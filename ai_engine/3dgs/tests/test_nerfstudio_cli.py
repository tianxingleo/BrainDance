import importlib.util
from pathlib import Path


def _load_nerfstudio_cli_module():
    project_root = Path(__file__).resolve().parent.parent
    module_path = project_root / "src" / "utils" / "nerfstudio_cli.py"
    spec = importlib.util.spec_from_file_location("test_nerfstudio_cli_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_nerfstudio_env_prepends_repo_fork_and_torch_flag():
    module = _load_nerfstudio_cli_module()

    env = module.patch_nerfstudio_env({"PYTHONPATH": "/tmp/existing"})

    parts = env["PYTHONPATH"].split(module.os.pathsep)
    assert parts[0] == str(module.get_repo_nerfstudio_root())
    assert parts[1] == "/tmp/existing"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_build_nerfstudio_cli_command_uses_python_module_entrypoint(monkeypatch):
    module = _load_nerfstudio_cli_module()

    monkeypatch.setattr(module, "resolve_nerfstudio_python", lambda preferred_envs=None: "/tmp/fake-python")

    cmd = module.build_nerfstudio_cli_command("ns-export")

    assert cmd == ["/tmp/fake-python", "-m", "nerfstudio.scripts.exporter"]
