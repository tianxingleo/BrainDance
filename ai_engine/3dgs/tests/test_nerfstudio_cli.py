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


def test_resolve_nerfstudio_python_respects_isolated_probe_env(monkeypatch):
    module = _load_nerfstudio_cli_module()

    candidates = [
        Path("/fake/braindance/bin/python"),
        Path("/fake/urban/bin/python"),
    ]

    monkeypatch.setattr(module, "_candidate_env_bin_dirs", lambda preferred_envs=None: [p.parent for p in candidates])
    monkeypatch.setattr(module, "sys", type("FakeSys", (), {"executable": str(candidates[0])})())
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.os, "access", lambda path, mode: True)
    monkeypatch.setattr(module.Path, "exists", lambda self: str(self) in {str(p) for p in candidates})

    seen = []

    def fake_has_modules(python_path, required_modules, probe_env=None):
        seen.append((str(python_path), probe_env))
        if str(python_path) == str(candidates[0]):
            return probe_env != {"PYTHONNOUSERSITE": "1"}
        return True

    monkeypatch.setattr(module, "_python_has_modules", fake_has_modules)

    resolved = module.resolve_nerfstudio_python(["Braindance", "urban_fine_grained_modeling"])

    assert resolved == str(candidates[1])
    assert any(env == {"PYTHONNOUSERSITE": "1"} for _, env in seen)
