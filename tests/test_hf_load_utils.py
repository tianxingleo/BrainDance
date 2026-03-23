from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "hf_load_utils.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hf_load_utils_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_from_pretrained_prefers_local_cache(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "model_is_cached_locally", lambda repo_id: True)

    calls: list[dict[str, object]] = []

    def factory(repo_id: str, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        return {"repo_id": repo_id, "kwargs": kwargs}

    result = module.safe_from_pretrained(factory, "Qwen/Qwen3-1.7B", trust_remote_code=True)

    assert result["repo_id"] == "Qwen/Qwen3-1.7B"
    assert calls == [{"repo_id": "Qwen/Qwen3-1.7B", "trust_remote_code": True, "local_files_only": True}]


def test_safe_from_pretrained_falls_back_from_mirror_to_official(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "model_is_cached_locally", lambda repo_id: False)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.delenv("HUGGINGFACE_HUB_ENDPOINT", raising=False)

    calls: list[str] = []

    def factory(repo_id: str, **kwargs):
        endpoint = os.environ.get("HF_ENDPOINT")
        calls.append(endpoint or "<unset>")
        if endpoint == "https://hf-mirror.com":
            raise OSError("502 Bad Gateway")
        return {"repo_id": repo_id, "endpoint": endpoint}

    result = module.safe_from_pretrained(factory, "Qwen/Qwen3-0.6B", trust_remote_code=True)

    assert result["endpoint"] == module.OFFICIAL_HF_ENDPOINT
    assert calls == ["https://hf-mirror.com", module.OFFICIAL_HF_ENDPOINT]
    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"
