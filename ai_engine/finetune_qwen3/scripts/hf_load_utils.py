#!/usr/bin/env python3
"""Utilities for robust Hugging Face loading in environments with flaky mirrors."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
MIRROR_HOST_HINTS = ("hf-mirror.com",)


def _repo_cache_dir_name(repo_id: str) -> str:
    namespace, _, name = repo_id.partition("/")
    if namespace and name:
        return f"models--{namespace}--{name}"
    return f"models--{repo_id}"


def _candidate_cache_roots() -> list[Path]:
    roots: list[Path] = []
    env_candidates = [
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("TRANSFORMERS_CACHE"),
    ]
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        env_candidates.append(str(Path(hf_home) / "hub"))
    env_candidates.append(str(Path.home() / ".cache" / "huggingface" / "hub"))

    for raw in env_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path not in roots:
            roots.append(path)
    return roots


def model_is_cached_locally(repo_id: str) -> bool:
    cache_dir_name = _repo_cache_dir_name(repo_id)
    for root in _candidate_cache_roots():
        try:
            cache_dir = root / cache_dir_name
            if not cache_dir.exists():
                continue
            if (cache_dir / "snapshots").exists() or (cache_dir / "refs").exists():
                return True
        except PermissionError:
            continue
    return False


@contextmanager
def temporary_hf_endpoint(endpoint: str | None):
    previous_hf_endpoint = os.environ.get("HF_ENDPOINT")
    previous_hub_endpoint = os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
    try:
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
            os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
            os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
        yield
    finally:
        if previous_hf_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous_hf_endpoint
        if previous_hub_endpoint is None:
            os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
        else:
            os.environ["HUGGINGFACE_HUB_ENDPOINT"] = previous_hub_endpoint


def _using_mirror_endpoint() -> bool:
    endpoint = (os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT") or "").strip().lower()
    return any(host in endpoint for host in MIRROR_HOST_HINTS)


def _offline_requested() -> bool:
    return any(
        str(os.getenv(key) or "").strip().lower() in {"1", "true", "yes", "on"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def safe_from_pretrained(
    factory: Callable[..., Any],
    repo_id: str,
    /,
    **kwargs: Any,
) -> Any:
    """Load a HF artifact robustly with local-cache and official-endpoint fallback."""

    cached = model_is_cached_locally(repo_id)
    errors: list[Exception] = []

    if cached:
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        try:
            return factory(repo_id, **local_kwargs)
        except Exception as exc:
            errors.append(exc)
            if _offline_requested():
                raise RuntimeError(
                    f"本地缓存加载失败，且当前处于离线模式：repo={repo_id}"
                ) from exc

    try:
        return factory(repo_id, **kwargs)
    except Exception as exc:
        errors.append(exc)
        if not _using_mirror_endpoint():
            raise

    with temporary_hf_endpoint(OFFICIAL_HF_ENDPOINT):
        try:
            return factory(repo_id, **kwargs)
        except Exception as exc:
            errors.append(exc)

    raise RuntimeError(
        f"Hugging Face 资源加载失败：repo={repo_id}。"
        f"已尝试本地缓存、当前镜像，以及官方端点回退。"
    ) from errors[-1]
