from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import os
import urllib.parse

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE if ENV_FILE.exists() else None, override=False)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"TOML root must be a table: {path}")
    return data


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


_RAW_SETTINGS = _merge_dict(
    _load_toml(CONFIG_DIR / "default.toml"),
    _load_toml(CONFIG_DIR / "local.toml"),
)


def _config_value(*path: str, default: Any = None) -> Any:
    node: Any = _RAW_SETTINGS
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_or_config(env_key: str, *path: str, default: Any = None) -> Any:
    value = os.getenv(env_key)
    if value is not None and value != "":
        return value
    return _config_value(*path, default=default)


def _get_str(env_key: str, *path: str, default: str = "") -> str:
    value = _env_or_config(env_key, *path, default=default)
    if value is None:
        return default
    return str(value)


def _get_int(env_key: str, *path: str, default: int) -> int:
    value = _env_or_config(env_key, *path, default=default)
    return int(value)


def _get_float(env_key: str, *path: str, default: float) -> float:
    value = _env_or_config(env_key, *path, default=default)
    return float(value)


def _get_bool(env_key: str, *path: str, default: bool) -> bool:
    value = _env_or_config(env_key, *path, default=default)
    return _coerce_bool(value)


def _get_path(env_key: str, *path: str, default: str) -> Path:
    raw = _env_or_config(env_key, *path, default=default)
    if raw in (None, ""):
        return Path("")
    value = Path(str(raw)).expanduser()
    if not value.is_absolute():
        value = (BASE_DIR / value).resolve()
    return value


def _normalize_supabase_url(url: str) -> str:
    if not url:
        return ""
    return url.rstrip("/") + "/"


def _merge_no_proxy_entries(existing: str, *entries: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for raw in (existing, *entries):
        for item in str(raw or "").split(","):
            value = item.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return ",".join(merged)


def ensure_no_proxy_for_url(url: str, *extra_hosts: str) -> str:
    """把目标 URL 的 host 与 host:port 同步加入 NO_PROXY/no_proxy，绕过全局代理。"""
    entries = ["localhost", "127.0.0.1", "::1", *extra_hosts]
    if url:
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname:
            entries.append(parsed.hostname)
            if parsed.port:
                entries.append(f"{parsed.hostname}:{parsed.port}")

    merged = _merge_no_proxy_entries(
        _merge_no_proxy_entries(os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")),
        *entries,
    )
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    return merged


def _apply_module_runtime_env() -> None:
    hf_endpoint = _get_str("HF_ENDPOINT", "runtime", "hf_endpoint", default="")
    proxy_url = _get_str("PROXY_URL", "runtime", "proxy_url", default="")
    pytorch_alloc_conf = _get_str(
        "PYTORCH_ALLOC_CONF",
        "runtime",
        "pytorch_alloc_conf",
        default="",
    )
    no_proxy = _get_str("NO_PROXY", "runtime", "no_proxy", default="")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    if proxy_url:
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["ALL_PROXY"] = proxy_url
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["all_proxy"] = proxy_url
    if pytorch_alloc_conf:
        os.environ["PYTORCH_ALLOC_CONF"] = pytorch_alloc_conf
    if no_proxy:
        merged_no_proxy = _merge_no_proxy_entries(
            _merge_no_proxy_entries(os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")),
            no_proxy,
        )
        os.environ["NO_PROXY"] = merged_no_proxy
        os.environ["no_proxy"] = merged_no_proxy
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"


_apply_module_runtime_env()


@dataclass
class PipelineConfig:
    project_name: str = "default_project"
    video_path: Optional[Path] = None
    work_root: Path = field(
        default_factory=lambda: _get_path("WORK_ROOT", "paths", "work_root", default="./temp_workspace")
    )

    app_env: str = field(default_factory=lambda: _get_str("APP_ENV", "app", "env", default="development"))
    openai_api_key: str = field(default_factory=lambda: _get_str("OPENAI_API_KEY", "api", "openai_api_key", default=""))
    dashscope_api_key: str = field(default_factory=lambda: _get_str("DASHSCOPE_API_KEY", "api", "dashscope_api_key", default=""))
    dashscope_vl_model: str = field(
        default_factory=lambda: _get_str("DASHSCOPE_VL_MODEL", "api", "dashscope_vl_model", default="qwen3-vl-plus")
    )
    dashscope_embedding_model: str = field(
        default_factory=lambda: _get_str(
            "DASHSCOPE_EMBEDDING_MODEL",
            "api",
            "dashscope_embedding_model",
            default="text-embedding-v2",
        )
    )
    dashscope_base_url: str = field(
        default_factory=lambda: _get_str(
            "DASHSCOPE_BASE_URL",
            "api",
            "dashscope_base_url",
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    dashscope_timeout_seconds: float = field(
        default_factory=lambda: _get_float(
            "DASHSCOPE_TIMEOUT_SECONDS",
            "api",
            "dashscope_timeout_seconds",
            default=45.0,
        )
    )
    redis_url: str = field(default_factory=lambda: _get_str("REDIS_URL", "api", "redis_url", default="redis://localhost:6379/0"))
    chroma_db_path: Path = field(
        default_factory=lambda: _get_path("CHROMA_DB_PATH", "storage", "chroma_db_path", default="./data_storage")
    )

    supabase_url: str = field(default_factory=lambda: _normalize_supabase_url(_get_str("SUPABASE_URL", "supabase", "url", default="")))
    supabase_key: str = field(default_factory=lambda: _get_str("SUPABASE_KEY", "supabase", "key", default=""))
    supabase_bucket: str = field(
        default_factory=lambda: _get_str("SUPABASE_BUCKET", "supabase", "bucket", default="braindance-assets")
    )
    supabase_table: str = field(
        default_factory=lambda: _get_str("SUPABASE_TABLE", "supabase", "table", default="processing_tasks")
    )

    gpu_index: int = field(default_factory=lambda: _get_int("GPU_INDEX", "training", "gpu_index", default=0))
    iterations: int = field(default_factory=lambda: _get_int("ITERATIONS", "training", "iterations", default=30000))
    max_images: int = field(default_factory=lambda: _get_int("MAX_IMAGES", "training", "max_images", default=300))
    training_iterations: int = field(
        default_factory=lambda: _get_int("TRAINING_ITERATIONS", "training", "training_iterations", default=15000)
    )
    min_quality_score: int = field(
        default_factory=lambda: _get_int("MIN_QUALITY_SCORE", "training", "min_quality_score", default=40)
    )
    mapper_type: str = field(default_factory=lambda: _get_str("MAPPER_TYPE", "training", "mapper_type", default="glomap"))
    colmap_use_gpu: bool = field(
        default_factory=lambda: _get_bool("COLMAP_USE_GPU", "training", "colmap_use_gpu", default=True)
    )
    colmap_gpu_index: str = field(
        default_factory=lambda: _get_str("COLMAP_GPU_INDEX", "training", "colmap_gpu_index", default="0")
    )

    enable_ai: bool = False
    enable_scene_analysis: bool = True
    force_spherical_culling: bool = False
    scene_radius_scale: float = 1.0
    keep_percentile: float = 0.8

    shared_model_dir: Path = field(
        default_factory=lambda: _get_path("SHARED_MODEL_DIR", "paths", "shared_model_dir", default="../../models")
    )
    sam3d_repo_path: Path = field(
        default_factory=lambda: _get_path("SAM3D_REPO_PATH", "repos", "sam3d_repo_path", default="src/libs/sam-3d-objects")
    )
    sam3d_checkpoint_dir: Path = field(
        default_factory=lambda: _get_path(
            "SAM3D_CHECKPOINT_DIR",
            "repos",
            "sam3d_checkpoint_dir",
            default="../../models/sam3d/checkpoints",
        )
    )
    sharp_repo_path: Path = field(
        default_factory=lambda: _get_path("SHARP_REPO_PATH", "repos", "sharp_repo_path", default="src/libs/ml-sharp")
    )
    da3_repo_path: Path = field(
        default_factory=lambda: _get_path("DA3_REPO_PATH", "repos", "da3_repo_path", default="src/libs/Depth-Anything-3")
    )
    sugar_repo_path: Path = field(
        default_factory=lambda: _get_path("SUGAR_REPO_PATH", "repos", "sugar_repo_path", default="src/libs/SuGaR")
    )
    dgs2_repo_path: Path = field(
        default_factory=lambda: _get_path("DGS2_REPO_PATH", "repos", "dgs2_repo_path", default="src/libs/2d-gaussian-splatting")
    )
    sparse2dgs_repo_path: Path = field(
        default_factory=lambda: _get_path(
            "SPARSE2DGS_REPO_PATH",
            "repos",
            "sparse2dgs_repo_path",
            default="src/libs/Sparse2DGS",
        )
    )
    sparse2dgs_conda_env: str = field(
        default_factory=lambda: _get_str(
            "SPARSE2DGS_CONDA_ENV",
            "repos",
            "sparse2dgs_conda_env",
            default="Braindance",
        )
    )

    model_delivery_format: str = field(
        default_factory=lambda: _get_str("MODEL_DELIVERY_FORMAT", "delivery", "model_delivery_format", default="splat")
    )
    compression_opacity_threshold: float = field(
        default_factory=lambda: _get_float(
            "COMPRESSION_OPACITY_THRESHOLD",
            "delivery",
            "compression_opacity_threshold",
            default=0.05,
        )
    )
    ksplat_alpha_threshold: int = field(
        default_factory=lambda: _get_int("KSPLAT_ALPHA_THRESHOLD", "delivery", "ksplat_alpha_threshold", default=1)
    )
    ksplat_script_path: str = field(
        default_factory=lambda: _get_str("KSPLAT_SCRIPT_PATH", "delivery", "ksplat_script_path", default="")
    )

    hf_endpoint: str = field(
        default_factory=lambda: _get_str("HF_ENDPOINT", "runtime", "hf_endpoint", default="https://hf-mirror.com")
    )
    proxy_url: str = field(
        default_factory=lambda: _get_str("PROXY_URL", "runtime", "proxy_url", default="")
    )
    pytorch_alloc_conf: str = field(
        default_factory=lambda: _get_str(
            "PYTORCH_ALLOC_CONF",
            "runtime",
            "pytorch_alloc_conf",
            default="expandable_segments:True",
        )
    )
    no_proxy: str = field(
        default_factory=lambda: _get_str("NO_PROXY", "runtime", "no_proxy", default="huggingface.co,hf-mirror.com")
    )
    colmap_bin: str = field(default_factory=lambda: _get_str("COLMAP_BIN", "executables", "colmap_bin", default=""))
    glomap_bin: str = field(default_factory=lambda: _get_str("GLOMAP_BIN", "executables", "glomap_bin", default=""))

    @property
    def project_dir(self) -> Path:
        return self.work_root

    @project_dir.setter
    def project_dir(self, value: Path) -> None:
        self.work_root = Path(value)

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "masks"

    @property
    def transforms_file(self) -> Path:
        return self.data_dir / "transforms.json"

    @property
    def vocab_tree_path(self) -> Path:
        return self.work_root / "vocab_tree_flickr100k_words.bin"

    def __post_init__(self) -> None:
        if self.video_path is not None and not isinstance(self.video_path, Path):
            self.video_path = Path(self.video_path)

        path_fields = [
            "work_root",
            "chroma_db_path",
            "shared_model_dir",
            "sam3d_repo_path",
            "sam3d_checkpoint_dir",
            "sharp_repo_path",
            "da3_repo_path",
            "sugar_repo_path",
            "dgs2_repo_path",
            "sparse2dgs_repo_path",
        ]
        for field_name in path_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))

        self.supabase_url = _normalize_supabase_url(self.supabase_url)
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        if self.proxy_url:
            os.environ["HTTP_PROXY"] = self.proxy_url
            os.environ["HTTPS_PROXY"] = self.proxy_url
            os.environ["ALL_PROXY"] = self.proxy_url
            os.environ["http_proxy"] = self.proxy_url
            os.environ["https_proxy"] = self.proxy_url
            os.environ["all_proxy"] = self.proxy_url
        if self.pytorch_alloc_conf:
            os.environ["PYTORCH_ALLOC_CONF"] = self.pytorch_alloc_conf
        if self.no_proxy:
            os.environ["NO_PROXY"] = self.no_proxy
            os.environ["no_proxy"] = self.no_proxy
        if self.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)
        if self.dashscope_api_key:
            os.environ.setdefault("DASHSCOPE_API_KEY", self.dashscope_api_key)

        if os.getenv("GPU_INDEX", "").strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("GPU_INDEX", "").strip()
        elif not os.getenv("CUDA_VISIBLE_DEVICES", "").strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
