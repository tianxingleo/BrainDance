import os
import shutil
from pathlib import Path


class NerfstudioCliNotFoundError(FileNotFoundError):
    """未找到可用的 Nerfstudio CLI 可执行文件。"""


def _candidate_env_bin_dirs() -> list[Path]:
    env_bin_dirs: list[Path] = []
    home = Path.home()
    roots = [
        home / "miniconda3" / "envs",
        home / "miniforge3" / "envs",
        home / "mambaforge" / "envs",
        Path("/opt/conda/envs"),
    ]
    preferred_envs = [
        "Braindance",
        "urban_fine_grained_modeling",
    ]

    for root in roots:
        if not root.exists():
            continue
        for env_name in preferred_envs:
            bin_dir = root / env_name / "bin"
            if bin_dir.exists():
                env_bin_dirs.append(bin_dir)

        for env_dir in sorted(root.iterdir()):
            bin_dir = env_dir / "bin"
            if not bin_dir.exists() or bin_dir in env_bin_dirs:
                continue
            env_bin_dirs.append(bin_dir)

    return env_bin_dirs


def resolve_nerfstudio_cli(command_name: str) -> str:
    """
    解析 `ns-process-data` / `ns-train` / `ns-export`。

    约束：
    - 优先使用专用 Conda 环境中的绝对路径。
    - 最后才回退到 PATH 中的结果。
    - 避免直接命中 ~/.local/bin 下被系统 Python 污染的脚本。
    """
    explicit_path = os.getenv(command_name.upper().replace("-", "_"), "").strip()
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)

    explicit_bin_dir = os.getenv("NERFSTUDIO_BIN_DIR", "").strip()
    if explicit_bin_dir:
        candidate = Path(explicit_bin_dir).expanduser() / command_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    for bin_dir in _candidate_env_bin_dirs():
        candidate = bin_dir / command_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    path_hit = shutil.which(command_name)
    if path_hit:
        return path_hit

    raise NerfstudioCliNotFoundError(
        f"❌ 找不到 Nerfstudio 可执行文件: {command_name}。"
        "请设置 NERFSTUDIO_BIN_DIR 或直接设置对应环境变量。"
    )


def patch_nerfstudio_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """
    为 Nerfstudio 子进程构造隔离环境，阻止 ~/.local site-packages 污染 Conda/系统解释器。
    """
    env = dict(base_env or os.environ.copy())
    env["PYTHONNOUSERSITE"] = "1"
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    return env
