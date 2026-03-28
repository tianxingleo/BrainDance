import os
import shutil
import subprocess
import sys
from pathlib import Path


class NerfstudioCliNotFoundError(FileNotFoundError):
    """未找到可用的 Nerfstudio CLI 可执行文件。"""


def _candidate_env_bin_dirs(preferred_envs: list[str] | None = None) -> list[Path]:
    env_bin_dirs: list[Path] = []
    home = Path.home()
    roots = [
        home / "miniconda3" / "envs",
        home / "miniforge3" / "envs",
        home / "mambaforge" / "envs",
        Path("/opt/conda/envs"),
    ]
    preferred_envs = preferred_envs or [
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


def _python_has_modules(python_path: Path, required_modules: list[str]) -> bool:
    if not required_modules:
        return True

    probe = (
        "import importlib.util, sys\n"
        f"mods = {required_modules!r}\n"
        "missing = [name for name in mods if importlib.util.find_spec(name) is None]\n"
        "sys.exit(0 if not missing else 1)\n"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def resolve_python_executable(
    required_modules: list[str] | None = None,
    preferred_envs: list[str] | None = None,
) -> str:
    """
    为子进程解析 Python 解释器。

    约束：
    - 优先满足 required_modules，避免主进程落在错误环境时把缺包问题带给子进程。
    - 3DGS 默认优先选择 Braindance 环境。
    """
    required_modules = required_modules or []
    candidates: list[Path] = []

    explicit_path = os.getenv("PYTHON_EXECUTABLE", "").strip()
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    candidates.append(Path(sys.executable))

    for bin_dir in _candidate_env_bin_dirs(preferred_envs):
        candidates.append(bin_dir / "python")

    path_hit = shutil.which("python")
    if path_hit:
        candidates.append(Path(path_hit))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists() or not os.access(candidate, os.X_OK):
            continue
        seen.add(key)
        deduped.append(candidate)

    for candidate in deduped:
        if _python_has_modules(candidate, required_modules):
            return str(candidate)

    if deduped:
        return str(deduped[0])

    raise FileNotFoundError("❌ 找不到可用的 Python 解释器。")


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
