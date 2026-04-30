"""
Smoke tests for BrainDance AI Engine -- 不含 GPU / 网络依赖。

验证:
1. 核心纯工具函数可被正常 import 与调用
2. 源码目录结构完整（__init__.py 存在）
3. 纯函数逻辑正确

运行: pytest ai_engine/tests/test_config_smoke.py -v
"""

import sys
from pathlib import Path

# 将 3dgs/src 加入 sys.path，使其可直接 import（不依赖 pip install -e）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ai_engine/
_SRC_DIR = _PROJECT_ROOT / "3dgs" / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ─── 源码目录结构 ──────────────────────────────────────────────────

class TestSourceStructure:
    """确保关键 __init__.py 存在，项目结构完整。"""

    def test_src_dir_exists(self):
        assert _SRC_DIR.is_dir(), f"src 目录不存在: {_SRC_DIR}"

    def test_src_init_exists(self):
        assert (_SRC_DIR / "__init__.py").exists(), "src/__init__.py 缺失"

    def test_utils_init_exists(self):
        assert (_SRC_DIR / "utils" / "__init__.py").exists(), \
            "utils/__init__.py 缺失"

    def test_core_init_exists(self):
        assert (_SRC_DIR / "core" / "__init__.py").exists(), \
            "core/__init__.py 缺失"

    def test_modules_init_exists(self):
        assert (_SRC_DIR / "modules" / "__init__.py").exists(), \
            "modules/__init__.py 缺失"

    def test_pipelines_init_exists(self):
        assert (_SRC_DIR / "pipelines" / "__init__.py").exists(), \
            "pipelines/__init__.py 缺失"


# ─── format_duration 纯函数 ────────────────────────────────────────

class TestFormatDuration:
    """common.format_duration -- 纯 stdlib，无 GPU/网络依赖。"""

    @staticmethod
    def _fn():
        from utils.common import format_duration  # noqa: F811
        return format_duration

    def test_zero_seconds(self):
        assert self._fn()(0) == "0:00:00"

    def test_less_than_a_minute(self):
        assert self._fn()(30) == "0:00:30"

    def test_exactly_one_minute(self):
        assert self._fn()(60) == "0:01:00"

    def test_one_hour(self):
        assert self._fn()(3600) == "1:00:00"

    def test_mixed_units(self):
        # 3661s = 1h 1m 1s
        assert self._fn()(3661) == "1:01:01"

    def test_large_value(self):
        # 86400s = 24h (一天)
        result = self._fn()(86400)
        assert "1 day" in result or result == "24:00:00"

    def test_float_input_truncates_to_int(self):
        # 90.7s -> int(90.7) = 90s -> 0:01:30
        assert self._fn()(90.7) == "0:01:30"

    def test_negative_input(self):
        # datetime.timedelta(seconds=-5) => '-1 day, 23:59:55'
        result = self._fn()(-5)
        assert isinstance(result, str)


# ─── utils.common 模块可达性 ───────────────────────────────────────

class TestModuleReachability:
    """确认 import 路径正确，模块可正常 import（不触发 GPU 依赖）。"""

    def test_import_common_does_not_crash(self):
        import utils.common
        assert hasattr(utils.common, "format_duration")

    def test_common_module_has_no_side_effects(self):
        """import utils.common 不应自动加载 cv2/torch 等重型库"""
        import utils.common
        # common.py 只 import datetime
        assert "cv2" not in dir(utils.common)
        assert "torch" not in dir(utils.common)
