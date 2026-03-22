import tempfile
import unittest
from pathlib import Path
import sys
import importlib.util

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

ply_utils_path = project_root / "src" / "utils" / "ply_utils.py"
spec = importlib.util.spec_from_file_location("test_ply_utils", ply_utils_path)
ply_utils = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(ply_utils)
rotate_gaussian_ply_x180 = ply_utils.rotate_gaussian_ply_x180


@unittest.skipUnless(HAS_PLYFILE, "plyfile 未安装，跳过 PLY 旋转测试")
class TestSharpOrientationFix(unittest.TestCase):
    def test_rotate_gaussian_ply_x180_updates_positions_and_quaternions(self):
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("opacity", "f4"),
        ]
        vertex = np.array(
            [
                (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5),
                (-4.0, 5.0, -6.0, 0.5, 0.5, 0.5, 0.5, 0.9),
            ],
            dtype=dtype,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "input.ply"
            PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))

            rotate_gaussian_ply_x180(str(ply_path))

            rotated = PlyData.read(str(ply_path))["vertex"].data

        np.testing.assert_allclose(rotated["x"], np.array([1.0, -4.0], dtype=np.float32))
        np.testing.assert_allclose(rotated["y"], np.array([-2.0, -5.0], dtype=np.float32))
        np.testing.assert_allclose(rotated["z"], np.array([-3.0, 6.0], dtype=np.float32))

        np.testing.assert_allclose(rotated["rot_0"], np.array([0.0, -0.5], dtype=np.float32))
        np.testing.assert_allclose(rotated["rot_1"], np.array([1.0, 0.5], dtype=np.float32))
        np.testing.assert_allclose(rotated["rot_2"], np.array([0.0, -0.5], dtype=np.float32))
        np.testing.assert_allclose(rotated["rot_3"], np.array([0.0, 0.5], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
