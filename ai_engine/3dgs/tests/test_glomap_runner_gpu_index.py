import sys
from pathlib import Path
from unittest.mock import patch

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.config import PipelineConfig
from src.modules.glomap_runner import GlomapRunner


def build_cfg():
    return PipelineConfig(
        colmap_use_gpu=True,
        colmap_gpu_index="1",
        colmap_bin="/usr/local/bin/colmap",
        glomap_bin="/usr/local/bin/glomap",
    )


def test_colmap_gpu_index_maps_physical_id_to_local_visible_index():
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
        runner = GlomapRunner(build_cfg())
    assert runner.colmap_gpu_index == "0"


def test_colmap_gpu_index_preserves_zero_index():
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
        runner = GlomapRunner(
            PipelineConfig(
                colmap_use_gpu=True,
                colmap_gpu_index="0",
                colmap_bin="/usr/local/bin/colmap",
                glomap_bin="/usr/local/bin/glomap",
            )
        )
    assert runner.colmap_gpu_index == "0"


def test_colmap_gpu_index_falls_back_when_out_of_visible_range():
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
        runner = GlomapRunner(
            PipelineConfig(
                colmap_use_gpu=True,
                colmap_gpu_index="3",
                colmap_bin="/usr/local/bin/colmap",
                glomap_bin="/usr/local/bin/glomap",
            )
        )
    assert runner.colmap_gpu_index == "0"
