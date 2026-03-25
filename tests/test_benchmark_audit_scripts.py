from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeSeries(list):
    def sum(self):
        return sum(self)

    def mean(self):
        return sum(self) / len(self) if self else 0.0

    def tolist(self):
        return list(self)


class FakeILoc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        return self._rows[index]


class FakeDataFrame:
    def __init__(self, rows):
        self._rows = list(rows)
        self.iloc = FakeILoc(self._rows)

    @property
    def empty(self):
        return len(self._rows) == 0

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return FakeSeries(row[key] for row in self._rows)


def install_fake_plotting_modules():
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = FakeDataFrame

    fake_plotly = types.ModuleType("plotly")
    fake_plotly_express = types.ModuleType("plotly.express")

    sys.modules.setdefault("pandas", fake_pandas)
    sys.modules.setdefault("plotly", fake_plotly)
    sys.modules.setdefault("plotly.express", fake_plotly_express)


def load_module(name: str, relative_path: str):
    install_fake_plotting_modules()
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_recent_scene_candidates_adds_contrastive_and_semantic_rows():
    module = load_module(
        "build_recent_scene_expansion_candidates_test",
        "ai_engine/finetune_qwen3/scripts/build_recent_scene_expansion_candidates.py",
    )

    recent_rows = [
        {
            "scene_id": "scene_a",
            "objects": ["猫", "风扇", "冰箱"],
            "tags": ["室内", "静物"],
        },
        {
            "scene_id": "scene_b",
            "objects": ["吉他", "贝斯"],
            "tags": ["乐队", "动漫"],
        },
    ]

    rows = module.build_candidate_rows(recent_rows, covered_scene_ids=set())

    groups = [row["group"] for row in rows]
    assert groups.count("must_answer") == 2
    assert groups.count("partial_coverage") == 4
    assert groups.count("abstract_semantic") == 2
    contrastive = [row for row in rows if row["notes"].endswith("contrastive candidate")]
    assert contrastive
    assert contrastive[0]["unsupported_objects"]


def test_extract_summary_rows_supports_multi_candidate_eval_payload():
    module = load_module(
        "build_benchmark_audit_dashboard_test",
        "ai_engine/finetune_qwen3/scripts/build_benchmark_audit_dashboard.py",
    )

    payload = {
        "summary": [
            {
                "candidate_id": "qwen3_1p7b_lora",
                "candidate_label": "1.7B LoRA",
                "end_to_end_pass_rate": 0.55,
                "avg_total_ms": 123.4,
            },
            {
                "candidate_id": "qwen3_0p6b_lora",
                "candidate_label": "0.6B LoRA",
                "end_to_end_pass_rate": 0.33,
                "avg_total_ms": 98.7,
            },
        ]
    }

    rows = module.extract_summary_rows("unseen_ood_benchmark_20260324_summary.json", payload, "unseen_ood")

    assert [row["label"] for row in rows] == ["1.7B LoRA", "0.6B LoRA"]
    assert rows[0]["benchmark_kind"] == "unseen_ood"
    assert rows[1]["avg_total_ms"] == 98.7


def test_build_model_coverage_tracks_all_major_eval_kinds():
    module = load_module(
        "build_benchmark_audit_dashboard_coverage_test",
        "ai_engine/finetune_qwen3/scripts/build_benchmark_audit_dashboard.py",
    )

    local_versions = [
        {
            "version_name": "qwen3_1p7b_lora_sft_round4_1_patch_mixed",
            "label": "1.7B LoRA",
            "root": "outputs",
            "path": "/tmp/model",
            "version_type": "lora",
            "is_smoke_only": False,
        }
    ]
    logged_rows = [
        {"benchmark_kind": "original_benchmark", "label": "1.7B LoRA", "log_file": "benchmark_qwen3_1p7b_lora.json"},
        {"benchmark_kind": "strict_v3", "label": "1.7B LoRA", "log_file": "benchmark_strict_v3_lora_20260322.json"},
        {"benchmark_kind": "unseen_ood", "label": "1.7B LoRA", "log_file": "unseen_ood_benchmark_20260324_summary.json"},
        {"benchmark_kind": "spatial_hardcases", "label": "1.7B LoRA", "log_file": "spatial_hardcases_candidates_20260324_summary.json"},
        {"benchmark_kind": "deployment_eval", "label": "1.7B LoRA", "log_file": "deployment_eval_part29_summary.json"},
    ]

    df = module.build_model_coverage_df(local_versions, logged_rows)

    row = df.iloc[0]
    assert row["major_eval_count"] == 5
    assert bool(row["strict_v3_logged"]) is True
    assert bool(row["deployment_eval_logged"]) is True
    assert row["missing_major_eval_kinds"] == []
