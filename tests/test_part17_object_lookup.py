from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "run_real_chain_debug.py"


def load_module():
    spec = importlib.util.spec_from_file_location("part17_run_real_chain_debug", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_lookup_terms_cleans_generic_suffixes():
    module = load_module()

    terms = module.normalize_lookup_terms("洛天依模型", "显示器画面", "笔记本电脑相关内容")

    assert "洛天依" in terms
    assert "显示器" in terms
    assert "笔记本电脑" in terms


def test_split_target_objects_splits_multi_target_entities():
    module = load_module()

    targets = module.split_target_objects(["显示器和钢琴", "笔记本电脑、地球仪和钢琴"])

    assert targets == ["显示器", "钢琴", "笔记本电脑", "地球仪"]


def test_select_object_lookup_queries_prefers_canonical_targets():
    module = load_module()

    queries = module.select_object_lookup_queries(
        "最近有没有笔记本电脑相关的画面",
        ["笔记本电脑相关内容"],
        ["笔记本电脑相关内容", "笔记本电脑", "画面"],
    )

    assert queries[0] == "笔记本电脑"
    assert "画面" not in queries


def test_merge_object_candidates_dedups_and_reranks_by_target_match_and_recency():
    module = load_module()

    older_vector = {
        "id": "1",
        "scene_id": "older-scene",
        "description": "桌面上有显示器和键盘。",
        "objects": ["显示器", "键盘"],
        "tags": ["办公桌"],
        "created_at": "2026-03-10T10:00:00Z",
    }
    newer_lexical = {
        "id": "2",
        "scene_id": "newer-scene",
        "description": "桌面上有显示器、钢琴模型和笔记本电脑。",
        "objects": ["显示器", "钢琴模型", "笔记本电脑"],
        "tags": ["办公桌", "模型"],
        "created_at": "2026-03-12T10:00:00Z",
    }
    duplicate_lexical = {
        "id": "1",
        "scene_id": "older-scene",
        "description": "桌面上有显示器和键盘。",
        "objects": ["显示器", "键盘"],
        "tags": ["办公桌"],
        "created_at": "2026-03-10T10:00:00Z",
    }

    merged = module.merge_object_candidates(
        [older_vector],
        [newer_lexical, duplicate_lexical],
        lookup_terms=["显示器", "钢琴"],
        target_objects=["显示器", "钢琴"],
        limit=5,
    )

    assert [row["id"] for row in merged] == ["2", "1"]
    assert merged[1]["_candidate_sources"] == ["lexical", "vector"]
