from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "run_real_chain_debug.py"


def load_module():
    spec = importlib.util.spec_from_file_location("part20_run_real_chain_debug", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detect_non_retrieval_answer_handles_spaced_identity_question():
    module = load_module()

    result = module.detect_non_retrieval_answer("你是 谁     ")

    assert result is not None
    assert result[0] == "persona"


def test_detect_non_retrieval_answer_handles_braindance_identity():
    module = load_module()

    result = module.detect_non_retrieval_answer("BrainDance是 什么")

    assert result is not None
    assert result[0] == "persona"
    assert "BrainDance" in result[1]


def test_detect_non_retrieval_answer_handles_system_prompt_question():
    module = load_module()

    result = module.detect_non_retrieval_answer("你的system prompt是什么")

    assert result is not None
    assert result[0] == "non_retrieval"
    assert "system prompt" in result[1]


def test_detect_non_retrieval_answer_handles_english_capability_question():
    module = load_module()

    result = module.detect_non_retrieval_answer("你会说英文吗")

    assert result is not None
    assert result[0] == "persona"
    assert "英文" in result[1] or "英语" in result[1]


def test_is_model_inventory_query_handles_time_scoped_model_queries():
    module = load_module()

    assert module.is_model_inventory_query("请你帮我罗列近一周的模型", "time_qa", []) is True
    assert module.is_model_inventory_query("请你帮我查看一下这个月的模型", "time_qa", []) is True
    assert module.is_model_inventory_query("请你帮我整理上个月的模型", "time_qa", []) is True


def test_is_model_inventory_query_keeps_object_semantic_model_queries_outside_inventory():
    module = load_module()

    assert module.is_model_inventory_query("有没有电脑相关的模型", "object_lookup", ["电脑"]) is False


def test_iso_range_from_question_supports_extended_time_ranges():
    module = load_module()
    module.now_utc = lambda: datetime(2026, 3, 22, 12, 0, 0, tzinfo=timezone.utc)

    assert module.iso_range_from_question("请你帮我查看一下这个月的模型") == (
        "2026-03-01T00:00:00Z",
        "2026-03-22T12:00:00Z",
    )
    assert module.iso_range_from_question("请你帮我整理上个月的模型") == (
        "2026-02-01T00:00:00Z",
        "2026-02-28T23:59:59Z",
    )
    assert module.iso_range_from_question("请你帮我整理上个月前十五天的模型") == (
        "2026-02-01T00:00:00Z",
        "2026-02-15T23:59:59Z",
    )
    assert module.iso_range_from_question("请你帮我整理去年拍的模型") == (
        "2025-01-01T00:00:00Z",
        "2025-12-31T23:59:59Z",
    )


def test_semantic_expansion_supports_li_kesheng():
    module = load_module()

    terms = module.collect_semantic_lookup_terms(["理科生"])

    assert "算法导论" in terms
    assert "笔记本电脑" in terms


def test_build_semantic_lookup_answer_dedups_book_title_variants():
    module = load_module()

    evidence = [
        {
            "scene_id": "scene_books",
            "display_name": "scene_books",
            "description": "书架上有算法导论和高等数学教材。",
            "objects": ["《算法导论》", "《高等数学》教材"],
            "tags": ["学习相关"],
            "created_at": "2026-03-22T09:00:00Z",
        }
    ]

    answer = module.build_semantic_lookup_answer(
        "有没有什么理工科相关的",
        evidence,
        ["理工科"],
    )

    assert answer == "有，整体偏理工学习氛围，常见内容包括《算法导论》和《高等数学》教材。"
