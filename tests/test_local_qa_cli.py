from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "local_qa_cli.py"


def load_module():
    spec = importlib.util.spec_from_file_location("local_qa_cli_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_format_chain_preview_renders_core_routes():
    module = load_module()

    chain = {
        "query_class": "object_lookup",
        "retrieval": {
            "intent": "object_lookup",
            "hit_count": 1,
            "retrieval_route": "lexical_fallback",
            "fallback_trigger_reason": "rpc_empty",
            "answer_route": "must_answer_focus_formatter",
            "evidence": [
                {
                    "display_name": "scene_a",
                    "scene_id": "scene_a",
                    "description": "桌面上有笔记本电脑和显示器。",
                    "objects": ["笔记本电脑", "显示器"],
                    "created_at": "2026-03-22T10:00:00Z",
                }
            ],
        },
    }

    preview = module.format_chain_preview(chain, include_evidence=True)

    assert "query_class: object_lookup" in preview
    assert "retrieval_route: lexical_fallback" in preview
    assert "answer_route: must_answer_focus_formatter" in preview
    assert "objects: 笔记本电脑、显示器" in preview


def test_resolve_answer_prefers_special_answer():
    module = load_module()

    answer, latency = module.resolve_answer(
        chain={"special_answer": "最近拍到过地球仪。", "retrieval": {}},
        question="最近拍到过什么地球仪相关画面？",
        tokenizer=None,
        model=None,
        device="cpu",
        max_new_tokens=96,
    )

    assert answer == "最近拍到过地球仪。"
    assert latency == 0.0
