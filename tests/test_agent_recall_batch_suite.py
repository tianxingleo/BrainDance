from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "scripts"
    / "run_agent_recall_batch_suite.py"
)
SUITE_PATH = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "data"
    / "agent_recall_batch_suite.json"
)


def load_module():
    script_dir = str(MODULE_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("agent_recall_batch_suite_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_suite_flattens_multi_turn_cases():
    module = load_module()

    cases = module.load_suite(SUITE_PATH)

    assert len(cases) == 33
    multi_turn = [case for case in cases if case.case_id == "multi_turn_write_confirm_001"]
    assert len(multi_turn) == 2
    assert multi_turn[0].turn_index == 1
    assert multi_turn[1].turn_index == 2
    assert multi_turn[1].request["execution_mode"] == "execute"


def test_build_cli_args_can_inherit_previous_state():
    module = load_module()
    case = module.TurnCase(
        suite_name="demo",
        case_id="case-1",
        title="demo",
        category="multi_turn",
        tags=("multi_turn",),
        turn_index=2,
        total_turns=2,
        request={
            "query": "继续",
            "inherit_previous_state": True,
        },
        expect={},
        notes="",
    )
    args = SimpleNamespace(accept="", timeout=30)
    previous = {
        "session_state": {"lastMode": "asset_metadata"},
        "conversation_summary": "上一轮已经给出候选。",
        "session_id": "session-demo",
    }

    cli_args = module.build_cli_args(case, args, previous)

    assert cli_args.session_id == "session-demo"
    assert cli_args.conversation_summary == "上一轮已经给出候选。"
    assert cli_args.session_state_file
    module.cleanup_session_state_file(cli_args.session_state_file)


def test_validate_result_checks_basic_success_contract():
    module = load_module()
    case = module.TurnCase(
        suite_name="demo",
        case_id="case-1",
        title="demo",
        category="persona",
        tags=("baseline",),
        turn_index=1,
        total_turns=1,
        request={"query": "你是谁"},
        expect={
            "status_code": 200,
            "require_answer": True,
            "require_done": True,
            "require_mode": True,
            "require_session_state": True,
        },
        notes="",
    )
    result = {
        "response_meta": {"status_code": 200},
        "events": [{"event": "status"}, {"event": "done"}],
        "result": {
            "answer": "我是 BrainDance 助手。",
            "mode": "creative",
            "session_state": {"lastMode": "creative"},
            "tool_trace": [],
            "actions": [],
        },
        "stats": {"timing": {}, "event_counts": {}},
    }

    failures = module.validate_result(case, result)

    assert failures == []
