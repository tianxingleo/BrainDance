from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "agent_recall_debug_cli.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agent_recall_debug_cli_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_drain_streaming_events_supports_sse_chunks():
    module = load_module()

    raw = (
        "event: status\n"
        'data: {"phase":"request_received","summary":"已收到请求"}\n\n'
        "event: message\n"
        'data: {"delta":"你好"}\n\n'
    )
    parsed = module.drain_streaming_events(raw)

    assert parsed.remaining == ""
    assert [item["event"] for item in parsed.events] == ["status", "message"]
    assert parsed.events[0]["data"]["phase"] == "request_received"
    assert parsed.events[1]["data"]["delta"] == "你好"


def test_drain_streaming_events_supports_ndjson_and_remaining_tail():
    module = load_module()

    raw = (
        '{"event":"status","data":{"summary":"阶段一"}}\n'
        '{"event":"message","data":{"delta":"半截"}}'
    )
    parsed = module.drain_streaming_events(raw)

    assert len(parsed.events) == 1
    assert parsed.events[0]["event"] == "status"
    assert parsed.remaining == '{"event":"message","data":{"delta":"半截"}}'


def test_build_payload_matches_flutter_fields(tmp_path: Path):
    module = load_module()
    session_state_file = tmp_path / "session_state.json"
    session_state_file.write_text(
        '{"lastMode":"spatial_search","lastSelectedModelIds":["m1"]}',
        encoding="utf-8",
    )
    args = module.parse_args.__globals__["argparse"].Namespace(
        execution_mode="preview",
        current_scene_id="scene-1",
        current_model_id="model-1",
        current_mode="batch_edit",
        selected_model_ids="m1,m2",
        candidate_scene_ids="scene-2,scene-3",
        session_id="session-x",
        conversation_summary="上一轮已确认目标",
        session_state_file=str(session_state_file),
    )

    payload = module.build_payload("把最新两个模型都改名", args)

    assert payload == {
        "query": "把最新两个模型都改名",
        "executionMode": "preview",
        "selectedModelIds": ["m1", "m2"],
        "currentSceneId": "scene-1",
        "currentModelId": "model-1",
        "currentMode": "batch_edit",
        "candidateSceneIds": ["scene-2", "scene-3"],
        "sessionId": "session-x",
        "conversationSummary": "上一轮已确认目标",
        "sessionState": {
            "lastMode": "spatial_search",
            "lastSelectedModelIds": ["m1"],
        },
    }
