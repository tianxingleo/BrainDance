from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests.exceptions import ChunkedEncodingError


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


def test_summarize_event_for_timeline_formats_core_events():
    module = load_module()

    status_summary = module.summarize_event_for_timeline(
        {"event": "status", "data": {"summary": "已完成模式判断"}}
    )
    tool_summary = module.summarize_event_for_timeline(
        {"event": "tool_call", "data": {"toolName": "read_model_assets"}}
    )
    done_summary = module.summarize_event_for_timeline(
        {"event": "done", "data": {"answer": "当前没有找到匹配的模型资产。"}}
    )

    assert status_summary == "已完成模式判断"
    assert tool_summary == "read_model_assets"
    assert done_summary == "当前没有找到匹配的模型资产。"


def test_compute_timing_stats_picks_first_occurrence():
    module = load_module()

    timeline = [
        {"event": "ping", "elapsed_ms": 12.5},
        {"event": "status", "elapsed_ms": 30.0},
        {"event": "tool_call", "elapsed_ms": 80.0},
        {"event": "message", "elapsed_ms": 180.0},
        {"event": "message", "elapsed_ms": 220.0},
        {"event": "done", "elapsed_ms": 300.0},
    ]
    stats = module.compute_timing_stats(timeline)

    assert stats["first_event_ms"] == 12.5
    assert stats["first_status_ms"] == 30.0
    assert stats["first_tool_call_ms"] == 80.0
    assert stats["first_message_ms"] == 180.0
    assert stats["done_ms"] == 300.0
    assert stats["total_events"] == 6


def test_resolve_log_target_supports_directory_mode(tmp_path: Path):
    module = load_module()

    log_target = module.resolve_log_target(tmp_path, "请你找一下洛天依相关的模型")
    event_log_target = module.resolve_event_log_target(tmp_path, "请你找一下洛天依相关的模型")

    assert log_target is not None
    assert event_log_target is not None
    assert log_target.parent == tmp_path
    assert event_log_target.parent == tmp_path
    assert log_target.suffix == ".json"
    assert event_log_target.suffix == ".jsonl"


def test_run_single_turn_reports_partial_context_when_stream_interrupts(monkeypatch: pytest.MonkeyPatch):
    module = load_module()

    class FakeResponse:
        status_code = 200
        reason = "OK"
        headers = {"content-type": "text/event-stream; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=None, decode_unicode=True):
            yield 'event: ping\ndata: {"message":"ok"}\n\n'
            yield 'event: status\ndata: {"phase":"request_received","summary":"已收到请求"}\n\n'
            raise ChunkedEncodingError("Response ended prematurely")

    monkeypatch.setattr(module, "extract_supabase_config", lambda: ("https://example.supabase.co", "test-key"))
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: FakeResponse())

    args = SimpleNamespace(
        accept="sse",
        execution_mode="preview",
        current_scene_id="",
        current_model_id="",
        current_mode="",
        selected_model_ids="",
        candidate_scene_ids="",
        session_id="",
        conversation_summary="",
        session_state_file="",
        show_request=False,
        show_response_meta=False,
        show_raw_event=False,
        show_event_timeline=False,
        show_full_result=False,
        hide_candidates=False,
        hide_tool_trace=False,
        timeout=10,
    )

    with pytest.raises(module.StreamInterruptedError) as exc_info:
        module.run_single_turn("你是谁", args)

    partial_result = exc_info.value.partial_result
    assert "流式响应在收到 done 事件前中断" in str(exc_info.value)
    assert partial_result["response_meta"]["status_code"] == 200
    assert partial_result["stream_error"]["type"] == "chunked_encoding_error"
    assert [item["event"] for item in partial_result["events"]] == ["ping", "status"]
