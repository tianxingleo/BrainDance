#!/usr/bin/env python3
"""Desktop debug CLI for the Flutter agent-recall flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_real_chain_debug import extract_supabase_config


DEFAULT_TIMEOUT_SEC = 300
DEFAULT_ACCEPT = "text/event-stream, application/x-ndjson"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "agent_recall_debug"


@dataclass
class ParsedEvents:
    events: list[dict[str, Any]]
    remaining: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the Flutter agent-recall flow on desktop")
    parser.add_argument("--query", default="", help="单轮问题；传入后执行一次即退出")
    parser.add_argument("--execution-mode", choices=("preview", "execute"), default="preview")
    parser.add_argument("--current-scene-id", default="")
    parser.add_argument("--current-model-id", default="")
    parser.add_argument("--current-mode", choices=("search", "compare", "batch_edit", "collection"), default="")
    parser.add_argument("--selected-model-ids", default="", help="逗号分隔的 model id 列表")
    parser.add_argument("--candidate-scene-ids", default="", help="逗号分隔的 scene id 列表")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--conversation-summary", default="")
    parser.add_argument("--session-state-file", default="", help="传入 JSON 文件，内容应与 Flutter session_state 一致")
    parser.add_argument("--accept", choices=("sse", "ndjson"), default="sse")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--log-file", default="", help="把完整事件与最终结果落盘为 JSON")
    parser.add_argument("--show-raw-event", action="store_true", help="额外打印原始事件 JSON")
    parser.add_argument("--hide-candidates", action="store_true")
    parser.add_argument("--hide-tool-trace", action="store_true")
    parser.add_argument("--non-interactive", action="store_true", help="未传 --query 时直接退出")
    return parser.parse_args()


def parse_csv_arg(text: str) -> list[str] | None:
    values = [item.strip() for item in (text or "").split(",") if item.strip()]
    return values or None


def load_session_state(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("session_state_file 必须是 JSON 对象")
    return payload


def build_endpoint(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return f"{trimmed}/functions/v1/agent-recall"


def build_payload(question: str, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": question.strip(),
        "executionMode": args.execution_mode,
    }
    selected_model_ids = parse_csv_arg(args.selected_model_ids)
    candidate_scene_ids = parse_csv_arg(args.candidate_scene_ids)
    session_state = load_session_state(args.session_state_file)
    if selected_model_ids:
        payload["selectedModelIds"] = selected_model_ids
    if args.current_scene_id:
        payload["currentSceneId"] = args.current_scene_id
    if args.current_model_id:
        payload["currentModelId"] = args.current_model_id
    if args.current_mode:
        payload["currentMode"] = args.current_mode
    if candidate_scene_ids:
        payload["candidateSceneIds"] = candidate_scene_ids
    if args.session_id:
        payload["sessionId"] = args.session_id
    if args.conversation_summary:
        payload["conversationSummary"] = args.conversation_summary
    if session_state:
        payload["sessionState"] = session_state
    return payload


def parse_event_chunk(chunk: str) -> list[dict[str, Any]]:
    trimmed = chunk.strip()
    if not trimmed:
        return []

    if trimmed.startswith("{"):
        decoded = json.loads(trimmed)
        if isinstance(decoded, dict) and "event" in decoded:
            return [decoded]
        return [{"event": "message", "data": decoded}]

    event_name = "message"
    data_lines: list[str] = []
    for line in trimmed.splitlines():
        normalized = line.rstrip()
        if not normalized or normalized.startswith(":"):
            continue
        if normalized.startswith("event:"):
            event_name = normalized[6:].strip() or "message"
            continue
        if normalized.startswith("data:"):
            data_lines.append(normalized[5:].lstrip())

    if not data_lines:
        return []

    data_text = "\n".join(data_lines).strip()
    if not data_text:
        return []

    try:
        data = json.loads(data_text)
    except json.JSONDecodeError:
        data = data_text
    return [{"event": event_name, "data": data}]


def drain_streaming_events(raw: str) -> ParsedEvents:
    events: list[dict[str, Any]] = []
    cursor = 0

    while cursor < len(raw):
        sse_boundary = raw.find("\n\n", cursor)
        line_boundary = raw.find("\n", cursor)
        looks_like_sse = raw.startswith("event:", cursor) or raw.startswith("data:", cursor)

        if looks_like_sse:
            if sse_boundary == -1:
                break
            chunk = raw[cursor:sse_boundary].strip()
            cursor = sse_boundary + 2
            events.extend(parse_event_chunk(chunk))
            continue

        if line_boundary == -1:
            break

        chunk = raw[cursor:line_boundary].strip()
        cursor = line_boundary + 1
        if not chunk:
            continue
        events.extend(parse_event_chunk(chunk))

    return ParsedEvents(events=events, remaining=raw[cursor:])


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def summarize_tool_result(payload: dict[str, Any]) -> str:
    for key in ("summary", "resultSummary", "result_summary", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, dict):
        for key in ("summary", "message", "description"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return stringify(payload)


def print_section(title: str) -> None:
    print(f"\n[{title}]")


def print_event(event: dict[str, Any], *, show_raw_event: bool) -> None:
    event_name = str(event.get("event") or "message")
    payload = event.get("data")
    if show_raw_event:
        print_section(f"原始事件 {event_name}")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    if event_name == "ping":
        print("PING: 流式连接已建立")
        return

    if event_name == "status" and isinstance(payload, dict):
        phase = str(payload.get("phase") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        detail = str(payload.get("detail") or "").strip()
        text = summary or phase or "状态更新"
        if detail:
            text = f"{text} | {detail}"
        print(f"STATUS: {text}")
        return

    if event_name in {"plan", "thought"}:
        print(f"{event_name.upper()}: {stringify(payload).strip()}")
        return

    if event_name == "tool_call" and isinstance(payload, dict):
        tool_name = str(payload.get("toolName") or payload.get("tool_name") or "未命名工具")
        args = payload.get("args") or {}
        print_section(f"工具调用 {tool_name}")
        print(json.dumps(args, ensure_ascii=False, indent=2))
        return

    if event_name == "tool_result" and isinstance(payload, dict):
        tool_name = str(payload.get("toolName") or payload.get("tool_name") or "未命名工具")
        print_section(f"工具结果 {tool_name}")
        print(summarize_tool_result(payload))
        return

    if event_name == "message" and isinstance(payload, dict):
        delta = str(payload.get("delta") or "")
        if delta:
            print(delta, end="", flush=True)
        return

    if event_name == "error":
        print_section("执行异常")
        print(stringify(payload).strip())
        return

    if event_name == "done":
        print("\nDONE: 已收到最终结果")
        return

    print_section(f"未分类事件 {event_name}")
    print(stringify(payload).strip())


def print_candidates(candidates: Iterable[dict[str, Any]]) -> None:
    items = list(candidates)
    if not items:
        print("无候选结果")
        return
    for index, item in enumerate(items, start=1):
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        scene_id = item.get("scene_id") or item.get("sceneId") or "-"
        model_id = item.get("model_id") or item.get("modelId") or "-"
        description = str(item.get("description") or "").strip()
        pose_image_id = item.get("pose_image_id") or item.get("poseImageId") or "-"
        print(f"{index}. score={score_text} scene={scene_id} model={model_id} pose={pose_image_id}")
        if description:
            print(f"   {description}")


def print_tool_trace(entries: Iterable[dict[str, Any]]) -> None:
    items = list(entries)
    if not items:
        print("无 tool_trace")
        return
    for index, item in enumerate(items, start=1):
        tool_name = item.get("tool_name") or item.get("toolName") or "-"
        result_summary = item.get("result_summary") or item.get("resultSummary") or ""
        args = item.get("args") or {}
        print(f"{index}. {tool_name}")
        if result_summary:
            print(f"   result: {result_summary}")
        if args:
            print("   args:")
            print(indent_text(json.dumps(args, ensure_ascii=False, indent=2), prefix="     "))


def indent_text(text: str, *, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def print_final_result(result: dict[str, Any], *, hide_candidates: bool, hide_tool_trace: bool) -> None:
    print_section("最终回答")
    print(str(result.get("answer") or "").strip() or "<empty>")

    print_section("摘要")
    mode = result.get("mode") or "-"
    reason = result.get("selected_candidate_reason") or "-"
    print(f"mode: {mode}")
    print(f"selected_candidate_reason: {reason}")

    actions = result.get("actions") or []
    print_section("动作")
    if actions:
        print(json.dumps(actions, ensure_ascii=False, indent=2))
    else:
        print("无动作")

    follow_up = result.get("follow_up")
    if follow_up:
        print_section("续聊状态")
        print(json.dumps(follow_up, ensure_ascii=False, indent=2))

    session_state = result.get("session_state")
    if session_state:
        print_section("会话状态")
        print(json.dumps(session_state, ensure_ascii=False, indent=2))

    if not hide_candidates:
        print_section("候选结果")
        print_candidates(result.get("top_candidates") or result.get("candidates") or [])

    if not hide_tool_trace:
        print_section("Tool Trace")
        print_tool_trace(result.get("tool_trace") or [])


def run_single_turn(question: str, args: argparse.Namespace) -> dict[str, Any]:
    supabase_url, supabase_key = extract_supabase_config()
    access_token = os.getenv("SUPABASE_ACCESS_TOKEN", "").strip()
    endpoint = build_endpoint(supabase_url)
    payload = build_payload(question, args)
    accept_header = "text/event-stream" if args.accept == "sse" else "application/x-ndjson"
    headers = {
        "apikey": supabase_key,
        "Accept": accept_header if args.accept != "sse" else DEFAULT_ACCEPT,
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    events: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    answer_started = False

    with requests.post(
        endpoint,
        params={"stream": "1"},
        headers=headers,
        json=payload,
        stream=True,
        timeout=args.timeout,
    ) as response:
        response.raise_for_status()
        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            parsed = drain_streaming_events(buffer)
            buffer = parsed.remaining
            for event in parsed.events:
                events.append(event)
                if event.get("event") == "message" and not answer_started:
                    print_section("流式回答")
                    answer_started = True
                print_event(event, show_raw_event=args.show_raw_event)
                if event.get("event") == "done" and isinstance(event.get("data"), dict):
                    final_result = event["data"]

        tail = buffer.strip()
        if tail:
            for event in parse_event_chunk(tail):
                events.append(event)
                print_event(event, show_raw_event=args.show_raw_event)
                if event.get("event") == "done" and isinstance(event.get("data"), dict):
                    final_result = event["data"]

    if final_result is None:
        raise RuntimeError("流式结束后未收到 done 事件")

    print()
    print_final_result(
        final_result,
        hide_candidates=args.hide_candidates,
        hide_tool_trace=args.hide_tool_trace,
    )

    return {
        "query": question,
        "payload": payload,
        "events": events,
        "result": final_result,
    }


def persist_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_question_interactive() -> str:
    try:
        return input("\n你> ").strip()
    except EOFError:
        return ""


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else None

    if args.query.strip():
        result = run_single_turn(args.query.strip(), args)
        if log_path:
            persist_log(log_path, result)
        return

    if args.non_interactive:
        print("未传 --query，且指定了 --non-interactive，直接退出。")
        return

    print("BrainDance Agent Recall Debug CLI")
    print("命令: /quit 退出")
    while True:
        question = read_question_interactive()
        if not question:
            continue
        if question == "/quit":
            break
        result = run_single_turn(question, args)
        if log_path:
            target = log_path
            if log_path.is_dir() or log_path.suffix == "":
                target = log_path / "latest.json"
            persist_log(target, result)


if __name__ == "__main__":
    main()
