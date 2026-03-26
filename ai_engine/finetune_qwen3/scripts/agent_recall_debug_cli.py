#!/usr/bin/env python3
"""Desktop debug CLI for the Flutter agent-recall flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    parser.add_argument("--event-log-file", default="", help="把事件时间线额外落盘为 JSONL")
    parser.add_argument("--show-raw-event", action="store_true", help="额外打印原始事件 JSON")
    parser.add_argument("--show-request", action="store_true", help="打印请求 endpoint、headers 摘要与 payload")
    parser.add_argument("--show-response-meta", action="store_true", help="打印 HTTP 状态码和关键响应头")
    parser.add_argument("--show-event-timeline", action="store_true", help="在结束后打印事件时间线摘要")
    parser.add_argument("--show-full-result", action="store_true", help="在结束后打印完整 done payload")
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_headers_for_display(headers: dict[str, str]) -> dict[str, str]:
    displayed = dict(headers)
    if "apikey" in displayed:
        displayed["apikey"] = mask_secret(displayed["apikey"])
    if "Authorization" in displayed:
        displayed["Authorization"] = mask_secret(displayed["Authorization"])
    return displayed


def mask_secret(value: str, keep_start: int = 6, keep_end: int = 4) -> str:
    text = (value or "").strip()
    if len(text) <= keep_start + keep_end:
        return "*" * max(4, len(text))
    return f"{text[:keep_start]}...{text[-keep_end:]}"


def extract_response_meta(response: requests.Response) -> dict[str, Any]:
    wanted_headers = {}
    for key in (
        "content-type",
        "cache-control",
        "transfer-encoding",
        "x-accel-buffering",
        "server",
    ):
        value = response.headers.get(key)
        if value:
            wanted_headers[key] = value
    return {
        "status_code": response.status_code,
        "reason": response.reason,
        "headers": wanted_headers,
    }


def summarize_event_for_timeline(event: dict[str, Any]) -> str:
    event_name = str(event.get("event") or "message")
    payload = event.get("data")
    if event_name == "status" and isinstance(payload, dict):
        return str(payload.get("summary") or payload.get("phase") or "status").strip()
    if event_name in {"plan", "thought"}:
        if isinstance(payload, dict):
            return str(payload.get("content") or payload).strip()
        return stringify(payload).strip()
    if event_name == "tool_call" and isinstance(payload, dict):
        return str(payload.get("toolName") or payload.get("tool_name") or "未命名工具")
    if event_name == "tool_result" and isinstance(payload, dict):
        tool_name = str(payload.get("toolName") or payload.get("tool_name") or "未命名工具")
        return f"{tool_name}: {summarize_tool_result(payload)}".strip()
    if event_name == "message" and isinstance(payload, dict):
        delta = str(payload.get("delta") or "")
        return delta.strip()
    if event_name == "done":
        result = payload if isinstance(payload, dict) else {}
        return str(result.get("answer") or "done").strip()
    return stringify(payload).strip()


def build_timeline_entry(
    *,
    seq: int,
    event: dict[str, Any],
    started_at_monotonic: float,
) -> dict[str, Any]:
    payload = event.get("data")
    return {
        "seq": seq,
        "event": str(event.get("event") or "message"),
        "at": now_iso(),
        "elapsed_ms": round((time.perf_counter() - started_at_monotonic) * 1000, 1),
        "summary": summarize_event_for_timeline(event),
        "payload_type": type(payload).__name__,
    }


def build_event_counts(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_name = str(event.get("event") or "message")
        counts[event_name] = counts.get(event_name, 0) + 1
    return counts


def compute_timing_stats(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    def first_elapsed(event_name: str) -> float | None:
        for item in timeline:
            if item["event"] == event_name:
                return item["elapsed_ms"]
        return None

    first_event_ms = first_elapsed("ping")
    if first_event_ms is None and timeline:
        first_event_ms = timeline[0]["elapsed_ms"]

    stats = {
        "first_event_ms": first_event_ms,
        "first_status_ms": first_elapsed("status"),
        "first_tool_call_ms": first_elapsed("tool_call"),
        "first_message_ms": first_elapsed("message"),
        "done_ms": first_elapsed("done"),
        "total_events": len(timeline),
    }
    return stats


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


def print_request_summary(*, endpoint: str, headers: dict[str, str], payload: dict[str, Any]) -> None:
    print_section("请求摘要")
    print(f"endpoint: {endpoint}")
    print("headers:")
    print(indent_text(json.dumps(normalize_headers_for_display(headers), ensure_ascii=False, indent=2), prefix="  "))
    print("payload:")
    print(indent_text(json.dumps(payload, ensure_ascii=False, indent=2), prefix="  "))


def print_response_meta(meta: dict[str, Any]) -> None:
    print_section("响应元信息")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


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


def print_evidence(evidence: dict[str, Any] | None) -> None:
    if not evidence:
        print("无 evidence")
        return
    scene_id = evidence.get("sceneId") or evidence.get("scene_id") or "-"
    model_id = evidence.get("modelId") or evidence.get("model_id") or "-"
    similarity = evidence.get("similarity")
    similarity_text = f"{float(similarity):.4f}" if isinstance(similarity, (int, float)) else "-"
    description = str(evidence.get("description") or "").strip()
    print(f"scene: {scene_id}")
    print(f"model: {model_id}")
    print(f"similarity: {similarity_text}")
    if description:
        print(f"description: {description}")
    matched_frames = evidence.get("matchedFrames") or []
    if matched_frames:
        print("matchedFrames:")
        for index, item in enumerate(matched_frames[:5], start=1):
            image_name = item.get("imageName") or item.get("image_name") or "-"
            frame_similarity = item.get("similarity")
            frame_text = f"{float(frame_similarity):.4f}" if isinstance(frame_similarity, (int, float)) else "-"
            tag = item.get("tag")
            suffix = f" tag={tag}" if tag else ""
            print(f"  {index}. {image_name} similarity={frame_text}{suffix}")


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


def print_event_timeline(timeline: list[dict[str, Any]]) -> None:
    if not timeline:
        print("无事件时间线")
        return
    for item in timeline:
        summary = str(item.get("summary") or "").replace("\n", " ").strip()
        if len(summary) > 100:
            summary = summary[:97] + "..."
        print(f"{item['seq']:02d}. +{item['elapsed_ms']:>7}ms [{item['event']}] {summary}")


def print_stats(stats: dict[str, Any], event_counts: dict[str, int]) -> None:
    print_section("调试统计")
    print(json.dumps({"timing": stats, "event_counts": event_counts}, ensure_ascii=False, indent=2))


def print_final_result(
    result: dict[str, Any],
    *,
    hide_candidates: bool,
    hide_tool_trace: bool,
    show_full_result: bool,
) -> None:
    print_section("最终回答")
    print(str(result.get("answer") or "").strip() or "<empty>")

    print_section("摘要")
    mode = result.get("mode") or "-"
    reason = result.get("selected_candidate_reason") or "-"
    candidate_count = len(result.get("top_candidates") or result.get("candidates") or [])
    tool_trace_count = len(result.get("tool_trace") or [])
    action_count = len(result.get("actions") or [])
    print(f"mode: {mode}")
    print(f"selected_candidate_reason: {reason}")
    print(f"action_count: {action_count}")
    print(f"candidate_count: {candidate_count}")
    print(f"tool_trace_count: {tool_trace_count}")

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

    print_section("Evidence")
    print_evidence(result.get("evidence"))

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

    if show_full_result:
        print_section("完整结果")
        print(json.dumps(result, ensure_ascii=False, indent=2))


def resolve_log_target(path: Path | None, question: str) -> Path | None:
    if path is None:
        return None
    if path.suffix:
        return path
    slug = "".join(ch if ch.isalnum() else "_" for ch in question.strip())[:48].strip("_") or "latest"
    return path / f"{slug}.json"


def resolve_event_log_target(path: Path | None, question: str) -> Path | None:
    if path is None:
        return None
    if path.suffix:
        return path
    slug = "".join(ch if ch.isalnum() else "_" for ch in question.strip())[:48].strip("_") or "latest"
    return path / f"{slug}.jsonl"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    request_started_iso = now_iso()
    started_at_monotonic = time.perf_counter()
    events: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    answer_started = False
    response_meta: dict[str, Any] | None = None

    if args.show_request:
        print_request_summary(endpoint=endpoint, headers=headers, payload=payload)

    with requests.post(
        endpoint,
        params={"stream": "1"},
        headers=headers,
        json=payload,
        stream=True,
        timeout=args.timeout,
    ) as response:
        response.raise_for_status()
        response_meta = extract_response_meta(response)
        if args.show_response_meta:
            print_response_meta(response_meta)
        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            parsed = drain_streaming_events(buffer)
            buffer = parsed.remaining
            for event in parsed.events:
                events.append(event)
                timeline.append(
                    build_timeline_entry(
                        seq=len(events),
                        event=event,
                        started_at_monotonic=started_at_monotonic,
                    ),
                )
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
                timeline.append(
                    build_timeline_entry(
                        seq=len(events),
                        event=event,
                        started_at_monotonic=started_at_monotonic,
                    ),
                )
                print_event(event, show_raw_event=args.show_raw_event)
                if event.get("event") == "done" and isinstance(event.get("data"), dict):
                    final_result = event["data"]

    if final_result is None:
        raise RuntimeError("流式结束后未收到 done 事件")

    event_counts = build_event_counts(events)
    timing = compute_timing_stats(timeline)
    request_summary = {
        "started_at": request_started_iso,
        "ended_at": now_iso(),
        "endpoint": endpoint,
        "accept_mode": args.accept,
        "payload": payload,
    }

    print()
    print_stats(timing, event_counts)
    if args.show_event_timeline:
        print_section("事件时间线")
        print_event_timeline(timeline)
    print_final_result(
        final_result,
        hide_candidates=args.hide_candidates,
        hide_tool_trace=args.hide_tool_trace,
        show_full_result=args.show_full_result,
    )

    return {
        "request": request_summary,
        "response_meta": response_meta,
        "stats": {
            "timing": timing,
            "event_counts": event_counts,
        },
        "query": question,
        "payload": payload,
        "events": events,
        "timeline": timeline,
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
    event_log_path = Path(args.event_log_file) if args.event_log_file else None

    if args.query.strip():
        result = run_single_turn(args.query.strip(), args)
        final_log_path = resolve_log_target(log_path, args.query.strip())
        final_event_log_path = resolve_event_log_target(event_log_path, args.query.strip())
        if final_log_path:
            persist_log(final_log_path, result)
        if final_event_log_path:
            write_jsonl(final_event_log_path, result["timeline"])
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
        final_log_path = resolve_log_target(log_path, question)
        final_event_log_path = resolve_event_log_target(event_log_path, question)
        if final_log_path:
            persist_log(final_log_path, result)
        if final_event_log_path:
            write_jsonl(final_event_log_path, result["timeline"])


if __name__ == "__main__":
    main()
