#!/usr/bin/env python3
"""Batch-run diversified agent-recall cases against the Supabase LangChain chain."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import agent_recall_debug_cli as debug_cli


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE_FILE = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "data"
    / "agent_recall_batch_suite.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "logs"
    / "agent_recall_batch_suite"
)


@dataclass(frozen=True)
class TurnCase:
    suite_name: str
    case_id: str
    title: str
    category: str
    tags: tuple[str, ...]
    turn_index: int
    total_turns: int
    request: dict[str, Any]
    expect: dict[str, Any]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run agent-recall regression and exploration suites")
    parser.add_argument("--suite-file", default=str(DEFAULT_SUITE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cases", default="", help="逗号分隔 case_id")
    parser.add_argument("--categories", default="", help="逗号分隔 category")
    parser.add_argument("--tags", default="", help="逗号分隔 tags，命中任一即可")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--accept", choices=("sse", "ndjson"), default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--quiet-cli", action="store_true", help="默认吞掉单轮 CLI 输出，只保留日志")
    parser.add_argument("--print-failures", action="store_true", help="失败时回显单轮 CLI 输出")
    parser.add_argument("--summary-only", action="store_true", help="不打印逐条结果，只打印汇总")
    return parser.parse_args()


def parse_csv(text: str) -> set[str]:
    return {item.strip() for item in (text or "").split(",") if item.strip()}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_turn_case(
    *,
    suite_name: str,
    case_id: str,
    title: str,
    category: str,
    tags: list[str],
    turn_index: int,
    total_turns: int,
    request: dict[str, Any],
    expect: dict[str, Any],
    notes: str,
) -> TurnCase:
    return TurnCase(
        suite_name=suite_name,
        case_id=case_id,
        title=title,
        category=category,
        tags=tuple(tags),
        turn_index=turn_index,
        total_turns=total_turns,
        request=request,
        expect=expect,
        notes=notes,
    )


def load_suite(path: Path) -> list[TurnCase]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("suite 文件必须是 JSON 对象")

    suite_name = str(payload.get("suite_name") or path.stem).strip() or path.stem
    defaults = payload.get("defaults") or {}
    request_defaults = {k: v for k, v in defaults.items() if k != "expect"}
    expect_defaults = defaults.get("expect") or {}
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("suite 文件必须包含非空 cases 数组")

    cases: list[TurnCase] = []
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] 必须是对象")
        case_id = str(item.get("case_id") or f"case_{index:03d}").strip()
        title = str(item.get("title") or case_id).strip()
        category = str(item.get("category") or "general").strip()
        notes = str(item.get("notes") or "").strip()
        tags = [str(tag).strip() for tag in item.get("tags") or [] if str(tag).strip()]
        case_defaults = deep_merge(request_defaults, item.get("defaults") or {})
        case_expect = deep_merge(expect_defaults, item.get("expect") or {})
        turns = item.get("turns")

        if turns is not None:
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"{case_id} 的 turns 必须是非空数组")
            total_turns = len(turns)
            for turn_idx, turn in enumerate(turns, start=1):
                if not isinstance(turn, dict):
                    raise ValueError(f"{case_id} 第 {turn_idx} 轮必须是对象")
                request = deep_merge(case_defaults, turn)
                expect = deep_merge(case_expect, turn.get("expect") or {})
                cases.append(
                    make_turn_case(
                        suite_name=suite_name,
                        case_id=case_id,
                        title=title,
                        category=category,
                        tags=tags,
                        turn_index=turn_idx,
                        total_turns=total_turns,
                        request=request,
                        expect=expect,
                        notes=notes,
                    ),
                )
            continue

        request = deep_merge(case_defaults, item)
        cases.append(
            make_turn_case(
                suite_name=suite_name,
                case_id=case_id,
                title=title,
                category=category,
                tags=tags,
                turn_index=1,
                total_turns=1,
                request=request,
                expect=case_expect,
                notes=notes,
            ),
        )
    return cases


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def filter_cases(cases: list[TurnCase], args: argparse.Namespace) -> list[TurnCase]:
    selected_case_ids = parse_csv(args.cases)
    selected_categories = parse_csv(args.categories)
    selected_tags = parse_csv(args.tags)

    filtered: list[TurnCase] = []
    for case in cases:
        if selected_case_ids and case.case_id not in selected_case_ids:
            continue
        if selected_categories and case.category not in selected_categories:
            continue
        if selected_tags and not (set(case.tags) & selected_tags):
            continue
        filtered.append(case)

    if args.limit > 0:
        return filtered[:args.limit]
    return filtered


def build_cli_args(
    case: TurnCase,
    args: argparse.Namespace,
    previous_final_result: dict[str, Any] | None = None,
) -> SimpleNamespace:
    request = dict(case.request)
    if request.get("inherit_previous_state") and previous_final_result:
        request.setdefault("session_state", previous_final_result.get("session_state"))
        request.setdefault(
            "conversation_summary",
            previous_final_result.get("conversation_summary"),
        )
        request.setdefault("session_id", previous_final_result.get("session_id"))
    return SimpleNamespace(
        accept=args.accept or str(request.get("accept") or "sse"),
        execution_mode=str(request.get("execution_mode") or "preview"),
        current_scene_id=str(request.get("current_scene_id") or ""),
        current_model_id=str(request.get("current_model_id") or ""),
        current_mode=str(request.get("current_mode") or ""),
        selected_model_ids=",".join(request.get("selected_model_ids") or []),
        candidate_scene_ids=",".join(request.get("candidate_scene_ids") or []),
        session_id=str(request.get("session_id") or ""),
        conversation_summary=str(request.get("conversation_summary") or ""),
        session_state_file=create_session_state_file(request.get("session_state")),
        show_request=False,
        show_response_meta=False,
        show_raw_event=False,
        show_event_timeline=False,
        show_full_result=False,
        hide_candidates=False,
        hide_tool_trace=False,
        timeout=args.timeout,
    )


def create_session_state_file(session_state: Any) -> str:
    if session_state is None:
        return ""
    temp_dir = DEFAULT_OUTPUT_DIR / ".tmp_session_state"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(session_state, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def cleanup_session_state_file(path: str) -> None:
    if not path:
        return
    Path(path).unlink(missing_ok=True)


def execute_case_with_previous(
    case: TurnCase,
    args: argparse.Namespace,
    previous_final_result: dict[str, Any] | None,
) -> dict[str, Any]:
    cli_args = build_cli_args(case, args, previous_final_result)
    query = str(case.request.get("query") or "").strip()
    if not query:
        raise ValueError(f"{case.case_id} 第 {case.turn_index} 轮缺少 query")

    transcript_buffer = io.StringIO()
    use_capture = args.quiet_cli or args.summary_only or args.print_failures
    try:
        if use_capture:
            with contextlib.redirect_stdout(transcript_buffer), contextlib.redirect_stderr(transcript_buffer):
                result = debug_cli.run_single_turn(query, cli_args)
        else:
            result = debug_cli.run_single_turn(query, cli_args)
    except debug_cli.StreamInterruptedError as error:
        result = error.partial_result
        result["runtime_error"] = {
            "type": "StreamInterruptedError",
            "message": str(error),
        }
    except Exception as error:  # pragma: no cover - exercised in real chain runs.
        result = {
            "request": {
                "payload": {"query": query},
            },
            "events": [],
            "timeline": [],
            "result": None,
            "response_meta": None,
            "stats": {
                "timing": {},
                "event_counts": {},
            },
            "stream_error": None,
            "runtime_error": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
        }
    finally:
        cleanup_session_state_file(cli_args.session_state_file)

    result["captured_output"] = transcript_buffer.getvalue() if use_capture else ""
    return result


def list_tool_names(result: dict[str, Any]) -> list[str]:
    final = result.get("result") or {}
    items = final.get("tool_trace") or []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name") or item.get("toolName") or "").strip()
        if name:
            names.append(name)
    return names


def get_event_names(result: dict[str, Any]) -> list[str]:
    return [str(item.get("event") or "") for item in result.get("events") or []]


def validate_result(case: TurnCase, result: dict[str, Any]) -> list[str]:
    expect = case.expect or {}
    failures: list[str] = []
    response_meta = result.get("response_meta") or {}
    status_code = int(response_meta.get("status_code") or 200)
    final = result.get("result") or {}
    stream_error = result.get("stream_error")
    runtime_error = result.get("runtime_error")
    event_names = get_event_names(result)
    tool_names = list_tool_names(result)
    actions = final.get("actions") or []
    answer = str(final.get("answer") or "").strip()
    mode = str(final.get("mode") or "").strip()
    candidates = final.get("top_candidates") or final.get("candidates") or []
    follow_up = final.get("follow_up")
    session_state = final.get("session_state")

    expect_error = bool(expect.get("expect_error"))
    require_answer = expect.get("require_answer", not expect_error)
    require_done = expect.get("require_done", not expect_error)
    require_mode = expect.get("require_mode", not expect_error)

    expected_status = int(expect.get("status_code") or 200)
    if status_code != expected_status:
        failures.append(f"HTTP 状态码不符合预期: {status_code} != {expected_status}")

    if expect_error:
        if not stream_error and not runtime_error and final:
            failures.append("预期失败，但链路返回了正常结果")
    else:
        if stream_error:
            failures.append(f"链路出现 stream_error: {stringify(stream_error)}")
        if runtime_error:
            failures.append(f"链路出现 runtime_error: {stringify(runtime_error)}")

    if require_done and "done" not in event_names:
        failures.append("未收到 done 事件")
    if require_answer and not answer:
        failures.append("answer 为空")
    if require_mode and not mode:
        failures.append("mode 为空")

    mode_any_of = [str(item).strip() for item in expect.get("mode_any_of") or [] if str(item).strip()]
    if mode_any_of and mode not in mode_any_of:
        failures.append(f"mode 不在允许范围内: {mode} not in {mode_any_of}")

    required_events = [str(item).strip() for item in expect.get("required_events") or [] if str(item).strip()]
    missing_events = [item for item in required_events if item not in event_names]
    if missing_events:
        failures.append(f"缺少事件: {missing_events}")

    required_tools = [str(item).strip() for item in expect.get("required_tools") or [] if str(item).strip()]
    missing_tools = [item for item in required_tools if item not in tool_names]
    if missing_tools:
        failures.append(f"缺少工具调用: {missing_tools}")

    forbidden_tools = [str(item).strip() for item in expect.get("forbidden_tools") or [] if str(item).strip()]
    hit_forbidden_tools = [item for item in forbidden_tools if item in tool_names]
    if hit_forbidden_tools:
        failures.append(f"命中了禁用工具: {hit_forbidden_tools}")

    answer_includes = [str(item).strip() for item in expect.get("answer_includes_any") or [] if str(item).strip()]
    if answer_includes and not any(item in answer for item in answer_includes):
        failures.append(f"answer 未包含任一关键字: {answer_includes}")

    answer_excludes = [str(item).strip() for item in expect.get("answer_excludes") or [] if str(item).strip()]
    hit_excludes = [item for item in answer_excludes if item in answer]
    if hit_excludes:
        failures.append(f"answer 命中了禁用短语: {hit_excludes}")

    if "candidates_min" in expect and len(candidates) < int(expect["candidates_min"]):
        failures.append(f"候选数量不足: {len(candidates)} < {int(expect['candidates_min'])}")
    if "candidates_max" in expect and len(candidates) > int(expect["candidates_max"]):
        failures.append(f"候选数量过多: {len(candidates)} > {int(expect['candidates_max'])}")

    action_types_any = [str(item).strip() for item in expect.get("action_types_any") or [] if str(item).strip()]
    if action_types_any:
        actual_action_types = {str(item.get("type") or "").strip() for item in actions if isinstance(item, dict)}
        if not (actual_action_types & set(action_types_any)):
            failures.append(f"actions 未覆盖预期类型: {action_types_any}")

    if expect.get("require_follow_up") and not follow_up:
        failures.append("预期 follow_up，但结果缺失")
    if expect.get("require_session_state") and not session_state:
        failures.append("预期 session_state，但结果缺失")
    return failures


def stringify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def suite_slug(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    return slug.strip("_") or "suite"


def build_case_log_path(output_dir: Path, case: TurnCase) -> Path:
    suffix = f"{case.case_id}__turn_{case.turn_index:02d}.json"
    return output_dir / "cases" / suffix


def persist_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_case_line(case: TurnCase, failures: list[str], result: dict[str, Any]) -> None:
    final = result.get("result") or {}
    timing = ((result.get("stats") or {}).get("timing") or {})
    status = "PASS" if not failures else "FAIL"
    mode = str(final.get("mode") or "-")
    answer = str(final.get("answer") or "").replace("\n", " ").strip()
    if len(answer) > 80:
        answer = answer[:77] + "..."
    print(
        f"[{status}] {case.case_id}#{case.turn_index} "
        f"category={case.category} mode={mode} "
        f"done_ms={timing.get('done_ms')} answer={answer or '<empty>'}"
    )
    if failures:
        for item in failures:
            print(f"  - {item}")


def build_summary(
    *,
    suite_name: str,
    selected_cases: list[TurnCase],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    failed = total - passed
    categories: dict[str, dict[str, int]] = {}
    tags: dict[str, int] = {}
    tool_coverage: dict[str, int] = {}
    event_coverage: dict[str, int] = {}

    case_lookup = {(case.case_id, case.turn_index): case for case in selected_cases}
    for row in rows:
        case = case_lookup[(row["case_id"], row["turn_index"])]
        category_bucket = categories.setdefault(case.category, {"total": 0, "passed": 0, "failed": 0})
        category_bucket["total"] += 1
        category_bucket["passed" if row["passed"] else "failed"] += 1
        for tag in case.tags:
            tags[tag] = tags.get(tag, 0) + 1
        for tool_name in row.get("tool_names") or []:
            tool_coverage[tool_name] = tool_coverage.get(tool_name, 0) + 1
        for event_name in row.get("event_names") or []:
            event_coverage[event_name] = event_coverage.get(event_name, 0) + 1

    return {
        "suite_name": suite_name,
        "total_turns": total,
        "passed_turns": passed,
        "failed_turns": failed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "categories": categories,
        "tag_distribution": dict(sorted(tags.items())),
        "tool_coverage": dict(sorted(tool_coverage.items())),
        "event_coverage": dict(sorted(event_coverage.items())),
        "failures": [row for row in rows if not row["passed"]],
    }


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite_file)
    output_root = Path(args.output_dir) / suite_slug(suite_path.stem)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = load_suite(suite_path)
    selected_cases = filter_cases(cases, args)
    if not selected_cases:
        raise SystemExit("筛选后没有可执行的测试用例")

    rows: list[dict[str, Any]] = []
    suite_name = selected_cases[0].suite_name
    previous_results: dict[str, dict[str, Any]] = {}
    for case in selected_cases:
        previous_result = previous_results.get(case.case_id)
        result = execute_case_with_previous(case, args, previous_result)
        failures = validate_result(case, result)
        row = {
            "suite_name": suite_name,
            "case_id": case.case_id,
            "title": case.title,
            "category": case.category,
            "tags": list(case.tags),
            "turn_index": case.turn_index,
            "total_turns": case.total_turns,
            "notes": case.notes,
            "query": str(case.request.get("query") or ""),
            "passed": not failures,
            "failures": failures,
            "tool_names": list_tool_names(result),
            "event_names": get_event_names(result),
            "result_mode": str((result.get("result") or {}).get("mode") or ""),
            "answer": str((result.get("result") or {}).get("answer") or ""),
            "stats": result.get("stats") or {},
            "stream_error": result.get("stream_error"),
            "runtime_error": result.get("runtime_error"),
            "log_path": str(build_case_log_path(output_root, case)),
        }
        rows.append(row)
        persist_json(build_case_log_path(output_root, case), {"case": case.__dict__, "debug_result": result, "evaluation": row})

        if not args.summary_only:
            print_case_line(case, failures, result)
            if failures and args.print_failures and result.get("captured_output"):
                print("  [captured_output]")
                print(result["captured_output"].rstrip())

        if failures and args.stop_on_failure:
            break
        previous_results[case.case_id] = (result.get("result") or {}) if result.get("result") else {}

    summary = build_summary(suite_name=suite_name, selected_cases=selected_cases, rows=rows)
    summary_path = output_root / "summary.json"
    persist_json(summary_path, summary)

    print()
    print("[SUMMARY]")
    print(json.dumps(
        {
            "suite_name": summary["suite_name"],
            "total_turns": summary["total_turns"],
            "passed_turns": summary["passed_turns"],
            "failed_turns": summary["failed_turns"],
            "pass_rate": summary["pass_rate"],
            "summary_path": str(summary_path),
        },
        ensure_ascii=False,
        indent=2,
    ))

    if summary["failed_turns"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
