#!/usr/bin/env python3
"""Run Part 16-C batch probes and persist route-level interactive-style logs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from interactive_debug_chat import build_summary as build_interactive_summary
from run_real_chain_debug import (
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_MODEL_NAME,
    analyze_answer,
    extract_dashscope_key,
    extract_supabase_config,
    generate_answer,
    load_generator,
    now_utc,
    retrieve_real_chain_case,
    unload_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "interactive_debug_cases_part16_template.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "interactive_sessions"
DEFAULT_ADAPTER_PATH = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "outputs"
    / "qwen3_1p7b_lora_sft_round4_1_patch_mixed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch Part 16-C probes")
    parser.add_argument("--cases_file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--mode", choices=("lora", "base"), default="lora")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--session_name", default="part16c_batch_probe")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--groups", default="")
    parser.add_argument("--priority", default="")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases_file 必须是 JSON array")
    return [item for item in payload if isinstance(item, dict) and str(item.get("question") or "").strip()]


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    groups: str,
    priority: str,
    max_cases: int,
) -> list[dict[str, Any]]:
    selected = cases
    if groups.strip():
        allowed = {item.strip() for item in groups.split(",") if item.strip()}
        selected = [item for item in selected if str(item.get("group") or "").strip() in allowed]
    if priority.strip():
        allowed_priority = {item.strip() for item in priority.split(",") if item.strip()}
        selected = [item for item in selected if str(item.get("priority") or "").strip() in allowed_priority]
    if max_cases > 0:
        selected = selected[:max_cases]
    return selected


def build_paths(session_name: str) -> tuple[Path, Path]:
    log_file = DEFAULT_LOG_DIR / f"{session_name}.jsonl"
    summary_file = DEFAULT_LOG_DIR / f"{session_name}.summary.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file, summary_file


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_rpc_error_text(text: str) -> bool:
    normalized = (text or "").lower()
    return "match_memory_poses" in normalized or "530" in normalized or "server error" in normalized


def auto_triage(row: dict[str, Any], analysis: dict[str, Any]) -> tuple[str, str | None, str]:
    if row.get("error"):
        error_text = str(row.get("error") or "")
        if is_rpc_error_text(error_text):
            return "bad", "rpc_error", "外部 RPC 失败，当前轮次不计入正常 retrieval 质量。"
        return "bad", "retrieval_miss", "当前轮次执行失败，无法视为可接受回答。"

    query_class = str(row.get("query_class") or "").strip()
    hit_count = int(row.get("hit_count") or 0)
    retrieval_route = str(row.get("retrieval_route") or "").strip()
    rpc_error_count = int(row.get("rpc_error_count") or 0)
    fallback_after_rpc_error = bool(row.get("fallback_after_rpc_error"))

    if rpc_error_count > 0 and hit_count == 0:
        return "bad", "rpc_error", "外部 RPC 报错后未拿到有效证据，当前轮次判为 bad。"
    if query_class == "partial_coverage":
        if analysis.get("partial_false_negative"):
            return "bad", "retrieval_low_relevance", "部分命中被答成未命中或显式否定已命中目标。"
        if analysis.get("partial_missing_negation"):
            return "acceptable", "focus_drift", "只回答了命中部分，但未清楚说明未命中目标。"
    if hit_count == 0 and query_class not in {"greeting", "persona", "non_retrieval"}:
        return "bad", "retrieval_miss", "该类问题未命中证据，当前优先归为 retrieval miss。"
    if not analysis.get("natural_style", False):
        return "acceptable", "answer_style", "回答可用，但风格仍偏模板化或不够自然。"
    if query_class in {"object_lookup", "partial_coverage"} and not analysis.get("must_answer_focused", True):
        return "acceptable", "focus_drift", "回答有证据支撑，但焦点仍偏散。"
    if fallback_after_rpc_error:
        return "acceptable", "rpc_error", "当前回答可用，但链路依赖了 RPC 出错后的 fallback。"
    if retrieval_route == "lexical_fallback" and query_class == "object_lookup":
        return "acceptable", None, "回答可接受，但 object_lookup 仍明显依赖 lexical fallback。"
    return "good", None, "当前回答可接受，未观察到明显 retrieval/style 问题。"


def main() -> None:
    args = parse_args()
    session_name = args.session_name.strip() or "part16c_batch_probe"
    started_at = now_utc().isoformat().replace("+00:00", "Z")
    log_file, summary_file = build_paths(session_name)

    cases = load_cases(Path(args.cases_file))
    selected_cases = filter_cases(
        cases,
        groups=args.groups,
        priority=args.priority,
        max_cases=args.max_cases,
    )
    if not selected_cases:
        raise ValueError("筛选后没有可执行 case")

    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()
    active_adapter_path = args.adapter_path if args.mode == "lora" else ""

    tokenizer = None
    model = None
    device = "cpu"
    rows: list[dict[str, Any]] = []

    try:
        tokenizer, model, device = load_generator(args.model_name, active_adapter_path)
        for turn_index, case in enumerate(selected_cases, start=1):
            question = str(case.get("question") or "").strip()
            timestamp = now_utc().isoformat().replace("+00:00", "Z")
            print(f"[{turn_index}/{len(selected_cases)}] {question}")
            try:
                retrieval_started = time.time()
                chain = retrieve_real_chain_case(
                    question=question,
                    dashscope_key=dashscope_key,
                    dashscope_base_url=DEFAULT_DASHSCOPE_BASE_URL,
                    chat_model=args.dashscope_chat_model,
                    embedding_model=args.dashscope_embedding_model,
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                    match_threshold=args.match_threshold,
                    match_count=args.match_count,
                    recent_limit=args.recent_limit,
                )
                retrieval_latency = round(time.time() - retrieval_started, 3)

                special_answer = str(chain.get("special_answer") or "").strip()
                if special_answer:
                    answer = special_answer
                    generation_latency = 0.0
                else:
                    generation_started = time.time()
                    answer = generate_answer(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        question=question,
                        retrieval=chain["retrieval"],
                        max_new_tokens=args.max_new_tokens,
                    )
                    generation_latency = round(time.time() - generation_started, 3)

                analysis = analyze_answer(
                    answer,
                    {
                        "group": str(case.get("group") or ""),
                        "parsed_intent": chain["parsed_intent"],
                        "retrieval": chain["retrieval"],
                        "support_map": chain["support_map"],
                        "evidence": chain["retrieval"]["evidence"],
                    },
                )
                draft_row = {
                    "query_class": chain["query_class"],
                    "hit_count": chain["retrieval"]["hit_count"],
                    "retrieval_route": chain["retrieval"]["retrieval_route"],
                    "rpc_error_count": chain["retrieval"].get("rpc_error_count", 0),
                    "fallback_after_rpc_error": chain["retrieval"].get("fallback_after_rpc_error", False),
                }
                user_feedback_label, issue_bucket, feedback = auto_triage(draft_row, analysis)
                row = {
                    "session_name": session_name,
                    "turn_index": turn_index,
                    "turn_id": f"{session_name}_{turn_index:03d}",
                    "timestamp": timestamp,
                    "mode": args.mode,
                    "model_name": args.model_name,
                    "adapter_path": active_adapter_path or None,
                    "case_id": case.get("case_id"),
                    "group": case.get("group"),
                    "priority": case.get("priority"),
                    "expected_route_hint": case.get("expected_route_hint"),
                    "expected_focus": case.get("expected_focus"),
                    "notes": case.get("notes"),
                    "question": question,
                    "answer": answer,
                    "answer_route": chain["retrieval"]["answer_route"],
                    "feedback_source": "auto_heuristic",
                    "user_feedback_label": user_feedback_label,
                    "triage_label": user_feedback_label,
                    "triage_reason": feedback,
                    "issue_bucket": issue_bucket,
                    "feedback": feedback,
                    "parsed_intent": chain["parsed_intent"],
                    "query_class": chain["query_class"],
                    "intent": chain["retrieval"]["intent"],
                    "raw_target_objects": chain.get("raw_target_objects", []),
                    "normalized_lookup_terms": chain.get("normalized_lookup_terms", []),
                    "hit_count": chain["retrieval"]["hit_count"],
                    "retrieval_route": chain["retrieval"]["retrieval_route"],
                    "fallback_trigger_reason": chain["retrieval"]["fallback_trigger_reason"],
                    "route_reasons": chain["retrieval"].get("route_reasons", []),
                    "rpc_error_count": chain["retrieval"].get("rpc_error_count", 0),
                    "rpc_errors": chain["retrieval"].get("rpc_errors", []),
                    "fallback_after_rpc_error": chain["retrieval"].get("fallback_after_rpc_error", False),
                    "sample_valid_for_retrieval_analysis": True,
                    "evidence": chain["retrieval"]["evidence"],
                    "support_map": chain["support_map"],
                    "answer_analysis": analysis,
                    "retrieval_latency_sec": retrieval_latency,
                    "generation_latency_sec": generation_latency,
                }
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                label, issue_bucket, feedback = auto_triage({"error": error_text}, {})
                row = {
                    "session_name": session_name,
                    "turn_index": turn_index,
                    "turn_id": f"{session_name}_{turn_index:03d}",
                    "timestamp": timestamp,
                    "mode": args.mode,
                    "model_name": args.model_name,
                    "adapter_path": active_adapter_path or None,
                    "case_id": case.get("case_id"),
                    "group": case.get("group"),
                    "priority": case.get("priority"),
                    "expected_route_hint": case.get("expected_route_hint"),
                    "expected_focus": case.get("expected_focus"),
                    "notes": case.get("notes"),
                    "question": question,
                    "error": error_text,
                    "feedback_source": "auto_heuristic",
                    "user_feedback_label": label,
                    "triage_label": label,
                    "triage_reason": feedback,
                    "issue_bucket": issue_bucket,
                    "feedback": feedback,
                    "rpc_error_count": 1 if is_rpc_error_text(error_text) else 0,
                    "rpc_errors": [{"stage": "execution", "target": question, "error": error_text}] if is_rpc_error_text(error_text) else [],
                    "fallback_after_rpc_error": False,
                }
            write_jsonl(log_file, row)
            rows.append(row)
    finally:
        unload_model(model)

    summary = build_interactive_summary(
        session_name=session_name,
        args=args,
        log_file=log_file,
        rows=rows,
        started_at=started_at,
    )
    summary["feedback_source"] = "auto_heuristic"
    summary["case_count"] = len(selected_cases)
    write_json(summary_file, summary)
    print(f"saved_log={log_file}")
    print(f"saved_summary={summary_file}")
    print(f"turn_count={summary['turn_count']} feedback_count={summary['feedback_count']}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
