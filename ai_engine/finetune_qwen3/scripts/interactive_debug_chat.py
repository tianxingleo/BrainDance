#!/usr/bin/env python3
"""Interactive real-chain debug chat for BrainDance local QA.

This script is intended for manual user-style probing:
- real retrieval chain via DashScope + Supabase
- single visible assistant answer per turn
- optional per-turn free-form feedback
- JSONL logging for later analysis
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

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
DEFAULT_LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "interactive_sessions"
DEFAULT_INTERACTIVE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "outputs"
    / "qwen3_1p7b_lora_sft_round4_1_patch_mixed"
)

HELP_TEXT = """命令:
/help            显示帮助
/quit            退出并写入 summary
/last            查看上一轮的检索摘要
/log             显示当前日志文件路径

正常直接输入问题即可。
每轮回答后可选填写反馈标签、问题归因和补充备注，直接回车表示跳过。
"""

USER_FEEDBACK_LABELS = ("good", "acceptable", "bad")
ISSUE_BUCKETS = (
    "retrieval_miss",
    "retrieval_low_relevance",
    "answer_style",
    "focus_drift",
    "formatter_needed",
    "rpc_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive BrainDance real-chain debug chat")
    parser.add_argument("--mode", choices=("lora", "base"), default="lora")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=str(DEFAULT_INTERACTIVE_ADAPTER_PATH))
    parser.add_argument("--session_name", default="")
    parser.add_argument("--log_file", default="")
    parser.add_argument("--summary_file", default="")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    parser.add_argument("--show_evidence", action="store_true")
    parser.add_argument("--skip_feedback", action="store_true")
    return parser.parse_args()


def build_session_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    session_name = args.session_name.strip()
    if not session_name:
        session_name = now_utc().strftime("interactive_debug_%Y%m%dT%H%M%SZ")

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = DEFAULT_LOG_DIR / f"{session_name}.jsonl"

    if args.summary_file:
        summary_file = Path(args.summary_file)
    else:
        summary_file = DEFAULT_LOG_DIR / f"{session_name}.summary.json"

    log_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    return session_name, log_file, summary_file


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def truncate(text: str, limit: int = 80) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def format_last_turn(turn: dict[str, Any]) -> str:
    if turn.get("error"):
        return "\n".join([
            f"问题: {turn.get('question') or '-'}",
            f"turn_id: {turn.get('turn_id') or '-'}",
            f"error: {turn.get('error')}",
        ])
    lines = [
        f"问题: {turn['question']}",
        f"turn_id: {turn.get('turn_id') or '-'}",
        f"query_class: {turn.get('query_class') or 'unknown'}",
        f"intent: {turn['intent']}",
        f"hit_count: {turn['hit_count']}",
        f"retrieval_route: {turn.get('retrieval_route') or 'unknown'}",
        f"fallback_trigger_reason: {turn.get('fallback_trigger_reason') or '-'}",
        f"answer_route: {turn.get('answer_route') or 'unknown'}",
        f"user_feedback_label: {turn.get('user_feedback_label') or '-'}",
        f"issue_bucket: {turn.get('issue_bucket') or '-'}",
        f"retrieval_latency_sec: {turn['retrieval_latency_sec']}",
        f"generation_latency_sec: {turn['generation_latency_sec']}",
    ]
    evidence = turn.get("evidence") or []
    if evidence:
        lines.append("evidence:")
        for index, item in enumerate(evidence[:3], start=1):
            display_name = item.get("display_name") or item.get("scene_id") or "unknown"
            created_at = item.get("created_at") or ""
            desc = truncate(str(item.get("description") or ""), limit=70)
            lines.append(f"  {index}. {display_name} | {created_at}")
            if desc:
                lines.append(f"     {desc}")
    else:
        lines.append("evidence: <empty>")
    return "\n".join(lines)


def build_summary(
    *,
    session_name: str,
    args: argparse.Namespace,
    log_file: Path,
    rows: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    feedback_rows = [row for row in rows if row.get("feedback")]
    errors = [row for row in rows if row.get("error")]

    hit_count_distribution: dict[str, int] = {}
    query_class_counts: dict[str, int] = {}
    retrieval_route_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    answer_route_counts: dict[str, int] = {}
    feedback_label_counts: dict[str, int] = {}
    issue_bucket_counts: dict[str, int] = {}
    rpc_error_count = 0
    fallback_after_rpc_error_count = 0
    for row in rows:
        key = str(row.get("hit_count"))
        hit_count_distribution[key] = hit_count_distribution.get(key, 0) + 1
        query_class = str(row.get("query_class") or "").strip()
        if query_class:
            query_class_counts[query_class] = query_class_counts.get(query_class, 0) + 1
        retrieval_route = str(row.get("retrieval_route") or "").strip()
        if retrieval_route:
            retrieval_route_counts[retrieval_route] = retrieval_route_counts.get(retrieval_route, 0) + 1
        fallback_reason = str(row.get("fallback_trigger_reason") or "").strip()
        if fallback_reason:
            fallback_reason_counts[fallback_reason] = fallback_reason_counts.get(fallback_reason, 0) + 1
        answer_route = str(row.get("answer_route") or "").strip()
        if answer_route:
            answer_route_counts[answer_route] = answer_route_counts.get(answer_route, 0) + 1
        feedback_label = str(row.get("user_feedback_label") or "").strip()
        if feedback_label:
            feedback_label_counts[feedback_label] = feedback_label_counts.get(feedback_label, 0) + 1
        issue_bucket = str(row.get("issue_bucket") or "").strip()
        if issue_bucket:
            issue_bucket_counts[issue_bucket] = issue_bucket_counts.get(issue_bucket, 0) + 1
        rpc_error_count += int(row.get("rpc_error_count") or 0)
        if row.get("fallback_after_rpc_error"):
            fallback_after_rpc_error_count += 1

    return {
        "session_name": session_name,
        "started_at": started_at,
        "ended_at": now_utc().isoformat().replace("+00:00", "Z"),
        "mode": args.mode,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path if args.mode == "lora" else None,
        "log_file": str(log_file),
        "turn_count": len(rows),
        "feedback_count": len(feedback_rows),
        "error_count": len(errors),
        "avg_retrieval_latency_sec": round(
            sum(float(row.get("retrieval_latency_sec") or 0.0) for row in rows if not row.get("error"))
            / max(1, len([row for row in rows if not row.get("error")])),
            3,
        ),
        "avg_generation_latency_sec": round(
            sum(float(row.get("generation_latency_sec") or 0.0) for row in rows if not row.get("error"))
            / max(1, len([row for row in rows if not row.get("error")])),
            3,
        ),
        "hit_count_distribution": hit_count_distribution,
        "rpc_error_count": rpc_error_count,
        "rpc_error_rate": round(rpc_error_count / max(1, len([row for row in rows if not row.get("error")])), 4),
        "fallback_after_rpc_error_count": fallback_after_rpc_error_count,
        "query_class_counts": query_class_counts,
        "retrieval_route_counts": retrieval_route_counts,
        "fallback_reason_counts": fallback_reason_counts,
        "answer_route_counts": answer_route_counts,
        "user_feedback_label_counts": feedback_label_counts,
        "issue_bucket_counts": issue_bucket_counts,
        "feedback_examples": [
            {
                "turn_index": row["turn_index"],
                "question": row["question"],
                "user_feedback_label": row.get("user_feedback_label"),
                "issue_bucket": row.get("issue_bucket"),
                "feedback": row["feedback"],
            }
            for row in feedback_rows[:20]
        ],
    }


def prompt_optional_label(prompt_text: str, allowed_values: tuple[str, ...]) -> str | None:
    raw_value = input(prompt_text).strip().lower()
    if not raw_value:
        return None
    if raw_value not in allowed_values:
        print(f"忽略无效值: {raw_value}，允许值为 {', '.join(allowed_values)}")
        return None
    return raw_value


def main() -> None:
    args = parse_args()
    session_name, log_file, summary_file = build_session_paths(args)
    started_at = now_utc().isoformat().replace("+00:00", "Z")

    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()
    active_adapter_path = args.adapter_path if args.mode == "lora" else ""

    print(f"Session: {session_name}")
    print(f"Log file: {log_file}")
    print(f"Summary file: {summary_file}")
    print(f"Mode: {args.mode}")
    if active_adapter_path:
        print(f"Adapter: {active_adapter_path}")
    print("Loading model...")

    tokenizer = None
    model = None
    device = "cpu"
    rows: list[dict[str, Any]] = []
    last_turn: dict[str, Any] | None = None

    try:
        tokenizer, model, device = load_generator(args.model_name, active_adapter_path)
        print(f"Model loaded on: {device}")
        print(HELP_TEXT.strip())

        turn_index = 0
        while True:
            try:
                question = input("\n你> ").strip()
            except EOFError:
                print("\n收到 EOF，结束会话。")
                break
            except KeyboardInterrupt:
                print("\n输入已中断。使用 /quit 退出。")
                continue

            if not question:
                continue

            if question in {"/quit", "/exit"}:
                break
            if question == "/help":
                print(HELP_TEXT.strip())
                continue
            if question == "/log":
                print(log_file)
                continue
            if question == "/last":
                if last_turn is None:
                    print("当前还没有已记录轮次。")
                else:
                    print(format_last_turn(last_turn))
                continue
            if question.startswith("/"):
                print("未知命令，输入 /help 查看帮助。")
                continue

            turn_index += 1
            timestamp = now_utc().isoformat().replace("+00:00", "Z")

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
                        "group": "interactive",
                        "parsed_intent": chain["parsed_intent"],
                        "retrieval": chain["retrieval"],
                        "support_map": chain["support_map"],
                    },
                )

                print(f"\n助手> {answer}")
                if args.show_evidence:
                    preview = {
                        "query_class": chain["retrieval"]["query_class"],
                        "intent": chain["retrieval"]["intent"],
                        "hit_count": chain["retrieval"]["hit_count"],
                        "retrieval_route": chain["retrieval"]["retrieval_route"],
                        "raw_target_objects": chain["retrieval"].get("raw_target_objects", []),
                        "normalized_lookup_terms": chain["retrieval"].get("normalized_lookup_terms", []),
                        "fallback_trigger_reason": chain["retrieval"]["fallback_trigger_reason"],
                        "route_reasons": chain["retrieval"].get("route_reasons", []),
                        "answer_route": chain["retrieval"]["answer_route"],
                        "rpc_error_count": chain["retrieval"].get("rpc_error_count", 0),
                        "rpc_errors": chain["retrieval"].get("rpc_errors", []),
                        "fallback_after_rpc_error": chain["retrieval"].get("fallback_after_rpc_error", False),
                        "evidence": chain["retrieval"]["evidence"][:3],
                    }
                    print(json.dumps(preview, ensure_ascii=False, indent=2))

                feedback = ""
                user_feedback_label = None
                issue_bucket = None
                if not args.skip_feedback:
                    try:
                        user_feedback_label = prompt_optional_label(
                            f"反馈标签({ '/'.join(USER_FEEDBACK_LABELS) }，可回车跳过)> ",
                            USER_FEEDBACK_LABELS,
                        )
                        issue_bucket = prompt_optional_label(
                            f"问题归因({ '/'.join(ISSUE_BUCKETS) }，可回车跳过)> ",
                            ISSUE_BUCKETS,
                        )
                        feedback = input("反馈(可回车跳过)> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        feedback = ""
                        user_feedback_label = None
                        issue_bucket = None
                        print()

                row = {
                    "session_name": session_name,
                    "turn_index": turn_index,
                    "turn_id": f"{session_name}_{turn_index:03d}",
                    "timestamp": timestamp,
                    "mode": args.mode,
                    "model_name": args.model_name,
                    "adapter_path": active_adapter_path or None,
                    "question": question,
                    "answer": answer,
                    "answer_route": chain["retrieval"]["answer_route"],
                    "user_feedback_label": user_feedback_label,
                    "issue_bucket": issue_bucket,
                    "feedback": feedback or None,
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
                write_jsonl(log_file, [row])
                rows.append(row)
                last_turn = row
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                print(f"\n[error] {error_text}")
                row = {
                    "session_name": session_name,
                    "turn_index": turn_index,
                    "timestamp": timestamp,
                    "mode": args.mode,
                    "model_name": args.model_name,
                    "adapter_path": active_adapter_path or None,
                    "question": question,
                    "error": error_text,
                }
                write_jsonl(log_file, [row])
                rows.append(row)
                last_turn = row
    finally:
        unload_model(model)

    summary = build_summary(
        session_name=session_name,
        args=args,
        log_file=log_file,
        rows=rows,
        started_at=started_at,
    )
    write_json(summary_file, summary)
    print(f"\nSession finished. turns={summary['turn_count']} feedback={summary['feedback_count']}")
    print(f"Saved log: {log_file}")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
