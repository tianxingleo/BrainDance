#!/usr/bin/env python3
"""Evaluate Part 17 object_lookup retrieval quality on a fixed case set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "object_lookup_eval_cases_part17.json"
DEFAULT_LOG_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "object_lookup_eval_part17.jsonl"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "object_lookup_after_summary.json"
DEFAULT_COMPARE_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "object_lookup_before_after_compare.md"
DEFAULT_HARD_CASES_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "object_lookup_hard_cases_part17.json"
DEFAULT_MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "run_real_chain_debug.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate object_lookup retrieval for Part 17")
    parser.add_argument("--cases_file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--summary_file", default=str(DEFAULT_SUMMARY_FILE))
    parser.add_argument("--compare_md", default=str(DEFAULT_COMPARE_FILE))
    parser.add_argument("--baseline_summary", default="")
    parser.add_argument("--baseline_jsonl", default="")
    parser.add_argument("--hard_cases_file", default=str(DEFAULT_HARD_CASES_FILE))
    parser.add_argument("--retrieval_module", default=str(DEFAULT_MODULE_PATH))
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--hard_case_top_k", type=int, default=15)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases_file 必须是 JSON array")
    return [item for item in payload if isinstance(item, dict)]


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("part17_retrieval_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 retrieval module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_hit(row: dict[str, Any], module: Any, focus: str) -> bool:
    focus_terms = module.normalize_lookup_terms(focus)
    return module.row_supports_target(row, focus) or module.row_matches_lookup_terms(row, focus_terms)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad_rows = [row for row in rows if row.get("user_feedback_label") == "bad"]
    lexical_rows = [
        row
        for row in rows
        if str(row.get("retrieval_route") or "").strip() in {"lexical_fallback", "merged_vector_lexical"}
    ]
    summary = {
        "object_lookup_count": len(rows),
        "object_lookup_hit_rate": round(sum(1 for row in rows if int(row.get("hit_count") or 0) > 0) / max(1, len(rows)), 4),
        "object_lookup_bad_rate": round(len(bad_rows) / max(1, len(rows)), 4),
        "object_lookup_lexical_fallback_rate": round(len(lexical_rows) / max(1, len(rows)), 4),
        "object_lookup_retrieval_miss_bad_count": sum(1 for row in rows if row.get("issue_bucket") == "retrieval_miss"),
        "object_lookup_retrieval_low_relevance_bad_count": sum(1 for row in rows if row.get("issue_bucket") == "retrieval_low_relevance"),
        "object_lookup_rpc_empty_count": sum(
            1
            for row in rows
            if str(row.get("fallback_trigger_reason") or "").strip() == "rpc_empty"
            or "rpc_empty" in (row.get("route_reasons") or [])
        ),
        "object_lookup_post_filter_empty_count": sum(
            1
            for row in rows
            if str(row.get("fallback_trigger_reason") or "").strip() == "post_filter_empty"
            or "post_filter_empty" in (row.get("route_reasons") or [])
        ),
        "retrieval_route_counts": {},
        "route_reason_counts": {},
        "issue_bucket_counts": {},
    }
    for key in ("retrieval_route", "issue_bucket"):
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "").strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
        target_key = "retrieval_route_counts" if key == "retrieval_route" else "issue_bucket_counts"
        summary[target_key] = counts
    reason_counts: dict[str, int] = {}
    for row in rows:
        for value in row.get("route_reasons") or []:
            token = str(value or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1
    summary["route_reason_counts"] = reason_counts
    return summary


def compute_case_score(row: dict[str, Any]) -> tuple[int, int, int, int]:
    issue = str(row.get("issue_bucket") or "").strip()
    route = str(row.get("retrieval_route") or "").strip()
    reasons = {str(item).strip() for item in (row.get("route_reasons") or []) if str(item).strip()}
    return (
        int(str(row.get("user_feedback_label") or "").strip() == "bad"),
        int(issue == "retrieval_miss") + int("rpc_empty" in reasons) + int("post_filter_empty" in reasons),
        int(route in {"lexical_fallback", "merged_vector_lexical"}),
        -int(row.get("hit_count") or 0),
    )


def infer_manual_label(row: dict[str, Any]) -> str:
    issue = str(row.get("issue_bucket") or "").strip()
    reasons = {str(item).strip() for item in (row.get("route_reasons") or []) if str(item).strip()}
    if issue == "retrieval_miss":
        return "retrieval_miss"
    if issue == "retrieval_low_relevance":
        return "retrieval_low_relevance"
    if "post_filter_empty" in reasons:
        return "post_filter_too_strict"
    if str(row.get("retrieval_route") or "").strip() in {"lexical_fallback", "merged_vector_lexical"}:
        return "ok"
    return "ok"


def build_hard_cases(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    baseline_by_id = {str(row.get("case_id") or ""): row for row in baseline_rows if row.get("case_id")}
    ranked_rows = sorted(rows, key=compute_case_score, reverse=True)
    selected: list[dict[str, Any]] = []
    for row in ranked_rows[:top_k]:
        case_id = str(row.get("case_id") or "")
        before_row = baseline_by_id.get(case_id, {})
        selected.append({
            "case_id": case_id,
            "question": row.get("question"),
            "group": row.get("group"),
            "after_retrieval_route": row.get("retrieval_route"),
            "before_retrieval_route": before_row.get("retrieval_route"),
            "after_route_reasons": row.get("route_reasons", []),
            "before_route_reasons": before_row.get("route_reasons", []),
            "after_issue_bucket": row.get("issue_bucket"),
            "before_issue_bucket": before_row.get("issue_bucket"),
            "after_hit_count": row.get("hit_count"),
            "before_hit_count": before_row.get("hit_count"),
            "expected_focus": row.get("expected_focus", []),
            "matched_focus": row.get("matched_focus", []),
            "suggested_manual_label": infer_manual_label(row),
            "evidence_preview": [
                {
                    "display_name": item.get("display_name"),
                    "objects": item.get("objects", [])[:6],
                }
                for item in (row.get("evidence") or [])[:2]
            ],
        })
    return {
        "top_k": top_k,
        "cases": selected,
    }


def render_count_delta(before_counts: dict[str, Any], after_counts: dict[str, Any]) -> list[str]:
    keys = sorted(set(before_counts) | set(after_counts))
    if not keys:
        return ["<empty>"]
    lines: list[str] = []
    for key in keys:
        before_value = int(before_counts.get(key, 0) or 0)
        after_value = int(after_counts.get(key, 0) or 0)
        delta = after_value - before_value
        lines.append(f"- {key}: {before_value} -> {after_value} ({delta:+d})")
    return lines


def render_compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    keys = [
        "object_lookup_count",
        "object_lookup_hit_rate",
        "object_lookup_bad_rate",
        "object_lookup_lexical_fallback_rate",
        "object_lookup_retrieval_miss_bad_count",
        "object_lookup_retrieval_low_relevance_bad_count",
        "object_lookup_rpc_empty_count",
        "object_lookup_post_filter_empty_count",
    ]
    lines = [
        "# Object Lookup Before/After Compare",
        "",
        "| metric | before | after | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in keys:
        before_value = before.get(key, 0)
        after_value = after.get(key, 0)
        try:
            delta = round(float(after_value) - float(before_value), 4)
        except Exception:
            delta = "n/a"
        lines.append(f"| {key} | {before_value} | {after_value} | {delta} |")
    lines.extend([
        "",
        "## After Retrieval Routes",
        "",
        json.dumps(after.get("retrieval_route_counts", {}), ensure_ascii=False, indent=2),
        "",
        "## Retrieval Route Delta",
        "",
    ])
    lines.extend(render_count_delta(before.get("retrieval_route_counts", {}), after.get("retrieval_route_counts", {})))
    lines.extend([
        "",
        "## Route Reason Delta",
        "",
    ])
    lines.extend(render_count_delta(before.get("route_reason_counts", {}), after.get("route_reason_counts", {})))
    lines.extend([
        "",
        "## After Issue Buckets",
        "",
        json.dumps(after.get("issue_bucket_counts", {}), ensure_ascii=False, indent=2),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.cases_file))
    module = load_module(Path(args.retrieval_module))
    dashscope_key = module.extract_dashscope_key()
    supabase_url, supabase_key = module.extract_supabase_config()

    rows: list[dict[str, Any]] = []
    for case in cases:
        question = str(case.get("question") or "").strip()
        chain = module.retrieve_real_chain_case(
            question=question,
            dashscope_key=dashscope_key,
            dashscope_base_url=module.DEFAULT_DASHSCOPE_BASE_URL,
            chat_model=args.dashscope_chat_model,
            embedding_model=args.dashscope_embedding_model,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            match_threshold=args.match_threshold,
            match_count=args.match_count,
            recent_limit=args.recent_limit,
        )
        evidence = chain["retrieval"]["evidence"]
        expected_focus = [str(item).strip() for item in (case.get("expected_focus") or []) if str(item).strip()]
        matched_focus = [
            focus
            for focus in expected_focus
            if any(normalize_hit(item, module, focus) for item in evidence)
        ]
        if not evidence:
            user_feedback_label = "bad"
            issue_bucket = "retrieval_miss"
        elif expected_focus and not matched_focus:
            user_feedback_label = "bad"
            issue_bucket = "retrieval_low_relevance"
        elif chain["retrieval"]["retrieval_route"] in {"lexical_fallback", "merged_vector_lexical"}:
            user_feedback_label = "acceptable"
            issue_bucket = None
        else:
            user_feedback_label = "good"
            issue_bucket = None

        rows.append({
            "case_id": case.get("case_id"),
            "group": case.get("group", "object_lookup"),
            "question": question,
            "expected_focus": expected_focus,
            "matched_focus": matched_focus,
            "parsed_intent": chain["parsed_intent"],
            "query_class": chain["query_class"],
            "raw_target_objects": chain.get("raw_target_objects", []),
            "normalized_lookup_terms": chain.get("normalized_lookup_terms", []),
            "hit_count": chain["retrieval"]["hit_count"],
            "retrieval_route": chain["retrieval"]["retrieval_route"],
            "fallback_trigger_reason": chain["retrieval"]["fallback_trigger_reason"],
            "route_reasons": chain["retrieval"].get("route_reasons", []),
            "rpc_error_count": chain["retrieval"].get("rpc_error_count", 0),
            "user_feedback_label": user_feedback_label,
            "issue_bucket": issue_bucket,
            "sample_valid_for_retrieval_analysis": True,
            "evidence": evidence,
        })

    summary = summarize_rows(rows)
    write_jsonl(Path(args.output_file), rows)
    write_json(Path(args.summary_file), summary)

    baseline_rows: list[dict[str, Any]] = []
    if args.baseline_jsonl:
        baseline_rows = load_jsonl(Path(args.baseline_jsonl))
    hard_cases = build_hard_cases(rows, baseline_rows, args.hard_case_top_k)
    write_json(Path(args.hard_cases_file), hard_cases)

    if args.baseline_summary:
        before = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        compare_text = render_compare(before, summary)
        write_text(Path(args.compare_md), compare_text)
        print(compare_text)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
