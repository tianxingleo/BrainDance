#!/usr/bin/env python3
"""Aggregate interactive debug session logs into route-level summaries."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_GLOB = str(
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "interactive_sessions" / "*.jsonl"
)
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "interactive_route_summary.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "interactive_route_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize interactive BrainDance route logs")
    parser.add_argument("--input", dest="inputs", action="append", default=[], help="JSONL file, directory, or glob")
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def expand_inputs(inputs: list[str]) -> list[Path]:
    patterns = inputs or [DEFAULT_INPUT_GLOB]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        candidates: list[Path]
        if any(token in pattern for token in "*?[]"):
            candidates = sorted(Path(item) for item in glob.glob(pattern))
        elif path.is_dir():
            candidates = sorted(path.glob("*.jsonl"))
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


def load_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload["_source_file"] = str(path)
                rows.append(payload)
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "sample_valid_for_retrieval_analysis" not in normalized:
        normalized["sample_valid_for_retrieval_analysis"] = True
    query_class = str(normalized.get("query_class") or "").strip()
    retrieval_route = str(normalized.get("retrieval_route") or "").strip()
    intent = str(normalized.get("intent") or "").strip()
    parsed_intent = normalized.get("parsed_intent") or {}

    if not query_class:
        if retrieval_route == "inventory_special_case":
            query_class = "inventory"
        elif retrieval_route == "non_retrieval":
            query_class = "non_retrieval"
        elif intent:
            query_class = intent
        elif isinstance(parsed_intent, dict):
            query_class = str(parsed_intent.get("question_type") or "").strip()
    normalized["query_class"] = query_class or "unknown"

    answer_route = str(normalized.get("answer_route") or "").strip()
    if not answer_route:
        if normalized["query_class"] == "inventory":
            answer_route = "inventory_formatter"
        elif normalized["query_class"] in {"greeting", "persona", "non_retrieval"} or retrieval_route == "non_retrieval":
            answer_route = "fixed_response"
        elif normalized.get("answer"):
            answer_route = "lora_generation"
    normalized["answer_route"] = answer_route or "unknown"
    return normalized


def safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feedback_counts = Counter(str(row.get("user_feedback_label") or "").strip() for row in rows if row.get("user_feedback_label"))
    triage_counts = Counter(str(row.get("triage_label") or "").strip() for row in rows if row.get("triage_label"))
    issue_counts = Counter(str(row.get("issue_bucket") or "").strip() for row in rows if row.get("issue_bucket"))
    fallback_rows = [
        row
        for row in rows
        if str(row.get("retrieval_route") or "").strip() in {"lexical_fallback", "merged_vector_lexical"}
    ]
    bad_rows = [row for row in rows if str(row.get("user_feedback_label") or "").strip() == "bad"]
    rpc_error_total = sum(int(row.get("rpc_error_count") or 0) for row in rows)
    rpc_empty_count = sum(
        1
        for row in rows
        if str(row.get("fallback_trigger_reason") or "").strip() == "rpc_empty"
        or "rpc_empty" in (row.get("route_reasons") or [])
    )
    return {
        "count": len(rows),
        "hit_rate": round(sum(1 for row in rows if int(row.get("hit_count") or 0) > 0) / max(1, len(rows)), 4),
        "avg_hit_count": round(
            sum(int(row.get("hit_count") or 0) for row in rows) / max(1, len(rows)),
            3,
        ),
        "avg_retrieval_latency_sec": safe_avg([float(row.get("retrieval_latency_sec") or 0.0) for row in rows]),
        "avg_generation_latency_sec": safe_avg([float(row.get("generation_latency_sec") or 0.0) for row in rows]),
        "fallback_rate": round(len(fallback_rows) / max(1, len(rows)), 4),
        "bad_rate": round(len(bad_rows) / max(1, len(rows)), 4),
        "rpc_empty_count": rpc_empty_count,
        "rpc_error_count": rpc_error_total,
        "rpc_error_rate": round(rpc_error_total / max(1, len(rows)), 4),
        "fallback_after_rpc_error_count": sum(1 for row in rows if row.get("fallback_after_rpc_error")),
        "user_feedback_label_counts": counter_to_sorted_dict(feedback_counts),
        "triage_label_counts": counter_to_sorted_dict(triage_counts),
        "issue_bucket_counts": counter_to_sorted_dict(issue_counts),
    }


def build_summary(rows: list[dict[str, Any]], files: list[Path]) -> dict[str, Any]:
    normalized_rows = [normalize_row(row) for row in rows]
    invalid_rows = [row for row in normalized_rows if row.get("sample_valid_for_retrieval_analysis") is False]
    valid_rows = [
        row
        for row in normalized_rows
        if not row.get("error") and row.get("sample_valid_for_retrieval_analysis") is not False
    ]
    query_class_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_route_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    answer_route_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in valid_rows:
        query_class = str(row.get("query_class") or "").strip() or "unknown"
        retrieval_route = str(row.get("retrieval_route") or "").strip() or "unknown"
        answer_route = str(row.get("answer_route") or "").strip() or "unknown"
        query_class_groups[query_class].append(row)
        retrieval_route_groups[retrieval_route].append(row)
        answer_route_groups[answer_route].append(row)

    overall = {
        "session_count": len({str(row.get("session_name") or "") for row in valid_rows if row.get("session_name")}),
        "turn_count": len(valid_rows),
        "error_count": len([row for row in rows if row.get("error")]),
        "excluded_invalid_count": len(invalid_rows),
        "rpc_error_count": sum(int(row.get("rpc_error_count") or 0) for row in valid_rows),
        "rpc_error_rate": round(
            sum(int(row.get("rpc_error_count") or 0) for row in valid_rows) / max(1, len(valid_rows)),
            4,
        ),
        "fallback_after_rpc_error_count": sum(1 for row in valid_rows if row.get("fallback_after_rpc_error")),
        "query_class_counts": counter_to_sorted_dict(Counter(str(row.get("query_class") or "").strip() for row in valid_rows if row.get("query_class"))),
        "retrieval_route_counts": counter_to_sorted_dict(Counter(str(row.get("retrieval_route") or "").strip() for row in valid_rows if row.get("retrieval_route"))),
        "fallback_reason_counts": counter_to_sorted_dict(Counter(str(row.get("fallback_trigger_reason") or "").strip() for row in valid_rows if row.get("fallback_trigger_reason"))),
        "answer_route_counts": counter_to_sorted_dict(Counter(str(row.get("answer_route") or "").strip() for row in valid_rows if row.get("answer_route"))),
        "user_feedback_label_counts": counter_to_sorted_dict(Counter(str(row.get("user_feedback_label") or "").strip() for row in valid_rows if row.get("user_feedback_label"))),
        "issue_bucket_counts": counter_to_sorted_dict(Counter(str(row.get("issue_bucket") or "").strip() for row in valid_rows if row.get("issue_bucket"))),
        "avg_hit_count": round(sum(int(row.get("hit_count") or 0) for row in valid_rows) / max(1, len(valid_rows)), 3),
        "avg_retrieval_latency_sec": safe_avg([float(row.get("retrieval_latency_sec") or 0.0) for row in valid_rows]),
        "avg_generation_latency_sec": safe_avg([float(row.get("generation_latency_sec") or 0.0) for row in valid_rows]),
    }
    object_lookup_rows = query_class_groups.get("object_lookup", [])
    object_lookup_summary = summarize_subset(object_lookup_rows)
    object_lookup_summary["retrieval_miss_bad_count"] = sum(
        1
        for row in object_lookup_rows
        if str(row.get("user_feedback_label") or "").strip() == "bad"
        and str(row.get("issue_bucket") or "").strip() == "retrieval_miss"
    )
    object_lookup_summary["retrieval_low_relevance_bad_count"] = sum(
        1
        for row in object_lookup_rows
        if str(row.get("user_feedback_label") or "").strip() == "bad"
        and str(row.get("issue_bucket") or "").strip() == "retrieval_low_relevance"
    )

    return {
        "input_files": [str(path) for path in files],
        "overall": overall,
        "object_lookup_summary": object_lookup_summary,
        "by_query_class": {
            key: summarize_subset(value)
            for key, value in sorted(query_class_groups.items())
        },
        "by_retrieval_route": {
            key: summarize_subset(value)
            for key, value in sorted(retrieval_route_groups.items())
        },
        "by_answer_route": {
            key: summarize_subset(value)
            for key, value in sorted(answer_route_groups.items())
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Interactive Route Summary",
        "",
        "## Overall",
        "",
        f"- input_files: {len(summary['input_files'])}",
        f"- session_count: {overall['session_count']}",
        f"- turn_count: {overall['turn_count']}",
        f"- error_count: {overall['error_count']}",
        f"- excluded_invalid_count: {overall['excluded_invalid_count']}",
        f"- rpc_error_count: {overall['rpc_error_count']}",
        f"- rpc_error_rate: {overall['rpc_error_rate']}",
        f"- fallback_after_rpc_error_count: {overall['fallback_after_rpc_error_count']}",
        f"- avg_hit_count: {overall['avg_hit_count']}",
        f"- avg_retrieval_latency_sec: {overall['avg_retrieval_latency_sec']}",
        f"- avg_generation_latency_sec: {overall['avg_generation_latency_sec']}",
        "",
    ]
    object_lookup = summary["object_lookup_summary"]
    lines.extend([
        "## Object Lookup Focus",
        "",
        f"- object_lookup_count: {object_lookup['count']}",
        f"- object_lookup_hit_rate: {object_lookup['hit_rate']}",
        f"- object_lookup_lexical_fallback_rate: {object_lookup['fallback_rate']}",
        f"- object_lookup_bad_rate: {object_lookup['bad_rate']}",
        f"- object_lookup_rpc_empty_count: {object_lookup['rpc_empty_count']}",
        f"- object_lookup_rpc_error_count: {object_lookup['rpc_error_count']}",
        f"- object_lookup_fallback_after_rpc_error_count: {object_lookup['fallback_after_rpc_error_count']}",
        f"- object_lookup_retrieval_miss_bad_count: {object_lookup['retrieval_miss_bad_count']}",
        f"- object_lookup_retrieval_low_relevance_bad_count: {object_lookup['retrieval_low_relevance_bad_count']}",
        "",
    ])
    for title, key in (
        ("Query Class Counts", "query_class_counts"),
        ("Retrieval Route Counts", "retrieval_route_counts"),
        ("Fallback Reason Counts", "fallback_reason_counts"),
        ("Answer Route Counts", "answer_route_counts"),
        ("User Feedback Label Counts", "user_feedback_label_counts"),
        ("Issue Bucket Counts", "issue_bucket_counts"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        mapping = overall[key]
        if not mapping:
            lines.append("- <empty>")
        else:
            for name, count in mapping.items():
                lines.append(f"- {name}: {count}")
        lines.append("")

    for section_title, section_key in (
        ("By Query Class", "by_query_class"),
        ("By Retrieval Route", "by_retrieval_route"),
        ("By Answer Route", "by_answer_route"),
    ):
        lines.append(f"## {section_title}")
        lines.append("")
        section = summary[section_key]
        if not section:
            lines.append("- <empty>")
            lines.append("")
            continue
        for name, item in section.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- count: {item['count']}")
            lines.append(f"- hit_rate: {item['hit_rate']}")
            lines.append(f"- avg_hit_count: {item['avg_hit_count']}")
            lines.append(f"- fallback_rate: {item['fallback_rate']}")
            lines.append(f"- bad_rate: {item['bad_rate']}")
            lines.append(f"- rpc_empty_count: {item['rpc_empty_count']}")
            lines.append(f"- rpc_error_count: {item['rpc_error_count']}")
            lines.append(f"- rpc_error_rate: {item['rpc_error_rate']}")
            lines.append(f"- fallback_after_rpc_error_count: {item['fallback_after_rpc_error_count']}")
            lines.append(f"- avg_retrieval_latency_sec: {item['avg_retrieval_latency_sec']}")
            lines.append(f"- avg_generation_latency_sec: {item['avg_generation_latency_sec']}")
            lines.append(f"- user_feedback_label_counts: {json.dumps(item['user_feedback_label_counts'], ensure_ascii=False)}")
            lines.append(f"- triage_label_counts: {json.dumps(item['triage_label_counts'], ensure_ascii=False)}")
            lines.append(f"- issue_bucket_counts: {json.dumps(item['issue_bucket_counts'], ensure_ascii=False)}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("未找到可汇总的 interactive session JSONL")

    rows = load_rows(files)
    summary = build_summary(rows, files)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"input_files={len(files)} turn_count={summary['overall']['turn_count']}")
    print(f"saved_json={output_json}")
    print(f"saved_md={output_md}")


if __name__ == "__main__":
    main()
