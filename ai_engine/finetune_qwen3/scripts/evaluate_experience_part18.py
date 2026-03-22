#!/usr/bin/env python3
"""Evaluate Part 18 formatter experience quality on a small real-chain case set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "experience_eval_cases_part18.json"
DEFAULT_LOG_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "experience_eval_part18.jsonl"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "experience_eval_part18_summary.json"
DEFAULT_HARD_CASES_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "experience_eval_part18_hard_cases.json"
DEFAULT_MODULE_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts" / "run_real_chain_debug.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formatter experience quality for Part 18")
    parser.add_argument("--cases_file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--summary_file", default=str(DEFAULT_SUMMARY_FILE))
    parser.add_argument("--hard_cases_file", default=str(DEFAULT_HARD_CASES_FILE))
    parser.add_argument("--retrieval_module", default=str(DEFAULT_MODULE_PATH))
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
    return [item for item in payload if isinstance(item, dict)]


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("part18_eval_module", path)
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


def bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(row.get(key)) for row in rows) / max(1, len(rows)), 4)


def build_group_analysis(module: Any, group: str, answer: str, chain: dict[str, Any]) -> dict[str, Any]:
    analysis = module.analyze_answer(
        answer,
        {
            "group": group,
            "parsed_intent": chain["parsed_intent"],
            "retrieval": chain["retrieval"],
            "support_map": chain["support_map"],
        },
    )
    route = str(chain["retrieval"].get("answer_route") or "").strip()
    analysis["formatter_answered"] = bool(answer)
    analysis["formatter_route_ok"] = route in {
        "recent_answer_formatter",
        "must_answer_focus_formatter",
        "inventory_formatter",
        "semantic_summary_formatter",
    }
    analysis["inventory_humanized"] = (
        analysis["natural_style"]
        and "scene_" not in answer
        and "最近生成过" in answer
        and "主要包括" in answer
    )
    analysis["semantic_summary_readable"] = (
        analysis["natural_style"]
        and route == "semantic_summary_formatter"
        and ("常见内容包括" in answer or "相关内容里常见" in answer)
        and module.count_list_separators(answer) <= 3
    )
    analysis["recent_style_ok"] = (
        analysis["natural_style"]
        and route == "recent_answer_formatter"
        and "最近拍到的" in answer
        and module.count_list_separators(answer) <= 3
    )
    return analysis


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("group") or "unknown"), []).append(row)

    summary = {
        "case_count": len(rows),
        "group_counts": {group: len(group_rows) for group, group_rows in sorted(by_group.items())},
        "answer_route_counts": {},
        "formatter_answer_rate": bool_rate(rows, "formatter_answered"),
        "natural_style_rate": bool_rate(rows, "natural_style"),
        "recent_style_rate": bool_rate([row for row in rows if row.get("group") == "recent_hit"], "recent_style_ok"),
        "must_answer_focus_rate": bool_rate([row for row in rows if row.get("group") == "must_answer"], "must_answer_focused"),
        "inventory_humanized_rate": bool_rate([row for row in rows if row.get("group") == "inventory"], "inventory_humanized"),
        "semantic_summary_readability": bool_rate(
            [row for row in rows if row.get("group") == "abstract_semantic"],
            "semantic_summary_readable",
        ),
        "metrics_by_group": {},
    }

    route_counts: dict[str, int] = {}
    for row in rows:
        route = str(row.get("answer_route") or "").strip()
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
    summary["answer_route_counts"] = route_counts

    for group, group_rows in sorted(by_group.items()):
        summary["metrics_by_group"][group] = {
            "count": len(group_rows),
            "formatter_answer_rate": bool_rate(group_rows, "formatter_answered"),
            "natural_style_rate": bool_rate(group_rows, "natural_style"),
        }
        if group == "recent_hit":
            summary["metrics_by_group"][group]["recent_style_rate"] = bool_rate(group_rows, "recent_style_ok")
        if group == "must_answer":
            summary["metrics_by_group"][group]["must_answer_focus_rate"] = bool_rate(group_rows, "must_answer_focused")
        if group == "inventory":
            summary["metrics_by_group"][group]["inventory_humanized_rate"] = bool_rate(group_rows, "inventory_humanized")
        if group == "abstract_semantic":
            summary["metrics_by_group"][group]["semantic_summary_readability"] = bool_rate(
                group_rows,
                "semantic_summary_readable",
            )
    return summary


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.cases_file))
    module = load_module(Path(args.retrieval_module))
    dashscope_key = module.extract_dashscope_key()
    supabase_url, supabase_key = module.extract_supabase_config()

    rows: list[dict[str, Any]] = []
    for case in cases:
        question = str(case.get("question") or "").strip()
        group = str(case.get("group") or "").strip()
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
        answer = str(chain.get("special_answer") or "").strip()
        analysis = build_group_analysis(module, group, answer, chain)
        rows.append({
            "case_id": str(case.get("case_id") or "").strip(),
            "group": group,
            "question": question,
            "query_class": chain["query_class"],
            "intent": chain["retrieval"]["intent"],
            "hit_count": chain["retrieval"]["hit_count"],
            "retrieval_route": chain["retrieval"]["retrieval_route"],
            "answer_route": chain["retrieval"]["answer_route"],
            "fallback_trigger_reason": chain["retrieval"]["fallback_trigger_reason"],
            "route_reasons": chain["retrieval"].get("route_reasons", []),
            "answer": answer,
            "natural_style": analysis["natural_style"],
            "must_answer_focused": analysis["must_answer_focused"],
            "recent_style_ok": analysis["recent_style_ok"],
            "inventory_humanized": analysis["inventory_humanized"],
            "semantic_summary_readable": analysis["semantic_summary_readable"],
            "formatter_answered": analysis["formatter_answered"],
            "formatter_route_ok": analysis["formatter_route_ok"],
            "focus_terms": analysis.get("focus_terms", []),
            "evidence_preview": [
                {
                    "display_name": item.get("display_name"),
                    "objects": item.get("objects", [])[:6],
                    "created_at": item.get("created_at"),
                }
                for item in (chain["retrieval"]["evidence"] or [])[:2]
            ],
        })

    summary = summarize(rows)
    hard_cases = {
        "cases": [
            row
            for row in rows
            if not row["formatter_answered"]
            or not row["natural_style"]
            or (row["group"] == "must_answer" and not row["must_answer_focused"])
            or (row["group"] == "inventory" and not row["inventory_humanized"])
            or (row["group"] == "abstract_semantic" and not row["semantic_summary_readable"])
            or (row["group"] == "recent_hit" and not row["recent_style_ok"])
        ]
    }

    write_jsonl(Path(args.output_file), rows)
    write_json(Path(args.summary_file), summary)
    write_json(Path(args.hard_cases_file), hard_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
