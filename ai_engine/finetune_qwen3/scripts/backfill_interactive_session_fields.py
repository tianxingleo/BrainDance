#!/usr/bin/env python3
"""Backfill missing route metadata in interactive debug session logs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from run_real_chain_debug import (
    DEFAULT_DASHSCOPE_BASE_URL,
    coalesce_fallback_reason,
    detect_non_retrieval_answer,
    extract_dashscope_key,
    extract_supabase_config,
    infer_answer_route,
    is_model_inventory_query,
    normalize_lookup_terms,
    retrieve_real_chain_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_GLOB = str(
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "interactive_sessions" / "*.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing interactive session metadata")
    parser.add_argument("--input", dest="inputs", action="append", default=[], help="JSONL file, directory, or glob")
    parser.add_argument("--rehydrate_route", action="store_true", help="Rerun retrieval for rows missing route metadata")
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
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


def derive_query_class(row: dict[str, Any]) -> str:
    if str(row.get("query_class") or "").strip():
        return str(row["query_class"]).strip()

    question = str(row.get("question") or "")
    non_retrieval = detect_non_retrieval_answer(question)
    if non_retrieval:
        return non_retrieval[0]

    parsed_intent = row.get("parsed_intent") or {}
    question_type = str(parsed_intent.get("question_type") or "").strip()
    target_objects = [str(item).strip() for item in (parsed_intent.get("target_objects") or []) if str(item).strip()]
    lookup_terms = normalize_lookup_terms(str(parsed_intent.get("search_text") or ""), *target_objects)
    if is_model_inventory_query(question, question_type or "other", lookup_terms):
        return "inventory"

    intent = str(row.get("intent") or "").strip()
    if intent and intent not in {"no_hit"}:
        return intent
    if question_type:
        return question_type
    return "unknown"


def derive_retrieval_route(row: dict[str, Any], query_class: str) -> str:
    if str(row.get("retrieval_route") or "").strip():
        return str(row["retrieval_route"]).strip()

    fallback_reason = str(row.get("fallback_trigger_reason") or "").strip()
    parsed_intent = row.get("parsed_intent") or {}
    search_text = str(parsed_intent.get("search_text") or "").strip()
    target_objects = [str(item).strip() for item in (parsed_intent.get("target_objects") or []) if str(item).strip()]

    if query_class in {"greeting", "persona", "non_retrieval"}:
        return "non_retrieval"
    if query_class == "inventory" or fallback_reason == "inventory_query":
        return "inventory_special_case"
    if query_class in {"recent_capture", "time_qa"} and not target_objects and not search_text:
        return "recent_list"
    if str(row.get("retrieval_route") or "").strip() == "merged_vector_lexical":
        return "merged_vector_lexical"
    if fallback_reason in {"rpc_empty", "post_filter_empty", "low_confidence_vector"} and int(row.get("hit_count") or 0) > 0:
        return "lexical_fallback"
    if query_class in {"object_lookup", "partial_coverage", "no_hit"}:
        return "vector_plus_filter"
    return "unknown"


def derive_answer_route(row: dict[str, Any], query_class: str) -> str:
    if str(row.get("answer_route") or "").strip():
        return str(row["answer_route"]).strip()
    special_answer = str(row.get("answer") or "").strip() if query_class in {"inventory", "greeting", "persona"} else None
    return infer_answer_route(query_class=query_class, special_answer=special_answer)


def maybe_rehydrate_row(
    row: dict[str, Any],
    *,
    dashscope_key: str,
    supabase_url: str,
    supabase_key: str,
    chat_model: str,
    embedding_model: str,
) -> dict[str, Any]:
    has_missing_core_fields = not all(
        str(row.get(field) or "").strip()
        for field in ("query_class", "retrieval_route", "answer_route")
    )
    if not has_missing_core_fields or row.get("error"):
        return row

    chain = retrieve_real_chain_case(
        question=str(row.get("question") or ""),
        dashscope_key=dashscope_key,
        dashscope_base_url=DEFAULT_DASHSCOPE_BASE_URL,
        chat_model=chat_model,
        embedding_model=embedding_model,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        match_threshold=0.5,
        match_count=5,
        recent_limit=3,
    )
    updated = dict(row)
    updated["query_class"] = chain["query_class"]
    updated["intent"] = chain["retrieval"]["intent"]
    updated["hit_count"] = chain["retrieval"]["hit_count"]
    updated["retrieval_route"] = chain["retrieval"]["retrieval_route"]
    updated["fallback_trigger_reason"] = chain["retrieval"]["fallback_trigger_reason"]
    updated["answer_route"] = chain["retrieval"]["answer_route"]
    if not updated.get("evidence"):
        updated["evidence"] = chain["retrieval"]["evidence"]
    if not updated.get("support_map"):
        updated["support_map"] = chain["support_map"]
    return updated


def backfill_row(
    row: dict[str, Any],
    *,
    dashscope_key: str | None,
    supabase_url: str | None,
    supabase_key: str | None,
    chat_model: str,
    embedding_model: str,
    rehydrate_route: bool,
) -> dict[str, Any]:
    updated = dict(row)
    if updated.get("error"):
        return updated

    used_rehydrate = False
    if rehydrate_route and dashscope_key and supabase_url and supabase_key:
        try:
            updated = maybe_rehydrate_row(
                updated,
                dashscope_key=dashscope_key,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                chat_model=chat_model,
                embedding_model=embedding_model,
            )
            used_rehydrate = updated != row
            if used_rehydrate:
                updated["route_backfill_method"] = "rehydrated"
                updated.pop("route_backfill_error", None)
        except Exception as exc:
            updated["route_backfill_method"] = "heuristic"
            updated["route_backfill_error"] = f"{type(exc).__name__}: {exc}"

    session_name = str(updated.get("session_name") or "").strip()
    turn_index = int(updated.get("turn_index") or 0)
    if session_name and turn_index and not str(updated.get("turn_id") or "").strip():
        updated["turn_id"] = f"{session_name}_{turn_index:03d}"

    query_class = derive_query_class(updated)
    updated["query_class"] = query_class

    if not str(updated.get("fallback_trigger_reason") or "").strip():
        reasons = []
        fallback_reason = coalesce_fallback_reason(reasons)
        if fallback_reason:
            updated["fallback_trigger_reason"] = fallback_reason

    updated["retrieval_route"] = derive_retrieval_route(updated, query_class)
    updated["answer_route"] = derive_answer_route(updated, query_class)
    updated.setdefault("sample_valid_for_retrieval_analysis", True)
    if not used_rehydrate:
        updated.setdefault("route_backfill_method", "heuristic")
    updated.setdefault("user_feedback_label", None)
    updated.setdefault("issue_bucket", None)
    return updated


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("未找到需要回填的 interactive session JSONL")

    dashscope_key = None
    supabase_url = None
    supabase_key = None
    if args.rehydrate_route:
        dashscope_key = extract_dashscope_key()
        supabase_url, supabase_key = extract_supabase_config()

    updated_files = 0
    updated_rows = 0
    for path in files:
        original_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        new_rows: list[dict[str, Any]] = []
        file_changed = False
        for row in original_rows:
            updated = backfill_row(
                row,
                dashscope_key=dashscope_key,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                chat_model=args.dashscope_chat_model,
                embedding_model=args.dashscope_embedding_model,
                rehydrate_route=args.rehydrate_route,
            )
            if updated != row:
                updated_rows += 1
                file_changed = True
            new_rows.append(updated)
        if file_changed:
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in new_rows) + "\n",
                encoding="utf-8",
            )
            updated_files += 1

    print(f"files={len(files)} updated_files={updated_files} updated_rows={updated_rows}")


if __name__ == "__main__":
    main()
