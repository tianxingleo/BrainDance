#!/usr/bin/env python3
"""Evaluate unseen OOD benchmark built from post-training Supabase samples."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from evaluate_deployment_candidates import (
    DEFAULT_LLAMA_CLI,
    call_retrieval,
    gpu_uuid_for_index,
    load_cases as _unused_load_cases,
    run_gguf_answer,
    run_hf_answer,
)
from run_real_chain_debug import (
    DEFAULT_DASHSCOPE_BASE_URL,
    analyze_answer,
    contains_term,
    extract_dashscope_key,
    extract_supabase_config,
    is_negated_mention,
    is_positive_mention,
    load_generator,
    now_utc,
    unload_model,
)
from benchmark_candidate_registry import DEFAULT_UNSEEN_CANDIDATE_IDS, select_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "braindance_qwen3_unseen_ood_benchmark_20260324.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "unseen_ood_benchmark_20260324_results.json"
)
DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "unseen_ood_benchmark_20260324_summary.json"
)
NO_ANSWER_PATTERNS = (
    "暂无相关记录",
    "暂无",
    "未见相关记录",
    "未见",
    "未发现",
    "没有相关记录",
    "没有找到",
    "没找到",
    "没看到",
    "没有",
    "不清楚",
    "不知道",
)
EXTRA_NEGATIVE_MARKERS = ("未发现", "没发现", "没有", "未见", "没看到", "暂无")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unseen OOD benchmark on local candidates")
    parser.add_argument("--cases_file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--summary_file", default=str(DEFAULT_SUMMARY_FILE))
    parser.add_argument("--retrieval_snapshot_file", default="")
    parser.add_argument("--llama_cli_path", default=str(DEFAULT_LLAMA_CLI))
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--gpu_index", type=int, default=1)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ubatch_size", type=int, default=64)
    parser.add_argument("--ctx_size", type=int, default=2048)
    parser.add_argument(
        "--candidate_ids",
        default=",".join(DEFAULT_UNSEEN_CANDIDATE_IDS),
        help="Comma-separated candidate ids, or 'all' to evaluate every local QA candidate",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases_file must be a JSON array")
    return payload


def load_retrieval_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("retrieval_snapshot_file must contain a JSON object with rows")
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        chain = row.get("chain")
        if case_id and isinstance(chain, dict):
            mapping[case_id] = {
                "chain": chain,
                "retrieval_latency_ms": float(row.get("retrieval_latency_ms") or 0.0),
            }
    return mapping


def bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key)) / len(rows), 4)


def mean_rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return round(statistics.fmean(values), 2) if values else 0.0


def contains_no_answer(text: str) -> bool:
    value = text or ""
    return any(pattern in value for pattern in NO_ANSWER_PATTERNS)


def is_negated_or_missing(text: str, term: str) -> bool:
    if is_negated_mention(text, term):
        return True
    normalized_text = "".join((text or "").split())
    normalized_term = "".join((term or "").split())
    if not normalized_text or not normalized_term:
        return False
    for marker in EXTRA_NEGATIVE_MARKERS:
        if f"{marker}{normalized_term}" in normalized_text or f"{normalized_term}{marker}" in normalized_text:
            return True
    return False


def build_support_map(case: dict[str, Any]) -> dict[str, bool]:
    support_map: dict[str, bool] = {}
    for term in case.get("supported_objects") or []:
        support_map[str(term)] = True
    for term in case.get("unsupported_objects") or []:
        support_map[str(term)] = False
    return support_map


def normalize_focus_terms(case: dict[str, Any]) -> list[str]:
    return [str(term).strip() for term in (case.get("focus_terms") or []) if str(term).strip()]


def evidence_contains(evidence: list[dict[str, Any]], term: str) -> bool:
    simplified = term.strip()
    if not simplified:
        return False
    for item in evidence:
        haystacks = []
        haystacks.extend(str(obj) for obj in (item.get("objects") or []))
        haystacks.extend(str(tag) for tag in (item.get("tags") or []))
        haystacks.append(str(item.get("description") or ""))
        haystacks.append(str(item.get("scene_id") or ""))
        if any(contains_term(text, simplified) for text in haystacks):
            return True
    return False


def assess_retrieval(case: dict[str, Any], chain: dict[str, Any]) -> dict[str, Any]:
    evidence = chain["retrieval"].get("evidence") or []
    supported = [str(term) for term in (case.get("supported_objects") or [])]
    unsupported = [str(term) for term in (case.get("unsupported_objects") or [])]
    supported_found = [term for term in supported if evidence_contains(evidence, term)]
    unsupported_found = [term for term in unsupported if evidence_contains(evidence, term)]
    expected_hit = bool(supported)
    hit_count = int(chain["retrieval"].get("hit_count") or 0)

    if expected_hit:
        retrieval_ok = len(supported_found) == len(supported)
    else:
        retrieval_ok = hit_count == 0 and len(unsupported_found) == 0

    return {
        "expected_hit": expected_hit,
        "hit_count": hit_count,
        "supported_found": supported_found,
        "unsupported_found": unsupported_found,
        "retrieval_ok": retrieval_ok,
    }


def assess_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    scoreable = bool(case.get("scoreable"))
    if not scoreable:
        return {"answer_pass": None, "analysis": None}

    supported = [str(term) for term in (case.get("supported_objects") or [])]
    unsupported = [str(term) for term in (case.get("unsupported_objects") or [])]
    answer_supported = [str(term) for term in (case.get("answer_supported_terms") or supported)]
    answer_unsupported = [str(term) for term in (case.get("answer_unsupported_terms") or unsupported)]
    focus_terms = normalize_focus_terms(case)

    analysis_row = {
        "group": case.get("group") or "",
        "support_map": build_support_map(case),
        "parsed_intent": {"target_objects": focus_terms},
    }
    analysis = analyze_answer(answer, analysis_row)

    if case.get("group") == "no_hit":
        answer_pass = contains_no_answer(answer)
    elif case.get("group") == "partial_coverage":
        positive_supported = [term for term in answer_supported if is_positive_mention(answer, term)]
        negated_unsupported = [term for term in answer_unsupported if is_negated_or_missing(answer, term)]
        answer_pass = (
            len(positive_supported) == len(answer_supported)
            and len(negated_unsupported) == len(answer_unsupported)
            and not any(is_negated_or_missing(answer, term) for term in answer_supported)
        )
    else:
        focus_hit = any(contains_term(answer, term) for term in (focus_terms or supported))
        answer_pass = not contains_no_answer(answer) and focus_hit and bool(analysis["must_answer_focused"])

    return {"answer_pass": bool(answer_pass), "analysis": analysis}


def summarize_retrieval(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scoreable_rows = [row for row in case_rows if row["scoreable"]]
    return {
        "scoreable_case_count": len(scoreable_rows),
        "retrieval_ok_rate": bool_rate(scoreable_rows, "retrieval_ok"),
        "blocked_case_count": sum(1 for row in scoreable_rows if not row["retrieval_ok"]),
        "by_group": {
            group: {
                "count": len(group_rows),
                "retrieval_ok_rate": bool_rate(group_rows, "retrieval_ok"),
            }
            for group in sorted({row["group"] for row in scoreable_rows})
            for group_rows in [[row for row in scoreable_rows if row["group"] == group]]
        },
        "by_difficulty": {
            difficulty: {
                "count": len(diff_rows),
                "retrieval_ok_rate": bool_rate(diff_rows, "retrieval_ok"),
            }
            for difficulty in sorted({row["difficulty"] for row in scoreable_rows})
            for diff_rows in [[row for row in scoreable_rows if row["difficulty"] == difficulty]]
        },
    }


def summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scoreable_rows = [row for row in rows if row["scoreable"]]
    retrieval_ok_rows = [row for row in scoreable_rows if row["retrieval_ok"]]
    formatter_rows = [row for row in rows if row["answer_route"] != "lora_generation"]
    generated_rows = [row for row in rows if row["answer_route"] == "lora_generation"]

    return {
        "case_count": len(rows),
        "scoreable_case_count": len(scoreable_rows),
        "formatter_case_count": len(formatter_rows),
        "model_generated_case_count": len(generated_rows),
        "end_to_end_pass_rate": bool_rate(scoreable_rows, "overall_pass"),
        "answer_pass_rate_when_retrieval_ok": bool_rate(retrieval_ok_rows, "answer_pass"),
        "retrieval_block_rate": round(
            sum(1 for row in scoreable_rows if not row["retrieval_ok"]) / max(1, len(scoreable_rows)),
            4,
        ),
        "avg_retrieval_latency_ms": mean_rate(rows, "retrieval_latency_ms"),
        "avg_generation_latency_ms": mean_rate(rows, "generation_latency_ms"),
        "avg_total_ms": mean_rate(rows, "time_to_final_answer_ms"),
        "avg_output_chars": mean_rate(rows, "output_chars"),
        "by_group": {
            group: {
                "count": len(group_rows),
                "end_to_end_pass_rate": bool_rate([row for row in group_rows if row["scoreable"]], "overall_pass"),
                "answer_pass_rate_when_retrieval_ok": bool_rate(
                    [row for row in group_rows if row["scoreable"] and row["retrieval_ok"]],
                    "answer_pass",
                ),
            }
            for group in sorted({row["group"] for row in rows})
            for group_rows in [[row for row in rows if row["group"] == group]]
        },
        "by_difficulty": {
            difficulty: {
                "count": len(diff_rows),
                "end_to_end_pass_rate": bool_rate([row for row in diff_rows if row["scoreable"]], "overall_pass"),
                "answer_pass_rate_when_retrieval_ok": bool_rate(
                    [row for row in diff_rows if row["scoreable"] and row["retrieval_ok"]],
                    "answer_pass",
                ),
            }
            for difficulty in sorted({row["difficulty"] for row in rows})
            for diff_rows in [[row for row in rows if row["difficulty"] == difficulty]]
        },
    }


def main() -> None:
    args = parse_args()
    candidates = select_candidates(args.candidate_ids, default_ids=DEFAULT_UNSEEN_CANDIDATE_IDS)
    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()
    cases = load_cases(Path(args.cases_file))
    retrieval_snapshot = (
        load_retrieval_snapshot(Path(args.retrieval_snapshot_file))
        if str(args.retrieval_snapshot_file).strip()
        else {}
    )

    retrieval_rows: list[dict[str, Any]] = []
    retrieval_cache: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, start=1):
        cached = retrieval_snapshot.get(str(case["case_id"]))
        if cached is None:
            chain, retrieval_latency_ms = call_retrieval(
                str(case["question"]),
                args=args,
                dashscope_key=dashscope_key,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
            )
        else:
            chain = cached["chain"]
            retrieval_latency_ms = float(cached["retrieval_latency_ms"])
        retrieval_info = assess_retrieval(case, chain)
        row = {
            "case_id": case["case_id"],
            "group": case["group"],
            "difficulty": case["difficulty"],
            "question": case["question"],
            "scoreable": bool(case.get("scoreable")),
            "notes": case.get("notes") or "",
            "query_class": chain.get("query_class") or "",
            "intent": chain["retrieval"].get("intent") or "",
            "answer_route": chain["retrieval"].get("answer_route") or "",
            "retrieval_route": chain["retrieval"].get("retrieval_route") or "",
            "fallback_trigger_reason": chain["retrieval"].get("fallback_trigger_reason") or "",
            "retrieval_latency_ms": retrieval_latency_ms,
            "context_preview": (chain["retrieval"].get("evidence") or [])[:2],
            **retrieval_info,
        }
        retrieval_rows.append(row)
        retrieval_cache[case["case_id"]] = {"case": case, "chain": chain, "retrieval_row": row}
        print(f"[retrieval] {index}/{len(cases)} {case['case_id']}")

    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        tokenizer = model = device = None
        if candidate.backend == "hf":
            tokenizer, model, device = load_generator(candidate.model_name, candidate.adapter_path)
        try:
            candidate_rows: list[dict[str, Any]] = []
            for index, case in enumerate(cases, start=1):
                payload = retrieval_cache[case["case_id"]]
                chain = payload["chain"]
                retrieval_row = payload["retrieval_row"]
                question = str(case["question"])

                if candidate.backend == "hf":
                    answer, ttft_ms, total_ms, peak_mem_mb, peak_vram_mb = run_hf_answer(
                        question=question,
                        retrieval=chain,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        gpu_index=args.gpu_index,
                    )
                else:
                    answer, ttft_ms, total_ms, peak_mem_mb, peak_vram_mb = run_gguf_answer(
                        question=question,
                        retrieval=chain,
                        args=args,
                        candidate=candidate,
                        gpu_index=args.gpu_index,
                    )

                answer_info = assess_answer(case, answer)
                row = {
                    "candidate_id": candidate.candidate_id,
                    "candidate_label": candidate.label,
                    "backend": candidate.backend,
                    "case_id": case["case_id"],
                    "group": case["group"],
                    "difficulty": case["difficulty"],
                    "question": question,
                    "scoreable": bool(case.get("scoreable")),
                    "notes": case.get("notes") or "",
                    "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
                    "query_class": retrieval_row["query_class"],
                    "intent": retrieval_row["intent"],
                    "retrieval_route": retrieval_row["retrieval_route"],
                    "answer_route": retrieval_row["answer_route"],
                    "fallback_trigger_reason": retrieval_row["fallback_trigger_reason"],
                    "retrieval_ok": retrieval_row["retrieval_ok"],
                    "supported_found": retrieval_row["supported_found"],
                    "unsupported_found": retrieval_row["unsupported_found"],
                    "retrieval_latency_ms": retrieval_row["retrieval_latency_ms"],
                    "generation_latency_ms": round(total_ms, 1),
                    "time_to_first_token_ms": round(ttft_ms, 1),
                    "time_to_final_answer_ms": round(retrieval_row["retrieval_latency_ms"] + total_ms, 1),
                    "peak_memory_mb": round(peak_mem_mb, 2),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "output_chars": len(answer),
                    "answer": answer,
                    "answer_pass": answer_info["answer_pass"],
                    "overall_pass": (
                        None
                        if not case.get("scoreable")
                        else bool(retrieval_row["retrieval_ok"] and answer_info["answer_pass"])
                    ),
                    "answer_analysis": answer_info["analysis"],
                    "context_preview": retrieval_row["context_preview"],
                }
                output_rows.append(row)
                candidate_rows.append(row)
                print(f"[eval] {candidate.label} {index}/{len(cases)} {case['case_id']}")
            summary_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_label": candidate.label,
                    "backend": candidate.backend,
                    **summarize_candidate(candidate_rows),
                }
            )
        finally:
            if candidate.backend == "hf":
                unload_model(model)

    summary_payload = {
        "generated_at": now_utc().isoformat().replace("+00:00", "Z"),
        "cases_file": str(Path(args.cases_file)),
        "retrieval_snapshot_file": str(Path(args.retrieval_snapshot_file)) if str(args.retrieval_snapshot_file).strip() else "",
        "gpu_uuid": gpu_uuid_for_index(args.gpu_index),
        "retrieval_summary": summarize_retrieval(retrieval_rows),
        "candidates": [candidate.__dict__ for candidate in candidates],
        "summary": summary_rows,
    }

    Path(args.output_file).write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.summary_file).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
