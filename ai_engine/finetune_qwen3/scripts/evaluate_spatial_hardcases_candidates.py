#!/usr/bin/env python3
"""Run spatial hardcases across multiple local candidates and summarize route/answer patterns."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from evaluate_deployment_candidates import (
    DEFAULT_LLAMA_CLI,
    call_retrieval,
    gpu_uuid_for_index,
    run_gguf_answer,
    run_hf_answer,
)
from run_real_chain_debug import extract_dashscope_key, extract_supabase_config, load_generator, now_utc, unload_model
from benchmark_candidate_registry import DEFAULT_SPATIAL_CANDIDATE_IDS, select_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "spatial_hardcases_candidates_20260324_results.json"
)
DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "spatial_hardcases_candidates_20260324_summary.json"
)

NO_ANSWER_PATTERNS = ("暂无", "未见", "没有", "没找到", "不清楚", "不知道")
SPATIAL_MARKERS = (
    "左边",
    "右边",
    "左侧",
    "右侧",
    "上方",
    "下方",
    "前面",
    "后面",
    "前方",
    "后方",
    "中间",
    "之间",
    "更靠近",
    "靠近",
    "更近",
    "更远",
    "顺序",
    "从左到右",
    "位于",
)
GENERIC_SUMMARY_MARKERS = (
    "最近拍到的主要有",
    "最近拍到过",
    "最近生成过",
    "主要有",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate spatial hardcases across local candidates")
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
        default=",".join(DEFAULT_SPATIAL_CANDIDATE_IDS),
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


def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in (text or "") for pattern in patterns)


def classify_answer(answer: str) -> str:
    if contains_any(answer, NO_ANSWER_PATTERNS):
        return "refusal"
    if contains_any(answer, SPATIAL_MARKERS):
        return "spatial_direct"
    if contains_any(answer, GENERIC_SUMMARY_MARKERS):
        return "generic_scene_summary"
    return "object_summary"


def bool_rate(rows: list[dict[str, Any]], key: str, *, target: Any = True) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key) == target) / len(rows), 4)


def mean_rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return round(statistics.fmean(values), 2) if values else 0.0


def summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_classes = sorted({row["answer_class"] for row in rows})
    query_classes = sorted({row["query_class"] for row in rows})
    answer_routes = sorted({row["answer_route"] for row in rows})
    retrieval_routes = sorted({row["retrieval_route"] for row in rows})
    return {
        "case_count": len(rows),
        "spatial_direct_rate": bool_rate(rows, "answer_class", target="spatial_direct"),
        "refusal_rate": bool_rate(rows, "answer_class", target="refusal"),
        "generic_scene_summary_rate": bool_rate(rows, "answer_class", target="generic_scene_summary"),
        "avg_retrieval_latency_ms": mean_rate(rows, "retrieval_latency_ms"),
        "avg_generation_latency_ms": mean_rate(rows, "generation_latency_ms"),
        "avg_total_ms": mean_rate(rows, "time_to_final_answer_ms"),
        "avg_output_chars": mean_rate(rows, "output_chars"),
        "answer_class_distribution": {
            label: round(sum(1 for row in rows if row["answer_class"] == label) / len(rows), 4)
            for label in answer_classes
        },
        "query_class_distribution": {
            label: round(sum(1 for row in rows if row["query_class"] == label) / len(rows), 4)
            for label in query_classes
        },
        "answer_route_distribution": {
            label: round(sum(1 for row in rows if row["answer_route"] == label) / len(rows), 4)
            for label in answer_routes
        },
        "retrieval_route_distribution": {
            label: round(sum(1 for row in rows if row["retrieval_route"] == label) / len(rows), 4)
            for label in retrieval_routes
        },
    }


def main() -> None:
    args = parse_args()
    candidates = select_candidates(args.candidate_ids, default_ids=DEFAULT_SPATIAL_CANDIDATE_IDS)
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
        row = {
            "case_id": case["case_id"],
            "group": case["group"],
            "difficulty": case["difficulty"],
            "question": case["question"],
            "relation_type": case.get("relation_type") or "",
            "query_class": chain.get("query_class") or "",
            "intent": chain["retrieval"].get("intent") or "",
            "retrieval_route": chain["retrieval"].get("retrieval_route") or "",
            "answer_route": chain["retrieval"].get("answer_route") or "",
            "fallback_trigger_reason": chain["retrieval"].get("fallback_trigger_reason") or "",
            "hit_count": int(chain["retrieval"].get("hit_count") or 0),
            "retrieval_latency_ms": retrieval_latency_ms,
            "context_preview": (chain["retrieval"].get("evidence") or [])[:2],
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
                row = {
                    "candidate_id": candidate.candidate_id,
                    "candidate_label": candidate.label,
                    "backend": candidate.backend,
                    "case_id": case["case_id"],
                    "group": case["group"],
                    "difficulty": case["difficulty"],
                    "relation_type": case.get("relation_type") or "",
                    "question": question,
                    "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
                    "query_class": retrieval_row["query_class"],
                    "intent": retrieval_row["intent"],
                    "retrieval_route": retrieval_row["retrieval_route"],
                    "answer_route": retrieval_row["answer_route"],
                    "fallback_trigger_reason": retrieval_row["fallback_trigger_reason"],
                    "hit_count": retrieval_row["hit_count"],
                    "retrieval_latency_ms": retrieval_row["retrieval_latency_ms"],
                    "generation_latency_ms": round(total_ms, 1),
                    "time_to_first_token_ms": round(ttft_ms, 1),
                    "time_to_final_answer_ms": round(retrieval_row["retrieval_latency_ms"] + total_ms, 1),
                    "peak_memory_mb": round(peak_mem_mb, 2),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "output_chars": len(answer),
                    "answer": answer,
                    "answer_class": classify_answer(answer),
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
        "candidates": [candidate.__dict__ for candidate in candidates],
        "retrieval_cases": retrieval_rows,
        "summary": summary_rows,
    }
    Path(args.output_file).write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.summary_file).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
