#!/usr/bin/env python3
"""Evaluate cloud chat models on frozen spatial hardcases with real-chain retrieval payload."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

from run_real_chain_debug import DEFAULT_DASHSCOPE_BASE_URL, SYSTEM_PROMPT, extract_dashscope_key, now_utc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json"
)
DEFAULT_RETRIEVAL_SNAPSHOT_FILE = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "data"
    / "braindance_qwen3_unseen_ood_spatial_hardcases_retrieval_snapshot_20260324.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "cloud_spatial_hardcases_20260324_frozen_results.json"
)
DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "cloud_spatial_hardcases_20260324_frozen_summary.json"
)

DEFAULT_MODELS = ("qwen2.5-32b-instruct", "qwen3-32b", "qwen3-8b", "qwen-turbo")
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
GENERIC_SUMMARY_MARKERS = ("最近拍到的主要有", "最近拍到过", "最近生成过", "主要有")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cloud models on frozen spatial hardcases")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--cases_file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--retrieval_snapshot_file", default=str(DEFAULT_RETRIEVAL_SNAPSHOT_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--summary_file", default=str(DEFAULT_SUMMARY_FILE))
    parser.add_argument("--base_url", default=DEFAULT_DASHSCOPE_BASE_URL)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout_sec", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff_sec", type=float, default=1.2)
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


def cloud_answer(
    *,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    retrieval: dict[str, Any],
    max_tokens: int,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> tuple[str, float]:
    special_answer = str(retrieval.get("special_answer") or "").strip()
    if special_answer:
        return special_answer, 0.0

    user_payload = json.dumps(
        {"question": question, "retrieval": retrieval["retrieval"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    transient_codes = {408, 409, 429, 500, 502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if response.status_code >= 400:
                text = response.text[:600]
                error = RuntimeError(f"HTTP {response.status_code}: {text}")
                if response.status_code in transient_codes and attempt < max_retries:
                    last_error = error
                    time.sleep(retry_backoff_sec * attempt)
                    continue
                raise error
            data = response.json()
            answer = str(data["choices"][0]["message"]["content"] or "").strip()
            total_ms = round((time.perf_counter() - started) * 1000, 1)
            return answer, total_ms
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_sec * attempt)
    raise RuntimeError(f"call failed after retries: {last_error}")


def main() -> None:
    args = parse_args()
    api_key = extract_dashscope_key()
    cases = load_cases(Path(args.cases_file))
    retrieval_snapshot = load_retrieval_snapshot(Path(args.retrieval_snapshot_file))

    retrieval_rows: list[dict[str, Any]] = []
    retrieval_cache: dict[str, dict[str, Any]] = {}
    for case in cases:
        payload = retrieval_snapshot[str(case["case_id"])]
        chain = payload["chain"]
        retrieval_latency_ms = float(payload["retrieval_latency_ms"])
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
        retrieval_cache[str(case["case_id"])] = {"case": case, "chain": chain, "retrieval_row": row}

    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model in args.models:
        candidate_rows: list[dict[str, Any]] = []
        for case in cases:
            payload = retrieval_cache[str(case["case_id"])]
            chain = payload["chain"]
            retrieval_row = payload["retrieval_row"]
            answer, total_ms = cloud_answer(
                api_key=api_key,
                base_url=args.base_url,
                model=model,
                question=str(case["question"]),
                retrieval=chain,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
            )
            row = {
                "candidate_id": model,
                "candidate_label": model,
                "backend": "cloud",
                "case_id": case["case_id"],
                "group": case["group"],
                "difficulty": case["difficulty"],
                "relation_type": case.get("relation_type") or "",
                "question": str(case["question"]),
                "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
                "query_class": retrieval_row["query_class"],
                "intent": retrieval_row["intent"],
                "retrieval_route": retrieval_row["retrieval_route"],
                "answer_route": retrieval_row["answer_route"],
                "fallback_trigger_reason": retrieval_row["fallback_trigger_reason"],
                "hit_count": retrieval_row["hit_count"],
                "retrieval_latency_ms": retrieval_row["retrieval_latency_ms"],
                "generation_latency_ms": total_ms,
                "time_to_final_answer_ms": round(retrieval_row["retrieval_latency_ms"] + total_ms, 1),
                "output_chars": len(answer),
                "answer": answer,
                "answer_class": classify_answer(answer),
                "context_preview": retrieval_row["context_preview"],
            }
            output_rows.append(row)
            candidate_rows.append(row)
            print(f"[cloud-spatial] {model} {case['case_id']}")
        summary_rows.append(
            {
                "candidate_id": model,
                "candidate_label": model,
                "backend": "cloud",
                **summarize_candidate(candidate_rows),
            }
        )

    payload = {
        "generated_at": now_utc().isoformat().replace("+00:00", "Z"),
        "cases_file": str(Path(args.cases_file)),
        "retrieval_snapshot_file": str(Path(args.retrieval_snapshot_file)),
        "models": list(args.models),
        "retrieval_cases": retrieval_rows,
        "summary": summary_rows,
    }
    Path(args.output_file).write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.summary_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
