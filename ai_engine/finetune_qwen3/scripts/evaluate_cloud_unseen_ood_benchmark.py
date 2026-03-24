#!/usr/bin/env python3
"""Evaluate cloud chat models on frozen unseen OOD benchmark with real-chain retrieval payload."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

from run_real_chain_debug import (
    DEFAULT_DASHSCOPE_BASE_URL,
    SYSTEM_PROMPT,
    analyze_answer,
    contains_term,
    extract_dashscope_key,
    is_negated_mention,
    is_positive_mention,
    now_utc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "braindance_qwen3_unseen_ood_benchmark_20260324.json"
)
DEFAULT_RETRIEVAL_SNAPSHOT_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "braindance_qwen3_unseen_ood_retrieval_snapshot_20260324.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "cloud_unseen_ood_benchmark_20260324_frozen_results.json"
)
DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "cloud_unseen_ood_benchmark_20260324_frozen_summary.json"
)

DEFAULT_MODELS = ("qwen2.5-32b-instruct", "qwen3-32b", "qwen3-8b", "qwen-turbo")
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
    parser = argparse.ArgumentParser(description="Evaluate cloud models on frozen unseen OOD benchmark")
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
            answer_info = assess_answer(case, answer)
            row = {
                "candidate_id": model,
                "candidate_label": model,
                "backend": "cloud",
                "case_id": case["case_id"],
                "group": case["group"],
                "difficulty": case["difficulty"],
                "question": str(case["question"]),
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
                "generation_latency_ms": total_ms,
                "time_to_final_answer_ms": round(retrieval_row["retrieval_latency_ms"] + total_ms, 1),
                "output_chars": len(answer),
                "answer": answer,
                "answer_pass": answer_info["answer_pass"],
                "overall_pass": (
                    None if not case.get("scoreable") else bool(retrieval_row["retrieval_ok"] and answer_info["answer_pass"])
                ),
                "answer_analysis": answer_info["analysis"],
                "context_preview": retrieval_row["context_preview"],
            }
            output_rows.append(row)
            candidate_rows.append(row)
            print(f"[cloud-unseen] {model} {case['case_id']}")
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
        "retrieval_summary": summarize_retrieval(retrieval_rows),
        "summary": summary_rows,
    }
    Path(args.output_file).write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.summary_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
