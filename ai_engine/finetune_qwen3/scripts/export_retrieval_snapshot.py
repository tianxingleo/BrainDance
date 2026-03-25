#!/usr/bin/env python3
"""Export full retrieval-chain snapshots for benchmark cases to keep evals reproducible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_deployment_candidates import call_retrieval
from run_real_chain_debug import extract_dashscope_key, extract_supabase_config, now_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retrieval snapshots for benchmark cases")
    parser.add_argument("--cases_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases_file must be a JSON array")
    return payload


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.cases_file))
    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        chain, retrieval_latency_ms = call_retrieval(
            str(case["question"]),
            args=args,
            dashscope_key=dashscope_key,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "question": str(case["question"]),
                "group": str(case.get("group") or ""),
                "difficulty": str(case.get("difficulty") or ""),
                "captured_at": now_utc().isoformat().replace("+00:00", "Z"),
                "retrieval_latency_ms": retrieval_latency_ms,
                "chain": chain,
            }
        )
        print(f"[snapshot] {index}/{len(cases)} {case['case_id']}")

    payload = {
        "generated_at": now_utc().isoformat().replace("+00:00", "Z"),
        "cases_file": str(Path(args.cases_file)),
        "row_count": len(rows),
        "rows": rows,
    }
    Path(args.output_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(Path(args.output_file)), "row_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
