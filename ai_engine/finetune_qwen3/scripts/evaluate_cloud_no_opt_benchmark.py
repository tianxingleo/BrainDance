#!/usr/bin/env python3
"""Evaluate cloud chat models on benchmark in no-optimization mode.

No optimization means:
- no retrieval payload injected into prompt
- no system instruction from benchmark template
- only raw user question is sent to model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from evaluate_benchmark import aggregate, evaluate_case, normalize_text
from run_real_chain_debug import DEFAULT_DASHSCOPE_BASE_URL, extract_dashscope_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cloud models in no-opt/no-ft mode")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--benchmark_file", default="ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl")
    parser.add_argument("--output_dir", default="ai_engine/finetune_qwen3/logs")
    parser.add_argument("--base_url", default=DEFAULT_DASHSCOPE_BASE_URL)
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--timeout_sec", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff_sec", type=float, default=1.2)
    parser.add_argument("--date_tag", default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_raw_question(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    content = str(last.get("content") or "")
    try:
        payload = json.loads(content)
        q = str(payload.get("question") or "").strip()
        return q or content
    except json.JSONDecodeError:
        return content.strip()


def call_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    max_tokens: int,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    transient_codes = {408, 409, 429, 500, 502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
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
            return normalize_text(data["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_sec * attempt)
    raise RuntimeError(f"call failed after retries: {last_error}")


def run_model(model: str, rows: list[dict[str, Any]], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    started = time.time()
    results: list[dict[str, Any]] = []
    failed_calls = 0

    for index, row in enumerate(rows, start=1):
        question = extract_raw_question(row)
        try:
            answer = call_model(
                api_key=api_key,
                base_url=args.base_url,
                model=model,
                question=question,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                retry_backoff_sec=args.retry_backoff_sec,
            )
        except Exception as exc:
            failed_calls += 1
            answer = f"[ERROR] {exc}"

        analysis = evaluate_case(answer, row)
        results.append(
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "question": question,
                "answer": answer,
                "reference_answer": row["reference_answer"],
                "metadata": row["metadata"],
                "analysis": analysis,
            }
        )
        if index % 20 == 0 or index == len(rows):
            print(f"[{model}] {index}/{len(rows)}")

    return {
        "model_name": model,
        "mode": "no_opt_no_ft_raw_question_only",
        "metrics": aggregate(results),
        "results": results,
        "failed_calls": failed_calls,
        "elapsed_sec": round(time.time() - started, 2),
        "provider": "dashscope-compatible",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    args = parse_args()
    benchmark_file = Path(args.benchmark_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = args.date_tag.strip() or time.strftime("%Y%m%d", time.localtime())

    rows = load_jsonl(benchmark_file)
    api_key = extract_dashscope_key()
    print(f"Loaded benchmark: {benchmark_file} ({len(rows)} cases)")
    print(f"Models: {', '.join(args.models)}")

    for model in args.models:
        payload = run_model(model, rows, args, api_key)
        payload["benchmark_file"] = str(benchmark_file)
        out_file = output_dir / f"benchmark_cloud_no_opt_{model.replace('-', '_')}_{date_tag}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {model} (no-opt) ===")
        print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
        print(f"failed_calls={payload['failed_calls']} elapsed={payload['elapsed_sec']}s")
        print(f"output={out_file}")


if __name__ == "__main__":
    main()
