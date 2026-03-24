#!/usr/bin/env python3
"""Minimal local QA CLI built on top of the real retrieval chain."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_real_chain_debug import (
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_MODEL_NAME,
    analyze_answer,
    extract_dashscope_key,
    extract_supabase_config,
    generate_answer,
    load_generator,
    now_utc,
    retrieve_real_chain_case,
    unload_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTERACTIVE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "outputs"
    / "qwen3_1p7b_lora_sft_round4_1_patch_mixed"
)
HELP_TEXT = """命令:
/help            显示帮助
/quit            退出
/last            查看上一轮链路摘要

正常直接输入问题即可。
默认只输出最终短答；通过参数开关可以额外查看证据和链路信息。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local QA CLI for BrainDance")
    parser.add_argument("--mode", choices=("lora", "base"), default="lora")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=str(DEFAULT_INTERACTIVE_ADAPTER_PATH))
    parser.add_argument("--question", default="", help="单轮问题；传入后执行一次即退出")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    parser.add_argument("--show_evidence", action="store_true")
    parser.add_argument("--show_trace", action="store_true")
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(text: str, limit: int = 80) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def format_chain_preview(chain: dict[str, Any], *, include_evidence: bool) -> str:
    retrieval = chain["retrieval"]
    lines = [
        f"query_class: {chain.get('query_class') or 'unknown'}",
        f"intent: {retrieval.get('intent') or 'unknown'}",
        f"hit_count: {retrieval.get('hit_count') or 0}",
        f"retrieval_route: {retrieval.get('retrieval_route') or 'unknown'}",
        f"fallback_trigger_reason: {retrieval.get('fallback_trigger_reason') or '-'}",
        f"answer_route: {retrieval.get('answer_route') or 'unknown'}",
    ]
    if include_evidence:
        evidence = retrieval.get("evidence") or []
        if evidence:
            lines.append("evidence:")
            for index, item in enumerate(evidence[:3], start=1):
                display_name = item.get("display_name") or item.get("scene_id") or "unknown"
                created_at = item.get("created_at") or ""
                objects = "、".join((item.get("objects") or [])[:3])
                description = truncate(str(item.get("description") or ""), limit=70)
                lines.append(f"  {index}. {display_name} | {created_at}")
                if objects:
                    lines.append(f"     objects: {objects}")
                if description:
                    lines.append(f"     desc: {description}")
        else:
            lines.append("evidence: <empty>")
    return "\n".join(lines)


def resolve_answer(
    *,
    chain: dict[str, Any],
    question: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    special_answer = str(chain.get("special_answer") or "").strip()
    if special_answer:
        return special_answer, 0.0
    started = time.time()
    answer = generate_answer(
        tokenizer=tokenizer,
        model=model,
        device=device,
        question=question,
        retrieval=chain["retrieval"],
        max_new_tokens=max_new_tokens,
    )
    return answer, round(time.time() - started, 3)


def run_single_turn(
    *,
    question: str,
    args: argparse.Namespace,
    dashscope_key: str,
    supabase_url: str,
    supabase_key: str,
    tokenizer: Any,
    model: Any,
    device: str,
) -> dict[str, Any]:
    retrieval_started = time.time()
    chain = retrieve_real_chain_case(
        question=question,
        dashscope_key=dashscope_key,
        dashscope_base_url=DEFAULT_DASHSCOPE_BASE_URL,
        chat_model=args.dashscope_chat_model,
        embedding_model=args.dashscope_embedding_model,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        match_threshold=args.match_threshold,
        match_count=args.match_count,
        recent_limit=args.recent_limit,
    )
    retrieval_latency = round(time.time() - retrieval_started, 3)
    answer, generation_latency = resolve_answer(
        chain=chain,
        question=question,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    analysis = analyze_answer(
        answer,
        {
            "group": "local_qa",
            "parsed_intent": chain["parsed_intent"],
            "retrieval": chain["retrieval"],
            "support_map": chain["support_map"],
        },
    )
    return {
        "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
        "question": question,
        "answer": answer,
        "query_class": chain["query_class"],
        "intent": chain["retrieval"]["intent"],
        "answer_route": chain["retrieval"]["answer_route"],
        "retrieval_route": chain["retrieval"]["retrieval_route"],
        "fallback_trigger_reason": chain["retrieval"]["fallback_trigger_reason"],
        "route_reasons": chain["retrieval"].get("route_reasons", []),
        "hit_count": chain["retrieval"]["hit_count"],
        "evidence": chain["retrieval"]["evidence"],
        "parsed_intent": chain["parsed_intent"],
        "raw_target_objects": chain.get("raw_target_objects", []),
        "normalized_lookup_terms": chain.get("normalized_lookup_terms", []),
        "retrieval_latency_sec": retrieval_latency,
        "generation_latency_sec": generation_latency,
        "answer_analysis": analysis,
        "_chain": chain,
    }


def print_turn_result(row: dict[str, Any], *, show_trace: bool, show_evidence: bool) -> None:
    print(f"\n助手> {row['answer']}")
    if show_trace or show_evidence:
        print()
        print(
            format_chain_preview(
                row["_chain"],
                include_evidence=show_evidence,
            )
        )


def main() -> None:
    args = parse_args()
    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()
    active_adapter_path = args.adapter_path if args.mode == "lora" else ""

    tokenizer = None
    model = None
    device = "cpu"
    last_row: dict[str, Any] | None = None
    log_path = Path(args.log_file) if args.log_file else None

    try:
        tokenizer, model, device = load_generator(args.model_name, active_adapter_path)
        if args.question.strip():
            row = run_single_turn(
                question=args.question.strip(),
                args=args,
                dashscope_key=dashscope_key,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            print_turn_result(row, show_trace=args.show_trace, show_evidence=args.show_evidence)
            if log_path:
                persisted = dict(row)
                persisted.pop("_chain", None)
                write_jsonl(log_path, [persisted])
            return

        print("BrainDance Local QA CLI")
        print(f"Mode: {args.mode}")
        if active_adapter_path:
            print(f"Adapter: {active_adapter_path}")
        print(HELP_TEXT.strip())

        while True:
            try:
                question = input("\n你> ").strip()
            except EOFError:
                print("\n收到 EOF，结束会话。")
                break
            except KeyboardInterrupt:
                print("\n输入已中断。使用 /quit 退出。")
                continue

            if not question:
                continue
            if question in {"/quit", "/exit"}:
                break
            if question == "/help":
                print(HELP_TEXT.strip())
                continue
            if question == "/last":
                if last_row is None:
                    print("当前还没有已记录轮次。")
                else:
                    print(format_chain_preview(last_row["_chain"], include_evidence=True))
                continue
            if question.startswith("/"):
                print("未知命令，输入 /help 查看帮助。")
                continue

            row = run_single_turn(
                question=question,
                args=args,
                dashscope_key=dashscope_key,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            print_turn_result(row, show_trace=args.show_trace, show_evidence=args.show_evidence)
            if log_path:
                persisted = dict(row)
                persisted.pop("_chain", None)
                write_jsonl(log_path, [persisted])
            last_row = row
    finally:
        unload_model(model)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
