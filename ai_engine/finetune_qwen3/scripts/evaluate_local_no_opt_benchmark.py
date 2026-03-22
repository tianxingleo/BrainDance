#!/usr/bin/env python3
"""Evaluate local HF model (base or LoRA) in no-opt mode on benchmark.

No-opt mode:
- only pass raw question text to the model
- no system prompt
- no retrieval/evidence payload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_benchmark import aggregate, evaluate_case, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local model in no-opt mode")
    parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--benchmark_file", default="ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl")
    parser.add_argument("--output_file", default="")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
    content = str(messages[-1].get("content") or "")
    try:
        payload = json.loads(content)
        question = str(payload.get("question") or "").strip()
        return question or content.strip()
    except json.JSONDecodeError:
        return content.strip()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.benchmark_file))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question = extract_raw_question(row)
        inputs = tokenizer(question, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        answer = normalize_text(tokenizer.decode(answer_tokens, skip_special_tokens=True))
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
            print(f"[local-no-opt] {index}/{len(rows)}")

    payload = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "mode": "local_no_opt_no_ft_if_no_adapter",
        "benchmark_file": args.benchmark_file,
        "metrics": aggregate(results),
        "results": results,
    }

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
