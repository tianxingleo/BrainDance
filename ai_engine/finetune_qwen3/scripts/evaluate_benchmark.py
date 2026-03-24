#!/usr/bin/env python3
"""Evaluate Qwen3 base model or LoRA adapter on the fixed BrainDance benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


NO_ANSWER_PATTERNS = ["暂无相关记录", "暂无", "未见相关记录", "没有相关记录", "不知道", "不清楚"]
NEGATIVE_MARKERS = ["暂无", "未见", "没有", "未拍到", "没看到", "无", "不存在"]
GENERIC_POSITIVE_ANSWERS = {"有记录", "有相关记录", "有记录。", "有相关记录。", "有记录！", "有相关记录！"}
LIST_SEPARATORS = ("、", "，", ",", "；", ";")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 BrainDance benchmark")
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


def apply_chat(tokenizer: AutoTokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def normalize_text(text: str) -> str:
    return text.strip().replace("\n", " ")


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def contains_term(text: str, term: str) -> bool:
    return normalize_match_text(term) in normalize_match_text(text)


def contains_no_answer(text: str) -> bool:
    return any(pattern in text for pattern in NO_ANSWER_PATTERNS)


def is_natural_output(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    if "```" in stripped:
        return False
    if '"answer"' in stripped or '"output"' in stripped:
        return False
    if re.search(r'^\s*[A-Za-z_]+\s*:\s*', stripped):
        return False
    return True


def is_negated_mention(text: str, term: str) -> bool:
    normalized_text = normalize_match_text(text)
    clauses = [part for part in re.split(r"[，。；;！？!?,]", normalized_text) if part]
    escaped = re.escape(normalize_match_text(term))
    patterns = [
        rf"(?:{'|'.join(NEGATIVE_MARKERS)}).{{0,6}}{escaped}",
        rf"{escaped}.{{0,6}}(?:{'|'.join(NEGATIVE_MARKERS)})",
    ]
    for clause in clauses:
        if escaped and re.search(escaped, clause) and any(re.search(pattern, clause) for pattern in patterns):
            return True
    return False


def is_positive_mention(text: str, term: str) -> bool:
    return contains_term(text, term) and not is_negated_mention(text, term)


def uses_evidence(text: str, support_terms: list[str]) -> bool:
    return any(contains_term(text, term) for term in support_terms)


def is_generic_positive(text: str, support_terms: list[str]) -> bool:
    stripped = normalize_text(text).rstrip("。！!？?")
    if stripped in GENERIC_POSITIVE_ANSWERS:
        return True
    if len(stripped) <= 8 and "有记录" in stripped:
        return True
    if "有记录" in stripped and not uses_evidence(text, support_terms):
        return True
    return False


def count_list_separators(text: str) -> int:
    return sum(text.count(marker) for marker in LIST_SEPARATORS)


def is_natural_style(text: str) -> bool:
    if not is_natural_output(text):
        return False
    stripped = normalize_text(text)
    if "；" in stripped or ";" in stripped:
        return False
    if count_list_separators(stripped) >= 5:
        return False
    return True


def is_must_answer_focused(answer: str, case: dict[str, Any], generic_positive: bool) -> bool:
    if case["group"] != "must_answer":
        return True
    if generic_positive or not is_natural_style(answer):
        return False
    meta = case["metadata"]
    focus_terms = meta.get("supported_objects", []) or meta.get("support_terms", [])[:1]
    if not any(contains_term(answer, term) for term in focus_terms):
        return False
    extra_terms = [term for term in meta.get("support_terms", []) if term not in focus_terms and contains_term(answer, term)]
    over_broad = count_list_separators(answer) >= 4
    if over_broad and len(extra_terms) > 3:
        return False
    return True


def evaluate_case(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    meta = case["metadata"]
    hit_count = meta["hit_count"]
    support_terms = meta.get("support_terms", [])
    supported_objects = meta.get("supported_objects", [])
    unsupported_objects = meta.get("unsupported_objects", [])

    natural_output = is_natural_output(answer)
    natural_style = is_natural_style(answer)
    evidence_used = hit_count == 0 or uses_evidence(answer, support_terms)
    generic_positive = hit_count > 0 and meta.get("forbid_generic_positive", True) and is_generic_positive(answer, support_terms)

    positive_supported = [term for term in supported_objects if is_positive_mention(answer, term)]
    negative_supported = [term for term in supported_objects if is_negated_mention(answer, term)]
    positive_unsupported = [term for term in unsupported_objects if is_positive_mention(answer, term)]
    negative_unsupported = [term for term in unsupported_objects if is_negated_mention(answer, term)]
    partial_hallucination = case["group"] == "partial_coverage" and len(positive_unsupported) > 0
    partial_false_negative = (
        case["group"] == "partial_coverage"
        and len(supported_objects) > 0
        and (len(positive_supported) == 0 or len(negative_supported) > 0)
    )
    partial_missing_negation = (
        case["group"] == "partial_coverage"
        and len(supported_objects) > 0
        and len(unsupported_objects) > 0
        and len(positive_supported) > 0
        and len(negative_unsupported) < len(unsupported_objects)
        and not partial_false_negative
    )
    # A hit case should only count as "false no answer" when the model effectively refused the hit.
    # Partial-coverage answers are allowed to negate unsupported objects, so merely containing
    # "暂无相关记录" is not enough to mark the whole answer as a false refusal.
    false_no_answer = hit_count > 0 and contains_no_answer(answer) and not evidence_used and not positive_supported
    must_answer_focused = is_must_answer_focused(answer, case, generic_positive)

    return {
        "false_no_answer": false_no_answer,
        "natural_output": natural_output,
        "natural_style": natural_style,
        "evidence_used": evidence_used,
        "generic_positive": generic_positive,
        "positive_supported": positive_supported,
        "negative_supported": negative_supported,
        "positive_unsupported": positive_unsupported,
        "negative_unsupported": negative_unsupported,
        "partial_hallucination": partial_hallucination,
        "partial_false_negative": partial_false_negative,
        "partial_missing_negation": partial_missing_negation,
        "must_answer_focused": must_answer_focused,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    hit_cases = [row for row in results if row["metadata"]["hit_count"] > 0]
    partial_cases = [row for row in results if row["group"] == "partial_coverage"]
    must_cases = [row for row in results if row["group"] == "must_answer"]
    tp = sum(len(row["analysis"]["positive_supported"]) for row in partial_cases)
    fp = sum(len(row["analysis"]["positive_unsupported"]) for row in partial_cases)

    metrics = {
        "total_cases": len(results),
        "hit_cases": len(hit_cases),
        "false_no_answer_rate": round(sum(row["analysis"]["false_no_answer"] for row in hit_cases) / max(1, len(hit_cases)), 4),
        "partial_hallucination_rate": round(sum(row["analysis"]["partial_hallucination"] for row in partial_cases) / max(1, len(partial_cases)), 4),
        "natural_output_rate": round(sum(row["analysis"]["natural_output"] for row in results) / max(1, len(results)), 4),
        "natural_style_rate": round(sum(row["analysis"]["natural_style"] for row in results) / max(1, len(results)), 4),
        "evidence_utilization_rate": round(sum(row["analysis"]["evidence_used"] for row in hit_cases) / max(1, len(hit_cases)), 4),
        "partial_hit_precision": round(tp / max(1, tp + fp), 4),
        "partial_false_negative_rate": round(sum(row["analysis"]["partial_false_negative"] for row in partial_cases) / max(1, len(partial_cases)), 4),
        "partial_missing_negation_rate": round(sum(row["analysis"]["partial_missing_negation"] for row in partial_cases) / max(1, len(partial_cases)), 4),
        "must_answer_specific_rate": round(1 - (sum(row["analysis"]["generic_positive"] for row in must_cases) / max(1, len(must_cases))), 4),
        "must_answer_focus_rate": round(sum(row["analysis"]["must_answer_focused"] for row in must_cases) / max(1, len(must_cases)), 4),
    }

    metrics_by_group: dict[str, Any] = {}
    for group in sorted({row["group"] for row in results}):
        group_rows = [row for row in results if row["group"] == group]
        metrics_by_group[group] = {
            "count": len(group_rows),
            "natural_output_rate": round(sum(row["analysis"]["natural_output"] for row in group_rows) / max(1, len(group_rows)), 4),
            "natural_style_rate": round(sum(row["analysis"]["natural_style"] for row in group_rows) / max(1, len(group_rows)), 4),
            "evidence_utilization_rate": round(
                sum(row["analysis"]["evidence_used"] for row in group_rows if row["metadata"]["hit_count"] > 0)
                / max(1, len([row for row in group_rows if row["metadata"]["hit_count"] > 0])),
                4,
            ),
        }
        if group == "partial_coverage":
            metrics_by_group[group]["partial_false_negative_rate"] = round(
                sum(row["analysis"]["partial_false_negative"] for row in group_rows) / max(1, len(group_rows)),
                4,
            )
            metrics_by_group[group]["partial_missing_negation_rate"] = round(
                sum(row["analysis"]["partial_missing_negation"] for row in group_rows) / max(1, len(group_rows)),
                4,
            )
        if group == "must_answer":
            metrics_by_group[group]["must_answer_focus_rate"] = round(
                sum(row["analysis"]["must_answer_focused"] for row in group_rows) / max(1, len(group_rows)),
                4,
            )
    metrics["by_group"] = metrics_by_group
    return metrics


def main() -> None:
    args = parse_args()
    benchmark_rows = load_jsonl(Path(args.benchmark_file))

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
    for row in benchmark_rows:
        prompt = apply_chat(tokenizer, row["messages"], add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
        results.append({
            "case_id": row["case_id"],
            "group": row["group"],
            "answer": answer,
            "reference_answer": row["reference_answer"],
            "metadata": row["metadata"],
            "analysis": analysis,
        })

    payload = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "benchmark_file": args.benchmark_file,
        "metrics": aggregate(results),
        "results": results,
    }

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
