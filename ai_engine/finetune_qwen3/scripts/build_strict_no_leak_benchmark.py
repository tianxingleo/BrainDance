#!/usr/bin/env python3
"""Build strict no-leak benchmark: dedup + paraphrase + OOD."""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict no-leak benchmark for BrainDance")
    parser.add_argument("--benchmark_file", default="ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl")
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=[
            "ai_engine/finetune_qwen3/data/braindance_qwen3_round4_train.jsonl",
            "ai_engine/finetune_qwen3/data/braindance_qwen3_sft_train.jsonl",
            "ai_engine/finetune_qwen3/data/real_chain_failures_round4_1_patch_plus_round4_train.jsonl",
        ],
    )
    parser.add_argument(
        "--output_file",
        default="ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark_strict_no_leak_ood_20260322.jsonl",
    )
    parser.add_argument(
        "--summary_file",
        default="ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark_strict_no_leak_ood_20260322_summary.json",
    )
    parser.add_argument("--fuzzy_threshold", type=float, default=0.88)
    parser.add_argument("--ood_per_group", type=int, default=3)
    parser.add_argument("--min_group_count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260322)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_question(text: str) -> str:
    value = re.sub(r"\s+", "", text or "")
    value = re.sub(r"[，。！？、,.!?;；:：'\"（）()【】\[\]<>《》\-—_]+", "", value)
    return value.strip().lower()


def extract_question_from_user_content(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    return str(payload.get("question") or "").strip()


def extract_train_questions(paths: list[Path]) -> list[str]:
    values: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        for row in rows:
            messages = row.get("messages") or []
            user_messages = [item for item in messages if item.get("role") == "user"]
            if not user_messages:
                continue
            content = str(user_messages[-1].get("content") or "")
            question = extract_question_from_user_content(content)
            if question:
                values.append(question)
    return values


def paraphrase_question(question: str, idx: int) -> str:
    replacements = [
        ("最近", "这段时间"),
        ("上周", "前一周"),
        ("拍过", "记录过"),
        ("拍到", "看到"),
        ("有哪些", "都有哪些"),
        ("是什么", "具体是什么"),
        ("有没有", "是否有"),
        ("我", "我这边"),
    ]
    rewritten = question
    for old, new in replacements:
        if old in rewritten:
            rewritten = rewritten.replace(old, new, 1)
    prefixes = ["帮我确认下，", "我想核对一下，", "换个问法：", "再确认一遍："]
    suffixes = ["", "，请直接回答", "，一句话告诉我", "，别太泛"]
    if rewritten == question:
        rewritten = prefixes[idx % len(prefixes)] + rewritten + suffixes[idx % len(suffixes)]
    else:
        if idx % 2 == 0:
            rewritten = prefixes[idx % len(prefixes)] + rewritten
        if idx % 3 == 0:
            rewritten = rewritten + suffixes[(idx + 1) % len(suffixes)]
    return rewritten.strip()


def ood_rewrite(question: str, idx: int) -> str:
    patterns = [
        "急，{q} 只说结论",
        "bro，{q}",
        "{q} ???",
        "我随便问下：{q}",
        "请你像复盘一样回答：{q}",
        "{q}，中文短答",
    ]
    text = patterns[idx % len(patterns)].format(q=question)
    text = text.replace("什么", "啥")
    return text.strip()


def aggressive_group_rewrite(question: str, group: str, idx: int) -> str:
    q = question.strip()
    templates = {
        "recent_hit": [
            "请按时间倒序概括我近期新增的拍摄记录。",
            "最近一段时间我的镜头里主要出现了哪些内容？",
            "把我最近捕捉到的画面做一个简短清单。",
        ],
        "must_answer": [
            f"请围绕目标对象直接作答，不要泛化：{q}",
            f"我只关心核心命中项，请聚焦回复：{q}",
            f"请给出与问题最相关的命中对象：{q}",
        ],
        "partial_coverage": [
            f"请区分“有记录”和“无记录”两部分回答：{q}",
            f"请分别说明命中项与未命中项：{q}",
            f"请对每个候选对象逐一判断是否被拍到：{q}",
        ],
        "stability": [
            f"用稳定口径重复确认一次：{q}",
            f"请给出与前述一致的简洁答案：{q}",
            f"再做一次同口径确认：{q}",
        ],
        "no_hit": [
            f"若无记录请明确说明，不要猜测：{q}",
            f"只基于现有记录判断是否命中：{q}",
            f"如果没有证据请直接拒答：{q}",
        ],
    }
    bucket = templates.get(group) or [f"改写提问：{q}"]
    return bucket[idx % len(bucket)]


def best_fuzzy_ratio(query: str, train_questions: list[str]) -> float:
    value = normalize_question(query)
    if not value:
        return 0.0
    best = 0.0
    for train_q in train_questions:
        ratio = difflib.SequenceMatcher(None, value, normalize_question(train_q)).ratio()
        if ratio > best:
            best = ratio
    return best


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    benchmark_rows = load_jsonl(Path(args.benchmark_file))
    train_questions = extract_train_questions([Path(path) for path in args.train_files])
    train_norm_set = {normalize_question(item) for item in train_questions if item.strip()}

    filtered_rows: list[dict[str, Any]] = []
    removed_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    removed_exact = 0
    removed_fuzzy = 0

    for row in benchmark_rows:
        raw_question = extract_question_from_user_content(str(row["messages"][-1]["content"]))
        norm = normalize_question(raw_question)
        if norm in train_norm_set:
            removed_exact += 1
            removed_rows_by_group[row["group"]].append(row)
            continue
        fuzzy = best_fuzzy_ratio(raw_question, train_questions)
        if fuzzy >= args.fuzzy_threshold:
            removed_fuzzy += 1
            removed_rows_by_group[row["group"]].append(row)
            continue
        filtered_rows.append(row)

    group_counter = defaultdict(int)
    for row in filtered_rows:
        group_counter[row["group"]] += 1

    expected_groups = sorted({row["group"] for row in benchmark_rows})
    backfilled_count = 0
    for group in expected_groups:
        need = max(0, args.min_group_count - group_counter[group])
        if need <= 0:
            continue
        candidates = removed_rows_by_group.get(group, [])
        for idx, src in enumerate(candidates, start=1):
            if need <= 0:
                break
            payload = json.loads(src["messages"][-1]["content"])
            raw_q = str(payload.get("question") or "").strip()
            new_q = aggressive_group_rewrite(raw_q, group, idx)
            norm_q = normalize_question(new_q)
            if norm_q in train_norm_set:
                continue
            if best_fuzzy_ratio(new_q, train_questions) >= args.fuzzy_threshold:
                continue
            item = copy.deepcopy(src)
            payload["question"] = new_q
            item["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            item["case_id"] = f"{src['case_id']}_bf{idx:02d}"
            item["source_id"] = f"{src.get('source_id') or src['case_id']}|strict_backfill"
            filtered_rows.append(item)
            group_counter[group] += 1
            backfilled_count += 1
            need -= 1

    rewritten_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(filtered_rows, start=1):
        item = copy.deepcopy(row)
        payload = json.loads(item["messages"][-1]["content"])
        q = str(payload.get("question") or "").strip()
        new_q = paraphrase_question(q, idx)
        if normalize_question(new_q) in train_norm_set:
            new_q = f"换个说法确认：{new_q}"
        payload["question"] = new_q
        item["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        item["case_id"] = f"{row['case_id']}_rw"
        item["source_id"] = f"{row.get('source_id') or row['case_id']}|strict_rw"
        rewritten_rows.append(item)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rewritten_rows:
        groups[row["group"]].append(row)

    ood_rows: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        picks = rows[: args.ood_per_group]
        for idx, row in enumerate(picks, start=1):
            item = copy.deepcopy(row)
            payload = json.loads(item["messages"][-1]["content"])
            q = str(payload.get("question") or "").strip()
            payload["question"] = ood_rewrite(q, idx)
            item["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            item["case_id"] = f"{row['case_id']}_ood{idx:02d}"
            item["source_id"] = f"{row.get('source_id') or row['case_id']}|strict_ood"
            ood_rows.append(item)

    final_rows = rewritten_rows + ood_rows
    write_jsonl(Path(args.output_file), final_rows)

    summary = {
        "benchmark_file": args.benchmark_file,
        "train_files": args.train_files,
        "fuzzy_threshold": args.fuzzy_threshold,
        "input_case_count": len(benchmark_rows),
        "removed_exact_overlap_count": removed_exact,
        "removed_fuzzy_overlap_count": removed_fuzzy,
        "remaining_after_dedup_count": len(filtered_rows),
        "backfilled_case_count": backfilled_count,
        "rewritten_case_count": len(rewritten_rows),
        "ood_case_count": len(ood_rows),
        "final_case_count": len(final_rows),
        "group_counts": dict(sorted((k, sum(1 for row in final_rows if row["group"] == k)) for k in groups.keys())),
        "output_file": args.output_file,
    }
    Path(args.summary_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
