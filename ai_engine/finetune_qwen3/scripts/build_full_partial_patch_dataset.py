#!/usr/bin/env python3
"""Build a small partial_coverage patch dataset for 0.6B full SFT follow-up experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"
OUTPUT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "outputs" / "datasets"

SYSTEM_PROMPT = (
    "你是 BrainDance 的本地记忆问答助手。"
    "你只能根据 retrieval 提供的证据回答，不要猜测。"
    "规则："
    "1. hit_count > 0 时，必须回答具体内容，不能只说有记录。"
    "2. hit_count == 0 时，只能回答‘暂无相关记录’。"
    "3. 部分命中时，只能回答证据覆盖到的部分，对未命中部分明确说‘暂无相关记录’或‘未见相关记录’。"
    "4. 输出必须是自然语言短句，最多两句。不要输出 JSON、代码块、列表或键值对。"
    "5. 不复述问题，不解释规则，不说‘根据给定证据’。"
)

RAW_BENCHMARK = DATA_DIR / "braindance_qwen3_benchmark.jsonl"
STRICT_BENCHMARK = DATA_DIR / "braindance_qwen3_benchmark_strict_no_leak_ood_20260322_v3.jsonl"
BASE_TRAIN = DATA_DIR / "braindance_qwen3_sft_train.jsonl"
BASE_VAL = DATA_DIR / "braindance_qwen3_sft_val.jsonl"

PATCH_TRAIN = DATA_DIR / "qwen3_0p6b_full_partial_patch_v1_train.jsonl"
PATCH_VAL = DATA_DIR / "qwen3_0p6b_full_partial_patch_v1_val.jsonl"
MERGED_TRAIN = OUTPUT_DIR / "qwen3_0p6b_full_partial_patch_v1_train.jsonl"
MERGED_VAL = OUTPUT_DIR / "qwen3_0p6b_full_partial_patch_v1_val.jsonl"

TRAIN_CASE_IDS = [
    "partial_coverage_001",
    "partial_coverage_003",
    "partial_coverage_004",
    "partial_coverage_005",
    "partial_coverage_006",
    "partial_coverage_007",
    "partial_coverage_008",
    "partial_coverage_009",
    "partial_coverage_010",
    "partial_coverage_011",
    "partial_coverage_012",
    "partial_coverage_013",
    "partial_coverage_014",
    "partial_coverage_015",
    "partial_coverage_016",
    "partial_coverage_001_rw_ood01",
    "partial_coverage_003_rw_ood02",
    "partial_coverage_004_rw_ood03",
]

VAL_CASE_IDS = [
    "partial_coverage_002",
    "partial_coverage_006_rw",
    "partial_coverage_008_rw",
    "partial_coverage_015_rw",
]

QUESTION_TEMPLATES = [
    "请直接回答：最近有{supported}，{unsupported}也有吗？",
    "最近和{supported}有关的内容能看到，{unsupported}呢？",
]

ANSWER_TEMPLATES = [
    "{supported}有相关记录，未见{unsupported}相关记录。",
    "目前只找到{supported}相关内容，{unsupported}暂无相关记录。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a targeted partial_coverage patch dataset")
    parser.add_argument("--patch_train_file", default=str(PATCH_TRAIN))
    parser.add_argument("--patch_val_file", default=str(PATCH_VAL))
    parser.add_argument("--merged_train_file", default=str(MERGED_TRAIN))
    parser.add_argument("--merged_val_file", default=str(MERGED_VAL))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_user_payload(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][1]["content"])


def make_source_id(case_id: str, variant_idx: int, question: str, answer: str) -> str:
    payload = json.dumps(
        {"case_id": case_id, "variant_idx": variant_idx, "question": question, "answer": answer},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"full_partial_patch_{case_id}_{digest}"


def make_variant(base_row: dict[str, Any], variant_idx: int) -> dict[str, Any]:
    payload = parse_user_payload(base_row)
    meta = base_row["metadata"]
    supported = meta["supported_objects"][0]
    unsupported = meta["unsupported_objects"][0]
    question = QUESTION_TEMPLATES[variant_idx].format(supported=supported, unsupported=unsupported)
    answer = ANSWER_TEMPLATES[variant_idx].format(supported=supported, unsupported=unsupported)
    user_content = json.dumps(
        {
            "question": question,
            "retrieval": payload["retrieval"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "source_id": make_source_id(base_row["case_id"], variant_idx, question, answer),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "category": "partial_coverage_patch_v1",
        "meta": {
            "patch_source_case_id": base_row["case_id"],
            "supported_objects": meta["supported_objects"],
            "unsupported_objects": meta["unsupported_objects"],
            "support_terms": meta["support_terms"],
            "benchmark_group": "partial_coverage",
            "reference_answer": base_row["reference_answer"],
        },
    }


def collect_case_rows(case_ids: list[str]) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for source in (RAW_BENCHMARK, STRICT_BENCHMARK):
        for row in load_jsonl(source):
            rows_by_id[row["case_id"]] = row
    missing = [case_id for case_id in case_ids if case_id not in rows_by_id]
    if missing:
        raise SystemExit(f"Missing case ids: {missing}")
    patch_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        base_row = rows_by_id[case_id]
        for variant_idx in range(len(QUESTION_TEMPLATES)):
            patch_rows.append(make_variant(base_row, variant_idx))
    return patch_rows


def main() -> None:
    args = parse_args()
    patch_train_file = Path(args.patch_train_file)
    patch_val_file = Path(args.patch_val_file)
    merged_train_file = Path(args.merged_train_file)
    merged_val_file = Path(args.merged_val_file)

    patch_train_rows = collect_case_rows(TRAIN_CASE_IDS)
    patch_val_rows = collect_case_rows(VAL_CASE_IDS)
    write_jsonl(patch_train_file, patch_train_rows)
    write_jsonl(patch_val_file, patch_val_rows)

    merged_train_rows = load_jsonl(BASE_TRAIN) + patch_train_rows
    merged_val_rows = load_jsonl(BASE_VAL) + patch_val_rows
    write_jsonl(merged_train_file, merged_train_rows)
    write_jsonl(merged_val_file, merged_val_rows)

    manifest = {
        "patch_train_file": str(patch_train_file),
        "patch_val_file": str(patch_val_file),
        "merged_train_file": str(merged_train_file),
        "merged_val_file": str(merged_val_file),
        "patch_train_count": len(patch_train_rows),
        "patch_val_count": len(patch_val_rows),
        "merged_train_count": len(merged_train_rows),
        "merged_val_count": len(merged_val_rows),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
