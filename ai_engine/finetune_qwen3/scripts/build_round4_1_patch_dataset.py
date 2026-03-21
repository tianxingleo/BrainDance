#!/usr/bin/env python3
"""Build the narrow round4.1 patch dataset."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"

SEED_FILE = DATA_DIR / "real_chain_failures_round4_seed.jsonl"
ROUND4_PATCH_TRAIN_FILE = DATA_DIR / "real_chain_failures_round4_patch_train.jsonl"
ROUND4_PATCH_VAL_FILE = DATA_DIR / "real_chain_failures_round4_patch_val.jsonl"
PATCH_ALL_FILE = DATA_DIR / "real_chain_failures_round4_1_patch.jsonl"
PATCH_TRAIN_FILE = DATA_DIR / "real_chain_failures_round4_1_patch_train.jsonl"
PATCH_VAL_FILE = DATA_DIR / "real_chain_failures_round4_1_patch_val.jsonl"
COMBINED_TRAIN_FILE = DATA_DIR / "real_chain_failures_round4_1_patch_plus_round4_train.jsonl"
COMBINED_VAL_FILE = DATA_DIR / "real_chain_failures_round4_1_patch_plus_round4_val.jsonl"
MANIFEST_FILE = DATA_DIR / "real_chain_failures_round4_1_patch_manifest.json"

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

PATCH_TARGETS = {
    "partial_missing_negation": 12,
    "multi_hit_must_answer_style": 8,
}

PATCH_VAL_COUNTS = {
    "partial_missing_negation": 2,
    "multi_hit_must_answer_style": 2,
}

RANDOM_SEED = 42

STUDY_ROOM_EVIDENCE = [
    {
        "scene_id": "test_study_room",
        "display_name": "书房场景",
        "description": "明亮的书房，有一张写字台、椅子和书架。墙上挂着风景画。",
        "objects": ["写字台", "椅子", "书架", "风景画"],
        "tags": ["书房", "室内", "明亮"],
        "created_at": "2026-03-14T10:00:00Z",
    }
]

NOTEBOOK_MULTI_EVIDENCE = [
    {
        "scene_id": "frame_00146",
        "display_name": "frame_00146.jpg",
        "description": "一个现代办公/学习桌面场景，画面里有 AOC 显示器、浅蓝色 HONOR 笔记本电脑、定制渐变色机械键盘和 Elaina 手办。",
        "objects": [
            "AOC显示器",
            "HONOR笔记本电脑",
            "定制渐变色机械键盘",
            "Elaina婚纱手办（白色裙装，紫色底座）",
            "浅灰色桌面",
        ],
        "tags": ["室内", "办公桌", "笔记本电脑", "显示器", "手办"],
        "created_at": "2026-03-18T12:46:00.003458Z",
    },
    {
        "scene_id": "frame_00198",
        "display_name": "frame_00198.jpg",
        "description": "一张深灰色办公桌画面，桌上有白色联想笔记本电脑、戴尔显示器、高脚杯和黄色螺旋记事本。",
        "objects": [
            "深灰色办公桌",
            "白色联想笔记本电脑（带多款动漫贴纸）",
            "戴尔显示器",
            "透明玻璃高脚杯",
            "黄色螺旋记事本",
        ],
        "tags": ["室内", "办公桌", "笔记本电脑", "显示器"],
        "created_at": "2026-03-04T13:49:54.931673Z",
    },
]

PARTIAL_CASE_SPECS = [
    {
        "seed_key": "partial_missing_negation",
        "question": "我最近拍过笔记本电脑和钢琴吗？",
        "answer": "最近拍到过 HONOR 笔记本电脑，钢琴暂无相关记录。",
        "supported": ["笔记本电脑"],
        "unsupported": ["钢琴"],
    },
    {
        "seed_key": "partial_missing_negation",
        "question": "最近记录里同时有显示器和小提琴吗？",
        "answer": "有 AOC 显示器相关内容，未见小提琴相关记录。",
        "supported": ["显示器"],
        "unsupported": ["小提琴"],
    },
    {
        "seed_key": "partial_missing_negation",
        "question": "最近拍过手办，钢琴也拍到过吗？",
        "answer": "最近拍到过 Elaina 手办，没有找到钢琴相关记录。",
        "supported": ["手办"],
        "unsupported": ["钢琴"],
    },
    {
        "seed_key": "partial_missing_negation",
        "question": "请直接告诉我最近有没有拍到键盘和茶几。",
        "answer": "记录里出现过机械键盘，但没有茶几。",
        "supported": ["键盘"],
        "unsupported": ["茶几"],
    },
    {
        "seed_key": "partial_missing_negation",
        "question": "最近和笔记本电脑、冰箱有关的内容都有吗？",
        "answer": "最近拍到过 HONOR 笔记本电脑，冰箱暂无相关记录。",
        "supported": ["笔记本电脑"],
        "unsupported": ["冰箱"],
    },
    {
        "seed_key": "partial_missing_negation",
        "question": "最近拍过显示器，那钢琴呢？",
        "answer": "有 AOC 显示器相关内容，没有找到钢琴相关记录。",
        "supported": ["显示器"],
        "unsupported": ["钢琴"],
    },
    {
        "evidence_key": "study_room",
        "question": "最近记录里同时有椅子和打印机吗？",
        "answer": "最近拍到过椅子，打印机暂无相关记录。",
        "supported": ["椅子"],
        "unsupported": ["打印机"],
    },
    {
        "evidence_key": "study_room",
        "question": "上周的记录里有写字台和猫吗？",
        "answer": "有写字台相关内容，未见猫相关记录。",
        "supported": ["写字台"],
        "unsupported": ["猫"],
    },
    {
        "evidence_key": "study_room",
        "question": "最近拍过书架，吉他也拍到过吗？",
        "answer": "最近拍到过书架，没有找到吉他相关记录。",
        "supported": ["书架"],
        "unsupported": ["吉他"],
    },
    {
        "evidence_key": "study_room",
        "question": "最近和风景画、键盘有关的内容都有吗？",
        "answer": "记录里能看到风景画，但没有键盘。",
        "supported": ["风景画"],
        "unsupported": ["键盘"],
    },
    {
        "seed_key": "partial_false_negative",
        "question": "我最近拍过地球仪和钢琴吗？",
        "answer": "最近拍到过地球仪，钢琴暂无相关记录。",
        "supported": ["地球仪"],
        "unsupported": ["钢琴"],
    },
    {
        "seed_key": "partial_false_negative",
        "question": "最近记录里同时有帆船模型和咖啡机吗？",
        "answer": "有帆船模型相关内容，未见咖啡机相关记录。",
        "supported": ["帆船模型"],
        "unsupported": ["咖啡机"],
    },
]

MULTI_HIT_CASE_SPECS = [
    {
        "question": "最近拍到过什么笔记本电脑相关画面？",
        "answer": "最近拍到过多次笔记本电脑相关画面，包括 3 月 18 日的 HONOR 笔记本电脑和 3 月 4 日的联想笔记本电脑。",
    },
    {
        "question": "最近有哪些笔记本电脑相关记录？",
        "answer": "最近和笔记本电脑有关的记录不止一条，较新的是 3 月 18 日那台 HONOR 笔记本电脑，另外还有 3 月 4 日的联想笔记本电脑。",
    },
    {
        "question": "最近拍到的笔记本电脑内容主要是什么？",
        "answer": "最近的笔记本电脑相关画面主要有两次，一次是 3 月 18 日桌面上的 HONOR 笔记本电脑，一次是 3 月 4 日办公桌上的联想笔记本电脑。",
    },
    {
        "question": "最近和笔记本电脑有关的画面都是什么样的？",
        "answer": "最近拍到的笔记本电脑内容包括 3 月 18 日的 HONOR 笔记本电脑，以及 3 月 4 日办公桌上的联想笔记本电脑。",
    },
    {
        "question": "最近笔记本电脑相关的记录多吗？",
        "answer": "笔记本电脑相关记录有多条，最近能看到 3 月 18 日的 HONOR 笔记本电脑，也能看到 3 月 4 日的联想笔记本电脑。",
    },
    {
        "question": "最近两条笔记本电脑相关记录是什么？",
        "answer": "最近两条笔记本电脑相关画面分别是 3 月 18 日带 AOC 显示器的 HONOR 笔记本电脑场景，以及 3 月 4 日的联想笔记本电脑办公桌场景。",
    },
    {
        "question": "最近拍到过哪些笔记本电脑相关内容？",
        "answer": "最近拍到过两次笔记本电脑相关内容，一条是 3 月 18 日的 HONOR 笔记本电脑，另一条是 3 月 4 日的联想笔记本电脑。",
    },
    {
        "question": "最近的笔记本电脑相关记录里有什么重点内容？",
        "answer": "最近的笔记本电脑相关记录里，3 月 18 日这条画面是 HONOR 笔记本电脑，3 月 4 日那条是联想笔记本电脑。",
    },
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def simplify_object_name(text: str) -> str:
    base = re.split(r"[（(]", text, maxsplit=1)[0].strip()
    return re.sub(r"\s+", "", base)


def build_support_terms(evidence: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for item in evidence:
        display_name = str(item.get("display_name") or "").strip()
        if display_name and display_name not in terms:
            terms.append(display_name)
        created_at = str(item.get("created_at") or "").strip()
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                date_term = f"{dt.month}月{dt.day}日"
                if date_term not in terms:
                    terms.append(date_term)
            except ValueError:
                pass
        for key in ("objects", "tags"):
            for raw in item.get(key) or []:
                term = simplify_object_name(str(raw))
                if term and term not in terms:
                    terms.append(term)
    return terms


def make_input(question: str, evidence: list[dict[str, Any]], intent: str) -> str:
    payload = {
        "question": question,
        "retrieval": {
            "intent": intent,
            "hit_count": len(evidence),
            "evidence": evidence,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_source_id(failure_type: str, question: str, answer: str, evidence: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {
            "failure_type": failure_type,
            "question": question,
            "answer": answer,
            "scene_ids": [str(item.get("scene_id") or "") for item in evidence],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"round4_1_patch_{failure_type}_{digest}"


def make_case(
    *,
    failure_type: str,
    question: str,
    answer: str,
    intent: str,
    evidence: list[dict[str, Any]],
    supported_objects: list[str],
    unsupported_objects: list[str],
    patch_source: str,
) -> dict[str, Any]:
    return {
        "source_id": make_source_id(failure_type, question, answer, evidence),
        "category": "round4_1_patch",
        "failure_type": failure_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_input(question, evidence, intent)},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "hit_count": len(evidence),
            "supported_objects": supported_objects,
            "unsupported_objects": unsupported_objects,
            "support_terms": build_support_terms(evidence),
            "patch_source": patch_source,
        },
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["source_id"]] = row
    return list(deduped.values())


def split_patch_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["failure_type"]].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for failure_type, expected in PATCH_TARGETS.items():
        items = grouped[failure_type]
        if len(items) != expected:
            raise ValueError(f"{failure_type} expected {expected} rows, got {len(items)}")
        val_count = PATCH_VAL_COUNTS[failure_type]
        val_rows.extend(items[-val_count:])
        train_rows.extend(items[:-val_count])
    return train_rows, val_rows


def build_partial_rows(seed_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_map = {
        "study_room": STUDY_ROOM_EVIDENCE,
        "partial_missing_negation": seed_map["partial_missing_negation"]["evidence"],
        "partial_false_negative": seed_map["partial_false_negative"]["evidence"],
    }
    rows: list[dict[str, Any]] = []
    for spec in PARTIAL_CASE_SPECS:
        evidence_key = spec.get("evidence_key") or spec["seed_key"]
        patch_source = spec.get("evidence_key") or f"round4_seed:{spec['seed_key']}"
        rows.append(
            make_case(
                failure_type="partial_missing_negation",
                question=spec["question"],
                answer=spec["answer"],
                intent="partial_coverage",
                evidence=evidence_map[evidence_key],
                supported_objects=spec["supported"],
                unsupported_objects=spec["unsupported"],
                patch_source=patch_source,
            )
        )
    return rows


def build_multi_hit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MULTI_HIT_CASE_SPECS:
        rows.append(
            make_case(
                failure_type="multi_hit_must_answer_style",
                question=spec["question"],
                answer=spec["answer"],
                intent="time_qa",
                evidence=NOTEBOOK_MULTI_EVIDENCE,
                supported_objects=["笔记本电脑"],
                unsupported_objects=[],
                patch_source="real_chain:notebook_multi_hit",
            )
        )
    return rows


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    seeds = load_jsonl(SEED_FILE)
    seed_map = {row["failure_type"]: row for row in seeds}

    required_seeds = {"partial_missing_negation", "partial_false_negative"}
    missing = required_seeds - set(seed_map)
    if missing:
        raise ValueError(f"缺少 round4.1 seed failure_type: {sorted(missing)}")

    patch_rows = build_partial_rows(seed_map) + build_multi_hit_rows()
    patch_rows = dedupe_rows(patch_rows)
    total_expected = sum(PATCH_TARGETS.values())
    if len(patch_rows) != total_expected:
        raise ValueError(f"round4.1 patch 行数不符：expected {total_expected}, got {len(patch_rows)}")

    grouped_rows: list[dict[str, Any]] = []
    for failure_type in PATCH_TARGETS:
        items = [row for row in patch_rows if row["failure_type"] == failure_type]
        rng.shuffle(items)
        grouped_rows.extend(items)

    patch_train_rows, patch_val_rows = split_patch_rows(grouped_rows)
    round4_patch_train_rows = load_jsonl(ROUND4_PATCH_TRAIN_FILE)
    round4_patch_val_rows = load_jsonl(ROUND4_PATCH_VAL_FILE)
    combined_train_rows = round4_patch_train_rows + patch_train_rows
    combined_val_rows = round4_patch_val_rows + patch_val_rows
    rng.shuffle(combined_train_rows)
    rng.shuffle(combined_val_rows)

    write_jsonl(PATCH_ALL_FILE, grouped_rows)
    write_jsonl(PATCH_TRAIN_FILE, patch_train_rows)
    write_jsonl(PATCH_VAL_FILE, patch_val_rows)
    write_jsonl(COMBINED_TRAIN_FILE, combined_train_rows)
    write_jsonl(COMBINED_VAL_FILE, combined_val_rows)

    manifest = {
        "generated_at": iso_now(),
        "seed_file": str(SEED_FILE.relative_to(PROJECT_ROOT)),
        "patch_count": len(grouped_rows),
        "patch_train_count": len(patch_train_rows),
        "patch_val_count": len(patch_val_rows),
        "round4_patch_train_count": len(round4_patch_train_rows),
        "round4_patch_val_count": len(round4_patch_val_rows),
        "combined_patch_train_count": len(combined_train_rows),
        "combined_patch_val_count": len(combined_val_rows),
        "patch_targets": PATCH_TARGETS,
        "patch_val_targets": PATCH_VAL_COUNTS,
        "patch_sources": {
            "round4_seed_partial_missing_negation": 6,
            "study_room_partial_guard": 4,
            "round4_seed_partial_false_negative_guard": 2,
            "real_chain_multi_hit_notebook": 8,
        },
        "output_files": {
            "patch_all": str(PATCH_ALL_FILE.relative_to(PROJECT_ROOT)),
            "patch_train": str(PATCH_TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "patch_val": str(PATCH_VAL_FILE.relative_to(PROJECT_ROOT)),
            "combined_train": str(COMBINED_TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "combined_val": str(COMBINED_VAL_FILE.relative_to(PROJECT_ROOT)),
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
