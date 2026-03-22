#!/usr/bin/env python3
"""Build the round4 patch dataset from real-chain failure seeds."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"

SEED_FILE = DATA_DIR / "real_chain_failures_round4_seed.jsonl"
PATCH_ALL_FILE = DATA_DIR / "real_chain_failures_round4_patch.jsonl"
PATCH_TRAIN_FILE = DATA_DIR / "real_chain_failures_round4_patch_train.jsonl"
PATCH_VAL_FILE = DATA_DIR / "real_chain_failures_round4_patch_val.jsonl"
MERGED_TRAIN_FILE = DATA_DIR / "braindance_qwen3_round4_train.jsonl"
MERGED_VAL_FILE = DATA_DIR / "braindance_qwen3_round4_val.jsonl"
MANIFEST_FILE = DATA_DIR / "braindance_qwen3_round4_manifest.json"

ROUND3_TRAIN_FILE = DATA_DIR / "braindance_qwen3_sft_train.jsonl"
ROUND3_VAL_FILE = DATA_DIR / "braindance_qwen3_sft_val.jsonl"

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
    "partial_false_negative": 30,
    "partial_missing_negation": 20,
    "must_answer_too_broad": 20,
    "style_not_natural": 10,
}

PATCH_VAL_COUNTS = {
    "partial_false_negative": 3,
    "partial_missing_negation": 2,
    "must_answer_too_broad": 2,
    "style_not_natural": 1,
}

RANDOM_SEED = 42


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
    return f"round4_patch_{failure_type}_{digest}"


def make_case(
    *,
    seed: dict[str, Any],
    failure_type: str,
    question: str,
    answer: str,
    intent: str,
    evidence: list[dict[str, Any]],
    supported_objects: list[str],
    unsupported_objects: list[str],
    support_map: dict[str, bool],
) -> dict[str, Any]:
    return {
        "source_id": make_source_id(failure_type, question, answer, evidence),
        "category": "round4_patch",
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
            "patch_source_question": seed["question"],
            "patch_source_failure_type": seed["failure_type"],
            "support_map": support_map,
        },
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["source_id"]] = row
    return list(deduped.values())


def build_partial_false_negative_rows(seed: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = seed["evidence"]
    supported_specs = [
        ("地球仪", "地球仪"),
        ("闹钟", "红色闹钟"),
        ("音箱", "JBL 音箱"),
        ("帆船模型", "帆船模型"),
        ("初音未来手办", "初音未来手办"),
        ("耳机", "骨传导耳机"),
    ]
    missing_terms = ["钢琴", "小提琴", "吉他", "咖啡机", "洗衣机"]
    question_templates = [
        "我最近拍过{found}和{missing}吗？",
        "最近有{found}和{missing}的记录吗？",
        "这几天看到过{found}和{missing}吗？",
        "最近拍到的画面里有{found}，也有{missing}吗？",
        "最近和{found}、{missing}有关的内容都有吗？",
        "最近拍过{found}，那{missing}呢？",
    ]
    answer_templates = [
        "最近拍到过{found}，{missing}暂无相关记录。",
        "有{found}相关内容，没有找到{missing}相关记录。",
        "记录里能看到{found}，但没有{missing}。",
        "目前查到过{found}，未见{missing}。",
        "{found}是有记录的，{missing}没有查到。",
    ]

    rows: list[dict[str, Any]] = []
    for idx, ((asked, answer_name), missing) in enumerate(product(supported_specs, missing_terms)):
        question = question_templates[idx % len(question_templates)].format(found=asked, missing=missing)
        answer = answer_templates[idx % len(answer_templates)].format(found=answer_name, missing=missing)
        rows.append(
            make_case(
                seed=seed,
                failure_type="partial_false_negative",
                question=question,
                answer=answer,
                intent="partial_coverage",
                evidence=evidence,
                supported_objects=[asked],
                unsupported_objects=[missing],
                support_map={asked: True, missing: False},
            )
        )
    return rows


def build_partial_missing_negation_rows(seed: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = seed["evidence"]
    supported_specs = [
        ("笔记本电脑", "HONOR 笔记本电脑"),
        ("显示器", "AOC 显示器"),
        ("键盘", "机械键盘"),
        ("手办", "Elaina 手办"),
    ]
    missing_terms = ["钢琴", "小提琴", "吉他", "冰箱", "茶几"]
    question_templates = [
        "我最近拍过{found}和{missing}吗？",
        "最近记录里同时有{found}和{missing}吗？",
        "请直接告诉我最近有没有拍到{found}和{missing}。",
        "这几天看到过{found}和{missing}吗？",
        "最近和{found}相关的内容有，{missing}也有吗？",
    ]
    answer_templates = [
        "最近拍到过{found}，{missing}暂无相关记录。",
        "有{found}相关内容，未见{missing}相关记录。",
        "目前只查到{found}，没有找到{missing}。",
        "记录里出现过{found}，但没有{missing}。",
    ]

    rows: list[dict[str, Any]] = []
    for idx, ((asked, answer_name), missing) in enumerate(product(supported_specs, missing_terms)):
        question = question_templates[idx % len(question_templates)].format(found=asked, missing=missing)
        answer = answer_templates[idx % len(answer_templates)].format(found=answer_name, missing=missing)
        rows.append(
            make_case(
                seed=seed,
                failure_type="partial_missing_negation",
                question=question,
                answer=answer,
                intent="partial_coverage",
                evidence=evidence,
                supported_objects=[asked],
                unsupported_objects=[missing],
                support_map={asked: True, missing: False},
            )
        )
    return rows


def build_must_answer_rows(seed: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = seed["evidence"]
    questions = [
        "最近拍到过什么办公桌上的东西？",
        "最近办公桌画面里主要有什么？",
        "最近桌上拍到了哪些物品？",
        "最近那个办公桌场景里有什么？",
        "最近拍到过什么桌面摆件？",
        "最近办公桌那条记录里有什么？",
        "最近和办公桌有关的画面里主要拍到了什么？",
        "最近拍到的桌上物品有哪些？",
        "最近办公桌上最显眼的是什么？",
        "最近办公桌场景里都看到了什么？",
        "最近那张白色办公桌的画面里有什么？",
        "最近桌面场景里主要拍到了哪些东西？",
        "最近办公桌上的物品是什么样的？",
        "最近拍到过什么放在办公桌上的东西？",
        "最近那个桌面记录里最主要的内容是什么？",
        "最近桌面上有什么比较显眼的物品？",
        "最近办公桌相关的画面里能看到什么？",
        "最近白色办公桌那条记录主要拍到了什么？",
        "最近办公桌上的摆件和旁边物品有哪些？",
        "最近桌上那条记录里有什么重点内容？",
    ]
    answers = [
        "最近拍到过放在办公桌上的 Elaina 手办，旁边还有笔记本电脑和纸巾盒。",
        "办公桌画面里主要是 Elaina 手办，旁边能看到笔记本电脑和绿色纸巾盒。",
        "最近那条办公桌记录里有 Elaina 手办，桌边还有笔记本电脑和纸巾盒。",
        "最近拍到的办公桌场景以 Elaina 手办为主，旁边还有笔记本电脑和纸巾盒。",
        "最近办公桌上最显眼的是 Elaina 手办，附近还能看到笔记本电脑和纸巾盒。",
    ]

    rows: list[dict[str, Any]] = []
    for idx, question in enumerate(questions):
        rows.append(
            make_case(
                seed=seed,
                failure_type="must_answer_too_broad",
                question=question,
                answer=answers[idx % len(answers)],
                intent=seed["intent"],
                evidence=evidence,
                supported_objects=["Elaina手办"],
                unsupported_objects=[],
                support_map={},
            )
        )
    return rows


def build_style_rows(seed: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = seed["evidence"]
    questions = [
        "这几天我拍了什么？",
        "最近拍到的主要内容是什么？",
        "最近新增了哪些拍摄内容？",
        "最近两三条记录是什么？",
        "最近拍过哪些场景？",
        "这几天主要拍了哪些东西？",
        "最近更新的记录里有什么？",
        "最近的新照片大概都是什么？",
        "最近拍到过什么内容？",
        "最近这几次拍摄分别是什么？",
    ]
    answers = [
        "最近拍到过洛天依主题展台，还有带蓝色地球仪的书架角落。",
        "这几天主要拍了洛天依展台，以及带地球仪的书架场景。",
        "最近的记录里有洛天依主题展台，也有带蓝色地球仪的书架画面。",
        "最近拍到的内容主要是洛天依展台和地球仪书架角落。",
        "最近新增的画面包括洛天依主题展台和带地球仪的书架角落。",
    ]

    rows: list[dict[str, Any]] = []
    for idx, question in enumerate(questions):
        rows.append(
            make_case(
                seed=seed,
                failure_type="style_not_natural",
                question=question,
                answer=answers[idx % len(answers)],
                intent=seed["intent"],
                evidence=evidence,
                supported_objects=["洛天依毛绒玩偶", "地球仪"],
                unsupported_objects=[],
                support_map={},
            )
        )
    return rows


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


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    seeds = load_jsonl(SEED_FILE)
    seed_map = {row["failure_type"]: row for row in seeds}

    required = set(PATCH_TARGETS)
    missing = required - set(seed_map)
    if missing:
        raise ValueError(f"缺少 round4 seed failure_type: {sorted(missing)}")

    patch_rows = []
    patch_rows.extend(build_partial_false_negative_rows(seed_map["partial_false_negative"]))
    patch_rows.extend(build_partial_missing_negation_rows(seed_map["partial_missing_negation"]))
    patch_rows.extend(build_must_answer_rows(seed_map["must_answer_too_broad"]))
    patch_rows.extend(build_style_rows(seed_map["style_not_natural"]))
    patch_rows = dedupe_rows(patch_rows)

    total_expected = sum(PATCH_TARGETS.values())
    if len(patch_rows) != total_expected:
        raise ValueError(f"round4 patch 行数不符：expected {total_expected}, got {len(patch_rows)}")

    patch_train_rows, patch_val_rows = split_patch_rows(patch_rows)
    base_train_rows = load_jsonl(ROUND3_TRAIN_FILE)
    base_val_rows = load_jsonl(ROUND3_VAL_FILE)

    merged_train_rows = base_train_rows + patch_train_rows
    merged_val_rows = base_val_rows + patch_val_rows
    rng.shuffle(merged_train_rows)
    rng.shuffle(merged_val_rows)

    write_jsonl(PATCH_ALL_FILE, patch_rows)
    write_jsonl(PATCH_TRAIN_FILE, patch_train_rows)
    write_jsonl(PATCH_VAL_FILE, patch_val_rows)
    write_jsonl(MERGED_TRAIN_FILE, merged_train_rows)
    write_jsonl(MERGED_VAL_FILE, merged_val_rows)

    manifest = {
        "generated_at": iso_now(),
        "seed_file": str(SEED_FILE.relative_to(PROJECT_ROOT)),
        "patch_count": len(patch_rows),
        "patch_train_count": len(patch_train_rows),
        "patch_val_count": len(patch_val_rows),
        "base_train_count": len(base_train_rows),
        "base_val_count": len(base_val_rows),
        "merged_train_count": len(merged_train_rows),
        "merged_val_count": len(merged_val_rows),
        "patch_targets": PATCH_TARGETS,
        "patch_val_targets": PATCH_VAL_COUNTS,
        "output_files": {
            "patch_all": str(PATCH_ALL_FILE.relative_to(PROJECT_ROOT)),
            "patch_train": str(PATCH_TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "patch_val": str(PATCH_VAL_FILE.relative_to(PROJECT_ROOT)),
            "merged_train": str(MERGED_TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "merged_val": str(MERGED_VAL_FILE.relative_to(PROJECT_ROOT)),
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
