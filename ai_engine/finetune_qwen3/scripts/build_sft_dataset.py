#!/usr/bin/env python3
"""
Build the round-3 BrainDance SFT dataset and fixed benchmark.

Goals:
- increase coverage for partial_coverage and must_answer
- force natural-language outputs instead of JSON
- keep a fixed benchmark for regression checking
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"
FRAME_DIR = PROJECT_ROOT / "ai_engine" / "demo" / "rag" / "data" / "output_analyzed"

RANDOM_SEED = 42
NOW = datetime(2026, 3, 21, 10, 0, 0, tzinfo=timezone.utc)

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

QUOTAS = {
    "partial_coverage": 350,
    "must_answer": 300,
    "recent_time": 200,
    "nohit_stability": 150,
}

BENCHMARK_GROUP_SIZES = {
    "recent_hit": 16,
    "no_hit": 16,
    "partial_coverage": 16,
    "must_answer": 16,
    "stability": 16,
}

FOCUS_SUPPORTED = ["触控笔", "键盘", "鼠标", "书架", "冰箱", "沙发", "办公椅", "床", "写字台", "茶几"]
NO_HIT_KEYWORDS = ["自行车", "海边", "吉他", "猫", "咖啡机", "打印机", "杯子", "耳机", "电视", "花瓶"]
ALT_MISSING = ["冰箱", "沙发", "办公椅", "床", "写字台", "茶几", "键盘", "书架"] + NO_HIT_KEYWORDS
GENERIC_POSITIVE_ANSWERS = {"有记录。", "有相关记录。", "有记录", "有相关记录"}


@dataclass
class MemoryRecord:
    scene_id: str
    display_name: str
    description: str
    objects: list[str]
    tags: list[str]
    created_at: str

    def to_evidence(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "display_name": self.display_name,
            "description": self.description,
            "objects": self.objects,
            "tags": self.tags,
            "created_at": self.created_at,
        }


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def fmt_date(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return f"{dt.month}月{dt.day}日"


def compact_desc(description: str) -> str:
    text = description.strip().rstrip("。；")
    if len(text) <= 30:
        return text
    return text[:30].rstrip("，, ") + "……"


def infer_objects_and_tags(description: str) -> tuple[list[str], list[str]]:
    object_map = [
        ("触控笔", "触控笔"),
        ("HUAWEI", "触控笔"),
        ("键盘", "键盘"),
        ("鼠标", "鼠标"),
        ("耳机盒", "耳机盒"),
        ("鼠标垫", "鼠标垫"),
        ("桌面", "桌面"),
        ("木纹桌面", "桌面"),
    ]
    tag_map = [
        ("桌面", "桌面"),
        ("木纹", "木纹"),
        ("室内", "室内"),
        ("键盘", "办公"),
        ("鼠标", "办公"),
        ("触控笔", "设备"),
        ("反光", "弱反光"),
        ("背光", "低照度"),
    ]
    objects: list[str] = []
    tags: list[str] = ["室内"]
    for needle, value in object_map:
        if needle in description and value not in objects:
            objects.append(value)
    for needle, value in tag_map:
        if needle in description and value not in tags:
            tags.append(value)
    if not objects:
        objects.append("场景物体")
    return objects, tags


def load_frame_records() -> list[MemoryRecord]:
    frame_files = sorted(FRAME_DIR.glob("frame_*.json"))
    records: list[MemoryRecord] = []
    start = NOW - timedelta(days=1, hours=16)
    for idx, path in enumerate(frame_files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        description = payload["description"]
        objects, tags = infer_objects_and_tags(description)
        created_at = start - timedelta(hours=6 * idx)
        records.append(
            MemoryRecord(
                scene_id=path.stem,
                display_name=f"触控笔桌面采集 {idx + 1:02d}",
                description=description,
                objects=objects,
                tags=tags,
                created_at=iso(created_at),
            )
        )
    return records


def build_manual_records() -> list[MemoryRecord]:
    base = NOW - timedelta(days=7)
    items = [
        ("test_study_room", "书房场景", "明亮的书房，有一张写字台、椅子和书架。墙上挂着风景画。", ["写字台", "椅子", "书架", "风景画"], ["书房", "室内", "明亮"]),
        ("test_bedroom", "卧室场景", "温馨的卧室，有大床、衣柜和床头柜。窗帘遮光良好。", ["床", "衣柜", "床头柜", "窗帘"], ["卧室", "室内", "温馨", "暗光"]),
        ("test_kitchen", "厨房场景", "现代化的厨房，有冰箱、灶台和洗碗机。台面整洁。", ["冰箱", "灶台", "洗碗机", "台面"], ["厨房", "室内", "现代"]),
        ("test_living_room", "客厅场景", "宽敞的客厅，有沙发、茶几和电视柜。地毯柔软舒适。", ["沙发", "茶几", "电视柜", "地毯"], ["客厅", "室内", "宽敞"]),
        ("test_office", "办公室场景", "简约的办公室，有电脑桌、办公椅和文件柜。", ["电脑桌", "办公椅", "文件柜"], ["办公室", "室内", "简约"]),
    ]
    records: list[MemoryRecord] = []
    for idx, (scene_id, display_name, description, objects, tags) in enumerate(items):
        records.append(
            MemoryRecord(
                scene_id=scene_id,
                display_name=display_name,
                description=description,
                objects=objects,
                tags=tags,
                created_at=iso(base - timedelta(days=idx)),
            )
        )
    return records


def all_supported_objects(records: list[MemoryRecord]) -> list[str]:
    objects = sorted({obj for record in records for obj in record.objects})
    return [obj for obj in objects if len(obj) >= 1]


def filter_by_keyword(records: list[MemoryRecord], keyword: str, start: datetime | None = None, end: datetime | None = None) -> list[MemoryRecord]:
    hits: list[MemoryRecord] = []
    for record in records:
        ts = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        haystack = " ".join([record.display_name, record.description, " ".join(record.objects), " ".join(record.tags)])
        if keyword in haystack:
            hits.append(record)
    return sorted(hits, key=lambda item: item.created_at, reverse=True)


def filter_by_range(records: list[MemoryRecord], start: datetime, end: datetime) -> list[MemoryRecord]:
    hits = []
    for record in records:
        ts = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
        if start <= ts < end:
            hits.append(record)
    return sorted(hits, key=lambda item: item.created_at, reverse=True)


def select_recent(records: list[MemoryRecord], n: int) -> list[MemoryRecord]:
    return sorted(records, key=lambda item: item.created_at, reverse=True)[:n]


def make_input(question: str, evidence: list[MemoryRecord], intent: str) -> str:
    payload = {
        "question": question,
        "retrieval": {
            "intent": intent,
            "hit_count": len(evidence),
            "evidence": [item.to_evidence() for item in evidence],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_support_terms(evidence: list[MemoryRecord]) -> list[str]:
    terms: list[str] = []
    for item in evidence:
        for term in [item.display_name, fmt_date(item.created_at), *item.objects]:
            if term and term not in terms:
                terms.append(term)
    return terms


def make_source_id(category: str, question: str, answer: str, evidence: list[MemoryRecord]) -> str:
    payload = json.dumps(
        {
            "category": category,
            "question": question,
            "answer": answer,
            "evidence": [item.scene_id for item in evidence],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"{category}_{digest}"


def make_case(category: str, question: str, answer: str, evidence: list[MemoryRecord], intent: str, meta: dict[str, Any]) -> dict[str, Any]:
    support_terms = build_support_terms(evidence)
    row = {
        "source_id": make_source_id(category, question, answer, evidence),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_input(question, evidence, intent)},
            {"role": "assistant", "content": answer},
        ],
        "category": category,
        "meta": {
            "hit_count": len(evidence),
            "supported_objects": meta.get("supported_objects", []),
            "unsupported_objects": meta.get("unsupported_objects", []),
            "support_terms": support_terms,
            "must_answer": len(evidence) > 0,
            "natural_output_required": True,
            "forbid_generic_positive": meta.get("forbid_generic_positive", len(evidence) > 0),
            "benchmark_group": meta.get("benchmark_group", category),
            "reference_answer": answer,
        },
    }
    return row


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["source_id"]] = row
    return list(deduped.values())


def sample_rows(rows: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) < target:
        raise ValueError(f"not enough rows: need {target}, got {len(rows)}")
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[:target]


def answer_recent(evidence: list[MemoryRecord], variant: int) -> str:
    if not evidence:
        return "暂无相关记录。"
    top = evidence[: min(3, len(evidence))]
    names = [item.display_name for item in top]
    descs = [compact_desc(item.description) for item in top]
    if variant % 4 == 0:
        return f"最近拍到的内容包括{'、'.join(names)}。"
    if variant % 4 == 1:
        snippets = [f"{fmt_date(item.created_at)}的{item.display_name}" for item in top]
        return f"最近的记录有{'；'.join(snippets)}。"
    if variant % 4 == 2:
        snippets = [f"{item.objects[0]}相关场景" for item in top]
        return f"最近拍到的主要内容有{'、'.join(snippets)}。"
    return f"最近两条较新的记录分别是{names[0]}和{names[-1]}。" if len(names) > 1 else f"最近拍到的是{names[0]}。"


def answer_time(evidence: list[MemoryRecord], label: str, variant: int) -> str:
    if not evidence:
        return "暂无相关记录。"
    top = evidence[: min(2, len(evidence))]
    if variant % 3 == 0:
        return f"{label}的记录包括{'、'.join(item.display_name for item in top)}。"
    if variant % 3 == 1:
        return f"{label}拍到过{top[0].objects[0]}，例如{compact_desc(top[0].description)}。"
    return f"{label}有相关记录，较新的内容是{compact_desc(top[0].description)}。"


def answer_must(keyword: str, evidence: list[MemoryRecord], variant: int) -> str:
    if not evidence:
        return "暂无相关记录。"
    first = evidence[0]
    if len(evidence) == 1:
        templates = [
            f"最近拍到过{keyword}，{compact_desc(first.description)}。",
            f"有{keyword}相关记录，{fmt_date(first.created_at)}的内容是{compact_desc(first.description)}。",
            f"最近和{keyword}有关的记录里，{compact_desc(first.description)}。",
            f"最近确实拍到过{keyword}，例如{first.display_name}这条记录。",
        ]
        return templates[variant % len(templates)]
    second = evidence[1]
    templates = [
        f"最近和{keyword}有关的内容包括{compact_desc(first.description)}；{compact_desc(second.description)}。",
        f"有{keyword}相关记录，较新的两条分别是{first.display_name}和{second.display_name}。",
        f"最近拍到过{keyword}，比如{fmt_date(first.created_at)}和{fmt_date(second.created_at)}都有相关记录。",
        f"最近的{keyword}相关内容不止一条，例如{compact_desc(first.description)}；{compact_desc(second.description)}。",
    ]
    return templates[variant % len(templates)]


def answer_partial(found: str, missing: str, evidence: list[MemoryRecord], variant: int) -> str:
    if not evidence:
        return "暂无相关记录。"
    first = evidence[0]
    templates = [
        f"有{found}相关记录，未见{missing}相关记录。",
        f"最近拍到过{found}，暂无{missing}相关记录。",
        f"记录里出现了{found}，没有看到{missing}。",
        f"目前只找到{found}相关内容，未见{missing}。",
        f"{fmt_date(first.created_at)}拍到过{found}，{missing}暂无相关记录。",
        f"有{found}的内容，例如{compact_desc(first.description)}；{missing}没有对应记录。",
    ]
    return templates[variant % len(templates)]


def answer_no_hit(keyword: str, variant: int) -> str:
    variants = [
        "暂无相关记录。",
        f"暂无{keyword}相关记录。",
        f"目前未见{keyword}相关记录。",
        f"现在没有查到{keyword}的相关记录。",
    ]
    return variants[variant % len(variants)]


def build_partial_candidates(records: list[MemoryRecord], supported_keywords: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = [
        "我最近拍过{found}和{missing}吗？",
        "最近记录里同时有{found}和{missing}吗？",
        "请直接告诉我最近有没有拍到{found}和{missing}。",
        "最近拍到的{found}和{missing}分别是什么？",
        "我最近看见过{found}，也看见过{missing}吗？",
        "最近和{found}、{missing}有关的记录都有吗？",
        "昨天看到{found}和{missing}了吗？",
        "上周的记录里有{found}和{missing}吗？",
        "最近拍过{found}，那{missing}呢？",
        "最近和{found}相关的内容有，{missing}也有吗？",
    ]
    missing_pool = [item for item in supported_keywords if item in ALT_MISSING] + NO_HIT_KEYWORDS
    for found in supported_keywords:
        hits = filter_by_keyword(records, found)
        if not hits:
            continue
        for missing in missing_pool:
            if missing == found:
                continue
            for take in (1, 2):
                evidence = hits[:take]
                if not evidence:
                    continue
                for idx, template in enumerate(templates):
                    question = template.format(found=found, missing=missing)
                    answer = answer_partial(found, missing, evidence, idx + take)
                    rows.append(
                        make_case(
                            "partial_coverage",
                            question,
                            answer,
                            evidence,
                            "partial_coverage",
                            {
                                "supported_objects": [found],
                                "unsupported_objects": [missing],
                                "benchmark_group": "partial_coverage",
                                "forbid_generic_positive": True,
                            },
                        )
                    )
    return dedupe_rows(rows)


def build_must_answer_candidates(records: list[MemoryRecord], supported_keywords: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = [
        "我最近拍过{keyword}吗？",
        "最近和{keyword}有关的记录有哪些？",
        "最近有{keyword}的拍摄记录吗？",
        "最近拍到的{keyword}是什么样的？",
        "请直接告诉我最近有没有拍到{keyword}。",
        "最近记录里有哪些{keyword}相关内容？",
        "最近有没有看到{keyword}？",
        "最近关于{keyword}的内容是什么？",
        "最近拍到过哪些{keyword}相关场景？",
        "最近和{keyword}有关的画面有哪些？",
    ]
    for keyword in supported_keywords:
        hits = filter_by_keyword(records, keyword)
        if not hits:
            continue
        for take in (1, 2, 3):
            evidence = hits[:take]
            if not evidence:
                continue
            for idx, template in enumerate(templates):
                question = template.format(keyword=keyword)
                answer = answer_must(keyword, evidence, idx + take)
                rows.append(
                    make_case(
                        "must_answer",
                        question,
                        answer,
                        evidence,
                        "object_lookup",
                        {
                            "supported_objects": [keyword],
                            "unsupported_objects": [],
                            "benchmark_group": "must_answer",
                            "forbid_generic_positive": True,
                        },
                    )
                )
    return dedupe_rows(rows)


def build_recent_time_candidates(records: list[MemoryRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recent_templates = [
        "我最近拍了什么？",
        "最近都拍了哪些内容？",
        "最近的拍摄记录是什么？",
        "最近记录里有什么？",
        "最近新拍了哪些场景？",
        "最近拍到的内容是什么？",
        "请直接告诉我最近拍了什么。",
        "最近的几条记录分别是什么？",
        "按时间倒序说说我最近拍了什么。",
        "最近前两三条记录是什么？",
        "最近拍到的主要场景有哪些？",
        "最近更新的记录里有什么内容？",
        "最近新增了哪些拍摄内容？",
        "最近这几次拍摄分别是什么？",
    ]
    for take in (1, 2, 3, 4, 5):
        evidence = select_recent(records, take)
        for idx, template in enumerate(recent_templates):
            for style in range(4):
                rows.append(
                    make_case(
                        "recent_list",
                        template,
                        answer_recent(evidence, idx + take + style),
                        evidence,
                        "recent_capture",
                        {
                            "supported_objects": sorted({obj for item in evidence for obj in item.objects}),
                            "unsupported_objects": [],
                            "benchmark_group": "recent_hit",
                        },
                    )
                )

    yesterday = filter_by_range(records, datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc), datetime(2026, 3, 21, 0, 0, 0, tzinfo=timezone.utc))
    last_week = filter_by_range(records, NOW - timedelta(days=7), NOW + timedelta(seconds=1))
    time_templates = [
        ("我昨天拍了什么？", yesterday, "昨天"),
        ("昨天的记录是什么？", yesterday, "昨天"),
        ("昨天拍到哪些内容？", yesterday, "昨天"),
        ("我上周拍过什么？", last_week[:3], "上周"),
        ("上周有哪几条记录？", last_week[:3], "上周"),
        ("上周的拍摄内容是什么？", last_week[:3], "上周"),
    ]
    for idx, (question, evidence, label) in enumerate(time_templates):
        for style in range(4):
            rows.append(
                make_case(
                    "time_qa",
                    question,
                    answer_time(evidence, label, idx + style),
                    evidence,
                    f"{label}_capture",
                    {
                        "supported_objects": sorted({obj for item in evidence for obj in item.objects}),
                        "unsupported_objects": [],
                        "benchmark_group": "recent_hit",
                    },
                )
            )
    return dedupe_rows(rows)


def build_no_hit_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = [
        "我最近拍过{keyword}吗？",
        "上周有{keyword}的记录吗？",
        "最近和{keyword}有关的内容有哪些？",
        "最近拍到过{keyword}吗？",
        "请直接告诉我最近有没有拍到{keyword}。",
        "昨天见过{keyword}吗？",
        "最近有没有{keyword}相关记录？",
        "最近记录里有{keyword}吗？",
        "最近拍到的场景里出现过{keyword}吗？",
        "我最近看见过{keyword}吗？",
    ]
    for keyword in NO_HIT_KEYWORDS:
        for idx, template in enumerate(templates):
            for style in range(4):
                rows.append(
                    make_case(
                        "no_hit",
                        template.format(keyword=keyword),
                        answer_no_hit(keyword, idx + style),
                        [],
                        "no_hit",
                        {
                            "supported_objects": [],
                            "unsupported_objects": [keyword],
                            "benchmark_group": "no_hit",
                            "forbid_generic_positive": False,
                        },
                    )
                )
    return dedupe_rows(rows)


def build_stability_candidates(records: list[MemoryRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recent_evidence = select_recent(records, 2)
    touch_evidence = filter_by_keyword(records, "触控笔")[:2]
    keyboard_evidence = filter_by_keyword(records, "键盘")[:2]
    partial_evidence = filter_by_keyword(records, "触控笔")[:1]
    cases = [
        ("只用自然语言回答我最近拍了什么。", recent_evidence, "output_stability", "最近拍到的内容包括触控笔桌面采集 01 和书房场景。", ["触控笔", "写字台"], []),
        ("直接回答最近拍了什么，不要输出 JSON。", recent_evidence, "output_stability", "最近拍到的内容包括触控笔桌面采集 01 和书房场景。", ["触控笔", "写字台"], []),
        ("简短告诉我最近和键盘有关的记录。", keyboard_evidence, "output_stability", "最近拍到过键盘，相关记录里有红色背光键盘的桌面场景。", ["键盘"], []),
        ("不要复述问题，直接说最近有没有拍到触控笔。", touch_evidence[:1], "output_stability", "最近拍到过触控笔，例如桌面中央的触控笔场景。", ["触控笔"], []),
        ("只用一句自然语言回答我最近拍过触控笔和冰箱吗？", partial_evidence, "output_stability", "有触控笔相关记录，未见冰箱相关记录。", ["触控笔"], ["冰箱"]),
        ("不要输出 JSON，回答我最近拍过触控笔和冰箱吗？", partial_evidence, "output_stability", "最近拍到过触控笔，暂无冰箱相关记录。", ["触控笔"], ["冰箱"]),
        ("直接回答最近有没有拍到办公椅，不要写成 JSON。", filter_by_keyword(records, "办公椅")[:1], "output_stability", "最近拍到过办公椅，例如简约办公室那条记录。", ["办公椅"], []),
        ("自然语言回答最近有没有拍到沙发。", filter_by_keyword(records, "沙发")[:1], "output_stability", "最近拍到过沙发，相关记录是宽敞客厅场景。", ["沙发"], []),
        ("只用自然语言回答我最近拍过键盘和猫吗？", keyboard_evidence[:1], "output_stability", "有键盘相关记录，未见猫相关记录。", ["键盘"], ["猫"]),
        ("不要输出键值对，直接说最近有没有拍到书架。", filter_by_keyword(records, "书架")[:1], "output_stability", "最近拍到过书架，例如明亮书房那条记录。", ["书架"], []),
    ]
    for idx, (question, evidence, intent, answer, supported, unsupported) in enumerate(cases):
        rows.append(
            make_case(
                "stability",
                question,
                answer,
                evidence,
                intent,
                {
                    "supported_objects": supported,
                    "unsupported_objects": unsupported,
                    "benchmark_group": "stability",
                    "forbid_generic_positive": True,
                },
            )
        )
        # Add lightweight paraphrases to strengthen anti-JSON behavior.
        rows.append(
            make_case(
                "stability",
                question.replace("最近", "刚才最接近现在的记录里" if "最近" in question else question),
                answer,
                evidence,
                intent,
                {
                    "supported_objects": supported,
                    "unsupported_objects": unsupported,
                    "benchmark_group": "stability",
                    "forbid_generic_positive": True,
                },
            )
        )
    return dedupe_rows(rows)


def reserve_benchmark_cases(pool: dict[str, list[dict[str, Any]]], rng: random.Random) -> tuple[list[dict[str, Any]], set[str]]:
    benchmark_rows: list[dict[str, Any]] = []
    reserved_ids: set[str] = set()
    mapping = {
        "recent_hit": [row for row in pool["recent_time"] if row["meta"]["benchmark_group"] == "recent_hit" and row["meta"]["hit_count"] > 0],
        "no_hit": [row for row in pool["nohit_stability"] if row["meta"]["benchmark_group"] == "no_hit"],
        "partial_coverage": [row for row in pool["partial_coverage"] if row["meta"]["benchmark_group"] == "partial_coverage"],
        "must_answer": [row for row in pool["must_answer"] if row["meta"]["benchmark_group"] == "must_answer"],
        "stability": [row for row in pool["nohit_stability"] if row["meta"]["benchmark_group"] == "stability"],
    }
    for group, size in BENCHMARK_GROUP_SIZES.items():
        candidates = mapping[group][:]
        rng.shuffle(candidates)
        selected = candidates[:size]
        for idx, row in enumerate(selected, start=1):
            reserved_ids.add(row["source_id"])
            benchmark_rows.append(
                {
                    "case_id": f"{group}_{idx:03d}",
                    "group": group,
                    "messages": row["messages"][:2],
                    "reference_answer": row["messages"][2]["content"],
                    "metadata": row["meta"],
                    "source_id": row["source_id"],
                }
            )
    return benchmark_rows, reserved_ids


def split_rows(rows: list[dict[str, Any]], val_ratio: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val_size = max(80, int(len(rows) * val_ratio))
    return rows[val_size:], rows[:val_size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_frame_records() + build_manual_records()
    supported_keywords = sorted({kw for kw in (FOCUS_SUPPORTED + all_supported_objects(records)) if filter_by_keyword(records, kw)})

    pool = {
        "partial_coverage": build_partial_candidates(records, supported_keywords),
        "must_answer": build_must_answer_candidates(records, supported_keywords),
        "recent_time": build_recent_time_candidates(records),
        "nohit_stability": build_no_hit_candidates() + build_stability_candidates(records),
    }
    for key, rows in pool.items():
        pool[key] = dedupe_rows(rows)

    benchmark_rows, reserved_ids = reserve_benchmark_cases(pool, rng)

    selected_rows: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for bucket, quota in QUOTAS.items():
        candidates = [row for row in pool[bucket] if row["source_id"] not in reserved_ids]
        chosen = sample_rows(candidates, quota, rng)
        selected_rows.extend(chosen)
        group_counts[bucket] = len(chosen)

    rng.shuffle(selected_rows)
    train_rows, val_rows = split_rows(selected_rows)
    manifest = {
        "generated_at": iso(datetime.now(timezone.utc)),
        "record_count": len(records),
        "example_count": len(selected_rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "benchmark_count": len(benchmark_rows),
        "bucket_counts": group_counts,
        "sources": [
            "ai_engine/demo/rag/data/output_analyzed/frame_*.json",
            "ai_engine/3dgs/test_search_data.py",
        ],
        "categories": sorted({row["category"] for row in selected_rows}),
        "benchmark_groups": BENCHMARK_GROUP_SIZES,
    }

    write_jsonl(OUTPUT_DIR / "braindance_qwen3_sft_train.jsonl", train_rows)
    write_jsonl(OUTPUT_DIR / "braindance_qwen3_sft_val.jsonl", val_rows)
    write_jsonl(OUTPUT_DIR / "braindance_qwen3_benchmark.jsonl", benchmark_rows)
    (OUTPUT_DIR / "braindance_qwen3_sft_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
