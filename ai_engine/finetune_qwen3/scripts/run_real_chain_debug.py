#!/usr/bin/env python3
"""Run BrainDance local QA debug cases against the real retrieval chain.

This script keeps the retrieval side close to the current BrainDance stack:
- query intent parsing via DashScope
- vector retrieval via Supabase RPC `match_memory_poses`
- recent/time queries handled in code by created_at ordering

The generation side is switchable:
- off
- base
- lora_round3
- compare
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - retrieval-only paths can run without torch.
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "real_chain_debug_cases.jsonl"
DEFAULT_SUMMARY_FILE = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "real_chain_debug_summary.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "outputs" / "qwen3_1p7b_lora_sft_round3"
DEFAULT_SUPABASE_URL = "https://supabase.tianxingleo.top"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
HTTP_RETRY_COUNT = 3
HTTP_RETRY_BACKOFF_SEC = 1.5

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

VALID_MODES = {"off", "base", "lora_round3", "compare"}
VALID_QUESTION_TYPES = {"recent_capture", "time_qa", "object_lookup", "partial_coverage", "other"}
GENERIC_SEARCH_TEXTS = {"什么", "哪些", "哪几个", "哪几条", "内容", "记录", "场景"}
RECENT_KEYWORDS = ("最近", "刚才", "最新", "这几天")
LAST_SEVEN_DAYS_KEYWORDS = ("近一周", "最近一周", "过去一周")
YESTERDAY_KEYWORDS = ("昨天",)
LAST_WEEK_KEYWORDS = ("上周",)
THIS_MONTH_KEYWORDS = ("这个月", "本月")
LAST_MONTH_KEYWORDS = ("上个月",)
LAST_YEAR_KEYWORDS = ("去年",)
PARTIAL_CONNECTORS = ("和", "以及", "还有", "、", "分别", "那")
NEGATIVE_MARKERS = ("暂无", "未见", "没有", "未拍到", "没看到", "无", "不存在")
LIST_SEPARATORS = ("、", "，", ",", "；", ";")
GENERIC_FOCUS_TERMS = {"办公桌", "桌面", "书桌", "桌子", "房间", "屋子"}
GENERIC_MODEL_TERMS = {"模型", "3d模型", "三维模型", "高斯模型", "3dgs", "高斯", "场景模型"}
MODEL_INVENTORY_HINTS = ("生成", "建模", "做了", "有没有", "有哪些", "最近", "刚刚")
GREETING_KEYWORDS = ("你好", "您好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好")
ENGLISH_MODEL_HINTS = ("model", "models")
ENGLISH_INVENTORY_HINTS = (
    "what model i have",
    "what models i have",
    "what model do i have",
    "what models do i have",
    "recent model",
    "recent models",
)
IDENTITY_PATTERNS = (
    "你是谁",
    "你是干什么的",
    "你有什么用",
    "你有什么用处",
    "你有什么作用",
    "你能做什么",
    "你会做什么",
    "你能帮我做什么",
    "你可以帮我做什么",
    "你可以做什么",
    "你有什么功能",
    "你从哪里来",
    "你会说英文吗",
    "你会英语吗",
    "你会英文吗",
)
SYSTEM_PROMPT_PATTERNS = ("systemprompt", "system提示", "系统提示", "prompt")
PRODUCT_IDENTITY_PATTERNS = ("braindance是什么", "braindance是啥", "braindance有什么用")
GENERIC_OBJECT_SUFFIXES = (
    "模型",
    "场景",
    "内容",
    "记录",
    "画面",
    "东西",
    "物品",
    "照片",
    "图片",
    "相关",
    "相关内容",
    "相关记录",
    "相关画面",
)
LOOKUP_SPLIT_SEPARATORS = ("、", "，", ",", "/", "|", "和", "及", "与")
GENERIC_LOOKUP_PREFIX_PATTERNS = (
    r"^(?:帮我)?找(?:一下|一找|找)?",
    r"^帮我查(?:一下)?",
    r"^查(?:一下)?",
    r"^看(?:一下)?",
    r"^有没有",
    r"^有没",
    r"^最近有没有",
    r"^最近有(?:没有)?",
    r"^最近拍(?:到|过)?(?:了)?",
    r"^最近记录里有哪些",
    r"^最近记录里有(?:没有)?",
    r"^最近关于",
    r"^关于",
)
GENERIC_LOOKUP_TRAILING_PATTERNS = (
    r"(?:是|有)?什么$",
    r"吗$",
    r"呢$",
    r"呀$",
    r"啊$",
)
GENERIC_LOOKUP_NOISE_TOKENS = {
    "相关的",
    "有关的",
    "一下",
    "一找",
    "一下子",
}
GENERIC_OBJECT_PREFIXES = (
    "关于",
    "有关",
    "相关",
    "最近",
    "最近有",
    "最近有没有",
    "最近拍到过",
    "最近拍过",
    "最近记录里",
)
SEMANTIC_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "理工": ("算法", "算法导论", "数学", "高等数学", "教材", "词典", "电脑", "笔记本电脑", "显示器", "白板"),
    "理工科": ("算法", "算法导论", "数学", "高等数学", "教材", "词典", "电脑", "笔记本电脑", "显示器", "白板"),
    "理科生": ("算法", "算法导论", "数学", "高等数学", "教材", "词典", "电脑", "笔记本电脑", "显示器", "白板"),
    "计算机科学": ("算法", "算法导论", "电脑", "笔记本电脑", "显示器", "机械键盘", "白板", "办公桌"),
    "计算机": ("电脑", "笔记本电脑", "显示器", "机械键盘", "办公桌", "白板"),
    "学习相关": ("教材", "词典", "算法导论", "高等数学", "白板", "笔记本电脑", "显示器"),
    "学习氛围": ("教材", "词典", "地球仪", "白板", "办公桌", "笔记本电脑"),
    "学习": ("教材", "词典", "地球仪", "白板", "办公桌", "笔记本电脑"),
    "学术": ("教材", "词典", "算法导论", "高等数学", "白板", "办公桌"),
}
OBJECT_LOOKUP_PHRASE_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "洛天依展台": ("洛天依",),
    "洛天依主题展台": ("洛天依",),
    "商业展台装置": ("洛天依", "展台"),
    "学习类摆件": ("学习相关", "地球仪", "手办", "书架", "词典"),
    "学习摆件": ("学习相关", "地球仪", "手办"),
    "动漫收藏角": ("手办", "书架", "地球仪"),
    "桌面设备": ("显示器", "笔记本电脑", "键盘", "办公桌"),
    "白色办公桌上的摆件": ("办公桌", "手办"),
    "办公桌摆件": ("办公桌", "手办"),
    "书架角落的模型": ("书架", "地球仪", "手办"),
    "词典和手办的收纳角": ("词典", "手办", "书架"),
}
ANCHOR_OBJECT_HINTS: tuple[str, ...] = (
    "洛天依",
    "地球仪",
    "显示器",
    "笔记本电脑",
    "办公桌",
    "手办",
    "书架",
    "词典",
    "键盘",
)
MUST_ANSWER_GROUPS = {"must_answer", "multi_hit_must_answer"}
RETRIEVAL_ROUTES = {
    "vector_only",
    "vector_plus_filter",
    "lexical_fallback",
    "merged_vector_lexical",
    "inventory_special_case",
    "recent_list",
    "non_retrieval",
}
ANSWER_ROUTES = {
    "fixed_response",
    "recent_answer_formatter",
    "must_answer_focus_formatter",
    "inventory_formatter",
    "semantic_summary_formatter",
    "lora_generation",
}


@dataclass(frozen=True)
class DebugCase:
    case_id: str
    group: str
    question: str
    expected_issue_tag: str = ""
    triage_label: str = ""


DEFAULT_CASES = [
    DebugCase(case_id="default_recent_001", group="recent_hit", question="我最近拍了什么？"),
    DebugCase(case_id="default_no_hit_001", group="no_hit", question="我最近拍过钢琴吗？"),
    DebugCase(case_id="default_partial_001", group="partial_coverage", question="我最近拍过笔记本电脑和钢琴吗？"),
    DebugCase(case_id="default_must_001", group="must_answer", question="最近拍到过什么笔记本电脑相关画面？"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BrainDance real-chain debug cases with local QA modes")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="compare")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--summary_file", default=str(DEFAULT_SUMMARY_FILE))
    parser.add_argument("--cases_file", default="")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    parser.add_argument("--overwrite_output", action="store_true")
    return parser.parse_args()


def http_request(method: str, url: str, *, timeout: int, **kwargs: Any) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRY_COUNT + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == HTTP_RETRY_COUNT:
                raise
            time.sleep(HTTP_RETRY_BACKOFF_SEC * attempt)
    if last_error:
        raise last_error
    raise RuntimeError(f"HTTP request failed without exception: {method} {url}")


def load_text_from_candidates(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def extract_dashscope_key() -> str:
    env_key = os.getenv("DASHSCOPE_API_KEY")
    if env_key:
        return env_key
    text = load_text_from_candidates([
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "supabase" / "functions" / "search-models" / ".env.local",
    ])
    match = re.search(r"DASHSCOPE_API_KEY=([^\n\r]+)", text)
    if match:
        return match.group(1).strip()
    raise RuntimeError("未找到 DASHSCOPE_API_KEY，请先配置环境变量或 .env 文件")


def extract_supabase_config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if url and key:
        return url.rstrip("/"), key
    # Reuse existing repository defaults before introducing new config.
    return DEFAULT_SUPABASE_URL, DEFAULT_SUPABASE_PUBLISHABLE_KEY


def load_cases(cases_file: str) -> list[DebugCase]:
    if not cases_file:
        return list(DEFAULT_CASES)
    path = Path(cases_file)
    if not path.exists():
        raise FileNotFoundError(f"未找到 cases_file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("cases_file 必须是 JSON 数组或 JSONL")
        rows = payload

    cases: list[DebugCase] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("cases_file 中的每条记录都必须是对象")
        case_id = str(row.get("case_id") or f"case_{index:03d}").strip()
        group = str(row.get("group") or "").strip()
        question = str(row.get("question") or "").strip()
        expected_issue_tag = str(row.get("expected_issue_tag") or "").strip()
        triage_label = str(row.get("triage_label") or "").strip()
        if not group or not question:
            raise ValueError("cases_file 中每条记录都必须包含非空的 group 和 question")
        cases.append(
            DebugCase(
                case_id=case_id,
                group=group,
                question=question,
                expected_issue_tag=expected_issue_tag,
                triage_label=triage_label,
            )
        )
    return cases


def is_must_answer_group(group: str) -> bool:
    return group in MUST_ANSWER_GROUPS


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_iso8601(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        value = parse_datetime(date_str)
    except ValueError:
        return None
    return value.isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty datetime string")

    normalized = text.replace("Z", "+00:00")
    if re.fullmatch(r".*[+-]\d{4}", normalized):
        normalized = normalized[:-5] + normalized[-5:-2] + ":" + normalized[-2:]

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        match = re.match(
            r"^(?P<head>.+?)(?:\.(?P<fraction>\d+))?(?P<tz>Z|[+-]\d{2}:?\d{2})?$",
            text,
        )
        if not match:
            raise

        head = match.group("head")
        fraction = match.group("fraction") or ""
        tz = match.group("tz") or "+00:00"
        if fraction:
            fraction = fraction[:6].ljust(6, "0")
        if tz == "Z":
            tz = "+00:00"
        elif re.fullmatch(r"[+-]\d{4}", tz):
            tz = tz[:-2] + ":" + tz[-2:]
        normalized = head + (f".{fraction}" if fraction else "") + tz
        parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_range_from_question(question: str) -> tuple[str | None, str | None]:
    current = now_utc()
    normalized = re.sub(r"\s+", "", question or "")
    if any(word in normalized for word in LAST_SEVEN_DAYS_KEYWORDS):
        start = (current - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = current.replace(microsecond=0)
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    if any(word in question for word in YESTERDAY_KEYWORDS):
        start = (current - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    if "上上周到上周" in normalized or "上上周至上周" in normalized:
        weekday = current.weekday()
        this_week_start = (current - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        start = this_week_start - timedelta(days=14)
        end = this_week_start - timedelta(seconds=1)
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    if any(word in question for word in LAST_WEEK_KEYWORDS):
        weekday = current.weekday()
        this_week_start = (current - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        start = this_week_start - timedelta(days=7)
        end = this_week_start - timedelta(seconds=1)
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    if any(word in normalized for word in THIS_MONTH_KEYWORDS):
        start = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = current.replace(microsecond=0)
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    if any(word in normalized for word in LAST_MONTH_KEYWORDS):
        first_day_this_month = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = first_day_this_month - timedelta(seconds=1)
        start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if "前十五天" in normalized:
            half_end_day = min(15, end.day)
            half_end = start.replace(day=half_end_day, hour=23, minute=59, second=59, microsecond=0)
            return start.isoformat().replace("+00:00", "Z"), half_end.isoformat().replace("+00:00", "Z")
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    if any(word in normalized for word in LAST_YEAR_KEYWORDS):
        start = current.replace(year=current.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = current.replace(year=current.year - 1, month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
        return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
    return None, None


def trim_search_text(text: str) -> str:
    value = text.strip()
    if value in GENERIC_SEARCH_TEXTS:
        return ""
    if value.endswith(("吗", "呢", "呀", "啊", "吧")) and len(value) > 1:
        value = value[:-1]
    return value.strip()


def parse_query_intent(
    dashscope_key: str,
    base_url: str,
    chat_model: str,
    question: str,
) -> dict[str, Any]:
    today = now_utc().date().isoformat()
    system_prompt = f"""你是 BrainDance 的查询解析器。当前日期是 {today}。
把用户问题解析成 JSON，只返回 JSON，不要多余文字。

字段要求：
- question_type: recent_capture | time_qa | object_lookup | partial_coverage | other
- search_text: 用于语义检索的核心文本；如果是“我最近拍了什么”这类纯时间问题，填空字符串
- target_objects: 用户明确提到的目标对象数组，没有就返回 []
- start_time: ISO8601 UTC 时间，无则 null
- end_time: ISO8601 UTC 时间，无则 null

示例1:
用户问题: 我最近拍了什么？
输出: {{"question_type":"recent_capture","search_text":"","target_objects":[],"start_time":null,"end_time":null}}

示例2:
用户问题: 我最近拍过笔记本电脑和钢琴吗？
输出: {{"question_type":"partial_coverage","search_text":"笔记本电脑 钢琴","target_objects":["笔记本电脑","钢琴"],"start_time":null,"end_time":null}}

示例3:
用户问题: 上周拍到过什么办公场景？
输出: {{"question_type":"time_qa","search_text":"办公场景","target_objects":[],"start_time":"2026-03-09T00:00:00Z","end_time":"2026-03-15T23:59:59Z"}}

示例4:
用户问题: 我最近拍过钢琴吗？
输出: {{"question_type":"object_lookup","search_text":"钢琴","target_objects":["钢琴"],"start_time":null,"end_time":null}}
"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_object"},
    }
    response = http_request(
        "POST",
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {dashscope_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    question_type = str(parsed.get("question_type") or "other").strip()
    if question_type not in VALID_QUESTION_TYPES:
        question_type = "other"
    target_objects = []
    for item in parsed.get("target_objects") or []:
        value = str(item).strip()
        if value and value not in target_objects:
            target_objects.append(value)
    search_text = trim_search_text(str(parsed.get("search_text") or ""))
    if not search_text and target_objects:
        search_text = " ".join(target_objects)
    start_time = normalize_iso8601(parsed.get("start_time"))
    end_time = normalize_iso8601(parsed.get("end_time"))
    inferred_start, inferred_end = iso_range_from_question(question)
    if not start_time:
        start_time = inferred_start
    if not end_time:
        end_time = inferred_end
    if question_type == "other":
        if any(token in question for token in RECENT_KEYWORDS):
            question_type = "recent_capture"
        elif any(
            token in question
            for token in YESTERDAY_KEYWORDS
            + LAST_WEEK_KEYWORDS
            + LAST_SEVEN_DAYS_KEYWORDS
            + THIS_MONTH_KEYWORDS
            + LAST_MONTH_KEYWORDS
            + LAST_YEAR_KEYWORDS
        ):
            question_type = "time_qa"
        elif target_objects and len(target_objects) >= 2 and any(token in question for token in PARTIAL_CONNECTORS):
            question_type = "partial_coverage"
        else:
            question_type = "object_lookup"
    return {
        "question_type": question_type,
        "search_text": search_text,
        "target_objects": target_objects,
        "start_time": start_time,
        "end_time": end_time,
    }


def create_embedding(
    dashscope_key: str,
    base_url: str,
    embedding_model: str,
    text: str,
) -> list[float]:
    response = http_request(
        "POST",
        f"{base_url.rstrip('/')}/embeddings",
        headers={
            "Authorization": f"Bearer {dashscope_key}",
            "Content-Type": "application/json",
        },
        json={"model": embedding_model, "input": [text]},
        timeout=30,
    )
    data = response.json()["data"][0]["embedding"]
    return [float(value) for value in data]


def supabase_headers(supabase_key: str) -> dict[str, str]:
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }


def rest_select_model_assets(
    supabase_url: str,
    supabase_key: str,
    *,
    limit: int,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict[str, Any]]:
    params = {
        "select": "id,scene_id,description,objects,tags,created_at,meta_info",
        "order": "created_at.desc",
        "limit": str(max(limit, 50) if start_time or end_time else limit),
    }
    response = http_request(
        "GET",
        f"{supabase_url.rstrip('/')}/rest/v1/model_assets",
        headers=supabase_headers(supabase_key),
        params=params,
        timeout=30,
    )
    rows = response.json()
    if start_time and end_time:
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
        filtered_rows = []
        for row in rows:
            try:
                created_at = parse_datetime(str(row.get("created_at") or ""))
            except ValueError:
                continue
            if start_dt <= created_at <= end_dt:
                filtered_rows.append(row)
        rows = filtered_rows
    return rows


def rest_rpc_match_memory_poses(
    supabase_url: str,
    supabase_key: str,
    *,
    query_embedding: list[float],
    match_threshold: float,
    match_count: int,
    start_time: str | None,
    end_time: str | None,
) -> list[dict[str, Any]]:
    response = http_request(
        "POST",
        f"{supabase_url.rstrip('/')}/rest/v1/rpc/match_memory_poses",
        headers=supabase_headers(supabase_key),
        json={
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_start": start_time,
            "filter_end": end_time,
        },
        timeout=30,
    )
    return response.json()


def safe_rpc_match_memory_poses(
    supabase_url: str,
    supabase_key: str,
    *,
    query_embedding: list[float],
    match_threshold: float,
    match_count: int,
    start_time: str | None,
    end_time: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        rows = rest_rpc_match_memory_poses(
            supabase_url,
            supabase_key,
            query_embedding=query_embedding,
            match_threshold=match_threshold,
            match_count=match_count,
            start_time=start_time,
            end_time=end_time,
        )
        return rows, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def rest_fetch_model_assets_by_ids(
    supabase_url: str,
    supabase_key: str,
    ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    id_filter = "in.(" + ",".join(ids) + ")"
    response = http_request(
        "GET",
        f"{supabase_url.rstrip('/')}/rest/v1/model_assets",
        headers=supabase_headers(supabase_key),
        params={
            "select": "id,scene_id,description,objects,tags,created_at,meta_info",
            "id": id_filter,
        },
        timeout=30,
    )
    rows = response.json()
    return {str(row["id"]): row for row in rows}


def infer_display_name(row: dict[str, Any]) -> str:
    matched_frames = row.get("matched_frames") or []
    if matched_frames:
        image_name = str(matched_frames[0].get("image_name") or "").strip()
        if image_name:
            return image_name
    meta_info = row.get("meta_info") or {}
    if isinstance(meta_info, dict):
        display_name = str(meta_info.get("display_name") or "").strip()
        if display_name:
            return display_name
    return str(row.get("scene_id") or "").strip()


def row_supports_target(row: dict[str, Any], target: str) -> bool:
    target = target.strip()
    if not target:
        return False
    haystacks: list[str] = []
    haystacks.append(str(row.get("scene_id") or ""))
    haystacks.append(str(row.get("description") or ""))
    for key in ("objects", "tags"):
        for item in row.get(key) or []:
            haystacks.append(str(item))
    return any(target in item for item in haystacks if item)


def expand_semantic_terms(term: str) -> list[str]:
    expanded: list[str] = []
    for key, values in SEMANTIC_QUERY_EXPANSIONS.items():
        if key in term:
            expanded.extend(values)
    return expanded


def infer_semantic_terms_from_question(question: str) -> list[str]:
    matched = [key for key in SEMANTIC_QUERY_EXPANSIONS if key in (question or "")]
    return dedupe_preserve_order(matched)


def collect_semantic_lookup_terms(lookup_terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in lookup_terms:
        expanded.extend(expand_semantic_terms(term))
    return dedupe_preserve_order(expanded)


def is_anchor_object_term(term: str) -> bool:
    normalized = canonicalize_lookup_term(term) or str(term).strip()
    if not normalized:
        return False
    return any(contains_term(normalized, anchor) or contains_term(anchor, normalized) for anchor in ANCHOR_OBJECT_HINTS)


def is_abstract_semantic_query(target_objects: list[str], lookup_terms: list[str]) -> bool:
    semantic_terms = collect_semantic_lookup_terms(lookup_terms)
    if not semantic_terms:
        return False
    concrete_targets = [term for term in target_objects if is_anchor_object_term(term)]
    return not concrete_targets


def split_lookup_fragments(text: str) -> list[str]:
    if not text:
        return []
    fragments = [text]
    for separator in LOOKUP_SPLIT_SEPARATORS + (" ",):
        next_fragments: list[str] = []
        for fragment in fragments:
            if separator in fragment:
                next_fragments.extend(part for part in fragment.split(separator))
            else:
                next_fragments.append(fragment)
        fragments = next_fragments
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def clean_lookup_fragment(text: str) -> str:
    cleaned = re.sub(r"[“”\"'（）()【】\[\]]", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    for pattern in GENERIC_LOOKUP_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
    cleaned = re.sub(r"^(?:有没有|有没|最近有没有|最近有|关于)", "", cleaned).strip()
    for pattern in GENERIC_LOOKUP_TRAILING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
    cleaned = re.sub(r"^(?:一)?(?:个|条|段)", "", cleaned).strip()
    if cleaned in GENERIC_LOOKUP_NOISE_TOKENS:
        return ""

    stripped = cleaned
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        for suffix in GENERIC_OBJECT_SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) > len(suffix):
                stripped = stripped[: -len(suffix)].strip()
                changed = True
                break

    stripped = re.sub(r"^(?:相关|有关)", "", stripped).strip()
    stripped = re.sub(r"(?:相关|有关)$", "", stripped).strip()
    return stripped


def canonicalize_lookup_term(text: str) -> str:
    value = clean_lookup_fragment(text)
    if not value:
        return ""
    value = re.sub(r"(?:相关的|有关的)$", "", value).strip()
    changed = True
    while changed and len(value) >= 2:
        changed = False
        for prefix in GENERIC_OBJECT_PREFIXES:
            if value.startswith(prefix) and len(value) > len(prefix):
                value = value[len(prefix):].strip()
                changed = True
        for suffix in GENERIC_OBJECT_SUFFIXES:
            if value.endswith(suffix) and len(value) > len(suffix):
                value = value[: -len(suffix)].strip()
                changed = True
    return value.strip()


def expand_lookup_aliases(text: str) -> list[str]:
    base = canonicalize_lookup_term(text)
    if not base:
        return []

    expanded: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = canonicalize_lookup_term(value)
        if not token or token == base or token in seen:
            return
        seen.add(token)
        expanded.append(token)

    for phrase, aliases in OBJECT_LOOKUP_PHRASE_EXPANSIONS.items():
        if phrase in base:
            for alias in aliases:
                add(alias)

    for hint in ANCHOR_OBJECT_HINTS:
        if hint in base:
            add(hint)

    if "展台" in base and "洛天依" in base:
        add("洛天依")
    if "摆件" in base and "办公桌" in base:
        add("办公桌")
        add("手办")
    if "摆件" in base and "学习" in base:
        for alias in ("学习相关", "地球仪", "手办", "书架"):
            add(alias)
    if "模型" in base and "书架" in base:
        for alias in ("书架", "地球仪", "手办"):
            add(alias)

    return expanded


def is_generic_lookup_token(text: str) -> bool:
    token = canonicalize_lookup_term(text)
    if not token:
        return True
    return token in GENERIC_SEARCH_TEXTS or token in GENERIC_OBJECT_SUFFIXES or len(token) < 2


def split_target_objects(target_objects: list[str], search_text: str = "") -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = canonicalize_lookup_term(value)
        if is_generic_lookup_token(token) or token in seen:
            return
        seen.add(token)
        normalized.append(token)

    inputs = [item for item in target_objects if str(item).strip()]
    if search_text:
        inputs.append(search_text)

    for raw in inputs:
        text = str(raw).strip()
        if not text:
            continue
        if not any(separator in text for separator in LOOKUP_SPLIT_SEPARATORS):
            add(text)
            for alias in expand_lookup_aliases(text):
                add(alias)
        for fragment in split_lookup_fragments(text):
            add(fragment)
            cleaned = clean_lookup_fragment(fragment)
            if cleaned and cleaned != fragment:
                add(cleaned)
            for alias in expand_lookup_aliases(fragment):
                add(alias)

    return normalized


def normalize_lookup_terms(*terms: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = value.strip()
        if len(token) < 2 or token in seen:
            return
        seen.add(token)
        normalized.append(token)

    for term in terms:
        base = term.strip()
        if not base:
            continue
        add(base)
        cleaned = clean_lookup_fragment(base)
        add(cleaned)
        add(canonicalize_lookup_term(base))
        add(canonicalize_lookup_term(cleaned))

        for fragment in split_lookup_fragments(base):
            add(fragment)
            add(clean_lookup_fragment(fragment))
            add(canonicalize_lookup_term(fragment))

        for fragment in split_lookup_fragments(cleaned):
            add(fragment)
            add(clean_lookup_fragment(fragment))
            add(canonicalize_lookup_term(fragment))

        for alias in expand_lookup_aliases(base):
            add(alias)
        for extra in expand_semantic_terms(base):
            add(extra)

    return normalized


def count_lookup_term_matches(row: dict[str, Any], lookup_terms: list[str]) -> int:
    if not lookup_terms:
        return 0
    meta_info = row.get("meta_info") or {}
    haystacks: list[str] = [
        str(row.get("scene_id") or ""),
        str(row.get("description") or ""),
        str(meta_info.get("display_name") or "") if isinstance(meta_info, dict) else "",
    ]
    for key in ("objects", "tags"):
        haystacks.extend(str(item) for item in (row.get(key) or []))
    matched = {
        term
        for term in lookup_terms
        for text in haystacks
        if text and term in text
    }
    return len(matched)


def count_lookup_field_matches(row: dict[str, Any], lookup_terms: list[str]) -> int:
    if not lookup_terms:
        return 0
    meta_info = row.get("meta_info") or {}
    fields: dict[str, list[str]] = {
        "scene_id": [str(row.get("scene_id") or "")],
        "description": [str(row.get("description") or "")],
        "display_name": [str(meta_info.get("display_name") or "")] if isinstance(meta_info, dict) else [""],
        "objects": [str(item) for item in (row.get("objects") or [])],
        "tags": [str(item) for item in (row.get("tags") or [])],
    }
    matched_fields = {
        field_name
        for field_name, values in fields.items()
        if any(term in value for term in lookup_terms for value in values if value)
    }
    return len(matched_fields)


def row_matches_lookup_terms(row: dict[str, Any], lookup_terms: list[str]) -> bool:
    return count_lookup_term_matches(row, lookup_terms) > 0


def lexical_fallback_model_assets(
    supabase_url: str,
    supabase_key: str,
    *,
    lookup_terms: list[str],
    start_time: str | None,
    end_time: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not lookup_terms:
        return []

    # Keep this bounded; the debug path values precision over completeness.
    rows = rest_select_model_assets(
        supabase_url,
        supabase_key,
        limit=max(limit * 20, 200),
        start_time=start_time,
        end_time=end_time,
    )
    matched = [row for row in rows if row_matches_lookup_terms(row, lookup_terms)]
    return matched[:limit]


def row_supports_any_target(row: dict[str, Any], target_objects: list[str]) -> bool:
    return any(row_supports_target(row, target) for target in target_objects if target.strip())


def score_object_candidate(
    row: dict[str, Any],
    *,
    lookup_terms: list[str],
    target_objects: list[str],
) -> tuple[int, int, int, int, str]:
    candidate_sources = set(row.get("_candidate_sources") or [])
    target_hit = int(row_supports_any_target(row, target_objects))
    term_hits = count_lookup_term_matches(row, lookup_terms)
    field_hits = count_lookup_field_matches(row, lookup_terms)
    vector_hit = int("vector" in candidate_sources)
    return (
        target_hit,
        term_hits,
        field_hits,
        vector_hit,
        str(row.get("created_at") or ""),
    )


def merge_object_candidates(
    rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    *,
    lookup_terms: list[str],
    target_objects: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    merged_by_id: dict[str, dict[str, Any]] = {}
    for source_name, group in (("vector", rows), ("lexical", lexical_rows)):
        for row in group:
            row_id = str(row.get("id") or row.get("scene_id") or id(row))
            existing = merged_by_id.get(row_id)
            if existing is None:
                merged = dict(row)
                merged["_candidate_sources"] = [source_name]
                merged_by_id[row_id] = merged
                continue
            sources = set(existing.get("_candidate_sources") or [])
            sources.add(source_name)
            existing["_candidate_sources"] = sorted(sources)
            for key in ("description", "objects", "tags", "meta_info", "created_at", "scene_id"):
                if not existing.get(key) and row.get(key):
                    existing[key] = row.get(key)

    ranked = sorted(
        merged_by_id.values(),
        key=lambda row: score_object_candidate(
            row,
            lookup_terms=lookup_terms,
            target_objects=target_objects,
        ),
        reverse=True,
    )
    return ranked[:limit]


def select_object_lookup_queries(
    search_text: str,
    target_objects: list[str],
    lookup_terms: list[str],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = canonicalize_lookup_term(value)
        if is_generic_lookup_token(token) or token in seen:
            return
        seen.add(token)
        ordered.append(token)

    for target in target_objects:
        for alias in expand_lookup_aliases(target):
            add(alias)
        add(target)
    for alias in expand_lookup_aliases(search_text):
        add(alias)
    add(search_text)
    for term in lookup_terms:
        add(term)
    if search_text:
        for fragment in split_lookup_fragments(search_text):
            add(fragment)
    return ordered[:3]


def build_object_lookup_candidates(
    *,
    search_text: str,
    target_objects: list[str],
    lookup_terms: list[str],
    dashscope_key: str,
    dashscope_base_url: str,
    embedding_model: str,
    supabase_url: str,
    supabase_key: str,
    match_threshold: float,
    match_count: int,
    recent_limit: int,
    start_time: str | None,
    end_time: str | None,
) -> tuple[list[dict[str, Any]], str, list[str], list[dict[str, str]]]:
    route_reasons: list[str] = []
    rpc_errors: list[dict[str, str]] = []
    search_query = canonicalize_lookup_term(search_text) or search_text or " ".join(target_objects)
    vector_queries = select_object_lookup_queries(search_query, target_objects, lookup_terms)
    if not vector_queries and search_query:
        vector_queries = [search_query]

    vector_result_groups: list[list[dict[str, Any]]] = []
    vector_any_hit = False
    for index, vector_query in enumerate(vector_queries):
        embedding = create_embedding(dashscope_key, dashscope_base_url, embedding_model, vector_query)
        rows, rpc_error = safe_rpc_match_memory_poses(
            supabase_url,
            supabase_key,
            query_embedding=embedding,
            match_threshold=match_threshold,
            match_count=match_count,
            start_time=start_time,
            end_time=end_time,
        )
        if rpc_error:
            route_reasons.append("rpc_exception")
            rpc_errors.append({
                "stage": "main_lookup" if index == 0 else "normalized_retry",
                "target": vector_query,
                "error": rpc_error,
            })
            continue
        if rows:
            vector_any_hit = True
            vector_result_groups.append(enrich_match_rows(supabase_url, supabase_key, rows))
            if index > 0:
                route_reasons.append("normalized_vector_retry")

    vector_rows = merge_object_candidates(
        [row for group in vector_result_groups for row in group],
        [],
        lookup_terms=lookup_terms,
        target_objects=target_objects,
        limit=max(match_count, recent_limit),
    ) if vector_result_groups else []
    if not vector_rows:
        route_reasons.append("rpc_empty")

    if target_objects:
        filtered_vector_rows = [
            row
            for row in vector_rows
            if row_supports_any_target(row, target_objects) or row_matches_lookup_terms(row, lookup_terms)
        ]
        if vector_rows and not filtered_vector_rows:
            route_reasons.append("post_filter_empty")
        candidate_rows = filtered_vector_rows if filtered_vector_rows else vector_rows
        retrieval_route = "vector_plus_filter"
    else:
        candidate_rows = vector_rows
        retrieval_route = "vector_only"

    entity_like = bool(target_objects or lookup_terms)
    should_add_lexical = bool(
        lookup_terms
        and (
            not candidate_rows
            or "post_filter_empty" in route_reasons
            or "rpc_empty" in route_reasons
            or (
                entity_like
                and target_objects
                and not any(row_supports_any_target(row, target_objects) for row in candidate_rows)
            )
        )
    )

    lexical_rows: list[dict[str, Any]] = []
    if should_add_lexical:
        lexical_rows = lexical_fallback_model_assets(
            supabase_url,
            supabase_key,
            lookup_terms=lookup_terms,
            start_time=start_time,
            end_time=end_time,
            limit=max(match_count, recent_limit),
        )

    if lexical_rows and candidate_rows:
        candidate_rows = merge_object_candidates(
            candidate_rows,
            lexical_rows,
            lookup_terms=lookup_terms,
            target_objects=target_objects,
            limit=max(match_count, recent_limit),
        )
        retrieval_route = "merged_vector_lexical"
    elif lexical_rows:
        candidate_rows = lexical_rows
        retrieval_route = "lexical_fallback"

    return candidate_rows, retrieval_route, route_reasons, rpc_errors


def detect_non_retrieval_answer(question: str) -> tuple[str, str] | None:
    normalized = re.sub(r"[\s\W_]+", "", question or "").lower()
    if not normalized:
        return None
    if normalized in {"你好", "您好", "嗨", "hello", "hi"} or any(keyword in normalized for keyword in GREETING_KEYWORDS):
        return (
            "greeting",
            "我是 BrainDance 的本地记忆问答助手，可以帮你查最近拍到的内容、物体线索和生成过的模型。",
        )
    if any(pattern in normalized for pattern in PRODUCT_IDENTITY_PATTERNS):
        return (
            "persona",
            "BrainDance 是一个面向本地记录检索的记忆问答助手，主要用来查询拍摄内容、物体线索和生成过的模型。",
        )
    if any(pattern in normalized for pattern in IDENTITY_PATTERNS):
        if "英文" in normalized or "英语" in normalized:
            return (
                "persona",
                "可以，我能处理简单英文问法，但当前主要面向本地记录检索，中文问法通常更稳。",
            )
        if "从哪里来" in normalized:
            return (
                "persona",
                "我是 BrainDance 项目里的本地记忆问答助手，基于你的本地记录和当前检索链路工作。",
            )
        return (
            "persona",
            "我是 BrainDance 的本地记忆问答助手，主要帮你根据本地记录查询拍摄内容、物体线索和模型资产。",
        )
    if any(pattern in normalized for pattern in SYSTEM_PROMPT_PATTERNS) and "你的" in normalized:
        return (
            "non_retrieval",
            "我不能直接透露内部 system prompt，但可以说明：我会优先依据本地检索证据回答问题。",
        )
    return None


def is_model_inventory_query(question: str, question_type: str, lookup_terms: list[str]) -> bool:
    if any(key in question for key in SEMANTIC_QUERY_EXPANSIONS):
        return False
    normalized_question = re.sub(r"\s+", "", question or "").lower()
    if any(hint in normalized_question for hint in ENGLISH_INVENTORY_HINTS):
        return True
    if question_type in {"recent_capture", "time_qa"} and "模型" in question:
        non_generic_terms = [term for term in lookup_terms if term not in GENERIC_MODEL_TERMS]
        if not non_generic_terms:
            return True
    if question_type != "object_lookup" or not lookup_terms:
        return (
            ("模型" in question and any(hint in question for hint in MODEL_INVENTORY_HINTS))
            or ("model" in normalized_question and any(hint in normalized_question for hint in ENGLISH_MODEL_HINTS))
        )
    if any(term not in GENERIC_MODEL_TERMS for term in lookup_terms):
        return False
    return any(hint in question for hint in MODEL_INVENTORY_HINTS)


def enrich_match_rows(
    supabase_url: str,
    supabase_key: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    asset_map = rest_fetch_model_assets_by_ids(
        supabase_url,
        supabase_key,
        [str(row["id"]) for row in rows],
    )
    merged = []
    for row in rows:
        asset = asset_map.get(str(row["id"]), {})
        combined = dict(row)
        for key in ("objects", "tags", "meta_info"):
            if key in asset:
                combined[key] = asset.get(key)
        if asset.get("description") and not combined.get("description"):
            combined["description"] = asset["description"]
        if asset.get("created_at"):
            combined["created_at"] = asset["created_at"]
        merged.append(combined)
    return merged


def normalize_retrieval_intent(
    question_type: str,
    *,
    hit_count: int,
    target_objects: list[str],
) -> str:
    if hit_count == 0:
        return "no_hit"
    if question_type == "partial_coverage" or len(target_objects) >= 2:
        return "partial_coverage"
    if question_type in {"recent_capture", "time_qa"}:
        return question_type
    return "object_lookup"


def build_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for row in rows:
        evidence.append({
            "scene_id": str(row.get("scene_id") or ""),
            "display_name": infer_display_name(row),
            "description": str(row.get("description") or ""),
            "objects": [str(item) for item in (row.get("objects") or [])],
            "tags": [str(item) for item in (row.get("tags") or [])],
            "created_at": normalize_iso8601(str(row.get("created_at") or "")) or str(row.get("created_at") or ""),
        })
    return evidence


def dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        normalized = re.sub(r"\s+", "", value)
        if value and normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped


def join_natural_list(items: list[str]) -> str:
    values = dedupe_preserve_order(items)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]}和{values[1]}"
    return "、".join(values[:-1]) + f"和{values[-1]}"


def join_answer_phrases(items: list[str]) -> str:
    values = dedupe_preserve_order(items)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]}，以及{values[1]}"
    return "，另外还有".join(values)


def evidence_haystacks(item: dict[str, Any]) -> list[str]:
    haystacks = [
        str(item.get("display_name") or ""),
        str(item.get("description") or ""),
    ]
    for key in ("objects", "tags"):
        haystacks.extend(str(value) for value in (item.get(key) or []))
    return [value for value in haystacks if value]


def evidence_supports_term(item: dict[str, Any], term: str) -> bool:
    normalized = normalize_match_text(term)
    if not normalized:
        return False
    return any(normalized in normalize_match_text(text) for text in evidence_haystacks(item))


def summarize_evidence_objects(
    item: dict[str, Any],
    *,
    focus_terms: list[str] | None = None,
    limit: int = 3,
) -> list[str]:
    focus_terms = [str(term).strip() for term in (focus_terms or []) if str(term).strip()]
    raw_objects = [
        simplify_object_name(str(obj))
        for obj in (item.get("objects") or [])
        if simplify_object_name(str(obj))
    ]
    objects = dedupe_preserve_order(raw_objects)
    if not objects:
        description = str(item.get("description") or "").strip()
        if description:
            first_clause = re.split(r"[。；;]", description, maxsplit=1)[0].strip()
            if first_clause:
                return [first_clause]
        display_name = str(item.get("display_name") or "").strip()
        return [display_name] if display_name else []

    prioritized: list[str] = []
    if focus_terms:
        for obj in objects:
            if any(
                contains_term(obj, term)
                or contains_term(term, obj)
                or evidence_supports_term({"objects": [obj]}, term)
                for term in focus_terms
            ):
                prioritized.append(obj)

    fallback_objects = [obj for obj in objects if obj not in prioritized]
    selected = prioritized + fallback_objects
    return selected[:limit]


def build_recent_answer(
    question: str,
    evidence: list[dict[str, Any]],
    lookup_terms: list[str],
) -> str | None:
    if not evidence:
        return None

    sorted_evidence = sorted(
        evidence,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )

    if lookup_terms:
        focus_objects: list[str] = []
        companions: list[str] = []
        for item in sorted_evidence:
            item_objects = summarize_evidence_objects(item, focus_terms=lookup_terms, limit=3)
            companions.extend(item_objects)
            for obj in item_objects:
                if any(contains_term(obj, term) or contains_term(term, obj) for term in lookup_terms):
                    focus_objects.append(obj)
        focus_objects = dedupe_preserve_order(focus_objects)[:3]
        companions = [obj for obj in dedupe_preserve_order(companions) if obj not in focus_objects][:2]
        focus_label = canonicalize_lookup_term(lookup_terms[0]) or lookup_terms[0]

        if focus_objects:
            if companions:
                return f"最近拍到的{focus_label}相关内容里，主要有{join_natural_list(focus_objects + companions)}。"
            return f"最近拍到的{focus_label}相关内容里，主要有{join_natural_list(focus_objects)}。"
        if companions:
            return f"最近拍到的{focus_label}相关内容里，主要有{join_natural_list(companions)}。"

    phrases: list[str] = []
    for item in sorted_evidence[:2]:
        objects = summarize_evidence_objects(item, limit=2)
        phrase = join_natural_list(objects)
        if phrase:
            phrases.append(phrase)
    phrases = dedupe_preserve_order(phrases)
    if not phrases:
        return None
    return f"最近拍到的主要有{join_answer_phrases(phrases)}。"


def build_model_inventory_answer(evidence: list[dict[str, Any]], *, include_scene_ids: bool = False) -> str | None:
    if not evidence:
        return None

    def summarize_item(item: dict[str, Any]) -> str:
        description = str(item.get("description") or "").strip()
        if description:
            first_clause = re.split(r"[。；;，,]", description, maxsplit=1)[0].strip()
            first_clause = re.sub(r"^(这是一个|这是|一个)", "", first_clause).strip("：:，, ")
            if first_clause:
                return first_clause
        objects = [
            simplify_object_name(str(obj))
            for obj in (item.get("objects") or [])
            if simplify_object_name(str(obj))
        ]
        if objects:
            return objects[0]
        return str(item.get("display_name") or item.get("scene_id") or "").strip()

    names: list[str] = []
    for item in evidence[:3]:
        name = summarize_item(item)
        if include_scene_ids:
            scene_id = str(item.get("scene_id") or "").strip()
            if name and scene_id:
                name = f"{name}（{scene_id}）"
        if name:
            names.append(name)
    names = dedupe_preserve_order(names)
    if not names:
        return None
    total_count = max(len(evidence), len(names))
    if len(names) == 1:
        return f"最近生成过{total_count}个模型，主要是{names[0]}。"
    return f"最近生成过{total_count}个模型，主要包括{join_natural_list(names[:3])}。"


def build_must_answer_focus_answer(
    question: str,
    evidence: list[dict[str, Any]],
    focus_terms: list[str],
) -> str | None:
    if not evidence or not focus_terms:
        return None

    ordered_focus_terms = dedupe_preserve_order(
        [canonicalize_lookup_term(term) or term for term in focus_terms]
    )
    if not ordered_focus_terms:
        return None
    focus = ordered_focus_terms[0]
    matched_items = [item for item in evidence if evidence_supports_term(item, focus)]
    if not matched_items:
        matched_items = evidence[:1]

    primary_objects: list[str] = []
    companion_objects: list[str] = []
    for item in matched_items[:2]:
        objects = summarize_evidence_objects(item, focus_terms=ordered_focus_terms, limit=3)
        for obj in objects:
            if contains_term(obj, focus) or contains_term(focus, obj):
                primary_objects.append(obj)
            else:
                companion_objects.append(obj)

    primary_objects = dedupe_preserve_order(primary_objects)
    companion_objects = dedupe_preserve_order([obj for obj in companion_objects if obj not in primary_objects])

    if focus in GENERIC_FOCUS_TERMS or focus in {"地球仪", "书架", "办公桌"}:
        objects = primary_objects + companion_objects
        objects = dedupe_preserve_order(objects)[:3]
        if objects:
            return f"最近拍到过{focus}相关内容，能看到{join_natural_list(objects)}。"
        return f"最近拍到过{focus}相关内容。"

    if primary_objects:
        if companion_objects:
            return f"最近拍到过{primary_objects[0]}，画面里也常一起出现{join_natural_list(companion_objects[:2])}。"
        return f"最近拍到过{primary_objects[0]}。"

    objects = dedupe_preserve_order(companion_objects)[:3]
    if objects:
        return f"最近拍到过{focus}相关内容，画面里主要有{join_natural_list(objects)}。"
    return f"最近拍到过{focus}相关内容。"


def build_semantic_lookup_answer(question: str, evidence: list[dict[str, Any]], lookup_terms: list[str]) -> str | None:
    if not evidence or not lookup_terms:
        return None
    semantic_terms = collect_semantic_lookup_terms(lookup_terms)
    if not semantic_terms:
        semantic_terms = lookup_terms

    def normalize_semantic_item(value: str) -> str:
        text = re.sub(r"[《》〈〉“”\"'（）()【】\[\]]", "", value or "")
        return re.sub(r"\s+", "", text)

    def add_unique(items: list[str], value: str, seen: set[str]) -> None:
        normalized = normalize_semantic_item(value)
        if not value or not normalized:
            return
        for existing in list(seen):
            if normalized == existing:
                return
            if normalized in existing:
                return
            if existing in normalized:
                seen.remove(existing)
                for index, item in enumerate(items):
                    if normalize_semantic_item(item) == existing:
                        items[index] = value
                        break
                seen.add(normalized)
                return
        seen.add(normalized)
        items.append(value)

    matched_items: list[str] = []
    seen_items: set[str] = set()
    for item in evidence:
        for candidate in item.get("objects") or []:
            text = simplify_object_name(str(candidate))
            if text and any(term in text for term in semantic_terms):
                add_unique(matched_items, text, seen_items)
        for candidate in item.get("tags") or []:
            text = str(candidate).strip()
            if text and any(term in text for term in semantic_terms):
                add_unique(matched_items, text, seen_items)
        description = str(item.get("description") or "").strip()
        for term in semantic_terms:
            if term and term in description:
                add_unique(matched_items, term, seen_items)

    if not matched_items:
        return None

    priority_order = [
        "算法导论",
        "高等数学",
        "笔记本电脑",
        "显示器",
        "机械键盘",
        "白板",
        "教材",
        "词典",
    ]

    def priority_score(text: str) -> tuple[int, int, str]:
        for index, keyword in enumerate(priority_order):
            if keyword in text:
                return (0, index, text)
        return (1, len(text), text)

    matched_items = sorted(matched_items, key=priority_score)
    examples = join_natural_list(matched_items[:3])
    if "计算机科学" in question or "计算机" in question:
        return f"有，整体偏计算机科学方向，常见内容包括{examples}。"
    if "理工" in question or "学术" in question or "学习" in question:
        return f"有，整体偏理工学习氛围，常见内容包括{examples}。"
    return f"有，相关内容里常见{examples}。"


def infer_answer_route(
    *,
    query_class: str,
    special_answer: str | None = None,
    special_answer_route: str | None = None,
) -> str:
    if query_class in {"greeting", "persona", "non_retrieval"}:
        return "fixed_response"
    if query_class == "inventory":
        return "inventory_formatter"
    if special_answer_route:
        return special_answer_route
    if special_answer:
        return "semantic_summary_formatter"
    return "lora_generation"


def coalesce_fallback_reason(reasons: list[str]) -> str | None:
    for reason in ("inventory_query", "post_filter_empty", "rpc_exception", "rpc_empty", "low_confidence_vector"):
        if reason in reasons:
            return reason
    return reasons[0] if reasons else None


def retrieve_real_chain_case(
    *,
    question: str,
    dashscope_key: str,
    dashscope_base_url: str,
    chat_model: str,
    embedding_model: str,
    supabase_url: str,
    supabase_key: str,
    match_threshold: float,
    match_count: int,
    recent_limit: int,
) -> dict[str, Any]:
    non_retrieval_result = detect_non_retrieval_answer(question)
    if non_retrieval_result:
        query_class, fixed_answer = non_retrieval_result
        return {
            "question": question,
            "parsed_intent": {
                "question_type": "other",
                "search_text": "",
                "target_objects": [],
                "start_time": None,
                "end_time": None,
            },
            "retrieval": {
                "query_class": query_class,
                "intent": query_class,
                "hit_count": 0,
                "retrieval_route": "non_retrieval",
                "fallback_trigger_reason": None,
                "answer_route": "fixed_response",
                "evidence": [],
            },
            "raw_rows": [],
            "support_map": {},
            "special_answer": fixed_answer,
            "query_class": query_class,
            "answer_route": "fixed_response",
        }

    intent = parse_query_intent(dashscope_key, dashscope_base_url, chat_model, question)
    question_type = intent["question_type"]
    raw_target_objects = list(intent["target_objects"])
    target_objects = split_target_objects(raw_target_objects, intent["search_text"])
    start_time = intent["start_time"]
    end_time = intent["end_time"]
    lookup_terms = normalize_lookup_terms(intent["search_text"], *target_objects)
    if not lookup_terms:
        lookup_terms = infer_semantic_terms_from_question(question)
    abstract_semantic_query = is_abstract_semantic_query(target_objects, lookup_terms)
    effective_target_objects = [] if abstract_semantic_query else target_objects
    effective_lookup_terms = collect_semantic_lookup_terms(lookup_terms) if abstract_semantic_query else lookup_terms

    raw_rows: list[dict[str, Any]] = []
    support_map: dict[str, bool] = {}
    route_reasons: list[str] = []
    retrieval_route = "vector_only"
    rpc_errors: list[dict[str, str]] = []

    model_inventory_query = is_model_inventory_query(question, question_type, lookup_terms)
    query_class = "inventory" if model_inventory_query else question_type

    if model_inventory_query:
        raw_rows = rest_select_model_assets(
            supabase_url,
            supabase_key,
            limit=recent_limit,
            start_time=start_time,
            end_time=end_time,
        )
        question_type = "recent_capture"
        retrieval_route = "inventory_special_case"
        route_reasons.append("inventory_query")
    elif question_type in {"recent_capture", "time_qa"} and not target_objects and not intent["search_text"]:
        raw_rows = rest_select_model_assets(
            supabase_url,
            supabase_key,
            limit=recent_limit,
            start_time=start_time,
            end_time=end_time,
        )
        retrieval_route = "recent_list"
    elif question_type == "partial_coverage" and len(target_objects) >= 2:
        seen: set[str] = set()
        partial_fallback_used = False
        partial_route_reasons: list[str] = []
        for target in target_objects:
            target_terms = normalize_lookup_terms(target)
            embedding = create_embedding(dashscope_key, dashscope_base_url, embedding_model, target)
            rows, rpc_error = safe_rpc_match_memory_poses(
                supabase_url,
                supabase_key,
                query_embedding=embedding,
                match_threshold=match_threshold,
                match_count=match_count,
                start_time=start_time,
                end_time=end_time,
            )
            if rpc_error:
                partial_route_reasons.append("rpc_exception")
                rpc_errors.append({
                    "stage": "partial_coverage",
                    "target": target,
                    "error": rpc_error,
                })
            rows = enrich_match_rows(supabase_url, supabase_key, rows) if rows else []
            if not rows:
                partial_route_reasons.append("rpc_empty")
            matched = next(
                (
                    row for row in rows
                    if row_supports_target(row, target) or row_matches_lookup_terms(row, target_terms)
                ),
                None,
            )
            if matched is None:
                if rows:
                    partial_route_reasons.append("post_filter_empty")
                lexical_rows = lexical_fallback_model_assets(
                    supabase_url,
                    supabase_key,
                    lookup_terms=target_terms,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1,
                )
                matched = lexical_rows[0] if lexical_rows else None
                partial_fallback_used = partial_fallback_used or matched is not None
            support_map[target] = matched is not None
            if matched and str(matched["id"]) not in seen:
                raw_rows.append(matched)
                seen.add(str(matched["id"]))
        retrieval_route = "lexical_fallback" if partial_fallback_used else "vector_plus_filter"
        route_reasons.extend(partial_route_reasons)
    else:
        search_text = intent["search_text"] or " ".join(target_objects) or question
        object_like_recent_query = (
            question_type in {"recent_capture", "time_qa"}
            and not effective_target_objects
            and bool(intent["search_text"])
            and bool(effective_lookup_terms)
        )
        if question_type == "object_lookup" or object_like_recent_query:
            raw_rows, retrieval_route, route_reasons, rpc_errors = build_object_lookup_candidates(
                search_text=search_text,
                target_objects=effective_target_objects,
                lookup_terms=effective_lookup_terms,
                dashscope_key=dashscope_key,
                dashscope_base_url=dashscope_base_url,
                embedding_model=embedding_model,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                match_threshold=match_threshold,
                match_count=match_count,
                recent_limit=recent_limit,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            embedding = create_embedding(dashscope_key, dashscope_base_url, embedding_model, search_text)
            rows, rpc_error = safe_rpc_match_memory_poses(
                supabase_url,
                supabase_key,
                query_embedding=embedding,
                match_threshold=match_threshold,
                match_count=match_count,
                start_time=start_time,
                end_time=end_time,
            )
            if rpc_error:
                route_reasons.append("rpc_exception")
                rpc_errors.append({
                    "stage": "main_lookup",
                    "target": search_text,
                    "error": rpc_error,
                })
            rows = enrich_match_rows(supabase_url, supabase_key, rows) if rows else []
            if not rows:
                route_reasons.append("rpc_empty")
            if target_objects:
                filtered = [
                    row for row in rows
                    if any(row_supports_target(row, target) for target in target_objects)
                    or row_matches_lookup_terms(row, lookup_terms)
                ]
                if rows and not filtered:
                    route_reasons.append("post_filter_empty")
                raw_rows = filtered if filtered else rows
                retrieval_route = "vector_plus_filter"
            else:
                raw_rows = rows

            semantic_query = abstract_semantic_query

            if (not raw_rows or (effective_target_objects and not any(row_matches_lookup_terms(row, effective_lookup_terms) for row in raw_rows))) and effective_lookup_terms:
                if raw_rows and not any(row_matches_lookup_terms(row, effective_lookup_terms) for row in raw_rows):
                    route_reasons.append("post_filter_empty")
                lexical_rows = lexical_fallback_model_assets(
                    supabase_url,
                    supabase_key,
                    lookup_terms=effective_lookup_terms,
                    start_time=start_time,
                    end_time=end_time,
                    limit=max(match_count, recent_limit),
                )
                if lexical_rows:
                    raw_rows = lexical_rows
                    retrieval_route = "lexical_fallback"
            elif semantic_query and effective_lookup_terms:
                lexical_rows = lexical_fallback_model_assets(
                    supabase_url,
                    supabase_key,
                    lookup_terms=effective_lookup_terms,
                    start_time=start_time,
                    end_time=end_time,
                    limit=max(match_count, recent_limit),
                )
                if lexical_rows:
                    raw_rows = merge_object_candidates(
                        raw_rows,
                        lexical_rows,
                        lookup_terms=effective_lookup_terms,
                        target_objects=effective_target_objects,
                        limit=max(match_count, recent_limit),
                    )
                    retrieval_route = "merged_vector_lexical"
                    route_reasons.append("low_confidence_vector")

    semantic_query = abstract_semantic_query

    if question_type in {"recent_capture", "time_qa"}:
        raw_rows = sorted(
            raw_rows,
            key=lambda row: str(row.get("created_at") or ""),
            reverse=True,
        )[:recent_limit]

    evidence = build_evidence(raw_rows)
    special_answer = None
    special_answer_route = None
    if model_inventory_query:
        special_answer = build_model_inventory_answer(evidence)
        special_answer_route = "inventory_formatter" if special_answer else None
    elif semantic_query:
        special_answer = build_semantic_lookup_answer(question, evidence, effective_lookup_terms)
        special_answer_route = "semantic_summary_formatter" if special_answer else None
    elif question_type in {"recent_capture", "time_qa"}:
        special_answer = build_recent_answer(question, evidence, effective_lookup_terms)
        special_answer_route = "recent_answer_formatter" if special_answer else None
    elif question_type == "object_lookup" and evidence:
        focus_terms = effective_target_objects or effective_lookup_terms
        special_answer = build_must_answer_focus_answer(question, evidence, focus_terms)
        special_answer_route = "must_answer_focus_formatter" if special_answer else None
    answer_route = infer_answer_route(
        query_class=query_class,
        special_answer=special_answer,
        special_answer_route=special_answer_route,
    )
    retrieval = {
        "query_class": query_class,
        "intent": normalize_retrieval_intent(
            question_type,
            hit_count=len(evidence),
            target_objects=effective_target_objects,
        ),
        "hit_count": len(evidence),
        "retrieval_route": retrieval_route,
        "fallback_trigger_reason": coalesce_fallback_reason(route_reasons),
        "answer_route": answer_route,
        "rpc_error_count": len(rpc_errors),
        "rpc_errors": rpc_errors,
        "fallback_after_rpc_error": bool(rpc_errors) and retrieval_route in {"lexical_fallback", "merged_vector_lexical"},
        "raw_target_objects": raw_target_objects,
        "normalized_lookup_terms": effective_lookup_terms,
        "route_reasons": route_reasons,
        "evidence": evidence,
    }
    return {
        "question": question,
        "parsed_intent": intent,
        "retrieval": retrieval,
        "raw_rows": raw_rows,
        "support_map": support_map,
        "special_answer": special_answer,
        "query_class": query_class,
        "answer_route": answer_route,
        "raw_target_objects": raw_target_objects,
        "normalized_lookup_terms": effective_lookup_terms,
    }


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def load_generator(model_name: str, adapter_path: str = "") -> tuple[Any, Any, str]:
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法加载生成模型")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def unload_model(model: Any | None) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_answer(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    question: str,
    retrieval: dict[str, Any],
    max_new_tokens: int,
) -> str:
    user_payload = json.dumps(
        {"question": question, "retrieval": retrieval},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = apply_chat(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法执行生成")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    answer_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def run_generation_pass(
    *,
    cases: list[dict[str, Any]],
    mode: str,
    model_name: str,
    adapter_path: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if mode == "off":
        return [{"answer": None, "generation_latency_sec": None} for _ in cases]
    load_adapter = mode == "lora_round3"
    tokenizer, model, device = load_generator(model_name, adapter_path if load_adapter else "")
    try:
        outputs = []
        for case in cases:
            special_answer = str(case.get("special_answer") or "").strip()
            if special_answer:
                outputs.append(
                    {
                        "answer": special_answer,
                        "generation_latency_sec": 0.0,
                    }
                )
                continue
            started = time.time()
            answer = generate_answer(
                tokenizer=tokenizer,
                model=model,
                device=device,
                question=case["question"],
                retrieval=case["retrieval"],
                max_new_tokens=max_new_tokens,
            )
            outputs.append(
                {
                    "answer": answer,
                    "generation_latency_sec": round(time.time() - started, 3),
                }
            )
        return outputs
    finally:
        unload_model(model)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def simplify_object_name(text: str) -> str:
    return re.split(r"[（(]", text or "", maxsplit=1)[0].strip()


def contains_term(text: str, term: str) -> bool:
    return normalize_match_text(term) in normalize_match_text(text)


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


def count_list_separators(text: str) -> int:
    return sum(text.count(marker) for marker in LIST_SEPARATORS)


def is_natural_style(text: str) -> bool:
    stripped = (text or "").strip()
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
    if "；" in stripped or ";" in stripped:
        return False
    if count_list_separators(stripped) >= 5:
        return False
    return True


def extract_focus_terms(row: dict[str, Any]) -> list[str]:
    parsed_intent = row.get("parsed_intent") or {}
    target_objects = [str(item).strip() for item in (parsed_intent.get("target_objects") or []) if str(item).strip()]
    if target_objects:
        return target_objects

    search_text = str(parsed_intent.get("search_text") or "").strip()
    normalized_search = search_text
    for suffix in ("相关画面", "相关内容", "相关记录", "画面", "内容", "记录", "场景"):
        if normalized_search.endswith(suffix):
            normalized_search = normalized_search[: -len(suffix)].strip()
            break
    if (
        normalized_search
        and normalized_search not in GENERIC_FOCUS_TERMS
        and "东西" not in search_text
        and "物品" not in search_text
        and len(normalized_search) <= 8
    ):
        return [normalized_search]

    evidence = row.get("evidence") or (row.get("retrieval") or {}).get("evidence") or []
    if evidence:
        first = evidence[0]
        primary = [
            simplify_object_name(str(item))
            for item in (first.get("objects") or [])
            if simplify_object_name(str(item))
        ]
        if primary:
            return primary[:1]
        display_name = str(first.get("display_name") or "").strip()
        if display_name:
            return [display_name]
    return []


def analyze_answer(answer: str | None, row: dict[str, Any]) -> dict[str, Any]:
    text = answer or ""
    support_map = row.get("support_map") or {}
    supported = [key for key, value in support_map.items() if value]
    unsupported = [key for key, value in support_map.items() if not value]

    positive_supported = [term for term in supported if is_positive_mention(text, term)]
    negative_supported = [term for term in supported if is_negated_mention(text, term)]
    negative_unsupported = [term for term in unsupported if is_negated_mention(text, term)]

    partial_false_negative = bool(supported) and (len(positive_supported) == 0 or len(negative_supported) > 0)
    partial_missing_negation = (
        bool(supported)
        and bool(unsupported)
        and len(positive_supported) > 0
        and len(negative_unsupported) < len(unsupported)
        and not partial_false_negative
    )

    focus_terms = extract_focus_terms(row)
    over_broad = count_list_separators(text) >= 4
    must_answer_focused = True
    if is_must_answer_group(str(row.get("group") or "")):
        must_answer_focused = (
            is_natural_style(text)
            and any(contains_term(text, term) for term in focus_terms)
            and not over_broad
        )

    return {
        "natural_style": is_natural_style(text),
        "partial_false_negative": partial_false_negative,
        "partial_missing_negation": partial_missing_negation,
        "must_answer_focused": must_answer_focused,
        "focus_terms": focus_terms,
    }


def main() -> None:
    args = parse_args()
    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()
    run_cases = load_cases(args.cases_file)

    retrieval_cases: list[dict[str, Any]] = []
    for case in run_cases:
        started = time.time()
        chain = retrieve_real_chain_case(
            question=case.question,
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
        chain["group"] = case.group
        chain["case_id"] = case.case_id
        chain["expected_issue_tag"] = case.expected_issue_tag
        chain["triage_label"] = case.triage_label
        chain["retrieval_latency_sec"] = round(time.time() - started, 3)
        retrieval_cases.append(chain)

    base_results: list[dict[str, Any]] = [{"answer": None, "generation_latency_sec": None} for _ in retrieval_cases]
    lora_results: list[dict[str, Any]] = [{"answer": None, "generation_latency_sec": None} for _ in retrieval_cases]

    if args.mode in {"base", "compare"}:
        base_results = run_generation_pass(
            cases=retrieval_cases,
            mode="base",
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )

    if args.mode in {"lora_round3", "compare"}:
        lora_results = run_generation_pass(
            cases=retrieval_cases,
            mode="lora_round3",
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )

    timestamp = now_utc().isoformat().replace("+00:00", "Z")
    run_id = datetime.now(timezone.utc).strftime("real_chain_debug_%Y%m%dT%H%M%SZ")
    output_rows = []
    for case, base_result, lora_result in zip(retrieval_cases, base_results, lora_results):
        base_analysis = analyze_answer(base_result["answer"], case)
        lora_analysis = analyze_answer(lora_result["answer"], case)
        output_rows.append({
            "run_id": run_id,
            "timestamp": timestamp,
            "case_id": case["case_id"],
            "group": case["group"],
            "question": case["question"],
            "local_qa_mode": args.mode,
            "expected_issue_tag": case["expected_issue_tag"],
            "triage_label": case["triage_label"],
            "parsed_intent": case["parsed_intent"],
            "query_class": case["query_class"],
            "intent": case["retrieval"]["intent"],
            "raw_target_objects": case.get("raw_target_objects", []),
            "normalized_lookup_terms": case.get("normalized_lookup_terms", []),
            "hit_count": case["retrieval"]["hit_count"],
            "retrieval_route": case["retrieval"]["retrieval_route"],
            "fallback_trigger_reason": case["retrieval"]["fallback_trigger_reason"],
            "route_reasons": case["retrieval"].get("route_reasons", []),
            "answer_route": case["retrieval"]["answer_route"],
            "sample_valid_for_retrieval_analysis": True,
            "evidence": case["retrieval"]["evidence"],
            "support_map": case["support_map"],
            "base_answer": base_result["answer"],
            "lora_answer": lora_result["answer"],
            "base_analysis": base_analysis,
            "lora_analysis": lora_analysis,
            "base_generation_latency_sec": base_result["generation_latency_sec"],
            "lora_generation_latency_sec": lora_result["generation_latency_sec"],
            "retrieval_latency_sec": case["retrieval_latency_sec"],
        })

    partial_rows = [row for row in output_rows if row["group"] == "partial_coverage" and row["support_map"]]
    must_rows = [row for row in output_rows if is_must_answer_group(row["group"])]
    group_counts = {
        group: len([row for row in output_rows if row["group"] == group])
        for group in sorted({row["group"] for row in output_rows})
    }
    triage_counts: dict[str, int] = {}
    query_class_counts: dict[str, int] = {}
    retrieval_route_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    answer_route_counts: dict[str, int] = {}
    for row in output_rows:
        triage_label = str(row.get("triage_label") or "").strip()
        if triage_label:
            triage_counts[triage_label] = triage_counts.get(triage_label, 0) + 1
        query_class = str(row.get("query_class") or "").strip()
        if query_class:
            query_class_counts[query_class] = query_class_counts.get(query_class, 0) + 1
        retrieval_route = str(row.get("retrieval_route") or "").strip()
        if retrieval_route:
            retrieval_route_counts[retrieval_route] = retrieval_route_counts.get(retrieval_route, 0) + 1
        fallback_reason = str(row.get("fallback_trigger_reason") or "").strip()
        if fallback_reason:
            fallback_reason_counts[fallback_reason] = fallback_reason_counts.get(fallback_reason, 0) + 1
        answer_route = str(row.get("answer_route") or "").strip()
        if answer_route:
            answer_route_counts[answer_route] = answer_route_counts.get(answer_route, 0) + 1

    metrics_by_group: dict[str, Any] = {}
    for group in sorted({row["group"] for row in output_rows}):
        group_rows = [row for row in output_rows if row["group"] == group]
        metrics_by_group[group] = {
            "count": len(group_rows),
            "natural_style_rate": round(
                sum(row["lora_analysis"]["natural_style"] for row in group_rows) / max(1, len(group_rows)),
                4,
            ),
        }
        if group == "partial_coverage":
            metrics_by_group[group]["partial_false_negative_rate"] = round(
                sum(row["lora_analysis"]["partial_false_negative"] for row in group_rows) / max(1, len(group_rows)),
                4,
            )
            metrics_by_group[group]["partial_missing_negation_rate"] = round(
                sum(row["lora_analysis"]["partial_missing_negation"] for row in group_rows) / max(1, len(group_rows)),
                4,
            )
        if is_must_answer_group(group):
            metrics_by_group[group]["must_answer_focus_rate"] = round(
                sum(row["lora_analysis"]["must_answer_focused"] for row in group_rows) / max(1, len(group_rows)),
                4,
            )
    summary = {
        "run_id": run_id,
        "generated_at": timestamp,
        "mode": args.mode,
        "supabase_url": supabase_url,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "cases_file": args.cases_file or None,
        "case_count": len(output_rows),
        "group_counts": group_counts,
        "triage_counts": triage_counts,
        "query_class_counts": query_class_counts,
        "retrieval_route_counts": retrieval_route_counts,
        "fallback_reason_counts": fallback_reason_counts,
        "answer_route_counts": answer_route_counts,
        "lora_metrics": {
            "natural_style_rate": round(
                sum(row["lora_analysis"]["natural_style"] for row in output_rows) / max(1, len(output_rows)),
                4,
            ),
            "partial_false_negative_rate": round(
                sum(row["lora_analysis"]["partial_false_negative"] for row in partial_rows) / max(1, len(partial_rows)),
                4,
            ),
            "partial_missing_negation_rate": round(
                sum(row["lora_analysis"]["partial_missing_negation"] for row in partial_rows) / max(1, len(partial_rows)),
                4,
            ),
            "must_answer_focus_rate": round(
                sum(row["lora_analysis"]["must_answer_focused"] for row in must_rows) / max(1, len(must_rows)),
                4,
            ),
        },
        "metrics_by_group": metrics_by_group,
        "cases": [
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "question": row["question"],
                "triage_label": row["triage_label"],
                "query_class": row["query_class"],
                "intent": row["intent"],
                "hit_count": row["hit_count"],
                "retrieval_route": row["retrieval_route"],
                "fallback_trigger_reason": row["fallback_trigger_reason"],
                "answer_route": row["answer_route"],
                "base_answer": row["base_answer"],
                "lora_answer": row["lora_answer"],
                "base_analysis": row["base_analysis"],
                "lora_analysis": row["lora_analysis"],
            }
            for row in output_rows
        ],
    }

    output_file = Path(args.output_file)
    summary_file = Path(args.summary_file)
    write_jsonl(output_file, output_rows, append=not args.overwrite_output)
    write_json(summary_file, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
