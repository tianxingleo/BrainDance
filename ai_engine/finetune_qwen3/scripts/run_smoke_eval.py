#!/usr/bin/env python3
"""
Run a few deterministic smoke-eval prompts against the base model or a LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_load_utils import safe_from_pretrained


SYSTEM_PROMPT = (
    "你是 BrainDance 的本地记忆问答助手。"
    "你只能根据 retrieval 提供的证据回答，不要猜测。"
    "规则："
    "1. hit_count > 0 时，必须根据证据作答，禁止回答不知道、没有记录。"
    "2. hit_count == 0 时，只能回答“暂无相关记录”。"
    "3. 遇到“最近/昨天/上周”等时间问题时，按 created_at 从近到远组织回答。"
    "4. 回答保持简短自然，不复述问题，不解释规则，不循环输出。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke eval for Qwen3 BrainDance QA")
    parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", default="")
    return parser.parse_args()


def build_cases() -> list[dict[str, str]]:
    return [
        {
            "name": "recent_hit",
            "input": json.dumps(
                {
                    "question": "我最近拍了什么？",
                    "retrieval": {
                        "intent": "recent_capture",
                        "hit_count": 2,
                        "evidence": [
                            {
                                "scene_id": "frame_00052",
                                "display_name": "触控笔桌面采集 01",
                                "description": "触控笔位于桌面中央，左侧为红色背光键盘，右侧有鼠标和耳机盒，整体环境真实，可作为光照变化参考帧。",
                                "objects": ["触控笔", "键盘", "鼠标", "耳机盒", "桌面"],
                                "tags": ["室内", "桌面", "办公", "设备", "低照度"],
                                "created_at": "2026-03-19T18:00:00Z",
                            },
                            {
                                "scene_id": "test_study_room",
                                "display_name": "书房场景",
                                "description": "明亮的书房，有一张写字台、椅子和书架。墙上挂着风景画。",
                                "objects": ["写字台", "椅子", "书架", "风景画"],
                                "tags": ["书房", "室内", "明亮"],
                                "created_at": "2026-03-14T10:00:00Z",
                            },
                        ],
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
        {
            "name": "no_hit",
            "input": json.dumps(
                {
                    "question": "我最近拍过自行车吗？",
                    "retrieval": {"intent": "no_hit", "hit_count": 0, "evidence": []},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
        {
            "name": "partial_hit",
            "input": json.dumps(
                {
                    "question": "我最近拍过触控笔和冰箱吗？",
                    "retrieval": {
                        "intent": "partial_coverage",
                        "hit_count": 1,
                        "evidence": [
                            {
                                "scene_id": "frame_00629",
                                "display_name": "触控笔桌面采集 14",
                                "description": "触控笔竖直放置于桌面中央，笔尖朝上，笔身完全对齐画幅中心，背景为纯木纹桌面，无其他物体干扰，适合用于顶面特征提取。",
                                "objects": ["触控笔", "桌面"],
                                "tags": ["室内", "桌面", "设备"],
                                "created_at": "2026-03-16T12:00:00Z",
                            }
                        ],
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


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


def main() -> None:
    args = parse_args()
    tokenizer = safe_from_pretrained(
        AutoTokenizer.from_pretrained,
        args.model_name,
        trust_remote_code=True,
    )
    model = safe_from_pretrained(
        AutoModelForCausalLM.from_pretrained,
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outputs: list[dict[str, str]] = []
    for case in build_cases():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["input"]},
        ]
        prompt = apply_chat(tokenizer, messages, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        outputs.append({"name": case["name"], "answer": answer})

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
