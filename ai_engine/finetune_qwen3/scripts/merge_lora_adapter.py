#!/usr/bin/env python3
"""
Merge a Qwen3 base model and a LoRA adapter into a standalone Hugging Face model dir.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hf_load_utils import safe_from_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a Qwen3 LoRA adapter into the base model")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--torch_dtype",
        default="auto_bf16",
        choices=["auto_bf16", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="cpu")
    parser.add_argument("--safe_serialization", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "auto_bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def build_merge_metadata(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "merged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "output_dir": args.output_dir,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "safe_serialization": args.safe_serialization,
    }


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_torch_dtype(args.torch_dtype)

    tokenizer = safe_from_pretrained(
        AutoTokenizer.from_pretrained,
        args.base_model,
        trust_remote_code=True,
    )
    model = safe_from_pretrained(
        AutoModelForCausalLM.from_pretrained,
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    peft_model = PeftModel.from_pretrained(model, args.adapter_path)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(output_dir)

    metadata = build_merge_metadata(args, dtype)
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
