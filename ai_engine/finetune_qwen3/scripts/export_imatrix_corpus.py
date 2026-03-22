#!/usr/bin/env python3
"""Export representative calibration text for llama.cpp importance-matrix runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibration corpus for llama-imatrix")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output text file")
    parser.add_argument("--max_records", type=int, default=256, help="Max records to export across all inputs")
    parser.add_argument(
        "--mode",
        choices=("chat", "prompt_only"),
        default="chat",
        help="chat: system+user+assistant reference; prompt_only: system+user",
    )
    return parser.parse_args()


def row_to_text(row: dict[str, Any], mode: str) -> str:
    sections: list[str] = []
    for message in row.get("messages") or []:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if mode == "prompt_only" and role == "assistant":
            continue
        sections.append(f"{role.upper()}:\n{content}")

    if mode == "chat":
        reference_answer = str(row.get("reference_answer") or "").strip()
        if reference_answer and not any(str(m.get("role")) == "assistant" for m in row.get("messages") or []):
            sections.append(f"ASSISTANT:\n{reference_answer}")

    return "\n\n".join(sections).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []
    exported = 0
    for input_path in [Path(item) for item in args.inputs]:
        rows = load_jsonl(input_path)
        for row in rows:
            if exported >= args.max_records:
                break
            text = row_to_text(row, mode=args.mode)
            if text:
                chunks.append(text)
                exported += 1
        if exported >= args.max_records:
            break

    output_path.write_text("\n\n<EOT>\n\n".join(chunks) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "records": exported,
                "mode": args.mode,
                "inputs": args.inputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
