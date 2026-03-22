#!/usr/bin/env python3
"""Mark known Part 16-D dirty interactive samples as invalid for retrieval analysis."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_FILE = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "logs"
    / "interactive_sessions"
    / "interactive_debug_20260321T140506Z.jsonl"
)
PREVIEW_SCOPE_ERROR = "UnboundLocalError: local variable 'preview' referenced before assignment"


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"未找到目标日志文件: {TARGET_FILE}")

    rows = [json.loads(line) for line in TARGET_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    updated_count = 0
    for row in rows:
        if str(row.get("error") or "").strip() == PREVIEW_SCOPE_ERROR:
            row["sample_valid_for_retrieval_analysis"] = False
            row["sample_invalid_reason"] = "part16d_preview_scope_bug"
            updated_count += 1
        else:
            row.setdefault("sample_valid_for_retrieval_analysis", True)

    TARGET_FILE.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"updated={updated_count} file={TARGET_FILE}")


if __name__ == "__main__":
    main()
