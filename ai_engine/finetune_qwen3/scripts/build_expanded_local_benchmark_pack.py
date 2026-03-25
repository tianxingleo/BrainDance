#!/usr/bin/env python3
"""Build an expanded local-only benchmark pack from current unseen OOD + recent scene expansions + spatial hardcases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dedupe_cases(rows):
    seen = set()
    out = []
    for row in rows:
        key = (str(row.get("question") or "").strip(), tuple(row.get("source_scene_ids") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    unseen_path = DATA_DIR / "braindance_qwen3_unseen_ood_benchmark_20260324.json"
    expansion_path = DATA_DIR / "braindance_qwen3_unseen_ood_expansion_candidates_20260324_local.json"
    spatial_path = DATA_DIR / "braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json"

    out_path = DATA_DIR / "braindance_qwen3_benchmark_expanded_20260324_local.json"
    summary_path = DATA_DIR / "braindance_qwen3_benchmark_expanded_20260324_local_summary.json"

    unseen_rows = load_json(unseen_path) if unseen_path.exists() else []
    expansion_rows = load_json(expansion_path) if expansion_path.exists() else []
    spatial_rows = load_json(spatial_path) if spatial_path.exists() else []

    for row in unseen_rows:
        row["benchmark_source"] = "unseen_ood_base"
    for row in expansion_rows:
        row["benchmark_source"] = "recent_scene_expansion"
    for row in spatial_rows:
        row["benchmark_source"] = "spatial_hardcases"

    merged = dedupe_cases(unseen_rows + expansion_rows + spatial_rows)

    summary = {
        "total_case_count": len(merged),
        "scoreable_case_count": sum(1 for row in merged if row.get("scoreable", True)),
        "by_source": dict(Counter(str(row.get("benchmark_source") or "") for row in merged)),
        "by_group": dict(Counter(str(row.get("group") or "") for row in merged)),
        "by_difficulty": dict(Counter(str(row.get("difficulty") or "") for row in merged)),
        "unique_scene_id_count": len(
            {str(scene_id) for row in merged for scene_id in (row.get("source_scene_ids") or [])}
        ),
    }

    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(out_path), "summary_file": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
