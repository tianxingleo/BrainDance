#!/usr/bin/env python3
"""Build a larger local-only unseen benchmark candidate set from recent Supabase scenes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_recent_scenes(limit: int = 60) -> list[dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts"))
    from run_real_chain_debug import extract_supabase_config, rest_select_model_assets

    supabase_url, supabase_key = extract_supabase_config()
    return rest_select_model_assets(supabase_url, supabase_key, limit=limit)


def main() -> None:
    benchmark_path = DATA_DIR / "braindance_qwen3_unseen_ood_benchmark_20260324.json"
    spatial_path = DATA_DIR / "braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json"
    out_path = DATA_DIR / "braindance_qwen3_unseen_ood_expansion_candidates_20260324_local.json"

    unseen_rows = load_json(benchmark_path) if benchmark_path.exists() else []
    spatial_rows = load_json(spatial_path) if spatial_path.exists() else []

    covered_scene_ids = {
        str(scene_id)
        for row in unseen_rows + spatial_rows
        for scene_id in (row.get("source_scene_ids") or [])
    }

    recent_rows = load_recent_scenes(limit=60)
    candidate_rows: list[dict[str, Any]] = []
    serial = 1

    for row in recent_rows:
        scene_id = str(row.get("scene_id") or "")
        if not scene_id or scene_id in covered_scene_ids:
            continue
        objects = [str(x).strip() for x in (row.get("objects") or []) if str(x).strip()]
        tags = [str(x).strip() for x in (row.get("tags") or []) if str(x).strip()]
        if not objects:
            continue

        focus_a = objects[0]
        candidate_rows.append(
            {
                "case_id": f"recent_expand_{serial:03d}",
                "difficulty": "medium" if len(objects) <= 5 else "hard",
                "group": "must_answer",
                "question": f"最近拍到过{focus_a}吗？",
                "scoreable": True,
                "supported_objects": [focus_a],
                "unsupported_objects": [],
                "focus_terms": [focus_a, *(tags[:2])],
                "source_scene_ids": [scene_id],
                "notes": "recent scene auto-expanded candidate",
            }
        )
        serial += 1

        if len(objects) >= 2:
            candidate_rows.append(
                {
                    "case_id": f"recent_expand_{serial:03d}",
                    "difficulty": "hard",
                    "group": "partial_coverage",
                    "question": f"最近那类画面里有{objects[0]}和{objects[1]}吗？",
                    "scoreable": True,
                    "supported_objects": [objects[0], objects[1]],
                    "unsupported_objects": [],
                    "focus_terms": [objects[0], objects[1]],
                    "source_scene_ids": [scene_id],
                    "notes": "recent scene auto-expanded pair candidate",
                }
            )
            serial += 1

    out_path.write_text(json.dumps(candidate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(out_path), "candidate_case_count": len(candidate_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
