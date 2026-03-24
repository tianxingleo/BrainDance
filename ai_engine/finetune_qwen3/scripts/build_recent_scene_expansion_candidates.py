#!/usr/bin/env python3
"""Build a larger local-only unseen benchmark candidate set from recent Supabase scenes."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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


def clean_terms(values: list[Any] | None) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values or []:
        item = str(value).strip()
        if not item or item in seen:
            continue
        rows.append(item)
        seen.add(item)
    return rows


def scene_difficulty(objects: list[str]) -> str:
    if len(objects) >= 6:
        return "hard"
    if len(objects) >= 3:
        return "medium"
    return "easy"


def build_negative_object_pool(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    negative_pool: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        scene_id = str(row.get("scene_id") or "")
        objects = clean_terms(row.get("objects") or [])
        for obj in objects:
            negative_pool[scene_id].append(obj)
    return dict(negative_pool)


def pick_negative_object(
    *,
    current_scene_id: str,
    current_objects: list[str],
    negative_pool: dict[str, list[str]],
) -> str:
    current_set = set(current_objects)
    for scene_id, objects in negative_pool.items():
        if scene_id == current_scene_id:
            continue
        for obj in objects:
            if obj not in current_set:
                return obj
    return "紫色蒸汽机械独角兽摆件"


def build_candidate_rows(
    recent_rows: list[dict[str, Any]],
    *,
    covered_scene_ids: set[str],
) -> list[dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []
    serial = 1
    negative_pool = build_negative_object_pool(recent_rows)

    for row in recent_rows:
        scene_id = str(row.get("scene_id") or "")
        if not scene_id or scene_id in covered_scene_ids:
            continue

        objects = clean_terms(row.get("objects") or [])
        tags = clean_terms(row.get("tags") or [])
        if not objects:
            continue

        focus_a = objects[0]
        difficulty = scene_difficulty(objects)
        focus_terms = clean_terms([focus_a, *tags[:2]])
        candidate_rows.append(
            {
                "case_id": f"recent_expand_{serial:03d}",
                "difficulty": difficulty,
                "group": "must_answer",
                "question": f"最近拍到过{focus_a}吗？",
                "scoreable": True,
                "supported_objects": [focus_a],
                "unsupported_objects": [],
                "focus_terms": focus_terms,
                "source_scene_ids": [scene_id],
                "notes": "recent scene auto-expanded single-object candidate",
            }
        )
        serial += 1

        if len(objects) >= 2:
            focus_b = objects[1]
            candidate_rows.append(
                {
                    "case_id": f"recent_expand_{serial:03d}",
                    "difficulty": "hard" if difficulty != "easy" else "medium",
                    "group": "partial_coverage",
                    "question": f"最近那类画面里有{focus_a}和{focus_b}吗？",
                    "scoreable": True,
                    "supported_objects": [focus_a, focus_b],
                    "unsupported_objects": [],
                    "focus_terms": clean_terms([focus_a, focus_b]),
                    "source_scene_ids": [scene_id],
                    "notes": "recent scene auto-expanded pair candidate",
                }
            )
            serial += 1

            negative_object = pick_negative_object(
                current_scene_id=scene_id,
                current_objects=objects,
                negative_pool=negative_pool,
            )
            candidate_rows.append(
                {
                    "case_id": f"recent_expand_{serial:03d}",
                    "difficulty": "hard",
                    "group": "partial_coverage",
                    "question": f"最近那类画面里有{focus_a}，也有{negative_object}吗？",
                    "scoreable": True,
                    "supported_objects": [focus_a],
                    "unsupported_objects": [negative_object],
                    "focus_terms": clean_terms([focus_a, negative_object]),
                    "answer_supported_terms": [focus_a],
                    "answer_unsupported_terms": [negative_object],
                    "source_scene_ids": [scene_id],
                    "notes": "recent scene auto-expanded contrastive candidate",
                }
            )
            serial += 1

        semantic_terms = clean_terms(tags[:2] or objects[:2])
        if semantic_terms:
            semantic_phrase = "、".join(semantic_terms)
            candidate_rows.append(
                {
                    "case_id": f"recent_expand_{serial:03d}",
                    "difficulty": "hard" if difficulty == "hard" else "medium",
                    "group": "abstract_semantic",
                    "question": f"最近有没有偏{semantic_phrase}这类风格或主题的内容？",
                    "scoreable": False,
                    "supported_objects": [],
                    "unsupported_objects": [],
                    "focus_terms": semantic_terms,
                    "source_scene_ids": [scene_id],
                    "notes": "recent scene auto-expanded semantic candidate",
                }
            )
            serial += 1

    return candidate_rows


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
    candidate_rows = build_candidate_rows(recent_rows, covered_scene_ids=covered_scene_ids)

    out_path.write_text(json.dumps(candidate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(out_path), "candidate_case_count": len(candidate_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
