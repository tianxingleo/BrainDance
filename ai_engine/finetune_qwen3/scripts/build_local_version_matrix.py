#!/usr/bin/env python3
"""Aggregate local benchmark logs into a unified version matrix and dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs"
AUDIT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pick_existing(*names: str) -> Path:
    for name in names:
        path = LOG_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no existing log file matched: {names}")


def infer_label(name: str) -> str:
    mapping = [
        ("benchmark_strict_v3_base_local_noopt", "1.7B Base local-noopt"),
        ("benchmark_strict_v3_lora_local_noopt", "1.7B LoRA local-noopt"),
        ("benchmark_qwen3_0p6b_full_round4_partial_patch_v1", "0.6B full round4 patch"),
        ("benchmark_qwen3_0p6b_full_round3_lr5e6", "0.6B full round3 lr5e-6"),
        ("benchmark_qwen3_0p6b_full_round2_lr8e6", "0.6B full round2 lr8e-6"),
        ("benchmark_qwen3_0p6b_full_round1", "0.6B full round1"),
        ("benchmark_qwen3_0p6b_round1", "0.6B LoRA"),
        ("benchmark_qwen3_1p7b_full_round1_lr5e6", "1.7B full lr5e-6"),
        ("benchmark_qwen3_1p7b_full_round1", "1.7B full round1"),
        ("benchmark_qwen3_1p7b_full_smoke", "1.7B full smoke"),
        ("benchmark_strict_v3_qwen3_1p7b_full_smoke", "1.7B full smoke"),
        ("benchmark_qwen3_1p7b_merged_round4_1_patch_mixed", "1.7B merged"),
        ("benchmark_qwen3_1p7b_q4_gguf_imatrix_v1_round4_1_patch_mixed", "1.7B Q4_K_M + imatrix"),
        ("benchmark_qwen3_1p7b_q5_gguf_imatrix_v1_round4_1_patch_mixed", "1.7B Q5_K_M + imatrix"),
        ("benchmark_qwen3_1p7b_q4_gguf_round4_1_patch_mixed", "1.7B Q4_K_M"),
        ("benchmark_qwen3_1p7b_q5_gguf_round4_1_patch_mixed", "1.7B Q5_K_M"),
        ("qwen3_0p6b_full_round4_partial_patch_v1", "0.6B full round4 patch"),
        ("qwen3_0p6b_full_round3_lr5e6", "0.6B full round3 lr5e-6"),
        ("qwen3_0p6b_full_round2_lr8e6", "0.6B full round2 lr8e-6"),
        ("qwen3_0p6b_full_gpu1", "0.6B full round1"),
        ("qwen3_0p6b_lora", "0.6B LoRA"),
        ("qwen3_1p7b_full_round1_lr5e6", "1.7B full lr5e-6"),
        ("qwen3_1p7b_full_gpu1", "1.7B full round1"),
        ("qwen3_1p7b_merged", "1.7B merged"),
        ("qwen3_1p7b_q4_gguf_imatrix_v1", "1.7B Q4 imatrix"),
        ("qwen3_1p7b_q5_gguf_imatrix_v1", "1.7B Q5 imatrix"),
        ("qwen3_1p7b_q4_gguf", "1.7B Q4"),
        ("qwen3_1p7b_q5_gguf", "1.7B Q5"),
        ("lora_20260322", "1.7B LoRA"),
        ("qwen3_1p7b_lora", "1.7B LoRA"),
        ("base", "1.7B Base"),
    ]
    for token, label in mapping:
        if token in name:
            return label
    return name


def main() -> None:
    rows = []

    for path in sorted(LOG_DIR.glob("benchmark_strict_v3_*.json")):
        obj = load_json(path)
        m = obj.get("metrics") or {}
        rows.append(
            {
                "benchmark": "strict_v3",
                "label": infer_label(path.stem),
                "source_file": path.name,
                "false_no_answer_rate": m.get("false_no_answer_rate"),
                "partial_hallucination_rate": m.get("partial_hallucination_rate"),
                "partial_hit_precision": m.get("partial_hit_precision"),
                "partial_missing_negation_rate": m.get("partial_missing_negation_rate"),
                "must_answer_focus_rate": m.get("must_answer_focus_rate"),
                "natural_style_rate": m.get("natural_style_rate"),
            }
        )

    for path in sorted(LOG_DIR.glob("benchmark_qwen3_*.json")):
        obj = load_json(path)
        m = obj.get("metrics") or {}
        rows.append(
            {
                "benchmark": "original_benchmark",
                "label": infer_label(path.stem),
                "source_file": path.name,
                "false_no_answer_rate": m.get("false_no_answer_rate"),
                "partial_hallucination_rate": m.get("partial_hallucination_rate"),
                "partial_hit_precision": m.get("partial_hit_precision"),
                "partial_missing_negation_rate": m.get("partial_missing_negation_rate"),
                "must_answer_focus_rate": m.get("must_answer_focus_rate"),
                "natural_style_rate": m.get("natural_style_rate"),
            }
        )

    unseen_path = pick_existing(
        "unseen_ood_benchmark_20260324_frozen_all_local_summary.json",
        "unseen_ood_benchmark_20260324_frozen_summary.json",
    )
    unseen = load_json(unseen_path)
    for item in unseen["summary"]:
        rows.append(
            {
                "benchmark": "unseen_ood_frozen",
                "label": item["candidate_label"],
                "source_file": unseen_path.name,
                "end_to_end_pass_rate": item.get("end_to_end_pass_rate"),
                "answer_pass_rate_when_retrieval_ok": item.get("answer_pass_rate_when_retrieval_ok"),
                "avg_total_ms": item.get("avg_total_ms"),
            }
        )

    spatial_path = pick_existing(
        "spatial_hardcases_candidates_20260324_frozen_all_local_summary.json",
        "spatial_hardcases_candidates_20260324_frozen_summary.json",
    )
    spatial = load_json(spatial_path)
    for item in spatial["summary"]:
        rows.append(
            {
                "benchmark": "spatial_hardcases_frozen",
                "label": item["candidate_label"],
                "source_file": spatial_path.name,
                "spatial_direct_rate": item.get("spatial_direct_rate"),
                "generic_scene_summary_rate": item.get("generic_scene_summary_rate"),
                "refusal_rate": item.get("refusal_rate"),
                "avg_total_ms": item.get("avg_total_ms"),
            }
        )

    df = pd.DataFrame(rows)
    matrix_path = AUDIT_DIR / "local_version_matrix_20260324.json"
    html_path = AUDIT_DIR / "local_version_matrix_dashboard_20260324.html"
    summary_path = AUDIT_DIR / "local_version_matrix_summary_20260324.json"
    matrix_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "row_count": int(len(df)),
        "benchmark_counts": {str(k): int(v) for k, v in df["benchmark"].value_counts().to_dict().items()},
        "unique_label_count": int(df["label"].nunique()),
        "labels": sorted(str(item) for item in df["label"].dropna().unique().tolist()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    subset = df[df["benchmark"].isin(["strict_v3", "unseen_ood_frozen", "spatial_hardcases_frozen"])].copy()
    heat_rows = []
    for _, row in subset.iterrows():
        if row["benchmark"] == "strict_v3":
            value = row.get("partial_hit_precision")
        elif row["benchmark"] == "unseen_ood_frozen":
            value = row.get("end_to_end_pass_rate")
        else:
            value = 1.0 - float(row.get("generic_scene_summary_rate") or 0.0)
        heat_rows.append({"label": row["label"], "benchmark": row["benchmark"], "value": value})
    heat_df = pd.DataFrame(heat_rows)

    fig = px.density_heatmap(
        heat_df,
        x="benchmark",
        y="label",
        z="value",
        histfunc="avg",
        color_continuous_scale="Viridis",
        title="Local Version Matrix Heatmap",
    )
    html_path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")

    print(
        json.dumps(
            {
                "matrix_file": str(matrix_path),
                "dashboard_file": str(html_path),
                "summary_file": str(summary_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
