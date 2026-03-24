#!/usr/bin/env python3
"""Audit benchmark coverage, local model coverage, and build a Plotly dashboard."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"
LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs"
OUTPUT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def infer_label_from_path(name: str) -> str:
    mapping = [
        ("qwen3_1p7b_lora", "1.7B LoRA"),
        ("qwen3_0p6b_lora", "0.6B LoRA"),
        ("qwen3_0p6b_full_round4_partial_patch_v1", "0.6B full SFT round4 partial patch"),
        ("qwen3_0p6b_full_round3_lr5e6", "0.6B full SFT round3 lr5e-6"),
        ("qwen3_0p6b_full_round2_lr8e6", "0.6B full SFT round2 lr8e-6"),
        ("qwen3_0p6b_full_round1", "0.6B full SFT round1"),
        ("qwen3_1p7b_full_round1_lr5e6", "1.7B full SFT lr5e-6"),
        ("qwen3_1p7b_full_round1", "1.7B full SFT round1"),
        ("qwen3_1p7b_lora_sft_round4_1_patch_mixed", "1.7B LoRA round4.1 mixed"),
        ("qwen3_1p7b_lora_sft_round4_1_patch", "1.7B LoRA round4.1 patch"),
        ("qwen3_1p7b_lora_sft_round4_patch", "1.7B LoRA round4 patch"),
        ("qwen3_1p7b_lora_sft_round3", "1.7B LoRA round3"),
        ("qwen3_1p7b_merged", "1.7B merged"),
        ("qwen3_1p7b_q4_gguf_imatrix_v1", "1.7B Q4_K_M + imatrix"),
        ("qwen3_1p7b_q5_gguf_imatrix_v1", "1.7B Q5_K_M + imatrix"),
        ("qwen3_1p7b_q4_gguf", "1.7B Q4_K_M"),
        ("qwen3_1p7b_q5_gguf", "1.7B Q5_K_M"),
        ("base", "1.7B Base"),
    ]
    for token, label in mapping:
        if token in name:
            return label
    return name


def benchmark_match_patterns(version_name: str) -> list[str]:
    patterns = [version_name]
    mapping = {
        "qwen3_0p6b_full_sft_round1": ["qwen3_0p6b_full_gpu1", "qwen3_0p6b_full_round1_gpu1"],
        "qwen3_0p6b_full_sft_round2_lr8e6": ["qwen3_0p6b_full_round2_lr8e6_gpu1"],
        "qwen3_0p6b_full_sft_round3_lr5e6": ["qwen3_0p6b_full_round3_lr5e6_gpu1"],
        "qwen3_0p6b_full_sft_round4_partial_patch_v1": ["qwen3_0p6b_full_round4_partial_patch_v1_gpu1"],
        "qwen3_0p6b_lora_sft_round1": ["qwen3_0p6b_lora_gpu1", "qwen3_0p6b_round1_gpu1", "qwen3_0p6b_round1_gpu0"],
        "qwen3_1p7b_full_sft_round1_gpu1": ["qwen3_1p7b_full_gpu1", "qwen3_1p7b_full_round1_gpu1"],
        "qwen3_1p7b_full_sft_round1_lr5e6_gpu1": ["qwen3_1p7b_full_round1_lr5e6_gpu1"],
        "qwen3_1p7b_lora_sft_round3": ["lora_20260322", "qwen3_1p7b_lora_round3", "qwen3_1p7b_lora"],
        "qwen3_1p7b_lora_sft_round4_patch": ["qwen3_1p7b_lora_round4_patch"],
        "qwen3_1p7b_lora_sft_round4_1_patch": ["qwen3_1p7b_lora_round4_1_patch"],
        "qwen3_1p7b_lora_sft_round4_1_patch_mixed": ["qwen3_1p7b_lora", "round4_1_patch_mixed"],
        "qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0": ["qwen3_1p7b_merged_gpu1", "qwen3_1p7b_merged_round4_1_patch_mixed_gpu1"],
        "qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0": [
            "qwen3_1p7b_q4_gguf",
            "qwen3_1p7b_q5_gguf",
            "qwen3_1p7b_q4_gguf_imatrix_v1",
            "qwen3_1p7b_q5_gguf_imatrix_v1",
        ],
    }
    patterns.extend(mapping.get(version_name, []))
    return patterns


def list_local_model_versions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_name in ("outputs", "releases"):
        root = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / root_name
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            if not name.startswith("qwen3_"):
                continue
            rows.append(
                {
                    "version_name": name,
                    "label": infer_label_from_path(name),
                    "root": root_name,
                    "path": str(path),
                }
            )
    return rows


def collect_logged_benchmarks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LOG_DIR.glob("*.json")):
        name = path.name
        payload = safe_read_json(path)
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue

        benchmark_kind = None
        if name.startswith("benchmark_strict_v3_"):
            benchmark_kind = "strict_v3"
        elif name.startswith("benchmark_qwen3_"):
            benchmark_kind = "original_benchmark"
        elif name.startswith("unseen_ood_benchmark_") and name.endswith("_summary.json"):
            benchmark_kind = "unseen_ood"
        elif name.startswith("spatial_hardcases_candidates_") and name.endswith("_summary.json"):
            benchmark_kind = "spatial_hardcases"
        if benchmark_kind is None:
            continue

        label = infer_label_from_path(name)
        rows.append(
            {
                "log_file": name,
                "benchmark_kind": benchmark_kind,
                "label": label,
                "false_no_answer_rate": metrics.get("false_no_answer_rate"),
                "partial_hallucination_rate": metrics.get("partial_hallucination_rate"),
                "partial_hit_precision": metrics.get("partial_hit_precision"),
                "partial_missing_negation_rate": metrics.get("partial_missing_negation_rate"),
                "must_answer_focus_rate": metrics.get("must_answer_focus_rate"),
                "natural_style_rate": metrics.get("natural_style_rate"),
                "evidence_utilization_rate": metrics.get("evidence_utilization_rate"),
            }
        )
    return rows


def load_recent_scenes(limit: int = 60) -> list[dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "scripts"))
    from run_real_chain_debug import extract_supabase_config, rest_select_model_assets

    supabase_url, supabase_key = extract_supabase_config()
    rows = rest_select_model_assets(supabase_url, supabase_key, limit=limit)
    return rows


def load_local_benchmark_cases() -> dict[str, list[dict[str, Any]]]:
    files = {
        "unseen_ood": DATA_DIR / "braindance_qwen3_unseen_ood_benchmark_20260324.json",
        "spatial_hardcases": DATA_DIR / "braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json",
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for key, path in files.items():
        payload = safe_read_json(path)
        if isinstance(payload, list):
            out[key] = payload
    return out


def build_scene_coverage(recent_scenes: list[dict[str, Any]], case_sets: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    covered_scene_ids = set()
    for rows in case_sets.values():
        for row in rows:
            for scene_id in row.get("source_scene_ids") or []:
                covered_scene_ids.add(str(scene_id))

    coverage_rows: list[dict[str, Any]] = []
    uncovered_rows: list[dict[str, Any]] = []
    for row in recent_scenes:
        scene_id = str(row.get("scene_id") or "")
        objects = [str(x) for x in (row.get("objects") or [])]
        tags = [str(x) for x in (row.get("tags") or [])]
        item = {
            "created_at": str(row.get("created_at") or ""),
            "scene_id": scene_id,
            "object_count": len(objects),
            "tag_count": len(tags),
            "objects_preview": " / ".join(objects[:6]),
            "tags_preview": " / ".join(tags[:6]),
            "covered_by_current_local_benchmarks": scene_id in covered_scene_ids,
        }
        coverage_rows.append(item)
        if scene_id not in covered_scene_ids:
            uncovered_rows.append(item)
    return pd.DataFrame(coverage_rows), pd.DataFrame(uncovered_rows)


def build_model_coverage_df(local_versions: list[dict[str, Any]], logged_rows: list[dict[str, Any]]) -> pd.DataFrame:
    strict_files = [row["log_file"] for row in logged_rows if row["benchmark_kind"] == "strict_v3"]
    original_files = [row["log_file"] for row in logged_rows if row["benchmark_kind"] == "original_benchmark"]

    rows: list[dict[str, Any]] = []
    for item in local_versions:
        patterns = benchmark_match_patterns(item["version_name"])
        strict_logged = any(any(pattern in name for pattern in patterns) for name in strict_files)
        original_logged = any(any(pattern in name for pattern in patterns) for name in original_files)
        rows.append(
            {
                **item,
                "strict_v3_logged": strict_logged,
                "original_benchmark_logged": original_logged,
            }
        )
    return pd.DataFrame(rows)


def build_oob_summary(case_sets: dict[str, list[dict[str, Any]]], recent_scene_df: pd.DataFrame) -> dict[str, Any]:
    unseen_rows = case_sets.get("unseen_ood", [])
    spatial_rows = case_sets.get("spatial_hardcases", [])
    all_covered = {
        str(scene_id)
        for row in unseen_rows + spatial_rows
        for scene_id in (row.get("source_scene_ids") or [])
    }
    return {
        "recent_scene_count": int(len(recent_scene_df)),
        "covered_recent_scene_count": int(sum(bool(v) for v in recent_scene_df["covered_by_current_local_benchmarks"].tolist()))
        if not recent_scene_df.empty
        else 0,
        "unseen_ood_case_count": len(unseen_rows),
        "spatial_hardcase_count": len(spatial_rows),
        "unique_scene_ids_in_local_benchmarks": len(all_covered),
    }


def write_dashboard(
    *,
    summary: dict[str, Any],
    model_coverage_df: pd.DataFrame,
    recent_scene_df: pd.DataFrame,
    uncovered_scene_df: pd.DataFrame,
    strict_metrics_df: pd.DataFrame,
    html_path: Path,
) -> None:
    figs: list[str] = []

    if not model_coverage_df.empty:
        fig = px.scatter(
            model_coverage_df,
            x="strict_v3_logged",
            y="original_benchmark_logged",
            color="root",
            hover_data=["version_name", "path", "label"],
            title="本地模型版本是否已进入主要 benchmark 日志覆盖",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    if not recent_scene_df.empty:
        coverage_counts = (
            recent_scene_df["covered_by_current_local_benchmarks"].value_counts().rename(index={True: "已覆盖", False: "未覆盖"})
        )
        fig = px.bar(
            x=list(coverage_counts.index),
            y=list(coverage_counts.values),
            title="最近 Supabase 场景在当前本地 benchmark 中的覆盖情况",
            labels={"x": "覆盖状态", "y": "场景数"},
            text=list(coverage_counts.values),
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not strict_metrics_df.empty:
        melted = strict_metrics_df.melt(
            id_vars=["label"],
            value_vars=[
                "false_no_answer_rate",
                "partial_hallucination_rate",
                "partial_hit_precision",
                "partial_missing_negation_rate",
                "must_answer_focus_rate",
                "natural_style_rate",
            ],
            var_name="metric",
            value_name="value",
        )
        fig = px.line(
            melted,
            x="metric",
            y="value",
            color="label",
            markers=True,
            title="Strict v3 各本地版本关键指标对比",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    uncovered_table = ""
    if not uncovered_scene_df.empty:
        top = uncovered_scene_df.head(20)[["created_at", "scene_id", "objects_preview", "tags_preview"]]
        uncovered_table = top.to_html(index=False, escape=False)

    summary_html = f"""
    <h1>Qwen3 Benchmark 审计总览</h1>
    <p>最近场景数：{summary['recent_scene_count']} | 当前本地 benchmark 覆盖的最近场景数：{summary['covered_recent_scene_count']} | 未见数据题数：{summary['unseen_ood_case_count']} | 空间 hardcase 题数：{summary['spatial_hardcase_count']}</p>
    <p>本地 benchmark 涉及唯一 scene_id 数：{summary['unique_scene_ids_in_local_benchmarks']}</p>
    <h2>最近但未纳入当前本地 benchmark 的场景（前 20）</h2>
    {uncovered_table}
    """

    html = "<html><head><meta charset='utf-8'><title>Qwen3 Benchmark Audit Dashboard</title></head><body>"
    html += summary_html
    html += "\n".join(figs)
    html += "</body></html>"
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    local_versions = list_local_model_versions()
    logged_rows = collect_logged_benchmarks()
    case_sets = load_local_benchmark_cases()
    recent_scenes = load_recent_scenes(limit=60)

    model_coverage_df = build_model_coverage_df(local_versions, logged_rows)
    recent_scene_df, uncovered_scene_df = build_scene_coverage(recent_scenes, case_sets)
    summary = build_oob_summary(case_sets, recent_scene_df)
    strict_metrics_df = pd.DataFrame([row for row in logged_rows if row["benchmark_kind"] == "strict_v3"]).drop_duplicates(
        subset=["label"], keep="last"
    )

    summary_path = OUTPUT_DIR / "benchmark_audit_summary_20260324.json"
    uncovered_path = OUTPUT_DIR / "benchmark_uncovered_recent_scenes_20260324.json"
    versions_path = OUTPUT_DIR / "benchmark_model_coverage_20260324.json"
    html_path = OUTPUT_DIR / "benchmark_audit_dashboard_20260324.html"

    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "local_model_version_count": len(local_versions),
                "logged_benchmark_row_count": len(logged_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    uncovered_path.write_text(uncovered_scene_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    versions_path.write_text(model_coverage_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    write_dashboard(
        summary=summary,
        model_coverage_df=model_coverage_df,
        recent_scene_df=recent_scene_df,
        uncovered_scene_df=uncovered_scene_df,
        strict_metrics_df=strict_metrics_df,
        html_path=html_path,
    )

    print(json.dumps({"summary_file": str(summary_path), "dashboard_file": str(html_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
