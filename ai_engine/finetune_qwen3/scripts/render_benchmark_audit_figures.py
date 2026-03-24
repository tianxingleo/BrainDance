#!/usr/bin/env python3
"""Render static benchmark audit figures with matplotlib + seaborn."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "audit"
LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_strict_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted(LOG_DIR.glob("benchmark_strict_v3_*.json")):
        obj = load_json(path)
        metrics = obj.get("metrics") or {}
        if not metrics:
            continue
        rows.append(
            {
                "model": path.stem.replace("benchmark_strict_v3_", ""),
                "false_no_answer_rate": metrics.get("false_no_answer_rate"),
                "partial_hallucination_rate": metrics.get("partial_hallucination_rate"),
                "partial_hit_precision": metrics.get("partial_hit_precision"),
                "partial_missing_negation_rate": metrics.get("partial_missing_negation_rate"),
                "must_answer_focus_rate": metrics.get("must_answer_focus_rate"),
                "natural_style_rate": metrics.get("natural_style_rate"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    summary = load_json(AUDIT_DIR / "benchmark_audit_summary_20260324.json")
    uncovered = pd.DataFrame(load_json(AUDIT_DIR / "benchmark_uncovered_recent_scenes_20260324.json"))
    model_coverage = pd.DataFrame(load_json(AUDIT_DIR / "benchmark_model_coverage_20260324.json"))
    strict_df = collect_strict_metrics()

    # Figure 1: recent scene coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    covered = summary["summary"]["covered_recent_scene_count"]
    recent = summary["summary"]["recent_scene_count"]
    coverage_df = pd.DataFrame(
        {"status": ["covered", "uncovered"], "count": [covered, max(recent - covered, 0)]}
    )
    sns.barplot(
        data=coverage_df,
        x="status",
        y="count",
        hue="status",
        palette=["#2E8B57", "#D95F02"],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Recent Scene Coverage in Current Local Benchmarks")
    ax.set_xlabel("")
    ax.set_ylabel("Scene Count")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "recent_scene_coverage_20260324.png", dpi=180)
    plt.close(fig)

    # Figure 2: model coverage heatmap
    if not model_coverage.empty:
        heat = model_coverage[["version_name", "strict_v3_logged", "original_benchmark_logged"]].copy()
        heat["strict_v3_logged"] = heat["strict_v3_logged"].astype(int)
        heat["original_benchmark_logged"] = heat["original_benchmark_logged"].astype(int)
        heat = heat.set_index("version_name")
        fig, ax = plt.subplots(figsize=(10, max(6, len(heat) * 0.35)))
        sns.heatmap(heat, annot=True, cmap="YlGnBu", cbar=False, linewidths=0.5, ax=ax, fmt="d")
        ax.set_title("Local Model Coverage Heatmap")
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / "model_coverage_heatmap_20260324.png", dpi=180)
        plt.close(fig)

    # Figure 3: strict metric heatmap
    if not strict_df.empty:
        heat = strict_df.set_index("model")
        fig, ax = plt.subplots(figsize=(12, max(6, len(heat) * 0.35)))
        sns.heatmap(heat, annot=True, cmap="mako", linewidths=0.5, ax=ax, fmt=".3f")
        ax.set_title("Strict v3 Metric Heatmap")
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / "strict_metrics_heatmap_20260324.png", dpi=180)
        plt.close(fig)

    # Figure 4: top uncovered recent scenes
    if not uncovered.empty:
        top = uncovered.head(12).copy()
        top["label"] = top["scene_id"]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=top, y="label", x="object_count", hue="label", palette="crest", dodge=False, legend=False, ax=ax)
        ax.set_title("Top Uncovered Recent Scenes by Object Count")
        ax.set_xlabel("Object Count")
        ax.set_ylabel("Scene ID")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / "top_uncovered_recent_scenes_20260324.png", dpi=180)
        plt.close(fig)

    print(json.dumps({"figure_dir": str(AUDIT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
