#!/usr/bin/env python3
"""Render high-density static figures for new benchmark local/cloud comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs"
AUDIT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "audit"
FIG_DIR = AUDIT_DIR / "new_benchmark_figures_20260324"
FIG_DIR.mkdir(parents=True, exist_ok=True)

UNSEEN_LOCAL_SUMMARY = LOG_DIR / "unseen_ood_benchmark_20260324_frozen_all_local_summary.json"
UNSEEN_CLOUD_SUMMARY = LOG_DIR / "cloud_unseen_ood_benchmark_20260324_frozen_summary.json"
SPATIAL_LOCAL_SUMMARY = LOG_DIR / "spatial_hardcases_candidates_20260324_frozen_all_local_summary.json"
SPATIAL_CLOUD_SUMMARY = LOG_DIR / "cloud_spatial_hardcases_20260324_frozen_summary.json"
MANIFEST_PATH = FIG_DIR / "manifest_20260324.json"

MODEL_ORDER_UNSEEN = [
    "1.7B Q5_K_M + imatrix",
    "qwen3-8b",
    "qwen2.5-32b-instruct",
    "1.7B merged",
    "qwen3-32b",
    "0.6B full SFT round2 lr8e-6",
    "1.7B full SFT round1",
    "1.7B LoRA round3",
    "1.7B LoRA round4.1 patch",
    "1.7B LoRA",
    "1.7B Q4_K_M + imatrix",
    "1.7B Q5_K_M",
    "0.6B full SFT round4 patch",
    "1.7B LoRA round4 patch",
    "1.7B Q4_K_M",
    "0.6B full SFT round3 lr5e-6",
    "qwen-turbo",
    "0.6B full SFT round1",
    "0.6B LoRA",
    "1.7B full SFT lr5e-6",
]

SELECTED_MODELS = [
    "1.7B Q5_K_M + imatrix",
    "qwen3-8b",
    "qwen2.5-32b-instruct",
    "1.7B merged",
    "qwen3-32b",
    "1.7B LoRA",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def backend_bucket(label: str, backend: str) -> str:
    if backend == "cloud":
        return "cloud"
    if "imatrix" in label:
        return "quant-imatrix"
    if "Q4_K_M" in label or "Q5_K_M" in label:
        return "quant"
    if "merged" in label:
        return "merged"
    if "LoRA" in label:
        return "lora"
    if "full SFT" in label:
        return "full-sft"
    return backend


def label_short(label: str) -> str:
    mapping = {
        "1.7B Q5_K_M + imatrix": "1.7B Q5+iM",
        "1.7B Q4_K_M + imatrix": "1.7B Q4+iM",
        "1.7B Q5_K_M": "1.7B Q5",
        "1.7B Q4_K_M": "1.7B Q4",
        "1.7B full SFT round1": "1.7B full r1",
        "1.7B full SFT lr5e-6": "1.7B full lr5e-6",
        "1.7B LoRA round4.1 patch": "1.7B LoRA r4.1",
        "1.7B LoRA round4 patch": "1.7B LoRA r4",
        "1.7B LoRA round3": "1.7B LoRA r3",
        "1.7B LoRA": "1.7B LoRA",
        "1.7B merged": "1.7B merged",
        "0.6B full SFT round2 lr8e-6": "0.6B full r2",
        "0.6B full SFT round3 lr5e-6": "0.6B full r3",
        "0.6B full SFT round4 patch": "0.6B full r4",
        "0.6B full SFT round1": "0.6B full r1",
        "0.6B LoRA": "0.6B LoRA",
        "qwen2.5-32b-instruct": "qwen2.5-32b",
        "qwen3-32b": "qwen3-32b",
        "qwen3-8b": "qwen3-8b",
        "qwen-turbo": "qwen-turbo",
    }
    return mapping.get(label, label)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["axes.labelsize"] = 11


def load_unseen_dataframe() -> tuple[pd.DataFrame, dict[str, Any]]:
    local_payload = load_json(UNSEEN_LOCAL_SUMMARY)
    cloud_payload = load_json(UNSEEN_CLOUD_SUMMARY)
    rows = local_payload["summary"] + cloud_payload["summary"]
    df = pd.DataFrame(rows)
    df["backend_bucket"] = [backend_bucket(row["candidate_label"], row["backend"]) for row in rows]
    df["label_short"] = df["candidate_label"].map(label_short)
    df["partial_coverage_pass_rate"] = df["by_group"].map(lambda x: float((x or {}).get("partial_coverage", {}).get("end_to_end_pass_rate", 0.0)))
    df["must_answer_pass_rate"] = df["by_group"].map(lambda x: float((x or {}).get("must_answer", {}).get("end_to_end_pass_rate", 0.0)))
    df["no_hit_pass_rate"] = df["by_group"].map(lambda x: float((x or {}).get("no_hit", {}).get("end_to_end_pass_rate", 0.0)))
    df["efficiency_score"] = df["end_to_end_pass_rate"] / df["avg_total_ms"] * 100000
    order_map = {label: idx for idx, label in enumerate(MODEL_ORDER_UNSEEN)}
    df["plot_order"] = df["candidate_label"].map(lambda x: order_map.get(x, 999))
    df = df.sort_values(["plot_order", "avg_total_ms"], ascending=[True, True]).reset_index(drop=True)
    return df, cloud_payload["retrieval_summary"]


def load_spatial_dataframe() -> pd.DataFrame:
    local_payload = load_json(SPATIAL_LOCAL_SUMMARY)
    cloud_payload = load_json(SPATIAL_CLOUD_SUMMARY)
    rows = local_payload["summary"] + cloud_payload["summary"]
    df = pd.DataFrame(rows)
    df["backend_bucket"] = [backend_bucket(row["candidate_label"], row["backend"]) for row in rows]
    df["label_short"] = df["candidate_label"].map(label_short)
    df["failure_pressure"] = df["generic_scene_summary_rate"] + df["refusal_rate"]
    df = df.sort_values(
        ["spatial_direct_rate", "generic_scene_summary_rate", "avg_total_ms"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return df


def render_unseen_leaderboard(df: pd.DataFrame) -> Path:
    palette = {
        "cloud": "#2F6BFF",
        "quant-imatrix": "#FF8C42",
        "quant": "#F2C14E",
        "merged": "#00A896",
        "lora": "#5E548E",
        "full-sft": "#7F5539",
    }
    chart = df.copy().sort_values(["end_to_end_pass_rate", "answer_pass_rate_when_retrieval_ok", "avg_total_ms"], ascending=[True, True, False])
    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    colors = [palette.get(value, "#888888") for value in chart["backend_bucket"]]
    ax.barh(chart["label_short"], chart["end_to_end_pass_rate"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("新增未见数据 OOD：本地与云端统一榜单")
    ax.set_xlabel("端到端通过率")
    ax.set_ylabel("")
    ax.set_xlim(0, 0.72)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    for idx, row in enumerate(chart.itertuples(index=False)):
        ax.text(
            min(row.end_to_end_pass_rate + 0.008, 0.70),
            idx,
            f"ans@ok {row.answer_pass_rate_when_retrieval_ok:.2f} | partial {row.partial_coverage_pass_rate:.2f} | {row.avg_total_ms:.0f}ms",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#23313f",
        )
    handles = [
        mpl.patches.Patch(color=color, label=label)
        for label, color in [
            ("云端", palette["cloud"]),
            ("量化+imatrix", palette["quant-imatrix"]),
            ("量化", palette["quant"]),
            ("merged", palette["merged"]),
            ("LoRA", palette["lora"]),
            ("full SFT", palette["full-sft"]),
        ]
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, title="模型类型")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "new_benchmark_unseen_leaderboard_20260324.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_unseen_scatter(df: pd.DataFrame, retrieval_summary: dict[str, Any]) -> Path:
    palette = {
        "cloud": "#2F6BFF",
        "quant-imatrix": "#FF8C42",
        "quant": "#F2C14E",
        "merged": "#00A896",
        "lora": "#5E548E",
        "full-sft": "#7F5539",
    }
    fig, ax = plt.subplots(figsize=(11.5, 8.3))
    scatter = sns.scatterplot(
        data=df,
        x="avg_total_ms",
        y="end_to_end_pass_rate",
        hue="backend_bucket",
        size="partial_coverage_pass_rate",
        sizes=(80, 560),
        palette=palette,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title("新增未见数据 OOD：速度、通过率与 partial 覆盖的三维对照")
    ax.set_xlabel("平均总时长 (ms)")
    ax.set_ylabel("端到端通过率")
    ax.axhline(0.6111, color="#C73E1D", linestyle="--", linewidth=1.0)
    ax.text(df["avg_total_ms"].min(), 0.619, "1.7B merged / qwen3-32b 分界线", color="#C73E1D", fontsize=8.5)
    texts = []
    for row in df.itertuples(index=False):
        if row.candidate_label in SELECTED_MODELS or row.end_to_end_pass_rate >= 0.6111 or row.backend == "cloud":
            texts.append(ax.text(row.avg_total_ms, row.end_to_end_pass_rate, row.label_short, fontsize=8.5))
    adjust_text(
        texts,
        ax=ax,
        expand=(1.15, 1.3),
        arrowprops=dict(arrowstyle="-", color="#6c757d", lw=0.6, alpha=0.8),
    )
    ax.text(
        0.015,
        0.02,
        f"Frozen retrieval: ok_rate={retrieval_summary['retrieval_ok_rate']:.4f}, blocked={retrieval_summary['blocked_case_count']}/{retrieval_summary['scoreable_case_count']}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
    )
    legend = scatter.legend_
    if legend is not None:
        legend.set_bbox_to_anchor((1.02, 1.0))
        legend._legend_box.align = "left"
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "new_benchmark_unseen_scatter_20260324.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_unseen_radar(df: pd.DataFrame) -> Path:
    chosen = df[df["candidate_label"].isin(SELECTED_MODELS)].copy()
    chosen["latency_score"] = 1.0 - (chosen["avg_total_ms"] - chosen["avg_total_ms"].min()) / (
        max(chosen["avg_total_ms"].max() - chosen["avg_total_ms"].min(), 1.0)
    )
    metrics = [
        ("end_to_end_pass_rate", "端到端"),
        ("answer_pass_rate_when_retrieval_ok", "回答@检索正确"),
        ("partial_coverage_pass_rate", "Partial"),
        ("must_answer_pass_rate", "Must"),
        ("no_hit_pass_rate", "No-hit"),
        ("latency_score", "时延逆向分"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9.2, 7.8), subplot_kw={"projection": "polar"})
    color_cycle = sns.color_palette("tab10", n_colors=len(chosen))
    for color, row in zip(color_cycle, chosen.itertuples(index=False)):
        values = [float(getattr(row, key)) for key, _ in metrics]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=row.label_short)
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1], [label for _, label in metrics], fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("新增未见数据 OOD：第一梯队模型画像", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.10), frameon=True, title="代表模型")
    fig.tight_layout()
    out = FIG_DIR / "new_benchmark_unseen_radar_20260324.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_spatial_bar(df: pd.DataFrame) -> Path:
    palette = {
        "cloud": "#2F6BFF",
        "quant-imatrix": "#FF8C42",
        "quant": "#F2C14E",
        "merged": "#00A896",
        "lora": "#5E548E",
        "full-sft": "#7F5539",
    }
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    colors = [palette.get(value, "#888888") for value in df["backend_bucket"]]
    ax.barh(df["label_short"], df["spatial_direct_rate"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("空间 hardcases：直接空间回答率全景")
    ax.set_xlabel("spatial_direct_rate")
    ax.set_ylabel("")
    ax.set_xlim(0, 0.12)
    for idx, row in enumerate(df.itertuples(index=False)):
        ax.text(
            min(row.spatial_direct_rate + 0.003, 0.115),
            idx,
            f"summary {row.generic_scene_summary_rate:.2f} | refusal {row.refusal_rate:.2f} | {row.avg_total_ms:.0f}ms",
            va="center",
            ha="left",
            fontsize=8.4,
            color="#243b53",
        )
    ax.axvline(0.0, color="#475569", linewidth=0.8)
    ax.text(0.002, len(df) - 0.75, "除 0.6B full r3 外，其余 19 个模型全部为 0", fontsize=8.7, color="#C2410C")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "new_benchmark_spatial_bar_20260324.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_cross_benchmark_heatmap(unseen_df: pd.DataFrame, spatial_df: pd.DataFrame) -> Path:
    unseen_cols = unseen_df.set_index("candidate_label")
    spatial_cols = spatial_df.set_index("candidate_label")
    rows = []
    for label in SELECTED_MODELS + ["qwen-turbo", "0.6B full SFT round3 lr5e-6"]:
        if label not in unseen_cols.index and label not in spatial_cols.index:
            continue
        row = {
            "模型": label_short(label),
            "OOD端到端": float(unseen_cols.loc[label, "end_to_end_pass_rate"]) if label in unseen_cols.index else np.nan,
            "OOD回答@检索正确": float(unseen_cols.loc[label, "answer_pass_rate_when_retrieval_ok"]) if label in unseen_cols.index else np.nan,
            "OOD Partial": float(unseen_cols.loc[label, "partial_coverage_pass_rate"]) if label in unseen_cols.index else np.nan,
            "空间直答": float(spatial_cols.loc[label, "spatial_direct_rate"]) if label in spatial_cols.index else np.nan,
            "空间摘要化": float(spatial_cols.loc[label, "generic_scene_summary_rate"]) if label in spatial_cols.index else np.nan,
            "空间拒答": float(spatial_cols.loc[label, "refusal_rate"]) if label in spatial_cols.index else np.nan,
        }
        rows.append(row)
    heat = pd.DataFrame(rows).set_index("模型")
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"shrink": 0.85}, ax=ax)
    ax.set_title("跨 benchmark 关键指标热力图")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=28, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "new_benchmark_cross_heatmap_20260324.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_manifest(paths: list[Path]) -> None:
    payload = {
        "figure_dir": str(FIG_DIR),
        "files": [str(path) for path in paths],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    configure_style()
    unseen_df, retrieval_summary = load_unseen_dataframe()
    spatial_df = load_spatial_dataframe()
    outputs = [
        render_unseen_leaderboard(unseen_df),
        render_unseen_scatter(unseen_df, retrieval_summary),
        render_unseen_radar(unseen_df),
        render_spatial_bar(spatial_df),
        render_cross_benchmark_heatmap(unseen_df, spatial_df),
    ]
    build_manifest(outputs)
    print(json.dumps({"figure_dir": str(FIG_DIR), "file_count": len(outputs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
