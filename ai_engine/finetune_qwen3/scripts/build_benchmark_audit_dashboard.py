#!/usr/bin/env python3
"""Audit benchmark coverage, local model coverage, and build a Plotly dashboard."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data"
LOG_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs"
OUTPUT_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_KINDS = (
    "original_benchmark",
    "strict_v3",
    "unseen_ood",
    "spatial_hardcases",
    "deployment_eval",
)

LOG_PREFIX_TO_KIND = {
    "benchmark_qwen3_": "original_benchmark",
    "benchmark_strict_v3_": "strict_v3",
    "unseen_ood_benchmark_": "unseen_ood",
    "spatial_hardcases_candidates_": "spatial_hardcases",
    "deployment_eval_part29_": "deployment_eval",
}

VERSION_MATCHES = {
    "qwen3_0p6b_full_sft_round1": {
        "labels": {"0.6B full SFT round1", "0.6B full SFT"},
        "file_tokens": {"qwen3_0p6b_full_round1_gpu1", "qwen3_0p6b_full_gpu1"},
    },
    "qwen3_0p6b_full_sft_round2_lr8e6": {"labels": {"0.6B full SFT round2 lr8e-6"}},
    "qwen3_0p6b_full_sft_round3_lr5e6": {"labels": {"0.6B full SFT round3 lr5e-6"}},
    "qwen3_0p6b_full_sft_round4_partial_patch_v1": {"labels": {"0.6B full SFT round4 partial patch"}},
    "qwen3_0p6b_lora_sft_round1": {"labels": {"0.6B LoRA"}, "file_tokens": {"qwen3_0p6b_round1_gpu0", "qwen3_0p6b_round1_gpu1"}},
    "qwen3_0p6b_braindance_round1": {"labels": {"0.6B LoRA"}, "file_tokens": {"qwen3_0p6b_round1_gpu0", "qwen3_0p6b_round1_gpu1"}},
    "qwen3_1p7b_full_sft_round1_gpu1": {
        "labels": {"1.7B full SFT round1"},
        "file_tokens": {"qwen3_1p7b_full_round1_gpu1", "qwen3_1p7b_full_gpu1"},
    },
    "qwen3_1p7b_full_sft_round1_lr5e6_gpu1": {"labels": {"1.7B full SFT lr5e-6"}},
    "qwen3_1p7b_lora_sft_round3": {"labels": {"1.7B LoRA round3"}},
    "qwen3_1p7b_lora_sft_round4_patch": {"labels": {"1.7B LoRA round4 patch"}},
    "qwen3_1p7b_lora_sft_round4_1_patch": {"labels": {"1.7B LoRA round4.1 patch"}},
    "qwen3_1p7b_lora_sft_round4_1_patch_mixed": {
        "labels": {"1.7B LoRA", "1.7B LoRA round4.1 mixed"},
        "file_tokens": {"round4_1_patch_mixed", "benchmark_strict_v3_lora_20260322.json"},
    },
    "qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0": {"labels": {"1.7B merged"}},
    "qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0": {
        "labels": {"1.7B Q4_K_M", "1.7B Q5_K_M", "1.7B Q4_K_M + imatrix", "1.7B Q5_K_M + imatrix"}
    },
    "qwen3_1p7b_lora_sft_smoke": {"labels": set()},
    "qwen3_1p7b_lora_sft_smoke_v2": {"labels": set()},
    "qwen3_0p6b_full_sft_smoke_gpu1": {"labels": set()},
    "qwen3_1p7b_full_sft_smoke_gpu1": {"labels": set()},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def infer_label_from_name(name: str) -> str:
    mapping = [
        ("qwen3_1p7b_lora_sft_round4_1_patch_mixed", "1.7B LoRA"),
        ("qwen3_1p7b_lora_sft_round4_1_patch", "1.7B LoRA round4.1 patch"),
        ("qwen3_1p7b_lora_sft_round4_patch", "1.7B LoRA round4 patch"),
        ("qwen3_1p7b_lora_sft_round3", "1.7B LoRA round3"),
        ("qwen3_0p6b_lora", "0.6B LoRA"),
        ("qwen3_0p6b_braindance_round1", "0.6B LoRA"),
        ("qwen3_0p6b_full_round4_partial_patch_v1", "0.6B full SFT round4 partial patch"),
        ("qwen3_0p6b_full_round3_lr5e6", "0.6B full SFT round3 lr5e-6"),
        ("qwen3_0p6b_full_round2_lr8e6", "0.6B full SFT round2 lr8e-6"),
        ("qwen3_0p6b_full_round1", "0.6B full SFT round1"),
        ("qwen3_1p7b_full_round1_lr5e6", "1.7B full SFT lr5e-6"),
        ("qwen3_1p7b_full_round1", "1.7B full SFT round1"),
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


def classify_version(name: str) -> str:
    if "smoke" in name:
        return "smoke"
    if "quantized" in name or "gguf" in name:
        return "quantized"
    if "merged" in name:
        return "merged"
    if "_lora_" in name or name.endswith("_lora"):
        return "lora"
    if "full_sft" in name:
        return "full_sft"
    return "other"


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
                    "label": infer_label_from_name(name),
                    "root": root_name,
                    "path": str(path),
                    "version_type": classify_version(name),
                    "is_smoke_only": "smoke" in name,
                }
            )
    return rows


def infer_benchmark_kind(name: str) -> str | None:
    for prefix, kind in LOG_PREFIX_TO_KIND.items():
        if name.startswith(prefix):
            return kind
    return None


def extract_summary_rows(name: str, payload: dict[str, Any], benchmark_kind: str) -> list[dict[str, Any]]:
    summary = payload.get("summary")
    if not isinstance(summary, list):
        return []

    rows: list[dict[str, Any]] = []
    multi_candidate = any(isinstance(item, dict) and item.get("candidate_label") for item in summary)
    for item in summary:
        if not isinstance(item, dict):
            continue
        label = str(item.get("candidate_label") or payload.get("candidate_label") or infer_label_from_name(name))
        row = {
            "log_file": name,
            "benchmark_kind": benchmark_kind,
            "label": label,
            "candidate_id": str(item.get("candidate_id") or ""),
            "end_to_end_pass_rate": item.get("end_to_end_pass_rate"),
            "answer_pass_rate_when_retrieval_ok": item.get("answer_pass_rate_when_retrieval_ok"),
            "retrieval_block_rate": item.get("retrieval_block_rate"),
            "spatial_direct_rate": item.get("spatial_direct_rate"),
            "refusal_rate": item.get("refusal_rate"),
            "generic_scene_summary_rate": item.get("generic_scene_summary_rate"),
            "avg_total_ms": item.get("avg_total_ms"),
            "avg_peak_vram_mb": item.get("avg_peak_vram_mb"),
            "avg_output_chars": item.get("avg_output_chars"),
            "formatter_case_count": item.get("formatter_case_count"),
            "model_generated_case_count": item.get("model_generated_case_count"),
        }
        if multi_candidate or any(value is not None for key, value in row.items() if key.endswith("_rate")):
            rows.append(row)
    return rows


def collect_logged_benchmarks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LOG_DIR.glob("*.json")):
        name = path.name
        payload = safe_read_json(path)
        if not isinstance(payload, dict):
            continue

        benchmark_kind = infer_benchmark_kind(name)
        if benchmark_kind is None:
            continue

        summary_rows = extract_summary_rows(name, payload, benchmark_kind)
        if summary_rows:
            rows.extend(summary_rows)
            continue

        metrics = payload.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "log_file": name,
                "benchmark_kind": benchmark_kind,
                "label": infer_label_from_name(name),
                "candidate_id": "",
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
    return rest_select_model_assets(supabase_url, supabase_key, limit=limit)


def load_local_benchmark_cases() -> dict[str, list[dict[str, Any]]]:
    files = {
        "unseen_ood": DATA_DIR / "braindance_qwen3_unseen_ood_benchmark_20260324.json",
        "spatial_hardcases": DATA_DIR / "braindance_qwen3_unseen_ood_spatial_hardcases_20260324.json",
        "expansion_candidates": DATA_DIR / "braindance_qwen3_unseen_ood_expansion_candidates_20260324_local.json",
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for key, path in files.items():
        payload = safe_read_json(path)
        if isinstance(payload, list):
            out[key] = payload
    return out


def build_scene_coverage(
    recent_scenes: list[dict[str, Any]],
    case_sets: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    covered_scene_ids = set()
    for key in ("unseen_ood", "spatial_hardcases"):
        for row in case_sets.get(key, []):
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


def version_matches_log(version_name: str, label: str, log_file: str) -> bool:
    spec = VERSION_MATCHES.get(version_name, {"labels": {infer_label_from_name(version_name)}})
    labels = set(spec.get("labels") or [])
    if labels and label in labels:
        return True
    for token in spec.get("file_tokens") or set():
        if token in log_file:
            return True
    return version_name in log_file


def build_model_coverage_df(local_versions: list[dict[str, Any]], logged_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in local_versions:
        coverage = {kind: False for kind in EVAL_KINDS}
        matched_logs: dict[str, list[str]] = {kind: [] for kind in EVAL_KINDS}
        for logged in logged_rows:
            kind = str(logged["benchmark_kind"])
            if kind not in coverage:
                continue
            if version_matches_log(item["version_name"], str(logged["label"]), str(logged["log_file"])):
                coverage[kind] = True
                matched_logs[kind].append(str(logged["log_file"]))

        row = {
            **item,
            **{f"{kind}_logged": coverage[kind] for kind in EVAL_KINDS},
            "major_eval_count": sum(int(value) for value in coverage.values()),
            "matched_logs": {kind: sorted(set(names)) for kind, names in matched_logs.items() if names},
        }
        row["missing_major_eval_kinds"] = [kind for kind, enabled in coverage.items() if not enabled]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_spatial_generation_stats() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ("spatial_hardcases_candidates_20260324_results.json", "spatial_hardcases_candidates_20260324_frozen_results.json"):
        path = LOG_DIR / name
        payload = safe_read_json(path)
        if not isinstance(payload, list):
            continue
        by_label: dict[str, list[dict[str, Any]]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("candidate_label") or "")
            if label:
                by_label.setdefault(label, []).append(item)
        mode = "frozen" if "frozen" in name else "live"
        for label, items in by_label.items():
            generated = [row for row in items if float(row.get("generation_latency_ms") or 0.0) > 0.0]
            rows.append(
                {
                    "label": label,
                    "mode": mode,
                    "case_count": len(items),
                    "model_generated_rate": round(len(generated) / len(items), 4) if items else 0.0,
                    "formatter_bypass_rate": round(1 - (len(generated) / len(items)), 4) if items else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_summary(
    case_sets: dict[str, list[dict[str, Any]]],
    recent_scene_df: pd.DataFrame,
    model_coverage_df: pd.DataFrame,
) -> dict[str, Any]:
    unseen_rows = case_sets.get("unseen_ood", [])
    spatial_rows = case_sets.get("spatial_hardcases", [])
    expansion_rows = case_sets.get("expansion_candidates", [])
    all_covered = {
        str(scene_id)
        for row in unseen_rows + spatial_rows
        for scene_id in (row.get("source_scene_ids") or [])
    }
    expandable_scene_ids = {
        str(scene_id)
        for row in expansion_rows
        for scene_id in (row.get("source_scene_ids") or [])
    }
    missing_rows = model_coverage_df[model_coverage_df["major_eval_count"] < len(EVAL_KINDS)] if not model_coverage_df.empty else pd.DataFrame()
    non_smoke_missing = missing_rows[missing_rows["is_smoke_only"] == False] if not missing_rows.empty else pd.DataFrame()
    summary = {
        "recent_scene_count": int(len(recent_scene_df)),
        "covered_recent_scene_count": int(recent_scene_df["covered_by_current_local_benchmarks"].sum()) if not recent_scene_df.empty else 0,
        "uncovered_recent_scene_count": int((~recent_scene_df["covered_by_current_local_benchmarks"]).sum()) if not recent_scene_df.empty else 0,
        "recent_scene_coverage_rate": round(float(recent_scene_df["covered_by_current_local_benchmarks"].mean()), 4)
        if not recent_scene_df.empty
        else 0.0,
        "unseen_ood_case_count": len(unseen_rows),
        "spatial_hardcase_count": len(spatial_rows),
        "expansion_candidate_case_count": len(expansion_rows),
        "expansion_candidate_scene_count": len(expandable_scene_ids),
        "unique_scene_ids_in_local_benchmarks": len(all_covered),
        "local_model_version_count": int(len(model_coverage_df)),
        "smoke_version_count": int(model_coverage_df["is_smoke_only"].sum()) if not model_coverage_df.empty else 0,
        "fully_covered_model_count": int((model_coverage_df["major_eval_count"] == len(EVAL_KINDS)).sum()) if not model_coverage_df.empty else 0,
        "non_smoke_missing_eval_count": int(len(non_smoke_missing)),
        "version_type_breakdown": Counter(model_coverage_df["version_type"]).copy() if not model_coverage_df.empty else Counter(),
        "missing_non_smoke_versions": non_smoke_missing["version_name"].tolist() if not non_smoke_missing.empty else [],
    }
    summary["version_type_breakdown"] = dict(summary["version_type_breakdown"])
    return summary


def build_highlight_tables(
    *,
    strict_metrics_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    deployment_df: pd.DataFrame,
) -> str:
    blocks: list[str] = []
    if not strict_metrics_df.empty:
        cols = [
            "label",
            "false_no_answer_rate",
            "partial_hallucination_rate",
            "partial_hit_precision",
            "partial_missing_negation_rate",
            "must_answer_focus_rate",
            "natural_style_rate",
        ]
        blocks.append("<h2>Strict v3 关键指标</h2>" + strict_metrics_df[cols].sort_values("partial_hit_precision", ascending=False).to_html(index=False))
    if not ood_df.empty:
        cols = ["label", "end_to_end_pass_rate", "answer_pass_rate_when_retrieval_ok", "retrieval_block_rate", "avg_total_ms"]
        blocks.append("<h2>未见 OOD 结果</h2>" + ood_df[cols].sort_values("end_to_end_pass_rate", ascending=False).to_html(index=False))
    if not spatial_df.empty:
        cols = ["label", "spatial_direct_rate", "generic_scene_summary_rate", "avg_total_ms"]
        blocks.append("<h2>Spatial Hardcase 结果</h2>" + spatial_df[cols].sort_values("spatial_direct_rate", ascending=False).to_html(index=False))
    if not deployment_df.empty:
        cols = ["label", "avg_total_ms", "avg_peak_vram_mb", "avg_output_chars"]
        blocks.append("<h2>部署候选耗时与资源</h2>" + deployment_df[cols].sort_values("avg_total_ms", ascending=True).to_html(index=False))
    return "\n".join(blocks)


def write_dashboard(
    *,
    summary: dict[str, Any],
    model_coverage_df: pd.DataFrame,
    recent_scene_df: pd.DataFrame,
    uncovered_scene_df: pd.DataFrame,
    strict_metrics_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    spatial_generation_df: pd.DataFrame,
    deployment_df: pd.DataFrame,
    html_path: Path,
) -> None:
    figs: list[str] = []

    if not model_coverage_df.empty:
        heat = model_coverage_df[["version_name", *[f"{kind}_logged" for kind in EVAL_KINDS]]].copy()
        rename_map = {f"{kind}_logged": kind for kind in EVAL_KINDS}
        heat = heat.rename(columns=rename_map).set_index("version_name").astype(int)
        fig = px.imshow(
            heat,
            color_continuous_scale=["#F2F2F2", "#0F766E"],
            aspect="auto",
            text_auto=True,
            title="本地版本评测覆盖矩阵",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    if not recent_scene_df.empty:
        coverage_counts = recent_scene_df["covered_by_current_local_benchmarks"].value_counts().rename(index={True: "已覆盖", False: "未覆盖"})
        fig = px.bar(
            x=list(coverage_counts.index),
            y=list(coverage_counts.values),
            text=list(coverage_counts.values),
            title="最近 Supabase 场景覆盖情况",
            labels={"x": "覆盖状态", "y": "场景数"},
            color=list(coverage_counts.index),
            color_discrete_map={"已覆盖": "#0F766E", "未覆盖": "#D97706"},
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
        fig = px.line(melted, x="metric", y="value", color="label", markers=True, title="Strict v3 各版本关键指标")
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not ood_df.empty:
        fig = px.bar(
            ood_df.sort_values("end_to_end_pass_rate", ascending=False),
            x="label",
            y=["end_to_end_pass_rate", "answer_pass_rate_when_retrieval_ok"],
            barmode="group",
            title="未见 OOD：端到端与 retrieval-ok 后回答能力",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not spatial_df.empty:
        fig = px.bar(
            spatial_df.sort_values("generic_scene_summary_rate", ascending=False),
            x="label",
            y=["spatial_direct_rate", "generic_scene_summary_rate"],
            barmode="group",
            title="Spatial Hardcase：空间直接回答率 vs 泛化总结率",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not spatial_generation_df.empty:
        fig = px.bar(
            spatial_generation_df.sort_values(["mode", "formatter_bypass_rate"], ascending=[True, False]),
            x="label",
            y=["formatter_bypass_rate", "model_generated_rate"],
            facet_col="mode",
            barmode="group",
            title="Spatial Hardcase：formatter 旁路率与模型生成率",
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not deployment_df.empty:
        fig = px.scatter(
            deployment_df,
            x="avg_total_ms",
            y="avg_peak_vram_mb",
            size="avg_output_chars",
            color="label",
            title="部署候选：延迟、显存、输出长度",
            hover_data=["label"],
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    summary_html = f"""
    <h1>Qwen3 Benchmark 审计总览</h1>
    <p>最近场景数：{summary['recent_scene_count']} | 已覆盖：{summary['covered_recent_scene_count']} | 未覆盖：{summary['uncovered_recent_scene_count']} | 覆盖率：{summary['recent_scene_coverage_rate']:.2%}</p>
    <p>未见 OOD：{summary['unseen_ood_case_count']} | spatial hardcase：{summary['spatial_hardcase_count']} | 自动扩容候选：{summary['expansion_candidate_case_count']} | 当前 benchmark 唯一 scene_id：{summary['unique_scene_ids_in_local_benchmarks']}</p>
    <p>本地版本：{summary['local_model_version_count']} | smoke 版本：{summary['smoke_version_count']} | 五类主评测全覆盖版本：{summary['fully_covered_model_count']} | 非 smoke 但未全覆盖版本：{summary['non_smoke_missing_eval_count']}</p>
    <p>版本类型分布：{json.dumps(summary['version_type_breakdown'], ensure_ascii=False)}</p>
    <p>未全覆盖的非 smoke 版本：{", ".join(summary['missing_non_smoke_versions']) or "无"}</p>
    """

    uncovered_table = ""
    if not uncovered_scene_df.empty:
        cols = ["created_at", "scene_id", "object_count", "objects_preview", "tags_preview"]
        uncovered_table = "<h2>最近但尚未纳入 benchmark 的场景（前 20）</h2>" + uncovered_scene_df.head(20)[cols].to_html(index=False)

    detail_tables = build_highlight_tables(
        strict_metrics_df=strict_metrics_df,
        ood_df=ood_df,
        spatial_df=spatial_df,
        deployment_df=deployment_df,
    )

    html = "<html><head><meta charset='utf-8'><title>Qwen3 Benchmark Audit Dashboard</title></head><body>"
    html += summary_html
    html += uncovered_table
    html += detail_tables
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
    summary = build_summary(case_sets, recent_scene_df, model_coverage_df)
    logged_df = pd.DataFrame(logged_rows)
    spatial_generation_df = compute_spatial_generation_stats()

    strict_metrics_df = pd.DataFrame([row for row in logged_rows if row["benchmark_kind"] == "strict_v3"]).drop_duplicates(
        subset=["label"], keep="last"
    )
    ood_df = logged_df[logged_df["benchmark_kind"] == "unseen_ood"].drop_duplicates(subset=["label"], keep="last") if not logged_df.empty else pd.DataFrame()
    spatial_df = (
        logged_df[logged_df["benchmark_kind"] == "spatial_hardcases"].drop_duplicates(subset=["label"], keep="last")
        if not logged_df.empty
        else pd.DataFrame()
    )
    deployment_df = (
        logged_df[logged_df["benchmark_kind"] == "deployment_eval"].drop_duplicates(subset=["label"], keep="last")
        if not logged_df.empty
        else pd.DataFrame()
    )

    summary_path = OUTPUT_DIR / "benchmark_audit_summary_20260324.json"
    uncovered_path = OUTPUT_DIR / "benchmark_uncovered_recent_scenes_20260324.json"
    versions_path = OUTPUT_DIR / "benchmark_model_coverage_20260324.json"
    html_path = OUTPUT_DIR / "benchmark_audit_dashboard_20260324.html"
    spatial_generation_path = OUTPUT_DIR / "spatial_generation_audit_20260324.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    uncovered_path.write_text(uncovered_scene_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    versions_path.write_text(model_coverage_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    spatial_generation_path.write_text(
        spatial_generation_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dashboard(
        summary=summary,
        model_coverage_df=model_coverage_df,
        recent_scene_df=recent_scene_df,
        uncovered_scene_df=uncovered_scene_df,
        strict_metrics_df=strict_metrics_df,
        ood_df=ood_df,
        spatial_df=spatial_df,
        spatial_generation_df=spatial_generation_df,
        deployment_df=deployment_df,
        html_path=html_path,
    )

    print(
        json.dumps(
            {
                "summary_file": str(summary_path),
                "dashboard_file": str(html_path),
                "spatial_generation_file": str(spatial_generation_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
