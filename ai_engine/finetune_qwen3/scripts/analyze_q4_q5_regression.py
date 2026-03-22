#!/usr/bin/env python3
"""Compare two benchmark result files and summarize case-level regressions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRACKED_FLAGS = (
    "false_no_answer",
    "partial_hallucination",
    "partial_false_negative",
    "partial_missing_negation",
    "must_answer_focused",
    "natural_style",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate-vs-baseline regressions for benchmark JSON files")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate benchmark JSON path")
    parser.add_argument("--output_json", default="", help="Optional output JSON path")
    parser.add_argument("--output_md", default="", help="Optional output Markdown path")
    parser.add_argument("--max_examples", type=int, default=12, help="Max example cases to keep")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in payload.get("results", [])}


def metric_delta(baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    keys = [
        "false_no_answer_rate",
        "partial_hallucination_rate",
        "partial_hit_precision",
        "partial_false_negative_rate",
        "partial_missing_negation_rate",
        "must_answer_focus_rate",
        "natural_style_rate",
    ]
    deltas: dict[str, dict[str, float]] = {}
    for key in keys:
        base = float(baseline_metrics.get(key, 0.0))
        cand = float(candidate_metrics.get(key, 0.0))
        deltas[key] = {"baseline": round(base, 4), "candidate": round(cand, 4), "delta": round(cand - base, 4)}
    return deltas


def summarize_cases(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    max_examples: int,
) -> dict[str, Any]:
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    by_group_counter: Counter[str] = Counter()
    by_flag_counter: Counter[str] = Counter()
    group_flag_counter: dict[str, Counter[str]] = defaultdict(Counter)

    shared_case_ids = sorted(set(baseline_results) & set(candidate_results))
    for case_id in shared_case_ids:
        baseline = baseline_results[case_id]
        candidate = candidate_results[case_id]
        base_analysis = baseline["analysis"]
        cand_analysis = candidate["analysis"]
        group = str(candidate.get("group") or baseline.get("group") or "unknown")

        changed_flags: list[dict[str, Any]] = []
        for flag in TRACKED_FLAGS:
            base_value = bool(base_analysis.get(flag, False))
            cand_value = bool(cand_analysis.get(flag, False))
            if base_value == cand_value:
                continue
            direction = "improved"
            if flag == "must_answer_focused" or flag == "natural_style":
                direction = "regressed" if base_value and not cand_value else "improved"
            else:
                direction = "regressed" if (not base_value and cand_value) else "improved"
            changed_flags.append(
                {
                    "flag": flag,
                    "baseline": base_value,
                    "candidate": cand_value,
                    "direction": direction,
                }
            )

        if not changed_flags:
            continue

        record = {
            "case_id": case_id,
            "group": group,
            "metadata": candidate.get("metadata") or baseline.get("metadata") or {},
            "baseline_answer": baseline.get("answer", ""),
            "candidate_answer": candidate.get("answer", ""),
            "changed_flags": changed_flags,
        }

        if any(item["direction"] == "regressed" for item in changed_flags):
            regressions.append(record)
            by_group_counter[group] += 1
            for item in changed_flags:
                if item["direction"] == "regressed":
                    by_flag_counter[item["flag"]] += 1
                    group_flag_counter[group][item["flag"]] += 1
        elif len(improvements) < max_examples:
            improvements.append(record)

    regressions.sort(
        key=lambda item: (
            -sum(1 for flag in item["changed_flags"] if flag["direction"] == "regressed"),
            item["group"],
            item["case_id"],
        )
    )
    improvements.sort(key=lambda item: (item["group"], item["case_id"]))

    return {
        "shared_cases": len(shared_case_ids),
        "regression_case_count": len(regressions),
        "improvement_case_count": len(improvements),
        "by_group": dict(sorted(by_group_counter.items())),
        "by_flag": dict(sorted(by_flag_counter.items())),
        "by_group_flag": {
            group: dict(sorted(counter.items()))
            for group, counter in sorted(group_flag_counter.items())
        },
        "regressions": regressions[:max_examples],
        "improvements": improvements[:max_examples],
    }


def build_markdown(summary: dict[str, Any], baseline_path: Path, candidate_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Q4 vs Q5 回退分析")
    lines.append("")
    lines.append("## 输入")
    lines.append("")
    lines.append(f"- baseline: `{baseline_path}`")
    lines.append(f"- candidate: `{candidate_path}`")
    lines.append("")
    lines.append("## 指标差异")
    lines.append("")
    lines.append("| 指标 | baseline | candidate | delta |")
    lines.append("|---|---:|---:|---:|")
    for key, payload in summary["metric_delta"].items():
        lines.append(f"| {key} | {payload['baseline']:.4f} | {payload['candidate']:.4f} | {payload['delta']:+.4f} |")
    lines.append("")
    lines.append("## 案例级总结")
    lines.append("")
    lines.append(f"- 共享 case 数：`{summary['case_summary']['shared_cases']}`")
    lines.append(f"- 回退 case 数：`{summary['case_summary']['regression_case_count']}`")
    lines.append(f"- 改善 case 样本数：`{summary['case_summary']['improvement_case_count']}`")
    lines.append(f"- 回退按 group 分布：`{json.dumps(summary['case_summary']['by_group'], ensure_ascii=False)}`")
    lines.append(f"- 回退按 flag 分布：`{json.dumps(summary['case_summary']['by_flag'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## 回退样例")
    lines.append("")
    if not summary["case_summary"]["regressions"]:
        lines.append("- 无")
    else:
        for item in summary["case_summary"]["regressions"]:
            changed = ", ".join(
                f"{flag['flag']}:{flag['baseline']}->{flag['candidate']}"
                for flag in item["changed_flags"]
                if flag["direction"] == "regressed"
            )
            unsupported = item["metadata"].get("unsupported_objects") or []
            supported = item["metadata"].get("supported_objects") or []
            lines.append(f"### {item['case_id']}")
            lines.append("")
            lines.append(f"- group: `{item['group']}`")
            lines.append(f"- supported: `{supported}`")
            lines.append(f"- unsupported: `{unsupported}`")
            lines.append(f"- regressions: `{changed}`")
            lines.append(f"- baseline: `{item['baseline_answer']}`")
            lines.append(f"- candidate: `{item['candidate_answer']}`")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline_payload = load_payload(baseline_path)
    candidate_payload = load_payload(candidate_path)

    summary = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "metric_delta": metric_delta(baseline_payload["metrics"], candidate_payload["metrics"]),
        "case_summary": summarize_cases(
            build_result_map(baseline_payload),
            build_result_map(candidate_payload),
            max_examples=args.max_examples,
        ),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(build_markdown(summary, baseline_path, candidate_path), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
