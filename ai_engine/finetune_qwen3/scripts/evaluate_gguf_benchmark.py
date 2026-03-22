#!/usr/bin/env python3
"""Evaluate a GGUF model with llama.cpp on the fixed BrainDance benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from evaluate_benchmark import aggregate, evaluate_case, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GGUF model on BrainDance benchmark")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--llama_cli_path", default="ai_engine/finetune_qwen3/tools/llama.cpp/build/bin/llama-cli")
    parser.add_argument("--benchmark_file", default="ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl")
    parser.add_argument("--output_file", default="")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--ctx_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ubatch_size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--device", default="")
    parser.add_argument("--gpu_layers", default="all")
    parser.add_argument("--main_gpu", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_chat_parts(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_prompt = ""
    user_prompt = ""
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content
    return system_prompt, user_prompt


def detect_llama_devices(llama_cli_path: str) -> list[str]:
    command = [llama_cli_path, "--list-devices"]
    completed = subprocess.run(command, capture_output=True, check=True)
    stdout = completed.stdout.decode("utf-8", errors="replace")
    lines = [line.strip() for line in stdout.splitlines()]
    return [line for line in lines if line and line != "Available devices:"]


def extract_answer(stdout: str) -> str:
    text = stdout.replace("\r\n", "\n")
    if "\n> " in text:
        text = text.split("\n> ", 1)[1]
        if "\n\n" in text:
            text = text.split("\n\n", 1)[1]
    for marker in (
        "llama_perf_sampler_print:",
        "llama_memory_breakdown_print:",
        "\n\nExiting...",
        "\nExiting...",
    ):
        if marker in text:
            text = text.split(marker, 1)[0]
    return normalize_text(text)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_command(args: argparse.Namespace, system_prompt: str, user_prompt: str, use_device: bool) -> list[str]:
    command = [
        args.llama_cli_path,
        "--model",
        args.model_path,
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--threads",
        str(args.threads),
        "--single-turn",
        "--conversation",
        "--simple-io",
        "--no-display-prompt",
        "--no-show-timings",
        "--no-warmup",
        "--reasoning",
        "off",
        "--system-prompt",
        system_prompt,
        "--prompt",
        user_prompt,
        "--predict",
        str(args.max_new_tokens),
    ]
    if use_device and args.device:
        command.extend(["--device", args.device, "--main-gpu", str(args.main_gpu), "--gpu-layers", args.gpu_layers])
    return command


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.benchmark_file))
    detected_devices = detect_llama_devices(args.llama_cli_path)
    requested_device = args.device.strip()
    use_device = bool(requested_device and detected_devices)
    runtime_backend = "llama.cpp_gpu" if use_device else "llama.cpp_cpu"
    backend_note = (
        f"detected devices: {detected_devices}"
        if detected_devices
        else "llama.cpp 当前构建未检测到可用加速设备，已回退 CPU-only 评测"
    )

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, row in enumerate(rows, start=1):
        system_prompt, user_prompt = extract_chat_parts(row.get("messages") or [])
        command = build_command(args, system_prompt, user_prompt, use_device=use_device)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=args.timeout,
            check=True,
        )
        latency = time.perf_counter() - started
        latencies.append(latency)
        stdout = completed.stdout.decode("utf-8", errors="replace")
        answer = extract_answer(stdout)
        analysis = evaluate_case(answer, row)
        results.append(
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "answer": answer,
                "reference_answer": row["reference_answer"],
                "metadata": row["metadata"],
                "analysis": analysis,
                "latency_seconds": round(latency, 4),
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"[gguf-benchmark] {index}/{len(rows)}")

    latency_summary = {
        "count": len(latencies),
        "mean_seconds": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "median_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
        "p95_seconds": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "max_seconds": round(max(latencies), 4) if latencies else 0.0,
    }

    payload = {
        "model_path": args.model_path,
        "llama_cli_path": args.llama_cli_path,
        "benchmark_file": args.benchmark_file,
        "runtime_backend": runtime_backend,
        "requested_device": requested_device,
        "detected_devices": detected_devices,
        "backend_note": backend_note,
        "metrics": aggregate(results),
        "latency_summary": latency_summary,
        "results": results,
    }

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"metrics": payload["metrics"], "latency_summary": latency_summary, "backend_note": backend_note}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
