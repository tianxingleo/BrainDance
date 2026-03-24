#!/usr/bin/env python3
"""Deployment-oriented small-sample evaluation for three local candidates."""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_gguf_benchmark import extract_answer
from run_real_chain_debug import (
    DEFAULT_DASHSCOPE_BASE_URL,
    SYSTEM_PROMPT,
    apply_chat,
    extract_dashscope_key,
    extract_supabase_config,
    generate_answer,
    load_generator,
    now_utc,
    retrieve_real_chain_case,
    unload_model,
)

try:
    import torch
    from transformers import TextIteratorStreamer
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    TextIteratorStreamer = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "data" / "braindance_qwen3_deployment_eval_part29.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "deployment_eval_part29_results.json"
)
DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "logs" / "deployment_eval_part29_summary.json"
)
DEFAULT_LORA_0P6B = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "releases" / "qwen3_0p6b_braindance_round1"
)
DEFAULT_MERGED_1P7B = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "releases" / "qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0"
)
DEFAULT_Q5_IMATRIX = (
    PROJECT_ROOT
    / "ai_engine"
    / "finetune_qwen3"
    / "releases"
    / "qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0"
    / "imatrix_v1"
    / "model-f16-q5_k_m-imatrix.gguf"
)
DEFAULT_LLAMA_CLI = (
    PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "tools" / "llama.cpp" / "build-cuda" / "bin" / "llama-cli"
)


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    label: str
    backend: str
    model_name: str = ""
    adapter_path: str = ""
    gguf_model_path: str = ""


CANDIDATES = (
    CandidateConfig(
        candidate_id="qwen3_0p6b_lora",
        label="0.6B LoRA",
        backend="hf",
        model_name="Qwen/Qwen3-0.6B",
        adapter_path=str(DEFAULT_LORA_0P6B),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_merged",
        label="1.7B merged",
        backend="hf",
        model_name=str(DEFAULT_MERGED_1P7B),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_q5_imatrix",
        label="1.7B Q5_K_M + imatrix",
        backend="gguf",
        gguf_model_path=str(DEFAULT_Q5_IMATRIX),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deployment candidates on a small real-chain set")
    parser.add_argument("--cases_file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--summary_file", default=str(DEFAULT_SUMMARY_FILE))
    parser.add_argument("--llama_cli_path", default=str(DEFAULT_LLAMA_CLI))
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--gpu_index", type=int, default=1, help="Physical GPU index for nvidia-smi sampling")
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--match_count", type=int, default=5)
    parser.add_argument("--recent_limit", type=int, default=3)
    parser.add_argument("--dashscope_chat_model", default="qwen-turbo")
    parser.add_argument("--dashscope_embedding_model", default="text-embedding-v2")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ubatch_size", type=int, default=64)
    parser.add_argument("--ctx_size", type=int, default=2048)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_proc_rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024, 2)
    except FileNotFoundError:
        return 0.0
    return 0.0


def read_pid_gpu_memory_mb(pid: int, gpu_index: int) -> float:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return 0.0

    gpu_uuid = gpu_uuid_for_index(gpu_index)
    if not gpu_uuid:
        return 0.0

    peak = 0.0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        row_gpu_uuid, row_pid, used_mb = parts
        if row_gpu_uuid != gpu_uuid:
            continue
        if row_pid != str(pid):
            continue
        try:
            peak = max(peak, float(used_mb))
        except ValueError:
            pass
    return peak


def gpu_uuid_for_index(gpu_index: int) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        if parts[0] == str(gpu_index):
            return parts[1]
    return ""


class PeakSampler:
    def __init__(self, pid: int, gpu_index: int, interval_sec: float = 0.05) -> None:
        self.pid = pid
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_mb = 0.0
        self.peak_vram_mb = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_mb = max(self.peak_rss_mb, read_proc_rss_mb(self.pid))
            self.peak_vram_mb = max(self.peak_vram_mb, read_pid_gpu_memory_mb(self.pid, self.gpu_index))
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def call_retrieval(question: str, args: argparse.Namespace, dashscope_key: str, supabase_url: str, supabase_key: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    chain = retrieve_real_chain_case(
        question=question,
        dashscope_key=dashscope_key,
        dashscope_base_url=DEFAULT_DASHSCOPE_BASE_URL,
        chat_model=args.dashscope_chat_model,
        embedding_model=args.dashscope_embedding_model,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        match_threshold=args.match_threshold,
        match_count=args.match_count,
        recent_limit=args.recent_limit,
    )
    return chain, round((time.perf_counter() - started) * 1000, 1)


def run_hf_answer(
    *,
    question: str,
    retrieval: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
    gpu_index: int,
) -> tuple[str, float, float, float, float]:
    special_answer = str(retrieval.get("special_answer") or "").strip()
    if special_answer:
        return special_answer, 0.0, 0.0, read_proc_rss_mb(os.getpid()), read_pid_gpu_memory_mb(os.getpid(), gpu_index)

    if torch is None or TextIteratorStreamer is None:
        raise RuntimeError("当前环境缺少 torch/transformers streamer，无法执行 HF 部署评测")

    user_payload = json.dumps(
        {"question": question, "retrieval": retrieval["retrieval"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = apply_chat(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sampler = PeakSampler(os.getpid(), gpu_index)
    generation_started = time.perf_counter()
    first_token_ms: float | None = None
    chunks: list[str] = []
    error_queue: queue.Queue[BaseException] = queue.Queue()

    def worker() -> None:
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except BaseException as exc:  # pragma: no cover
            error_queue.put(exc)

    thread = threading.Thread(target=worker, daemon=True)
    sampler.start()
    thread.start()
    for piece in streamer:
        if piece and piece.strip() and first_token_ms is None:
            first_token_ms = round((time.perf_counter() - generation_started) * 1000, 1)
        chunks.append(piece)
    thread.join()
    sampler.stop()

    if not error_queue.empty():
        raise error_queue.get()

    answer = "".join(chunks).strip()
    total_ms = round((time.perf_counter() - generation_started) * 1000, 1)
    peak_vram = sampler.peak_vram_mb
    if torch.cuda.is_available():
        peak_vram = max(peak_vram, round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2))
    return answer, first_token_ms or total_ms, total_ms, sampler.peak_rss_mb, peak_vram


def run_gguf_answer(
    *,
    question: str,
    retrieval: dict[str, Any],
    args: argparse.Namespace,
    candidate: CandidateConfig,
    gpu_index: int,
) -> tuple[str, float, float, float, float]:
    special_answer = str(retrieval.get("special_answer") or "").strip()
    if special_answer:
        return special_answer, 0.0, 0.0, 0.0, 0.0

    user_payload = json.dumps(
        {"question": question, "retrieval": retrieval["retrieval"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    command = [
        args.llama_cli_path,
        "--model",
        candidate.gguf_model_path,
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
        SYSTEM_PROMPT,
        "--prompt",
        user_payload,
        "--predict",
        str(args.max_new_tokens),
        "--device",
        args.device,
        "--main-gpu",
        "0",
        "--gpu-layers",
        "all",
    ]

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )
    sampler = PeakSampler(process.pid, gpu_index)
    sampler.start()
    stdout_chunks: list[str] = []
    first_token_ms: float | None = None
    assert process.stdout is not None
    while True:
        char = process.stdout.read(1)
        if char == "" and process.poll() is not None:
            break
        if not char:
            continue
        stdout_chunks.append(char)
        if first_token_ms is None and char.strip():
            first_token_ms = round((time.perf_counter() - started) * 1000, 1)
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    sampler.stop()
    if return_code != 0:
        raise RuntimeError(f"llama-cli failed: {stderr.strip()}")
    answer = extract_answer("".join(stdout_chunks))
    total_ms = round((time.perf_counter() - started) * 1000, 1)
    return answer, first_token_ms or total_ms, total_ms, sampler.peak_rss_mb, sampler.peak_vram_mb


def summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return round(statistics.fmean(values), 2) if values else 0.0

    groups: dict[str, dict[str, Any]] = {}
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        groups[group] = {
            "count": len(group_rows),
            "avg_ttft_ms": mean_from(group_rows, "time_to_first_token_ms"),
            "avg_total_ms": mean_from(group_rows, "time_to_final_answer_ms"),
            "avg_output_chars": mean_from(group_rows, "output_chars"),
        }

    return {
        "case_count": len(rows),
        "avg_retrieval_latency_ms": mean("retrieval_latency_ms"),
        "avg_generation_latency_ms": mean("generation_latency_ms"),
        "avg_ttft_ms": mean("time_to_first_token_ms"),
        "avg_total_ms": mean("time_to_final_answer_ms"),
        "avg_peak_memory_mb": mean("peak_memory_mb"),
        "avg_peak_vram_mb": mean("peak_vram_mb"),
        "avg_output_chars": mean("output_chars"),
        "groups": groups,
    }


def mean_from(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(statistics.fmean(values), 2) if values else 0.0


def main() -> None:
    args = parse_args()
    dashscope_key = extract_dashscope_key()
    supabase_url, supabase_key = extract_supabase_config()
    cases = load_cases(Path(args.cases_file))
    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        tokenizer = model = device = None
        if candidate.backend == "hf":
            tokenizer, model, device = load_generator(candidate.model_name, candidate.adapter_path)
        try:
            candidate_rows: list[dict[str, Any]] = []
            for index, case in enumerate(cases, start=1):
                question = str(case["question"])
                chain, retrieval_latency_ms = call_retrieval(
                    question,
                    args=args,
                    dashscope_key=dashscope_key,
                    supabase_url=supabase_url,
                    supabase_key=supabase_key,
                )
                if candidate.backend == "hf":
                    answer, ttft_ms, total_ms, peak_mem_mb, peak_vram_mb = run_hf_answer(
                        question=question,
                        retrieval=chain,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        gpu_index=args.gpu_index,
                    )
                else:
                    answer, ttft_ms, total_ms, peak_mem_mb, peak_vram_mb = run_gguf_answer(
                        question=question,
                        retrieval=chain,
                        args=args,
                        candidate=candidate,
                        gpu_index=args.gpu_index,
                    )
                row = {
                    "candidate_id": candidate.candidate_id,
                    "candidate_label": candidate.label,
                    "backend": candidate.backend,
                    "case_id": case["case_id"],
                    "group": case["group"],
                    "question": question,
                    "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
                    "query_class": chain["query_class"],
                    "intent": chain["retrieval"]["intent"],
                    "retrieval_route": chain["retrieval"]["retrieval_route"],
                    "answer_route": chain["retrieval"]["answer_route"],
                    "fallback_trigger_reason": chain["retrieval"]["fallback_trigger_reason"],
                    "hit_count": chain["retrieval"]["hit_count"],
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "generation_latency_ms": round(total_ms, 1),
                    "time_to_first_token_ms": round(ttft_ms, 1),
                    "time_to_final_answer_ms": round(retrieval_latency_ms + total_ms, 1),
                    "peak_memory_mb": round(peak_mem_mb, 2),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "output_chars": len(answer),
                    "answer": answer,
                    "context_preview": (chain["retrieval"].get("evidence") or [])[:2],
                }
                output_rows.append(row)
                candidate_rows.append(row)
                print(f"[part29] {candidate.label} {index}/{len(cases)} {case['case_id']}")
            summary_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_label": candidate.label,
                    "backend": candidate.backend,
                    **summarize_candidate(candidate_rows),
                }
            )
        finally:
            if candidate.backend == "hf":
                unload_model(model)

    Path(args.output_file).write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.summary_file).write_text(
        json.dumps(
            {
                "generated_at": now_utc().isoformat().replace("+00:00", "Z"),
                "cases_file": str(Path(args.cases_file)),
                "candidates": [candidate.__dict__ for candidate in CANDIDATES],
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
