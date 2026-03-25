#!/usr/bin/env python3
"""Shared local candidate registry for Qwen3 benchmark evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RELEASES_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "releases"
OUTPUTS_DIR = PROJECT_ROOT / "ai_engine" / "finetune_qwen3" / "outputs"
QUANT_DIR = RELEASES_DIR / "qwen3_1p7b_braindance_round4_1_patch_mixed_quantized_gpu0"
IMATRIX_DIR = QUANT_DIR / "imatrix_v1"


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    label: str
    backend: str
    model_name: str = ""
    adapter_path: str = ""
    gguf_model_path: str = ""


ALL_LOCAL_QA_CANDIDATES = (
    CandidateConfig(
        candidate_id="qwen3_0p6b_lora",
        label="0.6B LoRA",
        backend="hf",
        model_name="Qwen/Qwen3-0.6B",
        adapter_path=str(RELEASES_DIR / "qwen3_0p6b_braindance_round1"),
    ),
    CandidateConfig(
        candidate_id="qwen3_0p6b_full_round1",
        label="0.6B full SFT round1",
        backend="hf",
        model_name=str(OUTPUTS_DIR / "qwen3_0p6b_full_sft_round1"),
    ),
    CandidateConfig(
        candidate_id="qwen3_0p6b_full_round2_lr8e6",
        label="0.6B full SFT round2 lr8e-6",
        backend="hf",
        model_name=str(OUTPUTS_DIR / "qwen3_0p6b_full_sft_round2_lr8e6"),
    ),
    CandidateConfig(
        candidate_id="qwen3_0p6b_full_round3_lr5e6",
        label="0.6B full SFT round3 lr5e-6",
        backend="hf",
        model_name=str(OUTPUTS_DIR / "qwen3_0p6b_full_sft_round3_lr5e6"),
    ),
    CandidateConfig(
        candidate_id="qwen3_0p6b_full_round4_patch",
        label="0.6B full SFT round4 patch",
        backend="hf",
        model_name=str(OUTPUTS_DIR / "qwen3_0p6b_full_sft_round4_partial_patch_v1"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_lora_round3",
        label="1.7B LoRA round3",
        backend="hf",
        model_name="Qwen/Qwen3-1.7B",
        adapter_path=str(OUTPUTS_DIR / "qwen3_1p7b_lora_sft_round3"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_lora_round4_patch",
        label="1.7B LoRA round4 patch",
        backend="hf",
        model_name="Qwen/Qwen3-1.7B",
        adapter_path=str(OUTPUTS_DIR / "qwen3_1p7b_lora_sft_round4_patch"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_lora_round4_1_patch",
        label="1.7B LoRA round4.1 patch",
        backend="hf",
        model_name="Qwen/Qwen3-1.7B",
        adapter_path=str(OUTPUTS_DIR / "qwen3_1p7b_lora_sft_round4_1_patch"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_lora_round4_1_patch_mixed",
        label="1.7B LoRA",
        backend="hf",
        model_name="Qwen/Qwen3-1.7B",
        adapter_path=str(OUTPUTS_DIR / "qwen3_1p7b_lora_sft_round4_1_patch_mixed"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_full_round1",
        label="1.7B full SFT round1",
        backend="hf",
        model_name=str(OUTPUTS_DIR / "qwen3_1p7b_full_sft_round1_gpu1"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_full_round1_lr5e6",
        label="1.7B full SFT lr5e-6",
        backend="hf",
        model_name=str(OUTPUTS_DIR / "qwen3_1p7b_full_sft_round1_lr5e6_gpu1"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_merged",
        label="1.7B merged",
        backend="hf",
        model_name=str(RELEASES_DIR / "qwen3_1p7b_braindance_round4_1_patch_mixed_merged_gpu0"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_q4",
        label="1.7B Q4_K_M",
        backend="gguf",
        gguf_model_path=str(QUANT_DIR / "model-f16-q4_k_m.gguf"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_q5",
        label="1.7B Q5_K_M",
        backend="gguf",
        gguf_model_path=str(QUANT_DIR / "model-f16-q5_k_m.gguf"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_q4_imatrix",
        label="1.7B Q4_K_M + imatrix",
        backend="gguf",
        gguf_model_path=str(IMATRIX_DIR / "model-f16-q4_k_m-imatrix.gguf"),
    ),
    CandidateConfig(
        candidate_id="qwen3_1p7b_q5_imatrix",
        label="1.7B Q5_K_M + imatrix",
        backend="gguf",
        gguf_model_path=str(IMATRIX_DIR / "model-f16-q5_k_m-imatrix.gguf"),
    ),
)


DEFAULT_UNSEEN_CANDIDATE_IDS = (
    "qwen3_1p7b_lora_round4_1_patch_mixed",
    "qwen3_0p6b_lora",
    "qwen3_1p7b_q5_imatrix",
)

DEFAULT_SPATIAL_CANDIDATE_IDS = (
    "qwen3_1p7b_lora_round4_1_patch_mixed",
    "qwen3_0p6b_lora",
    "qwen3_0p6b_full_round1",
    "qwen3_1p7b_merged",
    "qwen3_1p7b_q4_imatrix",
    "qwen3_1p7b_q5_imatrix",
)


def select_candidates(candidate_ids: str | None, *, default_ids: tuple[str, ...]) -> tuple[CandidateConfig, ...]:
    if candidate_ids and candidate_ids.strip():
        wanted = [item.strip() for item in candidate_ids.split(",") if item.strip()]
    else:
        wanted = list(default_ids)

    if wanted == ["all"]:
        return ALL_LOCAL_QA_CANDIDATES

    by_id = {candidate.candidate_id: candidate for candidate in ALL_LOCAL_QA_CANDIDATES}
    missing = [candidate_id for candidate_id in wanted if candidate_id not in by_id]
    if missing:
        raise ValueError(f"unknown candidate_ids: {', '.join(missing)}")
    return tuple(by_id[candidate_id] for candidate_id in wanted)
