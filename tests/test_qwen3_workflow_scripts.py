from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_lora_helpers_support_qwen3_sizes():
    module = load_module(
        "train_lora_sft_test",
        "ai_engine/finetune_qwen3/scripts/train_lora_sft.py",
    )

    assert module.normalize_model_slug("Qwen/Qwen3-0.6B") == "qwen3_0p6b"
    assert module.normalize_model_slug("Qwen/Qwen3-1.7B") == "qwen3_1p7b"
    assert module.parse_target_modules("q_proj,k_proj , v_proj") == ["q_proj", "k_proj", "v_proj"]
    assert module.parse_target_modules("") == module.DEFAULT_TARGET_MODULES


def test_train_full_helpers_support_qwen3_sizes():
    module = load_module(
        "train_full_sft_test",
        "ai_engine/finetune_qwen3/scripts/train_full_sft.py",
    )

    assert module.normalize_model_slug("Qwen/Qwen3-0.6B") == "qwen3_0p6b"
    assert module.normalize_model_slug("Qwen/Qwen3-1.7B") == "qwen3_1p7b"


def test_merge_metadata_contains_core_fields():
    module = load_module(
        "merge_lora_adapter_test",
        "ai_engine/finetune_qwen3/scripts/merge_lora_adapter.py",
    )

    args = argparse.Namespace(
        base_model="Qwen/Qwen3-1.7B",
        adapter_path="ai_engine/finetune_qwen3/outputs/mock_adapter",
        output_dir="ai_engine/finetune_qwen3/releases/mock_merged",
        attn_implementation="sdpa",
        device_map="cpu",
        safe_serialization=True,
    )
    metadata = module.build_merge_metadata(args, module.torch.float16)
    assert metadata["base_model"] == "Qwen/Qwen3-1.7B"
    assert metadata["torch_dtype"] == "float16"
    assert metadata["safe_serialization"] is True


def test_quantization_manifest_builds_commands(tmp_path):
    module = load_module(
        "prepare_quantization_artifacts_test",
        "ai_engine/finetune_qwen3/scripts/prepare_quantization_artifacts.py",
    )

    merged_dir = tmp_path / "merged_model"
    merged_dir.mkdir()
    output_dir = tmp_path / "quantized"
    args = argparse.Namespace(
        merged_model_dir=str(merged_dir),
        output_dir=str(output_dir),
        gguf_name="model-f16.gguf",
        quant_type="Q4_K_M",
        quant_name="",
        llama_cpp_dir="",
        convert_script="/tmp/convert_hf_to_gguf.py",
        quantize_binary="/tmp/llama-quantize",
        execute=False,
        allow_missing_tools=False,
    )

    toolchain = module.detect_llama_cpp_toolchain("", args.convert_script, args.quantize_binary)
    commands = module.build_commands(args, toolchain)
    manifest = module.build_manifest(args, toolchain, commands)
    module.write_plan_files(output_dir, manifest)

    assert commands["convert"][0] == "python"
    assert "--outfile" in commands["convert"]
    outfile_index = commands["convert"].index("--outfile") + 1
    assert commands["convert"][outfile_index] == str((output_dir / "model-f16.gguf").resolve())
    assert commands["quantize"][-1] == "Q4_K_M"
    assert manifest["execution_ready"] is True
    assert (output_dir / "quantization_plan.json").exists()
    assert (output_dir / "run_quantization.sh").exists()
