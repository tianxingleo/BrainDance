# BrainDance Qwen3 Releases

This directory is for publishable LoRA adapter bundles.

Rules:

- Do not commit training workdirs from `ai_engine/finetune_qwen3/outputs/`.
- Export a clean adapter package here before committing or open-sourcing.
- `adapter_model.safetensors` will be handled by Git LFS because the repo root [`../../.gitattributes`](../../.gitattributes) already tracks `*.safetensors`.
- Keep benchmark data in normal Git at `ai_engine/finetune_qwen3/data/braindance_qwen3_benchmark.jsonl`.
- Keep generated train/val datasets ignored unless there is a specific reason to version a frozen snapshot.

Recommended workflow:

```bash
ai_engine/finetune_qwen3/scripts/export_release_adapter.sh \
  ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round3 \
  qwen3_1p7b_braindance_round3 \
  Qwen/Qwen3-1.7B
```

Merge + quantization workflow:

```bash
# 1) merge LoRA into a standalone HF model dir
ai_engine/finetune_qwen3/scripts/run_merge_qwen3_gpu0.sh \
  Qwen/Qwen3-1.7B \
  ai_engine/finetune_qwen3/outputs/qwen3_1p7b_lora_sft_round4_1_patch_mixed \
  ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged

# 2) generate GGUF/quantization plan; pass llama.cpp dir when available
ai_engine/finetune_qwen3/scripts/run_prepare_quantization_gpu0.sh \
  ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_merged \
  ai_engine/finetune_qwen3/releases/qwen3_1p7b_braindance_round4_1_patch_mixed_quantized \
  /path/to/llama.cpp \
  Q4_K_M
```

Expected release bundle:

- `adapter_model.safetensors`
- `adapter_config.json`
- `release_metadata.json`
- `final_metrics.json` if present
- `training_spec.json` if present
- `merge_metadata.json` if present
- `quantization_plan.json` if present
- `chat_template.jinja` if present
- `SHA256SUMS` if `sha256sum` is available

By default the export script does not copy `tokenizer.json`, `vocab.json`, or `merges.txt`.
Use the base model tokenizer unless you intentionally need to publish a tokenizer override.
