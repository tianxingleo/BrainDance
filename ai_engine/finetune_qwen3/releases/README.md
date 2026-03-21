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

Expected release bundle:

- `adapter_model.safetensors`
- `adapter_config.json`
- `release_metadata.json`
- `final_metrics.json` if present
- `chat_template.jinja` if present
- `SHA256SUMS` if `sha256sum` is available

By default the export script does not copy `tokenizer.json`, `vocab.json`, or `merges.txt`.
Use the base model tokenizer unless you intentionally need to publish a tokenizer override.
