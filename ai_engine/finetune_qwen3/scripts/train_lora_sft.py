#!/usr/bin/env python3
"""
Minimal LoRA SFT training script for BrainDance local QA on Qwen3-1.7B.

This script keeps the setup small and explicit:
- single GPU
- bf16 LoRA
- assistant-only loss masking
- chat template aligned with Qwen3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 LoRA SFT for BrainDance local QA")
    parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_chat(tokenizer: AutoTokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for tokenizer versions without enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_preprocess_fn(tokenizer: AutoTokenizer, cutoff_len: int):
    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        prompt_text = apply_chat(tokenizer, messages[:-1], add_generation_prompt=True)
        full_text = apply_chat(tokenizer, messages, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=cutoff_len,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess


def compute_trainable_ratio(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    trainable, total = compute_trainable_ratio(model)
    print(
        json.dumps(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_ratio": round(trainable / total * 100, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})
    preprocess = build_preprocess_fn(tokenizer, args.cutoff_len)
    dataset = dataset.map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing chat samples",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    steps_per_epoch = math.ceil(
        len(dataset["train"]) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    print(
        json.dumps(
            {
                "train_examples": len(dataset["train"]),
                "val_examples": len(dataset["validation"]),
                "steps_per_epoch_estimate": steps_per_epoch,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tensorboard"),
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
