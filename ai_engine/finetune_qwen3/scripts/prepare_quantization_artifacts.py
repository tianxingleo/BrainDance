#!/usr/bin/env python3
"""
Prepare or execute the GGUF conversion + quantization workflow for merged Qwen3 models.

This script works in two modes:
- planning mode: always available, writes a manifest and shell commands
- execution mode: only runs when llama.cpp tools are available
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GGUF conversion and quantization artifacts")
    parser.add_argument("--merged_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gguf_name", default="model-f16.gguf")
    parser.add_argument("--quant_type", default="Q4_K_M")
    parser.add_argument("--quant_name", default="")
    parser.add_argument("--llama_cpp_dir", default="")
    parser.add_argument("--convert_script", default="")
    parser.add_argument("--quantize_binary", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow_missing_tools", action="store_true")
    return parser.parse_args()


def detect_llama_cpp_toolchain(llama_cpp_dir: str, convert_script: str, quantize_binary: str) -> dict[str, str | None]:
    detected: dict[str, str | None] = {"llama_cpp_dir": None, "convert_script": None, "quantize_binary": None}
    if llama_cpp_dir:
        root = Path(llama_cpp_dir).expanduser().resolve()
        detected["llama_cpp_dir"] = str(root)
        candidate_convert = root / "convert_hf_to_gguf.py"
        if candidate_convert.exists():
            detected["convert_script"] = str(candidate_convert)
        for name in ("llama-quantize", "quantize"):
            candidate = root / "build" / "bin" / name
            if candidate.exists():
                detected["quantize_binary"] = str(candidate)
                break
    if convert_script:
        detected["convert_script"] = str(Path(convert_script).expanduser().resolve())
    if quantize_binary:
        detected["quantize_binary"] = str(Path(quantize_binary).expanduser().resolve())
    if not detected["convert_script"]:
        found = shutil.which("convert_hf_to_gguf.py")
        if found:
            detected["convert_script"] = found
    if not detected["quantize_binary"]:
        for name in ("llama-quantize", "quantize"):
            found = shutil.which(name)
            if found:
                detected["quantize_binary"] = found
                break
    return detected


def build_commands(args: argparse.Namespace, toolchain: dict[str, str | None]) -> dict[str, list[str]]:
    output_dir = Path(args.output_dir)
    gguf_path = output_dir / args.gguf_name
    quant_name = args.quant_name or gguf_path.name.replace(".gguf", f"-{args.quant_type.lower()}.gguf")
    quant_path = output_dir / quant_name

    commands: dict[str, list[str]] = {
        "convert": [],
        "quantize": [],
    }
    if toolchain["convert_script"]:
        commands["convert"] = [
            "python",
            toolchain["convert_script"],
            str(Path(args.merged_model_dir).resolve()),
            "--outfile",
            str(gguf_path.resolve()),
            "--outtype",
            "f16",
        ]
    if toolchain["quantize_binary"]:
        commands["quantize"] = [
            toolchain["quantize_binary"],
            str(gguf_path.resolve()),
            str(quant_path.resolve()),
            args.quant_type,
        ]
    return commands


def build_manifest(args: argparse.Namespace, toolchain: dict[str, str | None], commands: dict[str, list[str]]) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    gguf_path = output_dir / args.gguf_name
    quant_name = args.quant_name or gguf_path.name.replace(".gguf", f"-{args.quant_type.lower()}.gguf")
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "merged_model_dir": str(Path(args.merged_model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "gguf_path": str(gguf_path.resolve()),
        "quantized_path": str((output_dir / quant_name).resolve()),
        "quant_type": args.quant_type,
        "toolchain": toolchain,
        "commands": commands,
        "execution_ready": bool(commands["convert"] and commands["quantize"]),
    }


def write_plan_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quantization_plan.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    convert_cmd = manifest["commands"]["convert"]
    quantize_cmd = manifest["commands"]["quantize"]
    if convert_cmd:
        shell_lines.append(" ".join(shlex.quote(part) for part in convert_cmd))
    else:
        shell_lines.append("# Missing convert_hf_to_gguf.py, fill in the path before running.")
    shell_lines.append("")
    if quantize_cmd:
        shell_lines.append(" ".join(shlex.quote(part) for part in quantize_cmd))
    else:
        shell_lines.append("# Missing llama-quantize/quantize binary, fill in the path before running.")
    shell_lines.append("")
    (output_dir / "run_quantization.sh").write_text("\n".join(shell_lines), encoding="utf-8")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    toolchain = detect_llama_cpp_toolchain(args.llama_cpp_dir, args.convert_script, args.quantize_binary)
    commands = build_commands(args, toolchain)
    manifest = build_manifest(args, toolchain, commands)
    output_dir = Path(args.output_dir)
    write_plan_files(output_dir, manifest)

    if args.execute:
        if not manifest["execution_ready"]:
            if args.allow_missing_tools:
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
                return
            raise SystemExit("llama.cpp conversion or quantization tools are missing; inspect quantization_plan.json first.")
        run_command(commands["convert"])
        run_command(commands["quantize"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
