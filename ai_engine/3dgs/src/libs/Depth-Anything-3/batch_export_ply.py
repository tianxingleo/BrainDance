#!/usr/bin/env python3
"""批量将所有 checkpoint 导出为 PLY 文件。

原理：ns-export 只支持 --load-config，内部自动加载最新的 checkpoint。
为了导出指定 step 的 checkpoint，我们临时复制 config.yml 并修改其中的
load_step 字段，让它加载特定 step。
"""

import os
import sys
import yaml
import shutil
import subprocess
from pathlib import Path

# ===== 配置 =====
NS_PYTHON = "/home/ltx/my_envs/gs_linux_backup/bin/python"
NS_EXPORT = "/home/ltx/my_envs/gs_linux_backup/bin/ns-export"
PROJECT_ROOT = Path("/home/ltx/projects/Depth-Anything-3")

# 最新训练的输出目录
TRAIN_DIR = PROJECT_ROOT / "da3_dn_splatter_output/da3_dn_splatter/dn-splatter/2026-02-18_005746"
CONFIG_PATH = TRAIN_DIR / "config.yml"
CKPT_DIR = TRAIN_DIR / "nerfstudio_models"
EXPORT_BASE = PROJECT_ROOT / "da3_dn_splatter_output"


def get_all_steps():
    """扫描 checkpoint 目录，返回所有可用的 step 编号。"""
    steps = []
    for f in sorted(CKPT_DIR.glob("step-*.ckpt")):
        # 文件名格式: step-000005000.ckpt
        step_str = f.stem.split("-")[1]
        steps.append(int(step_str))
    return steps


def export_step(step: int):
    """导出指定 step 的 checkpoint 为 PLY。"""
    output_dir = EXPORT_BASE / f"export_step{step}"
    ply_path = output_dir / "splat.ply"

    # 跳过已导出的
    if ply_path.exists():
        size_mb = ply_path.stat().st_size / (1024 * 1024)
        print(f"  ⏭️  Step {step}: 已存在 ({size_mb:.1f} MB), 跳过")
        return True

    # 读取原始 config 文本 (不用 yaml 解析, 因为包含 Python 对象标签)
    with open(CONFIG_PATH, "r") as f:
        config_text = f.read()

    # 用文本替换设置 load_dir 和 load_step
    import re
    config_text = re.sub(
        r'^load_dir:.*$',
        f'load_dir: {CKPT_DIR}',
        config_text, flags=re.MULTILINE
    )
    config_text = re.sub(
        r'^load_step:.*$',
        f'load_step: {step}',
        config_text, flags=re.MULTILINE
    )

    # 写临时 config
    tmp_config = TRAIN_DIR / f"config_export_step{step}.yml"
    with open(tmp_config, "w") as f:
        f.write(config_text)

    # 执行导出
    cmd = [
        NS_PYTHON, NS_EXPORT, "gaussian-splat",
        "--load-config", str(tmp_config),
        "--output-dir", str(output_dir),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and ply_path.exists():
            size_mb = ply_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Step {step}: 导出成功 ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ❌ Step {step}: 导出失败")
            if result.stderr:
                # 只取最后几行错误信息
                err_lines = result.stderr.strip().split("\n")[-5:]
                for line in err_lines:
                    print(f"     {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ Step {step}: 超时")
        return False
    finally:
        # 清理临时 config
        if tmp_config.exists():
            tmp_config.unlink()


def main():
    steps = get_all_steps()
    print(f"🔍 发现 {len(steps)} 个 checkpoint: {steps}")
    print(f"📂 导出目录: {EXPORT_BASE}/export_stepXXXXX/")
    print()

    # 检查 step 29999 是否已经在 export/ 目录中
    existing_final = EXPORT_BASE / "export" / "splat.ply"
    if existing_final.exists():
        # 复制到统一命名目录
        final_step = max(steps)
        dest_dir = EXPORT_BASE / f"export_step{final_step}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_ply = dest_dir / "splat.ply"
        if not dest_ply.exists():
            shutil.copy2(existing_final, dest_ply)
            print(f"  📋 已将 export/splat.ply 复制到 export_step{final_step}/splat.ply")

    success = 0
    fail = 0
    for step in steps:
        ok = export_step(step)
        if ok:
            success += 1
        else:
            fail += 1

    print()
    print(f"🏁 完成: {success} 成功, {fail} 失败")

    # 列出所有导出结果
    print()
    print("📁 导出文件列表:")
    for step in steps:
        ply = EXPORT_BASE / f"export_step{step}" / "splat.ply"
        if ply.exists():
            size_mb = ply.stat().st_size / (1024 * 1024)
            print(f"  {ply}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
