#!/usr/bin/env python3
"""Rename labs from the messy decimal-style numbering (l00, l01_5, l05_8_5, ...)
to clean sequential numbering (l01..l35).

Strategy
--------
Phase 1 — Text replacement
    For every text file under SEARCH_DIRS / ROOT_FILES, replace each old lab ID
    with the corresponding new ID. We use a two-pass placeholder substitution to
    avoid substring collisions (e.g. `l01_5_nccl_ddp_smoke` contains `l01_`).

Phase 2 — Filesystem rename
    `git mv` the 35 lab directories (labs/<old> -> labs/<new>) and the 35
    quest YAML files (quests/<old>.yaml -> quests/<new>.yaml).

Phase 3 — Verification
    Re-scan all files and report any leftover occurrences of old IDs.

Usage
-----
    python scripts/rename_labs.py --dry-run        # preview every change
    python scripts/rename_labs.py --apply          # apply changes
    python scripts/rename_labs.py --verify-only    # scan for leftovers

The `level:` field inside quest YAMLs is intentionally left untouched — it
denotes phase (Evidence / Training / Serving / RL / Capstone), not lab order.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# (old_id, new_id) — order does not matter; placeholder-based two-pass
# replacement guarantees substring safety.
RENAME_MAP: list[tuple[str, str]] = [
    # Evidence (L00)
    ("l00_env_conda_cuda",                 "l01_env_conda_cuda"),
    # Training (L01..L06)
    ("l01_pytorch_systems",                "l02_pytorch_systems"),
    ("l01_5_nccl_ddp_smoke",               "l03_nccl_ddp_smoke"),
    ("l01_7_gpu_kernel",                   "l04_gpu_kernel"),
    ("l02_distributed_primitives",         "l05_distributed_primitives"),
    ("l03_torchtitan_training",            "l06_torchtitan_training"),
    ("l03_5_dataset_megatron_bin",         "l07_dataset_megatron_bin"),
    ("l04_megatron_text_pretrain",         "l08_megatron_text_pretrain"),
    ("l04_5_long_context_cp",              "l09_long_context_cp"),
    ("l04_8_megatron_pretrain_lifecycle",  "l10_megatron_pretrain_lifecycle"),
    ("l05_megatron_scale_optimization",    "l11_megatron_scale_optimization"),
    ("l05_3_fsdp2_llama",                  "l12_fsdp2_llama"),
    ("l05_5_moe_ep",                       "l13_moe_ep"),
    ("l05_7_pipeline_1f1b",                "l14_pipeline_1f1b"),
    ("l05_8_megatron_parallel_checkpoint", "l15_megatron_parallel_checkpoint"),
    ("l05_8_5_resume_after_crash",         "l16_resume_after_crash"),
    ("l06_megatron_multimodal_data",       "l17_megatron_multimodal_data"),
    ("l06_3_data_engineering",             "l18_data_engineering"),
    # Serving (L07..L09)
    ("l07_vllm_serving_baseline",          "l19_vllm_serving_baseline"),
    ("l07_5_vllm_scheduler_kv",            "l20_vllm_scheduler_kv"),
    ("l08_sglang_serving_core",            "l21_sglang_serving_core"),
    ("l08_3_flash_attn_v2_bench",          "l22_flash_attn_v2_bench"),
    ("l08_5_quant_serving",                "l23_quant_serving"),
    ("l08_7_spec_decode",                  "l24_spec_decode"),
    ("l08_8_serve_eval_lm_eval",           "l25_serve_eval_lm_eval"),
    ("l09_sglang_pd_observability",        "l26_sglang_pd_observability"),
    ("l09_5_sglang_pd_cache",              "l27_sglang_pd_cache"),
    ("l09_8_sft_qwen_alpaca",              "l28_sft_qwen_alpaca"),
    # RL (L10..L11)
    ("l10_verl_rl_baseline",               "l29_verl_rl_baseline"),
    ("l10_3_dpo_loss",                     "l30_dpo_loss"),
    ("l10_5_rollout_only_smoke",           "l31_rollout_only_smoke"),
    ("l11_slime_rl_core",                  "l32_slime_rl_core"),
    ("l11_5_rl_rollout_freshness",         "l33_rl_rollout_freshness"),
    ("l11_8_grpo",                         "l34_grpo"),
    # Capstone (L12)
    ("l12_multimodal_capstone",            "l35_multimodal_capstone"),
]

PLACEHOLDER_FMT = "\x00LAB_RENAME_TOKEN_{idx:02d}\x00"

# Directories to walk (relative to ROOT). Anything outside this list is left alone.
SEARCH_DIRS = [
    "labs",
    "quests",
    "mini_infra",
    "docs",
    "scripts",
    "notebooks",
    "tickets",
    "dashboards",
    "prompts",
    "app",
    "simulators",
]

# Plus these root-level files
ROOT_FILES = ["README.md", "Makefile", "pyproject.toml"]

# Text-file extensions to process. Files without an extension (Makefile, etc.)
# are checked by name in is_text_file().
TEXT_EXTS = {
    ".py", ".yaml", ".yml", ".md", ".mk", ".sh", ".txt",
    ".json", ".ipynb", ".cfg", ".toml", ".ini", ".html",
    ".js", ".ts", ".tsx", ".jsx", ".vue", ".css",
}

# Directory names to skip wherever they appear in the path. `runs/`, `reports/`
# and `final_artifacts/` contain historical run outputs we deliberately do NOT
# rewrite — they are immutable evidence of past lab executions.
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "envs",                # conda env yamls — generic, not lab-specific
    "runs", "reports", "final_artifacts", "checkpoints", "data",
    "github_repo",         # external upstream sources
    ".pytest_cache", ".mypy_cache", "dist", "build",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTS:
        return True
    if path.name in {"Makefile", "Dockerfile", ".gitignore", ".gitattributes"}:
        return True
    return False


# File paths (relative to ROOT) to skip entirely. The script must skip itself,
# otherwise its own RENAME_MAP gets rewritten in place and the historical record
# is destroyed.
SKIP_FILES = {
    "scripts/rename_labs.py",
}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if str(path) in SKIP_FILES:
        return True
    return False


def iter_target_files() -> list[Path]:
    files: list[Path] = []
    for d in SEARCH_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if should_skip(p.relative_to(ROOT)):
                continue
            if p.is_file() and is_text_file(p):
                files.append(p)
    for fname in ROOT_FILES:
        p = ROOT / fname
        if p.is_file():
            files.append(p)
    return files


def two_pass_replace(text: str) -> tuple[str, int]:
    """Two-pass placeholder substitution to avoid substring collisions."""
    n_total = 0
    # Pass 1: old -> placeholder
    for idx, (old, _) in enumerate(RENAME_MAP):
        placeholder = PLACEHOLDER_FMT.format(idx=idx)
        new_text, n = re.subn(re.escape(old), placeholder, text)
        text = new_text
        n_total += n
    # Pass 2: placeholder -> new
    for idx, (_, new) in enumerate(RENAME_MAP):
        placeholder = PLACEHOLDER_FMT.format(idx=idx)
        text = text.replace(placeholder, new)
    return text, n_total


def git_mv(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        print(f"  WARN: dst already exists, skipping: {dst.relative_to(ROOT)}",
              file=sys.stderr)
        return False
    rel_src = src.relative_to(ROOT)
    rel_dst = dst.relative_to(ROOT)
    print(f"  git mv {rel_src} -> {rel_dst}")
    if dry_run:
        return True
    # Try `git mv` first (preserves history); fall back to plain rename if not
    # in a git working tree or git complains.
    res = subprocess.run(
        ["git", "mv", str(rel_src), str(rel_dst)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"  (git mv failed: {res.stderr.strip()}; using os.rename)",
              file=sys.stderr)
        os.rename(src, dst)
    return True


# ----------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------


def phase1_replace_text(dry_run: bool) -> tuple[int, int]:
    n_files = 0
    n_repl = 0
    for path in iter_target_files():
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text, n = two_pass_replace(original)
        if n == 0:
            continue
        n_files += 1
        n_repl += n
        rel = path.relative_to(ROOT)
        print(f"  [{n:3d} edits] {rel}")
        if not dry_run:
            path.write_text(new_text, encoding="utf-8")
    return n_files, n_repl


def phase2_rename_paths(dry_run: bool) -> int:
    n = 0
    labs_dir = ROOT / "labs"
    quests_dir = ROOT / "quests"
    for old, new in RENAME_MAP:
        if git_mv(labs_dir / old, labs_dir / new, dry_run):
            n += 1
        if git_mv(quests_dir / f"{old}.yaml",
                  quests_dir / f"{new}.yaml", dry_run):
            n += 1
    return n


def phase3_verify() -> int:
    leftovers = 0
    for path in iter_target_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for old, _ in RENAME_MAP:
            if old in text:
                rel = path.relative_to(ROOT)
                # Count occurrences for clarity
                count = text.count(old)
                print(f"  LEFTOVER: {rel}  ({count}x '{old}')")
                leftovers += count
    return leftovers


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="Preview every change. Files are NOT modified.")
    g.add_argument("--apply", action="store_true",
                   help="Apply the rename in place.")
    g.add_argument("--verify-only", action="store_true",
                   help="Scan for leftover old IDs (run after --apply).")
    args = parser.parse_args()

    if args.verify_only:
        print("Scanning for leftover old lab IDs...")
        n = phase3_verify()
        if n == 0:
            print("\nClean — no leftover old IDs found.")
        else:
            print(f"\n{n} leftover occurrences. Investigate the LEFTOVER lines above.")
            sys.exit(1)
        return

    print("=" * 70)
    print(f"Phase 1/2: text replacement  (dry-run={args.dry_run})")
    print("=" * 70)
    n_files, n_repl = phase1_replace_text(dry_run=args.dry_run)
    print(f"\n  Phase 1 summary: {n_repl} replacements across {n_files} files")

    print()
    print("=" * 70)
    print(f"Phase 2/2: rename labs/ and quests/  (dry-run={args.dry_run})")
    print("=" * 70)
    n_paths = phase2_rename_paths(dry_run=args.dry_run)
    print(f"\n  Phase 2 summary: {n_paths} paths renamed")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        print("If the diff above looks right, re-run with: --apply")
    else:
        print("\n[applied]")
        print("Next: verify by running")
        print("  python scripts/rename_labs.py --verify-only")
        print("  python -m app.backend.manage list-missions")
        print("  make patch-test M=l01_env_conda_cuda")


if __name__ == "__main__":
    main()
