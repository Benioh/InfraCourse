"""L08 · 把 jsonl 切成 Megatron `.bin/.idx`，并打印真实 pretrain 命令。"""

from __future__ import annotations

import argparse
import json
import random
import os, sys
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB_DIR / "patch"))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l07_dataset_megatron_bin"


def _impl():
    name = os.environ.get("IMPL")
    if name in {"starter", "reference"}:
        return __import__(f"{name}.megatron_bin", fromlist=["megatron_bin"]), name
    try:
        from starter import megatron_bin as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except ImportError:
        from reference import megatron_bin as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _make_demo_jsonl(path: Path, lines: int) -> None:
    rng = random.Random(0)
    snippets = [
        "The quick brown fox jumps over the lazy dog.",
        "Megatron-LM trains transformer models at scale.",
        "FSDP2 shards parameters, gradients, and optimizer states.",
        "vLLM and SGLang both implement paged attention.",
        "Reinforcement learning from human feedback aligns models.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(lines):
            chunks = " ".join(rng.choice(snippets) for _ in range(rng.randint(1, 4)))
            fh.write(json.dumps({"text": chunks}) + "\n")


def _build_tokenizer(spec):
    kind = spec.get("kind", "char_mod_30k")
    if kind == "char_mod_30k":
        return lambda text: [(ord(ch) % 30000) + 1 for ch in text]
    if kind == "hf":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec["hf_name"])
        return lambda text: tok.encode(text, add_special_tokens=False)
    raise ValueError(f"unsupported tokenizer kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_demo.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    mod, impl_label = _impl()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "impl": impl_label, **config},
    )

    jsonl = ROOT / config["input"]["jsonl_path"]
    if not jsonl.exists():
        if config["input"].get("generate_demo_if_missing", False):
            _make_demo_jsonl(jsonl, int(config["input"].get("demo_lines", 100)))
        else:
            write_text(
                run_dir / "artifacts" / "fallback.txt",
                f"input jsonl not found: {jsonl}\n",
            )
            print(run_dir)
            return

    prefix = ROOT / config["output"]["prefix"]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = _build_tokenizer(config["tokenizer"])

    summary = mod.text_to_megatron_bin(
        jsonl, tokenizer, prefix, dtype_str=config.get("dtype", "int32")
    )
    write_json(run_dir / "artifacts" / "preprocess_summary.json", summary)

    dataset = mod.IndexedDataset(prefix)
    sample_dump = []
    for i in [0, 1, 2, max(0, len(dataset) - 1)]:
        if i < len(dataset):
            ids = dataset[i]
            sample_dump.append({"index": i, "len": len(ids), "head": ids[:8]})
    write_json(run_dir / "artifacts" / "sample_dump.json", sample_dump)

    accept = config.get("acceptance", {})
    accepted = (
        summary["n_samples"] >= int(accept.get("min_samples", 0))
        and summary["total_tokens"] >= int(accept.get("min_total_tokens", 0))
    )

    if "megatron_command_template" in config:
        cmd = " ".join(
            config["megatron_command_template"].format(prefix=str(prefix)).split()
        )
        write_text(
            run_dir / "artifacts" / "real_megatron_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n",
        )

    append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "timestamp": utc_now(),
            "metric_type": "preprocess",
            **summary,
            "accepted": accepted,
        },
    )

    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
把 `{jsonl}` 切成 Megatron 风格 `.bin/.idx`，并验证 IndexedDataset 读取一致。

## 2. 配置
- profile: {config.get('profile')}
- impl: {impl_label}
- tokenizer: {config['tokenizer']}
- dtype: {config.get('dtype', 'int32')}

## 3. 结果
- n_samples: {summary['n_samples']}
- total_tokens: {summary['total_tokens']}
- bin_path: {summary['bin_path']}
- idx_path: {summary['idx_path']}
- accepted: {accepted}

## 4. 真实 Megatron 启动命令
见 `artifacts/real_megatron_command.sh`。

## 5. 诊断
- 若 n_samples 偏少：检查 jsonl 是否大量空 text；调高 demo_lines
- 若 dtype 报错：换成 int32 或在 tokenize 阶段裁剪 vocab

## 6. 下一步
进入 L09 跑 Megatron pretrain；用 `--data-path` 指向上面的 prefix。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
