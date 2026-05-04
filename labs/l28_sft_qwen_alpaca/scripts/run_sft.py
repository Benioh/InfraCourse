"""L09.8 · SFT smoke：tokenize 一段 Alpaca-shaped 数据并跑 HF Trainer。

CPU mode 只验证 tokenize；GPU mode 跑 HuggingFace Trainer 100 步。
"""

from __future__ import annotations

import argparse
import json
import sys
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

MISSION_ID = "l28_sft_qwen_alpaca"


def _impl():
    try:
        from starter import sft_pipeline as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import sft_pipeline as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _synthetic_messages(n: int) -> list[list[dict]]:
    items = []
    for i in range(n):
        items.append(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is {i}+{i}?"},
                {"role": "assistant", "content": f"{i + i}"},
            ]
        )
    return items


def _alpaca_messages(jsonl_path: Path, n: int) -> list[list[dict]]:
    items = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(items) >= n:
                break
            obj = json.loads(line)
            items.append(
                [
                    {"role": "user", "content": obj.get("instruction", "") + (" " + obj.get("input", "") if obj.get("input") else "")},
                    {"role": "assistant", "content": obj.get("output", obj.get("response", ""))},
                ]
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
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
    if "torchrun_command" in config:
        write_text(
            run_dir / "artifacts" / "torchrun_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["torchrun_command"].split())
            + "\n",
        )

    mode = config.get("mode", "tokenize_only")
    if mode == "tokenize_only":
        class _CharTokenizer:
            eos_token_id = 1
            pad_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [(ord(ch) % 200) + 8 for ch in text]

        max_length = int(config.get("max_length", 64))
        messages_list = _synthetic_messages(int(config["data"]["num_samples"]))
        results = [
            mod.tokenize_chat_with_loss_mask(messages, _CharTokenizer(), max_length=max_length)
            for messages in messages_list
        ]
        assistant_token_counts = [
            sum(1 for label in result["labels"] if label != -100) for result in results
        ]
        accepted = min(assistant_token_counts) >= int(
            config.get("acceptance", {}).get("min_assistant_tokens", 1)
        )
        write_json(
            run_dir / "artifacts" / "tokenize_summary.json",
            {
                "samples": len(results),
                "min_assistant_tokens": min(assistant_token_counts),
                "accepted": accepted,
            },
        )
        for idx, count in enumerate(assistant_token_counts):
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "timestamp": utc_now(),
                    "metric_type": "tokenize",
                    "sample": idx,
                    "assistant_tokens": count,
                },
            )
        write_text(
            run_dir / "report.md",
            f"# {MISSION_ID}\nimpl={impl_label}; samples={len(results)}; accepted={accepted}.\n",
        )
        print(run_dir)
        return

    # train mode requires transformers + datasets + torch
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        write_text(run_dir / "artifacts" / "fallback.txt", f"missing dep: {exc}\n")
        write_text(run_dir / "report.md", f"# {MISSION_ID}\nfallback: {exc}\n")
        print(run_dir)
        return

    model_name = config["model"]["hf_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config["training"].get("precision", "bf16") == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    data_cfg = config["data"]
    if data_cfg["source"] == "synthetic":
        examples = _synthetic_messages(int(data_cfg["num_samples"]))
    else:
        jsonl = ROOT / data_cfg.get("jsonl_fallback", "")
        if jsonl.exists():
            examples = _alpaca_messages(jsonl, int(data_cfg["num_samples"]))
        else:
            try:
                from datasets import load_dataset

                ds = load_dataset(data_cfg["hf_name"], split=f"train[:{data_cfg['num_samples']}]")
                examples = [
                    [
                        {
                            "role": "user",
                            "content": (item.get("instruction", "") + (" " + item.get("input", "") if item.get("input") else "")).strip(),
                        },
                        {"role": "assistant", "content": item.get("output", item.get("response", ""))},
                    ]
                    for item in ds
                ]
            except Exception as exc:  # noqa: BLE001
                write_text(run_dir / "artifacts" / "fallback.txt", f"hf load failed: {exc}\n")
                examples = _synthetic_messages(int(data_cfg["num_samples"]))

    max_length = int(config["training"]["max_length"])
    tokenized = [
        mod.tokenize_chat_with_loss_mask(messages, tokenizer, max_length=max_length)
        for messages in examples
    ]
    dataset = Dataset.from_list(tokenized)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "artifacts" / "ckpt"),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["lr"]),
        max_steps=int(config["training"]["num_train_steps"]),
        warmup_steps=int(config["training"]["warmup_steps"]),
        bf16=config["training"].get("precision", "bf16") == "bf16",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()
    log_history = trainer.state.log_history
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    head = sum(losses[:3]) / max(1, len(losses[:3]))
    tail = sum(losses[-3:]) / max(1, len(losses[-3:]))
    drop_ratio = (head - tail) / max(1e-6, head)
    accept = drop_ratio >= float(config.get("acceptance", {}).get("loss_drop_ratio", 0.0))

    for entry in log_history:
        append_jsonl(
            run_dir / "metrics.jsonl",
            {"timestamp": utc_now(), "metric_type": "trainer", **entry},
        )
    write_json(
        run_dir / "artifacts" / "sft_summary.json",
        {"head_loss": head, "tail_loss": tail, "drop_ratio": drop_ratio, "accept": accept},
    )
    write_text(
        run_dir / "report.md",
        f"# {MISSION_ID}\nimpl={impl_label}; head={head:.3f}; tail={tail:.3f}; drop_ratio={drop_ratio:.3f}; accept={accept}\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
