from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


FIELD_PATTERNS: dict[str, list[str]] = {
    "iteration": [r"iteration\s+([0-9]+)", r"iter(?:ation)?[=: ]+([0-9]+)"],
    "lm_loss": [rf"lm loss[=: ]+({NUMBER})", rf"loss[=: ]+({NUMBER})"],
    "grad_norm": [
        rf"grad(?:ient)? norm[=: ]+({NUMBER})",
        rf"grad_norm[=: ]+({NUMBER})",
    ],
    "consumed_samples": [
        r"consumed samples[=: ]+([0-9]+)",
        r"consumed_samples[=: ]+([0-9]+)",
    ],
    "tokens_per_sec": [
        rf"tokens/sec[=: ]+({NUMBER})",
        rf"tokens_per_sec[=: ]+({NUMBER})",
    ],
    "mfu": [rf"MFU[=: ]+({NUMBER})", rf"model flops utilization[=: ]+({NUMBER})"],
}


def _last_match(text: str, patterns: list[str]) -> str | None:
    values: list[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return values[-1] if values else None


def _number(value: str | None) -> int | float | None:
    if value is None:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    return float(value)


def parse_megatron_log(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "expected_command_present": "expected_command=" in text,
        "fallback_validation": "fallback" in text.lower(),
        "line_count": len(text.splitlines()),
    }
    for field, patterns in FIELD_PATTERNS.items():
        payload[field] = _number(_last_match(text, patterns))
    payload["has_training_metrics"] = any(
        payload.get(field) is not None for field in ["lm_loss", "tokens_per_sec", "mfu"]
    )
    payload["status"] = (
        "parsed_training_log"
        if payload["has_training_metrics"]
        else "command_or_fallback_only"
    )
    missing = [
        field
        for field in ["iteration", "lm_loss", "tokens_per_sec"]
        if payload.get(field) is None
    ]
    payload["missing_core_fields"] = missing
    return payload


def self_test() -> dict[str, Any]:
    text = """
    [2026-05-01] expected_command=torchrun pretrain_gpt.py
    iteration 10 | lm loss: 4.25 | grad norm: 0.81 | consumed samples: 320
    iteration 11 | lm loss: 4.01 | tokens/sec: 12345.6 | MFU: 0.42
    """
    parsed = parse_megatron_log(text)
    expected = {
        "expected_command_present": True,
        "iteration": 11,
        "lm_loss": 4.01,
        "tokens_per_sec": 12345.6,
        "mfu": 0.42,
    }
    checks = {key: parsed.get(key) == value for key, value in expected.items()}
    return {"passed": all(checks.values()), "checks": checks, "parsed": parsed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Megatron training log into structured metrics"
    )
    parser.add_argument("log_path", nargs="?", help="Path to Megatron train.log")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--require-field",
        action="append",
        default=[],
        help="Fail if parsed field is missing",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        payload = self_test()
    else:
        if not args.log_path:
            raise SystemExit("log_path is required unless --self-test is used")
        payload = parse_megatron_log(Path(args.log_path).read_text(encoding="utf-8"))

    missing_required = [
        field for field in args.require_field if payload.get(field) is None
    ]
    if missing_required:
        payload["required_fields_missing"] = missing_required

    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    if missing_required:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
