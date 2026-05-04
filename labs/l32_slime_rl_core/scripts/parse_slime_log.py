from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_slime_log(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config_validation": "配置验证完成" in text or "config_valid" in text,
        "lines": len(text.splitlines()),
    }
    json_payload = re.search(r"SLiME 配置验证完成：(\{.*\})", text)
    if json_payload:
        try:
            payload.update(json.loads(json_payload.group(1)))
        except json.JSONDecodeError:
            payload["config_json_error"] = True
    for field, pattern in {
        "rollout_time_sec": rf"rollout_time_sec[=:： ]+({NUMBER})",
        "actor_update_time_sec": rf"actor_update_time_sec[=:： ]+({NUMBER})",
        "weight_sync_time_sec": rf"weight_sync_time_sec[=:： ]+({NUMBER})",
        "sglang_generation_tokens_per_sec": rf"sglang_generation_tokens_per_sec[=:： ]+({NUMBER})",
    }.items():
        matches = re.findall(pattern, text, flags=re.I)
        payload[field] = float(matches[-1]) if matches else None
    allocation = re.search(r"(\d+)actor_(\d+)rollout", text)
    if allocation:
        payload["actor_gpus"] = int(allocation.group(1))
        payload["rollout_gpus"] = int(allocation.group(2))
    payload["has_weight_sync_evidence"] = (
        payload.get("weight_sync_time_sec") is not None or "weight_sync" in text
    )
    payload["status"] = (
        "parsed_slime_metrics" if payload["has_weight_sync_evidence"] else "config_only"
    )
    return payload


def self_test() -> dict[str, Any]:
    text = '[2026] SLiME 配置验证完成：{"config_valid": true, "actor_gpus": 4, "rollout_gpus": 4}\nweight_sync_time_sec=0.9 rollout_time_sec=2.25 gpu_allocation=4actor_4rollout\n'
    parsed = parse_slime_log(text)
    checks = {
        "config": parsed.get("config_valid") is True,
        "actor": parsed.get("actor_gpus") == 4,
        "sync": parsed.get("weight_sync_time_sec") == 0.9,
        "status": parsed.get("status") == "parsed_slime_metrics",
    }
    return {"passed": all(checks.values()), "checks": checks, "parsed": parsed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SLiME RL logs for actor/rollout evidence"
    )
    parser.add_argument("log", nargs="?", default="rl.log")
    parser.add_argument("--output")
    parser.add_argument("--require-field", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    payload = (
        self_test()
        if args.self_test
        else parse_slime_log(
            Path(args.log).read_text(encoding="utf-8")
            if Path(args.log).exists()
            else ""
        )
    )
    missing = [field for field in args.require_field if payload.get(field) is None]
    if missing:
        payload["required_fields_missing"] = missing
    output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
