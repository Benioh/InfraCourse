from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


PATTERNS: dict[str, list[str]] = {
    "ttft_ms_p50": [rf"ttft_ms_p50[=: ]+({NUMBER})", rf"TTFT[^0-9]+({NUMBER})"],
    "itl_ms_p50": [rf"itl_ms_p50[=: ]+({NUMBER})", rf"ITL[^0-9]+({NUMBER})"],
    "requests_per_sec": [
        rf"requests_per_sec[=: ]+({NUMBER})",
        rf"Request throughput[^0-9]+({NUMBER})",
    ],
    "output_tokens_per_sec": [
        rf"output_tokens_per_sec[=: ]+({NUMBER})",
        rf"Output token throughput[^0-9]+({NUMBER})",
    ],
    "prompt_tokens_mean": [
        rf"prompt_tokens_mean[=: ]+({NUMBER})",
        rf"prompt tokens mean[=: ]+({NUMBER})",
    ],
    "output_tokens_mean": [
        rf"output_tokens_mean[=: ]+({NUMBER})",
        rf"output tokens mean[=: ]+({NUMBER})",
    ],
    "concurrency": [r"concurrency[=: ]+([0-9]+)", r"max_concurrency[=: ]+([0-9]+)"],
}


def _last_number(text: str, patterns: list[str]) -> float | int | None:
    for pattern in patterns:
        values = re.findall(pattern, text, flags=re.IGNORECASE)
        if values:
            value = values[-1]
            return int(value) if re.fullmatch(r"[-+]?\d+", value) else float(value)
    return None


def parse_vllm_log(text: str) -> dict[str, Any]:
    command = next(
        (
            line.split("command=", 1)[1]
            for line in text.splitlines()
            if "command=" in line
        ),
        None,
    )
    payload: dict[str, Any] = {
        "vllm_available": "vllm_available=True" in text,
        "server_port_open": "server_port_open=True" in text,
        "validation_only": "validation_only" in text
        or "server_port_open=False" in text,
        "command": command,
        "lines": len(text.splitlines()),
    }
    for field, patterns in PATTERNS.items():
        payload[field] = _last_number(text, patterns)
    workload_fields = [
        payload.get("prompt_tokens_mean"),
        payload.get("output_tokens_mean"),
        payload.get("concurrency"),
    ]
    if all(value is not None for value in workload_fields):
        fingerprint = json.dumps(workload_fields, sort_keys=True).encode()
        payload["workload_fingerprint"] = hashlib.sha256(fingerprint).hexdigest()[:12]
    else:
        payload["workload_fingerprint"] = None
    payload["has_latency_breakdown"] = (
        payload.get("ttft_ms_p50") is not None and payload.get("itl_ms_p50") is not None
    )
    payload["status"] = (
        "served_with_metrics"
        if payload["server_port_open"] and payload["has_latency_breakdown"]
        else "validation_or_health_only"
    )
    return payload


def self_test() -> dict[str, Any]:
    text = """
    vllm_available=True
    server_port_open=True
    command=vllm serve Qwen --port 31000
    prompt_tokens_mean=128 output_tokens_mean=64 concurrency=8
    ttft_ms_p50=123.4 itl_ms_p50=7.8 requests_per_sec=12.5 output_tokens_per_sec=800
    """
    parsed = parse_vllm_log(text)
    checks = {
        "server_port_open": parsed["server_port_open"] is True,
        "ttft": parsed["ttft_ms_p50"] == 123.4,
        "itl": parsed["itl_ms_p50"] == 7.8,
        "fingerprint": parsed["workload_fingerprint"] is not None,
    }
    return {"passed": all(checks.values()), "checks": checks, "parsed": parsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse vLLM serve/benchmark logs")
    parser.add_argument("log", nargs="?", default="serve.log")
    parser.add_argument("--output")
    parser.add_argument("--require-field", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    payload = (
        self_test()
        if args.self_test
        else parse_vllm_log(
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
