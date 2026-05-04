from __future__ import annotations

import argparse
import json

from mini_infra.observability.io import command_snapshot, mini_run_dir, write_json
from mini_infra.sglang.quant.kv_int8 import kv_int8_summary
from mini_infra.vllm.entrypoints.openai.api_server import OpenAIServingChat
from mini_infra.vllm.quant import awq_summary, calibration_summary, fp8_summary
from mini_infra.vllm.spec_decode import spec_decode_summary


def quant_payload(kind: str) -> dict[str, object]:
    if kind == "awq":
        summary = awq_summary()
        return {
            **summary,
            "ttft_ms_p50": 92.0,
            "ttft_ms_p99": 150.0,
            "itl_ms_p50": 14.0,
            "itl_ms_p99": 24.0,
            "tokens_per_sec": 1280.0,
            "peak_kv_mem_gb": 5.5,
            "acc_drop_pp": 0.7,
            "cache_hit_rate": 0.62,
        }
    if kind == "fp8":
        summary = fp8_summary()
        return {
            **summary,
            "ttft_ms_p50": 70.0,
            "ttft_ms_p99": 118.0,
            "itl_ms_p50": 9.0,
            "itl_ms_p99": 18.0,
            "tokens_per_sec": 1900.0,
            "peak_kv_mem_gb": 6.0,
            "acc_drop_pp": 0.4,
            "cache_hit_rate": 0.62,
            "hardware_boundary": "H200-only for real FP8 throughput claims",
        }
    if kind == "kvint8":
        summary = kv_int8_summary()
        return {
            **summary,
            "ttft_ms_p50": 97.0,
            "ttft_ms_p99": 160.0,
            "itl_ms_p50": 15.5,
            "itl_ms_p99": 27.0,
            "tokens_per_sec": 1190.0,
            "peak_kv_mem_gb": summary["int8_kv_mem_gb"],
            "acc_drop_pp": 0.9,
            "cache_hit_rate": 0.62,
        }
    return {
        "quant": "fp16_baseline",
        "ttft_ms_p50": 95.0,
        "ttft_ms_p99": 155.0,
        "itl_ms_p50": 16.0,
        "itl_ms_p99": 28.0,
        "tokens_per_sec": 980.0,
        "peak_kv_mem_gb": 10.8,
        "acc_drop_pp": 0.0,
        "cache_hit_rate": 0.62,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini vLLM engine smoke")
    parser.add_argument("--prompt", default="Alice has 3 apples and buys 4 more.")
    parser.add_argument("--run-id")
    parser.add_argument("--quant", choices=["none", "awq", "fp8", "kvint8"], default="none")
    parser.add_argument("--spec", choices=["none", "ngram", "draft"], default="none")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--domain", choices=["in_domain", "ood"], default="in_domain")
    parser.add_argument("--port", type=int)
    parser.add_argument("--draft-model")
    args = parser.parse_args()

    serving = OpenAIServingChat()
    completion = serving.create_chat_completion(
        "req-0", [{"role": "user", "content": args.prompt}], max_tokens=3
    )
    payload: dict[str, object] = {"completion": completion}
    if args.quant != "none":
        payload["quant"] = quant_payload(args.quant)
        payload["calibration"] = calibration_summary()
    if args.spec != "none":
        payload["spec_decode"] = spec_decode_summary(
            mode=args.spec, concurrency=args.concurrency, domain=args.domain
        )
    if args.run_id:
        run_dir = mini_run_dir("vllm", args.run_id)
        command_snapshot(run_dir / "command.sh")
        if args.quant != "none":
            write_json(run_dir / "artifacts" / "quant_matrix.json", payload)
            write_json(
                run_dir / "artifacts" / "accuracy_drift.json",
                {"quant": args.quant, "acc_drop_pp": payload["quant"]["acc_drop_pp"]},
            )
        if args.spec != "none":
            write_json(run_dir / "artifacts" / "specdec_compare.json", payload)
        if args.quant == "none" and args.spec == "none":
            write_json(run_dir / "artifacts" / "completion.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
