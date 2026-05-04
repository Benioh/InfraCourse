from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from typing import Any

from mini_infra.observability.io import (
    append_jsonl,
    command_snapshot,
    mini_run_dir,
    utc_now,
    write_json,
    write_text,
)
from mini_infra.serving.openai_server import complete


def workload_fingerprint(prompts: list[str], max_tokens: int, concurrency: int) -> str:
    payload = json.dumps(
        {"prompts": prompts, "max_tokens": max_tokens, "concurrency": concurrency},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def request_remote(base_url: str, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": "mini-infra-tiny-lm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    response = json.loads(urllib.request.urlopen(request, timeout=30).read().decode())
    return response["choices"][0]["message"]["content"]


def run_benchmark(
    prompts: list[str], max_tokens: int, concurrency: int, base_url: str | None = None
) -> dict[str, Any]:
    latencies = []
    outputs = []
    started = time.perf_counter()
    for prompt in prompts:
        request_started = time.perf_counter()
        output = (
            request_remote(base_url, prompt, max_tokens)
            if base_url
            else complete(prompt, max_tokens)
        )
        latencies.append((time.perf_counter() - request_started) * 1000)
        outputs.append(output)
    elapsed = max(time.perf_counter() - started, 1e-6)
    prompt_tokens = [len(prompt.split()) for prompt in prompts]
    output_tokens = [len(output.split()) for output in outputs]
    return {
        "request_count": len(prompts),
        "concurrency": concurrency,
        "prompt_tokens_mean": round(sum(prompt_tokens) / len(prompt_tokens), 3),
        "output_tokens_mean": round(sum(output_tokens) / len(output_tokens), 3),
        "ttft_ms_p50": round(sorted(latencies)[len(latencies) // 2], 3),
        "itl_ms_p50": round(
            (sum(latencies) / len(latencies))
            / max(sum(output_tokens) / len(output_tokens), 1),
            3,
        ),
        "e2e_ms_mean": round(sum(latencies) / len(latencies), 3),
        "requests_per_sec": round(len(prompts) / elapsed, 3),
        "workload_fingerprint": workload_fingerprint(prompts, max_tokens, concurrency),
        "mode": "remote_server" if base_url else "local_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra serving benchmark")
    parser.add_argument("--run-id")
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    run_dir = mini_run_dir("serving", args.run_id)
    command_snapshot(
        run_dir / "command.sh", ["python", "-m", "mini_infra.serving.benchmark"]
    )
    prompts = [
        "Alice has 3 apples and buys 4 more. Answer:",
        "There are 9 birds and 3 fly away. Answer:",
    ]
    row = run_benchmark(prompts, args.max_tokens, args.concurrency, args.base_url)
    append_jsonl(
        run_dir / "metrics.jsonl",
        {"timestamp": utc_now(), "metric_type": "serve", **row},
    )
    write_json(run_dir / "artifacts" / "benchmark.json", row)
    write_text(
        run_dir / "report.md",
        f"# MiniInfra Serving Benchmark\n\n{json.dumps(row, ensure_ascii=False, indent=2)}\n",
    )
    print(json.dumps({"run_dir": str(run_dir), **row}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
