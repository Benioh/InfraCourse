"""Tiny TTFT/ITL bench against an OpenAI-compatible vLLM endpoint.

Usage:
    python scripts/bench_ttft.py --base-url http://localhost:8000 \
        --model Qwen/Qwen2.5-0.5B-Instruct --concurrency 8 --total 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib import request

LAB_DIR = Path(__file__).resolve().parents[1]


def _send(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    sent = time.perf_counter()
    first = None
    completed = None
    tokens = 0
    with request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                completed = time.perf_counter()
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if first is None:
                first = time.perf_counter()
            for choice in payload.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    tokens += 1
    if first is None:
        first = time.perf_counter()
    if completed is None:
        completed = time.perf_counter()
    return {
        "ttft_s": first - sent,
        "total_s": completed - sent,
        "tokens": tokens,
        "itl_s": (completed - first) / max(1, tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--total", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="用一句话解释 paged attention 在 vLLM 里如何减少 KV 内存碎片。",
    )
    parser.add_argument("--out", default=None, help="optional jsonl path")
    args = parser.parse_args()

    samples: list[dict] = []
    lock = threading.Lock()

    def worker(_idx: int) -> None:
        try:
            sample = _send(args.base_url, args.model, args.prompt, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            sample = {"error": str(exc)}
        with lock:
            samples.append(sample)

    threads: list[threading.Thread] = []
    for idx in range(args.total):
        thread = threading.Thread(target=worker, args=(idx,))
        thread.start()
        threads.append(thread)
        while sum(1 for thread in threads if thread.is_alive()) >= args.concurrency:
            time.sleep(0.01)
    for thread in threads:
        thread.join()

    ok_samples = [sample for sample in samples if "error" not in sample]
    if not ok_samples:
        print("[bench] all requests failed:")
        for sample in samples[:5]:
            print(" ", sample)
        sys.exit(2)
    ttfts = sorted(s["ttft_s"] for s in ok_samples)
    itls = sorted(s["itl_s"] for s in ok_samples)
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "n_ok": len(ok_samples),
        "n_total": args.total,
        "p50_ttft_ms": round(statistics.median(ttfts) * 1000, 2),
        "p95_ttft_ms": round(ttfts[int(len(ttfts) * 0.95)] * 1000, 2) if ttfts else None,
        "p50_itl_ms": round(statistics.median(itls) * 1000, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
