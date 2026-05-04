from __future__ import annotations

import argparse
import json


def estimate(prompt_tokens: int, output_tokens: int, concurrency: int, cache_hit_rate: float) -> dict:
    ttft_ms = 150 + prompt_tokens * 0.08 * (1.0 - cache_hit_rate * 0.5)
    itl_ms = 12 + output_tokens * 0.01 / max(concurrency, 1)
    return {
        "ttft_ms_p50_estimate": round(ttft_ms, 2),
        "itl_ms_p50_estimate": round(itl_ms, 2),
        "cache_benefit": "high" if cache_hit_rate > 0.5 else "medium" if cache_hit_rate > 0.2 else "low",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--cache-hit-rate", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(estimate(args.prompt_tokens, args.output_tokens, args.concurrency, args.cache_hit_rate), indent=2))


if __name__ == "__main__":
    main()
