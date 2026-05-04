from __future__ import annotations

import argparse
import json
from typing import Any


def simulate_pd(
    prefill_gpus: int, decode_gpus: int, prompt_tokens: int, output_tokens: int
) -> dict[str, Any]:
    ttft = 40 + prompt_tokens * 0.08 / max(prefill_gpus, 1)
    itl = 3 + output_tokens * 0.02 / max(decode_gpus, 1)
    imbalance = abs(prefill_gpus - decode_gpus) / max(prefill_gpus + decode_gpus, 1)
    return {
        "prefill_gpus": prefill_gpus,
        "decode_gpus": decode_gpus,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms_estimate": round(ttft, 3),
        "itl_ms_estimate": round(itl, 3),
        "queue_risk": (
            "high" if imbalance > 0.5 else "medium" if imbalance > 0.25 else "low"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniInfra PD disaggregation simulator"
    )
    parser.add_argument("--prefill-gpus", type=int, default=2)
    parser.add_argument("--decode-gpus", type=int, default=6)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()
    print(
        json.dumps(
            simulate_pd(
                args.prefill_gpus,
                args.decode_gpus,
                args.prompt_tokens,
                args.output_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
