from __future__ import annotations

import argparse
import json
from typing import Any


def estimate(
    tp: int, pp: int, dp: int, seq_len: int, micro_batch: int, recompute: bool
) -> dict[str, Any]:
    activation_gb = (
        micro_batch * seq_len * 0.00018 / max(pp, 1) * (0.45 if recompute else 1.0)
    )
    model_gb = 2.2 / max(tp, 1)
    optimizer_gb = 6.5 / max(dp, 1)
    communication_penalty = 1.0 + 0.07 * max(tp - 1, 0) + 0.04 * max(pp - 1, 0)
    recompute_penalty = 1.18 if recompute else 1.0
    tokens_per_sec = 16000 * tp * pp * dp / communication_penalty / recompute_penalty
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "recompute": recompute,
        "estimated_peak_memory_gb": round(model_gb + optimizer_gb + activation_gb, 3),
        "estimated_tokens_per_sec": round(tokens_per_sec, 2),
        "risk": "checkpoint_topology_sensitive" if tp * pp > 1 else "single_partition",
    }


def sweep(gpus: int, seq_len: int, micro_batch: int) -> list[dict[str, Any]]:
    rows = []
    for tp in [1, 2, 4]:
        for pp in [1, 2]:
            if gpus % (tp * pp) != 0:
                continue
            dp = max(gpus // (tp * pp), 1)
            for recompute in [False, True]:
                rows.append(estimate(tp, pp, dp, seq_len, micro_batch, recompute))
    return sorted(
        rows,
        key=lambda item: (
            item["estimated_peak_memory_gb"],
            -item["estimated_tokens_per_sec"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniInfra scaling planner")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            sweep(args.gpus, args.seq_len, args.micro_batch),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
