from __future__ import annotations

import argparse
import json


def estimate(model_size_b: float, seq_len: int, micro_batch: int, tp: int, recompute: bool) -> dict:
    param_gb = model_size_b * 2.0 / max(tp, 1)
    activation_pressure = seq_len * micro_batch / 1024
    if recompute:
        activation_pressure *= 0.65
    return {
        "estimated_parameter_memory_gb": round(param_gb, 2),
        "estimated_optimizer_memory_gb": round(param_gb * 2, 2),
        "estimated_activation_pressure": round(activation_pressure, 2),
        "communication_risk": "high" if tp >= 4 else "medium" if tp >= 2 else "low",
        "4090_feasibility": activation_pressure < 12 and param_gb < 20,
        "h200_feasibility": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size-b", type=float, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(estimate(args.model_size_b, args.seq_len, args.micro_batch, args.tp, args.recompute), indent=2))


if __name__ == "__main__":
    main()
