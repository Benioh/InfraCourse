"""L05.8.5 · Crash drill：baseline vs crash+resume，要求最终 loss 一致。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB_DIR / "patch"))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l16_resume_after_crash"


def _impl():
    try:
        from starter import crash_safe as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import crash_safe as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _train_steps(state: dict, grads: list[float]) -> list[float]:
    losses = []
    for g in grads:
        state["w"] = state["w"] - 0.1 * g
        state["m"] = 0.9 * state["m"] + 0.1 * g
        losses.append(state["w"] ** 2)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    mod, impl_label = _impl()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "impl": impl_label, **config},
    )

    rng = random.Random(int(config.get("seed", 0)))
    grads = [rng.uniform(-1.0, 1.0) for _ in range(int(config["total_steps"]))]
    crash_at = int(config["crash_at_step"])

    state_b = {"w": 5.0, "m": 0.0}
    losses_b = _train_steps(state_b, grads)

    state_c = {"w": 5.0, "m": 0.0}
    losses_c_pre = _train_steps(state_c, grads[:crash_at])
    ckpt_dir = run_dir / "artifacts" / "checkpoint"
    mod.save_step(
        ckpt_dir,
        step=crash_at,
        model_state={"w": state_c["w"]},
        optimizer_state={"m": state_c["m"]},
        rng_state={},
    )
    payload = mod.load_latest(ckpt_dir)
    state_r = {
        "w": payload["model_state"]["w"],
        "m": payload["optimizer_state"]["m"],
    }
    losses_c_post = _train_steps(state_r, grads[crash_at:])
    losses_c = losses_c_pre + losses_c_post

    rel_diffs = [
        abs(b - c) / max(abs(b), 1e-9) for b, c in zip(losses_b, losses_c, strict=True)
    ]
    max_rel_diff = max(rel_diffs)
    accept = max_rel_diff <= float(config.get("acceptance", {}).get("rel_loss_diff_max", 0.01))

    for step, (b, c) in enumerate(zip(losses_b, losses_c, strict=True)):
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "crash_drill",
                "step": step,
                "loss_baseline": b,
                "loss_resumed": c,
                "rel_diff": abs(b - c) / max(abs(b), 1e-9),
            },
        )

    write_json(
        run_dir / "artifacts" / "crash_drill.json",
        {
            "impl": impl_label,
            "max_rel_diff": max_rel_diff,
            "accept": accept,
            "crash_at": crash_at,
            "total_steps": int(config["total_steps"]),
        },
    )
    write_text(
        run_dir / "report.md",
        f"# {MISSION_ID}\nimpl={impl_label}; max_rel_diff={max_rel_diff:.3e}; accept={accept}\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
