"""L11.8 · 在合成 reward + log-prob 上跑 GRPO 50 步，看 loss 下降。"""

from __future__ import annotations

import argparse
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

MISSION_ID = "l34_grpo"


def _impl():
    try:
        from starter import grpo as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import grpo as mod  # type: ignore[import-not-found]

        return mod, "reference"


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

    if "verl_command" in config:
        write_text(
            run_dir / "artifacts" / "verl_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["verl_command"].split())
            + "\n",
        )

    try:
        import torch
        from torch import nn
    except ImportError:
        write_text(run_dir / "artifacts" / "fallback.txt", "torch missing\n")
        write_text(run_dir / "report.md", f"# {MISSION_ID}\nfallback\n")
        print(run_dir)
        return

    torch.manual_seed(int(config.get("seed", 0)))
    G = int(config["group_size"])
    P = int(config["prompts"])
    T = int(config["seq_length"])

    # tiny "policy" parameter we will train: per-token logit shift
    policy = nn.Parameter(torch.zeros(T))
    optimizer = torch.optim.SGD([policy], lr=1e-2)

    # latent "good direction" — reward favors larger sum(log_probs along this direction)
    direction = torch.randn(T)
    direction = direction / direction.norm()

    losses: list[float] = []
    for step in range(50):
        # sample G responses per prompt, simulate log-probs
        log_probs_old = torch.randn(G, T) * 0.1
        log_probs = log_probs_old + policy.detach()
        log_probs_ref = log_probs_old.clone()
        # reward = projection onto direction
        rewards = (log_probs * direction).sum(dim=-1)
        advantages = mod.grpo_advantage(rewards)
        mask = torch.ones(G, T)
        # forward with grad through policy
        log_probs_train = log_probs_old + policy
        out = mod.grpo_loss(
            log_probs_train,
            log_probs_old,
            log_probs_ref,
            advantages,
            mask,
            clip_eps=float(config["clip_eps"]),
            kl_beta=float(config["kl_beta"]),
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "grpo",
                "step": step,
                "loss": losses[-1],
                "policy_loss": float(out["policy_loss"].detach()),
                "kl_loss": float(out["kl_loss"].detach()),
                "ratio_mean": float(out["ratio_mean"].detach()),
                "ratio_clipped_frac": float(out["ratio_clipped_frac"].detach()),
            },
        )

    drop = losses[0] - sum(losses[-5:]) / 5
    accept = drop >= float(config.get("acceptance", {}).get("loss_drop_first_50_steps", 0.0))
    write_json(
        run_dir / "artifacts" / "grpo_smoke.json",
        {"head_loss": losses[0], "tail_loss": losses[-1], "drop": drop, "accept": accept},
    )
    write_text(
        run_dir / "report.md",
        f"# {MISSION_ID}\nimpl={impl_label}; head={losses[0]:.4f}; tail={losses[-1]:.4f}; drop={drop:.4f}; accept={accept}\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
