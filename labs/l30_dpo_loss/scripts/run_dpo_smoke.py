"""L10.3 · 在合成数据上跑 DPO smoke：观察 loss 单调下降、reward_margin 拉开。"""

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

MISSION_ID = "l30_dpo_loss"


def _impl():
    try:
        from starter import dpo as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import dpo as mod  # type: ignore[import-not-found]

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

    try:
        import torch
        from torch import nn
    except ImportError:
        write_text(run_dir / "artifacts" / "fallback.txt", "torch missing\n")
        write_text(run_dir / "report.md", f"# {MISSION_ID}\nfallback: torch missing\n")
        print(run_dir)
        return

    if config.get("mode", "synthetic") != "synthetic":
        write_text(
            run_dir / "artifacts" / "fallback.txt",
            "real HF DPO requires transformers + datasets; see TRL DPOTrainer\n",
        )

    torch.manual_seed(int(config.get("seed", 0)))
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_dim"])
    seq = int(config["seq_length"])

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.body = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, ids):
            return self.head(torch.relu(self.body(self.embed(ids))))

    policy = TinyLM()
    ref = TinyLM()
    ref.load_state_dict(policy.state_dict())
    for param in ref.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    pairs = int(config["num_pairs"])
    chosen_ids = torch.randint(0, vocab, (pairs, seq))
    rejected_ids = torch.randint(0, vocab, (pairs, seq))
    labels_chosen = chosen_ids.clone()
    labels_chosen[:, : seq // 2] = -100
    labels_rejected = rejected_ids.clone()
    labels_rejected[:, : seq // 2] = -100

    losses: list[float] = []
    for step in range(50):
        idx = torch.randint(0, pairs, (8,))
        c_logits = policy(chosen_ids[idx])
        r_logits = policy(rejected_ids[idx])
        with torch.no_grad():
            rc_logits = ref(chosen_ids[idx])
            rr_logits = ref(rejected_ids[idx])
        c_logp = mod.compute_logps_for_completions(c_logits, labels_chosen[idx])
        r_logp = mod.compute_logps_for_completions(r_logits, labels_rejected[idx])
        rc_logp = mod.compute_logps_for_completions(rc_logits, labels_chosen[idx])
        rr_logp = mod.compute_logps_for_completions(rr_logits, labels_rejected[idx])
        out = mod.dpo_loss(c_logp, r_logp, rc_logp, rr_logp, beta=float(config["beta"]))
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(out["loss"].detach()))
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "dpo",
                "step": step,
                "loss": losses[-1],
                "reward_margin_mean": float(out["reward_margin"].mean().detach()),
            },
        )

    drop = losses[0] - sum(losses[-5:]) / 5
    accept = drop >= float(config.get("acceptance", {}).get("loss_drop_first_50_steps", 0.0))
    write_json(
        run_dir / "artifacts" / "dpo_smoke.json",
        {"head_loss": losses[0], "tail_loss": losses[-1], "drop": drop, "accept": accept},
    )
    write_text(
        run_dir / "report.md",
        f"# {MISSION_ID}\nimpl={impl_label}; head={losses[0]:.4f}; tail={losses[-1]:.4f}; drop={drop:.4f}; accept={accept}\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
