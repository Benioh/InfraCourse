"""Reference solution for L33 Patch."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def compute_logps_for_completions(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = (labels != -100).to(logits.dtype)
    safe_labels = labels.clone()
    safe_labels[labels == -100] = 0
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


def dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> dict:
    chosen_reward = beta * (policy_logp_chosen - ref_logp_chosen)
    rejected_reward = beta * (policy_logp_rejected - ref_logp_rejected)
    reward_margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(reward_margin).mean()
    return {
        "loss": loss,
        "reward_margin": reward_margin,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
    }
