"""L33 Patch · DPO loss + completion log-prob extraction."""

from __future__ import annotations

import torch


def compute_logps_for_completions(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log-probs of completion tokens only.

    Args:
        logits: [B, T, V]
        labels: [B, T] with prompt / pad positions set to -100.

    Returns:
        Tensor of shape [B] giving sum of log π(label_t | ...).
    """
    # TODO(student): build mask = (labels != -100)
    # TODO(student): replace -100 in labels with 0 (or any valid id) so gather doesn't blow up
    # TODO(student): logp = F.log_softmax(logits, dim=-1)
    # TODO(student): gather along V dim using labels.unsqueeze(-1) -> [B, T, 1] -> [B, T]
    # TODO(student): zero-out masked positions, sum along T to produce [B]
    raise NotImplementedError("L33: implement compute_logps_for_completions")


def dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> dict:
    """Direct Preference Optimization loss.

    Returns ``{"loss": scalar, "reward_margin": [B], "chosen_reward": [B], "rejected_reward": [B]}``.
    """
    # TODO(student): compute chosen_reward = beta * (policy_logp_chosen - ref_logp_chosen)
    # TODO(student): compute rejected_reward = beta * (policy_logp_rejected - ref_logp_rejected)
    # TODO(student): reward_margin = chosen_reward - rejected_reward
    # TODO(student): loss = -F.logsigmoid(reward_margin).mean()
    # TODO(student): return dict with all four entries (loss as scalar)
    raise NotImplementedError("L33: implement dpo_loss")
