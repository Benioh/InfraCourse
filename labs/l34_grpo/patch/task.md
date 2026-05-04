# L11.8 Patch · GRPO / RLOO advantage + loss

## 你要交付什么

```python
def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: [G] within one prompt's group; return [G] mean-0 std-1 advantages."""

def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline: A_i = r_i - mean_{j != i} r_j."""

def grpo_loss(
    log_probs: torch.Tensor,         # [G, T] current policy log-probs
    log_probs_old: torch.Tensor,     # [G, T] behavior policy
    log_probs_ref: torch.Tensor,     # [G, T] reference (SFT) policy
    advantages: torch.Tensor,        # [G] per-response advantage
    mask: torch.Tensor,              # [G, T] 1 on completion tokens
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> dict:
    """Return {"loss": scalar, "policy_loss": scalar, "kl_loss": scalar,
              "ratio_mean": scalar, "ratio_clipped_frac": scalar}."""
```

补丁规模目标：60–120 行。

## 不变量

1. `grpo_advantage` 输出均值 ≈ 0，std ≈ 1（除非全相等，则全 0）
2. G=1 时 advantage 必须返回全 0，不能 NaN
3. `rloo_advantage(r)[i]` 不依赖 `r[i]`（数值上确认）
4. `grpo_loss` 必须屏蔽 mask=0 的 token
5. ratio = exp(log_probs - log_probs_old) 必须做 clip：min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
6. KL 项使用 `kl = exp(log_probs_ref - log_probs) - (log_probs_ref - log_probs) - 1`（k3 estimator）
7. 总 loss = -policy_loss_term + kl_beta * kl_loss

## 怎么验证

```bash
make patch-test M=l34_grpo
```

## 写完之后你能做什么

- 在 verl / TRL 上跑 GRPO 训练
- 解释 DeepSeek-R1 不用 critic 的训练流水线
- 给 capstone Stage C 用 GRPO 替代 vanilla PPO
