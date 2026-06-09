# L39 Patch · GRPO / RLOO Advantage + Loss

## 你要交付什么

实现三个函数：

```python
def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: [G] within one prompt group; return [G] mean-0 std-1 advantages."""

def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out baseline: A_i = r_i - mean_{j != i} r_j."""

def grpo_loss(
    log_probs: torch.Tensor,         # [G, T] current policy log-probs
    log_probs_old: torch.Tensor,     # [G, T] behavior policy
    log_probs_ref: torch.Tensor,     # [G, T] reference policy
    advantages: torch.Tensor,        # [G] per-response advantage
    mask: torch.Tensor,              # [G, T] 1 on completion tokens
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> dict:
    """Return loss, policy_loss, kl_loss, ratio_mean, ratio_clipped_frac."""
```

## 不变量

1. `grpo_advantage` 对同一 prompt 的 G 个 reward 做组内归一化。
2. G=1 时 advantage 必须返回全 0，不能产生 NaN。
3. reward 全相等时 advantage 返回全 0。
4. `rloo_advantage(r)[i] = r[i] - mean(r[j] for j != i)`。
5. `ratio = exp(log_probs - log_probs_old)`。
6. policy loss 使用 `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`。
7. mask=0 的 token 不参与 policy loss、KL、ratio_mean 或 clipped fraction。
8. KL 使用 `exp(log_probs_ref - log_probs) - (log_probs_ref - log_probs) - 1`。
9. 总 loss = policy_loss + `kl_beta * kl_loss`。

## 怎么验证

```bash
make patch-test M=l34_grpo
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l34_grpo
IMPL=reference python labs/l34_grpo/scripts/run_grpo_smoke.py --run-id l39_local
```

## 写完之后你能做什么

- 解释 critic-free RL 中 advantage 从哪里来。
- 判断 GRPO loss 中 ratio clip、reference KL 和 completion mask 分别控制什么。
- 给真实 RL 复盘补上 reward 分布、advantage 分布、KL、ratio 和 mask 证据。
