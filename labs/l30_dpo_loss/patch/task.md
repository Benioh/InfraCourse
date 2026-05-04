# L10.3 Patch · DPO Loss & Per-Sequence Log-prob

## 你要交付什么

```python
def compute_logps_for_completions(
    logits: torch.Tensor,           # [B, T, V]
    labels: torch.Tensor,           # [B, T]，prompt 用 -100 屏蔽
) -> torch.Tensor:                  # [B] 每条样本 completion log-prob 之和

def dpo_loss(
    policy_logp_chosen: torch.Tensor,    # [B]
    policy_logp_rejected: torch.Tensor,  # [B]
    ref_logp_chosen: torch.Tensor,       # [B]
    ref_logp_rejected: torch.Tensor,     # [B]
    beta: float = 0.1,
) -> dict:                              # {"loss": [], "reward_margin": [B], "chosen_reward": [B], "rejected_reward": [B]}
```

补丁规模目标：40–80 行。

**禁止使用** `trl.DPOTrainer` 或 `trl.dpo_loss`，本关就是让你自己写。

**允许使用** `torch.nn.functional.log_softmax`、`torch.gather`、`torch.nn.functional.logsigmoid`。

## 不变量

1. `compute_logps_for_completions` 必须 mask 掉 `labels==-100` 的位置（prompt + pad）。
2. logp 在 completion 上是**和**，不是平均。
3. DPO loss 必须用 `F.logsigmoid` 保证数值稳定。
4. 当 policy_logp_chosen == ref_logp_chosen 且 policy_logp_rejected == ref_logp_rejected 时，
   reward_margin == 0，loss == log(2) ≈ 0.693。
5. 当 β = 0 时 loss == log(2)。
6. `chosen_reward = β·(policy_logp_chosen - ref_logp_chosen)`；`rejected_reward` 同理。
7. `reward_margin = chosen_reward - rejected_reward`。

## DPO 数学

```
margin = β·((log π_θ(yw|x) - log π_ref(yw|x)) - (log π_θ(yl|x) - log π_ref(yl|x)))
loss   = -E[log σ(margin)] = E[-logsigmoid(margin)]
```

## 怎么验证

```bash
make patch-test M=l30_dpo_loss
```

## 写完之后你能做什么

- 看懂 TRL `DPOTrainer.dpo_loss`
- 自己组合 SFT → DPO → DPO-iterative 流程
- 在 RL 段对比 PPO / GRPO / DPO 的工程取舍
