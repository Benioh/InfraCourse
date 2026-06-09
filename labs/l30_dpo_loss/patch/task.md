# L33 Patch · DPO Loss and Completion Logprob

## 你要交付什么

实现两个函数：

```python
def compute_logps_for_completions(
    logits: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,  # [B, T]，prompt / pad 位置为 -100
) -> torch.Tensor:        # [B]

def dpo_loss(
    policy_logp_chosen: torch.Tensor,    # [B]
    policy_logp_rejected: torch.Tensor,  # [B]
    ref_logp_chosen: torch.Tensor,       # [B]
    ref_logp_rejected: torch.Tensor,     # [B]
    beta: float = 0.1,
) -> dict:
```

禁止使用 `trl.DPOTrainer` 或外部 DPO loss。允许使用 `torch.nn.functional.log_softmax`、`torch.gather` 和 `torch.nn.functional.logsigmoid`。

## 接口契约

### 1. `compute_logps_for_completions`

输入是 batch logits 和 labels。labels 中 `-100` 表示 prompt、padding 或其他不计入 loss 的位置。

实现顺序：

1. `mask = labels != -100`
2. 把 `-100` 临时替换成合法 token id，避免 `gather` 失败。
3. `log_probs = F.log_softmax(logits, dim=-1)`
4. `gather` 出 label token 的 logprob。
5. masked 位置乘 0，沿时间维求和，返回 `[B]`。

### 2. `dpo_loss`

输入已经是四组 sequence-level logprob：

```text
chosen_reward = beta * (policy_logp_chosen - ref_logp_chosen)
rejected_reward = beta * (policy_logp_rejected - ref_logp_rejected)
reward_margin = chosen_reward - rejected_reward
loss = -F.logsigmoid(reward_margin).mean()
```

返回字典必须包含：

```python
{
    "loss": loss,
    "reward_margin": reward_margin,
    "chosen_reward": chosen_reward,
    "rejected_reward": rejected_reward,
}
```

## 不变量

1. `compute_logps_for_completions` 输出 shape 是 `[B]`。
2. `labels == -100` 的位置不能贡献 logprob。
3. policy 等于 reference 时，DPO loss 为 `log(2)`。
4. `beta=0` 时，DPO loss 为 `log(2)`。
5. policy 相对 reference 更偏向 chosen 时，loss 下降。
6. policy 相对 reference 更偏向 rejected 时，loss 上升。
7. 大 beta 下 loss 仍应有限，使用 `F.logsigmoid`。

## 怎么验证

```bash
make patch-test M=l30_dpo_loss
```

9 个 CPU 测试：

| 测试 | 验证 |
|---|---|
| `test_compute_logps_shape` | logprob 输出 shape 为 `[B]` |
| `test_compute_logps_masks_prompt` | prompt mask 会移除负 logprob 贡献 |
| `test_compute_logps_matches_manual` | 与手写 `log_softmax + gather` 一致 |
| `test_dpo_zero_beta_returns_log2` | `beta=0` 时 loss 为 `log(2)` |
| `test_dpo_policy_equals_ref_returns_log2` | policy 等于 reference 时 loss 为 `log(2)` |
| `test_dpo_chosen_better_lowers_loss` | chosen 相对变好时 loss 下降 |
| `test_dpo_rejected_better_raises_loss` | rejected 相对变好时 loss 上升 |
| `test_dpo_reward_margin_formula` | reward margin 公式正确 |
| `test_dpo_numerically_stable_with_large_beta` | 大 beta 下 loss 为 finite |

## Smoke

```bash
IMPL=reference bash labs/l30_dpo_loss/scripts/run_dpo_smoke.sh l33_reference
```

smoke 会写入 `runs/l30_dpo_loss/<run_id>/metrics.jsonl`、`artifacts/dpo_smoke.json` 和 `report.md`。它只验证最小训练循环和 artifact，不验证真实偏好数据质量。

## 卡住怎么办

1. 先用一个 batch 手算 `labels == -100` 的位置是否被清零。
2. 再检查 reward margin 的符号：chosen 应在减去 reference 后高于 rejected。
3. 最后检查是否使用 `F.logsigmoid`，避免大 beta 下数值下溢。
